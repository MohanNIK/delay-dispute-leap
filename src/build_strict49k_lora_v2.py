# -*- coding: utf-8 -*-
"""Prepare strict-49k Mimo labels and LoRA v2 export artifacts.

This script does not train a model. It builds a labeling queue from the
Chinese delay usable pool and exports LoRA-ready train/dev files from the
machine-assisted labels that are already available.

Leakage rule:
    LoRA inputs use pre-decision-style factual text only. Decision basis spans
    and post-decision label evidence are kept in manifests but not inserted
    into training prompts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "data/1_raw_text/combined_delay_dispute_corpus_20260527"
STRICT49K_MANIFEST = CORPUS_DIR / "chinese_delay_usable_manifest.csv"
BATCH_DIR = CORPUS_DIR / "mimo_batch_outcome_labels"
FAST_DIR = CORPUS_DIR / "mimo_fast_strong_labels"
V2_LABEL_DIR = CORPUS_DIR / "mimo_batch_outcome_labels_v2"
FROZEN_V1_DIR = PROJECT_ROOT / "data/lora_exports/lora_frozen_v1_2384"
FROZEN_TEST = PROJECT_ROOT / "data/gold/candidate_gold_extended_v2.csv"
V2_EXPORT_ROOT = PROJECT_ROOT / "data/lora_exports"

VALID_EXPORT_LABELS = {"support", "partial_support", "not_support"}
INSTRUCTION = (
    "Based only on the pre-decision information, predict the outcome label of "
    "the delay-related construction claim. Output only one label from: support, "
    "partial_support, not_support."
)
SYSTEM = (
    "You are a construction schedule-delay dispute analysis assistant. "
    "Use only pre-decision information for prediction."
)


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower().replace("-", "_")
    mapping = {
        "partial": "partial_support",
        "partial support": "partial_support",
        "partially_support": "partial_support",
        "partially supported": "partial_support",
        "not support": "not_support",
        "not_supported": "not_support",
        "unsupported": "not_support",
        "reject": "not_support",
        "rejected": "not_support",
        "supported": "support",
    }
    return mapping.get(label, label)


def as_bool_false(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"0", "false", "no", "否", ""}


def resolve_path(value: Any) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_text(path_value: Any) -> str:
    path = resolve_path(path_value)
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def compact_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def split_pre_decision_like_text(raw_text: str, max_chars: int = 6500) -> Tuple[str, str]:
    """Return pre-decision-style factual text and split anchor.

    For downloaded public judgments we do not have the original structured
    pre/post split. This conservative heuristic keeps the document head and
    cuts before judicial reasoning / disposition anchors.
    """
    text = compact_ws(raw_text)
    anchors = [
        "本院认为",
        "法院认为",
        "二审认为",
        "再审认为",
        "原审法院认为",
        "裁判结果",
        "判决如下",
        "裁定如下",
        "依照《",
        "综上",
    ]
    candidates: List[Tuple[int, str]] = []
    for anchor in anchors:
        idx = text.find(anchor)
        if idx >= 600:
            candidates.append((idx, anchor))
    if candidates:
        idx, anchor = sorted(candidates, key=lambda x: x[0])[0]
        pre = text[:idx]
    else:
        anchor = "head_only_no_anchor"
        pre = text[:max_chars]
    if len(pre) > max_chars:
        # Keep document head plus delay/evidence sentences, but never use tail.
        head = pre[: int(max_chars * 0.45)]
        keywords = [
            "工期",
            "延期",
            "延误",
            "逾期",
            "停工",
            "窝工",
            "签证",
            "通知",
            "索赔",
            "进度",
            "施工日志",
            "会议纪要",
            "关键线路",
            "鉴定",
            "证据",
        ]
        spans: List[str] = []
        for sent in re.split(r"(?<=[。；！？])", pre):
            if any(k in sent for k in keywords):
                spans.append(sent.strip())
            if sum(len(x) for x in spans) >= int(max_chars * 0.50):
                break
        pre = (head + "\n[Delay and evidence-related factual spans]\n" + "\n".join(spans))[:max_chars]
    return pre.strip(), anchor


def evidence_summary_from_pre_text(pre_text: str, max_chars: int = 1200) -> Dict[str, str]:
    roles = {
        "ENT": ["合同", "约定", "工期", "开工", "竣工", "价款"],
        "NOT": ["通知", "函", "签证", "报审", "索赔", "联系单"],
        "CAU": ["延期", "延误", "停工", "窝工", "原因", "影响"],
        "IMP": ["进度", "关键线路", "工期顺延", "逾期", "节点"],
        "DOC": ["证据", "施工日志", "会议纪要", "鉴定", "记录", "资料"],
    }
    sentences = [s.strip() for s in re.split(r"(?<=[。；！？])", compact_ws(pre_text)) if s.strip()]
    out: Dict[str, str] = {}
    for role, kws in roles.items():
        picked = [s for s in sentences if any(k in s for k in kws)]
        out[role] = " ".join(picked[:3])[:max_chars] if picked else "Not explicitly extracted from pre-decision text."
    return out


def build_input(row: Dict[str, Any], max_chars: int = 6500, evidence_conditioned: bool = False) -> str:
    title = compact_ws(row.get("title", ""))[:300]
    pre_text = compact_ws(row.get("pre_decision_text", ""))[:max_chars]
    parts = [
        f"Case ID: {row.get('case_id', '')}",
        f"Title: {title}",
        "Pre-decision factual information:",
        pre_text,
    ]
    if evidence_conditioned:
        ev = evidence_summary_from_pre_text(pre_text)
        parts += [
            "Evidence-role summary extracted from pre-decision text:",
            f"ENT contractual/entitlement basis: {ev['ENT']}",
            f"NOT notice/substantiation/procedure: {ev['NOT']}",
            f"CAU causality and delay event: {ev['CAU']}",
            f"IMP schedule impact: {ev['IMP']}",
            f"DOC documentation integrity: {ev['DOC']}",
        ]
    return "\n".join(parts)


def alpaca_record(row: Dict[str, Any], evidence_conditioned: bool = False) -> Dict[str, str]:
    return {
        "instruction": INSTRUCTION,
        "input": build_input(row, evidence_conditioned=evidence_conditioned),
        "output": normalize_label(row.get("outcome_label")),
        "system": SYSTEM,
    }


def raw_record(row: Dict[str, Any], evidence_conditioned: bool = False) -> str:
    return (
        "### Instruction:\n"
        f"{INSTRUCTION}\n\n"
        "### Input:\n"
        f"{build_input(row, evidence_conditioned=evidence_conditioned)}\n\n"
        "### Response:\n"
        f"{normalize_label(row.get('outcome_label'))}"
    )


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_existing_label_keys(extra_label_dir: Path | None = None) -> Tuple[Set[str], Set[str], Set[str]]:
    paths = [
        FAST_DIR / "mimo_fast_label_results.csv",
        BATCH_DIR / "mimo_batch_label_results.csv",
        BATCH_DIR / "mimo_batch_lora_usable_labels.csv",
    ]
    if extra_label_dir:
        paths += [
            extra_label_dir / "mimo_batch_label_results.csv",
            extra_label_dir / "mimo_batch_lora_usable_labels.csv",
        ]
    pageids: Set[str] = set()
    case_ids: Set[str] = set()
    text_hashes: Set[str] = set()
    for path in paths:
        df = read_csv(path)
        if df.empty:
            continue
        if "pageid" in df.columns:
            pageids.update(df["pageid"].dropna().astype(str))
        if "case_id" in df.columns:
            case_ids.update(df["case_id"].dropna().astype(str))
        if "text_sha256" in df.columns:
            text_hashes.update(df["text_sha256"].dropna().astype(str))
    frozen = read_csv(FROZEN_TEST)
    if not frozen.empty and "case_id" in frozen.columns:
        case_ids.update(frozen["case_id"].dropna().astype(str))
    return pageids, case_ids, text_hashes


def build_label_queue(queue_path: Path, skip_report_path: Path, config_path: Path, v2_label_dir: Path) -> Dict[str, Any]:
    manifest = read_csv(STRICT49K_MANIFEST)
    if manifest.empty:
        raise FileNotFoundError(f"Missing strict49k manifest: {STRICT49K_MANIFEST}")
    pageids, case_ids, text_hashes = collect_existing_label_keys(v2_label_dir)
    work = manifest.copy()
    for col in ["pageid", "case_id", "text_sha256"]:
        if col in work.columns:
            work[col] = work[col].astype(str)
    reasons: List[str] = []
    keep_mask = pd.Series([True] * len(work))
    if "pageid" in work.columns:
        mask = work["pageid"].isin(pageids)
        reasons.append(f"exclude_existing_pageid={int(mask.sum())}")
        keep_mask &= ~mask
    if "case_id" in work.columns:
        mask = work["case_id"].isin(case_ids)
        reasons.append(f"exclude_existing_or_frozen_case_id={int(mask.sum())}")
        keep_mask &= ~mask
    if "text_sha256" in work.columns:
        mask = work["text_sha256"].isin(text_hashes)
        reasons.append(f"exclude_existing_text_hash={int(mask.sum())}")
        keep_mask &= ~mask
        before = int(keep_mask.sum())
        dup_mask = work.duplicated("text_sha256", keep="first")
        keep_mask &= ~dup_mask
        reasons.append(f"exclude_duplicate_text_hash_after_first={before - int(keep_mask.sum())}")
    if "text_chars" in work.columns:
        chars = pd.to_numeric(work["text_chars"], errors="coerce").fillna(0)
        mask = chars < 1500
        reasons.append(f"exclude_text_chars_lt_1500={int((keep_mask & mask).sum())}")
        keep_mask &= ~mask
    if "chinese_procedural_only_hits" in work.columns:
        proc = pd.to_numeric(work["chinese_procedural_only_hits"], errors="coerce").fillna(0)
        # Keep substantive cases; a small number of procedural terms is allowed.
        mask = proc >= 4
        reasons.append(f"exclude_high_procedural_only_hits={int((keep_mask & mask).sum())}")
        keep_mask &= ~mask
    queue = work[keep_mask].copy()
    if "usable_tier" not in queue.columns:
        queue["usable_tier"] = "chinese_delay_usable_strict49k"
    sort_cols = [c for c in ["chinese_delay_hits", "chinese_evidence_hits", "chinese_decision_hits", "text_chars"] if c in queue.columns]
    for col in sort_cols:
        queue[col] = pd.to_numeric(queue[col], errors="coerce").fillna(0)
    if sort_cols:
        queue = queue.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(queue_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"metric": ["source_rows", "queue_rows"] + [r.split("=")[0] for r in reasons], "value": [len(manifest), len(queue)] + [r.split("=")[1] for r in reasons]}).to_csv(skip_report_path, index=False, encoding="utf-8-sig")
    cfg = {
        "input": {
            "combined_usable_manifest": str(queue_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "exclude_fast_label_results": "data/1_raw_text/combined_delay_dispute_corpus_20260527/nonexistent_exclude.csv",
        },
        "labeling": {
            "batch_size": 3,
            "max_chars_per_case": 2300,
            "min_strong_confidence": 0.8,
            "min_usable_confidence": 0.68,
            "min_pre_decision_sufficiency": 0.5,
            "min_usable_pre_decision_sufficiency": 0.35,
            "save_every_batches": 10,
        },
        "mimo": {
            "base_url_env": "MIMO_OPENAI_BASE_URL",
            "api_key_env": "MIMO_API_KEY",
            "model_name": "mimo-v2.5-pro",
            "temperature": 0.0,
            "max_tokens": 2200,
            "timeout_sec": 180,
            "retries": 6,
            "workers": 4,
            "enabled": True,
            "disable_thinking": True,
            "rate_limit_sleep_sec": 45,
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"source_rows": int(len(manifest)), "queue_rows": int(len(queue)), "config": str(config_path), "queue": str(queue_path)}


def rows_from_frozen_v1() -> pd.DataFrame:
    path = FROZEN_V1_DIR / "strong_label_master_v1_2384.csv"
    df = read_csv(path)
    if df.empty:
        return df
    out = pd.DataFrame()
    out["case_id"] = df["case_id"].astype(str)
    out["pageid"] = ""
    out["title"] = df.get("source_file", pd.Series([""] * len(df))).astype(str)
    out["raw_text_path"] = df.get("source_file", pd.Series([""] * len(df))).astype(str)
    out["text_sha256"] = df.get("text_hash", pd.Series([""] * len(df))).astype(str)
    out["outcome_label"] = df["outcome_label"].map(normalize_label)
    out["label_confidence"] = pd.to_numeric(df.get("label_confidence", 0.0), errors="coerce").fillna(0.0)
    out["label_model"] = df.get("label_model", "frozen_v1")
    out["label_source"] = "frozen_v1_2384"
    out["pre_decision_text"] = df.get("pre_decision_text", pd.Series([""] * len(df))).astype(str)
    out["pre_decision_chars"] = out["pre_decision_text"].str.len()
    out["split_anchor"] = "structured_pre_decision_text"
    out["decision_basis_span"] = df.get("decision_basis_span", "")
    out["needs_review"] = df.get("needs_review", 0)
    out["conflict_flag"] = df.get("conflict_flag", 0)
    return out


def rows_from_mimo_labels(paths: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        df = read_csv(path)
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "lora_usable_label_flag" in df.columns:
        df = df[pd.to_numeric(df["lora_usable_label_flag"], errors="coerce").fillna(0).astype(int).eq(1)].copy()
    if "label_status" in df.columns:
        df = df[df["label_status"].astype(str).eq("ok")].copy()
    df["outcome_label"] = df["outcome_label"].map(normalize_label)
    df = df[df["outcome_label"].isin(VALID_EXPORT_LABELS)].copy()
    if "conflict_flag" in df.columns:
        df = df[pd.to_numeric(df["conflict_flag"], errors="coerce").fillna(1).astype(int).eq(0)].copy()
    if "needs_review" in df.columns:
        df = df[pd.to_numeric(df["needs_review"], errors="coerce").fillna(1).astype(int).eq(0)].copy()
    records: List[Dict[str, Any]] = []
    for rec in df.to_dict("records"):
        raw = load_text(rec.get("raw_text_path", ""))
        pre, anchor = split_pre_decision_like_text(raw)
        records.append(
            {
                "case_id": str(rec.get("case_id", "")),
                "pageid": str(rec.get("pageid", "")),
                "title": rec.get("title", ""),
                "raw_text_path": rec.get("raw_text_path", ""),
                "text_sha256": rec.get("text_sha256", sha256_text(raw)),
                "outcome_label": rec.get("outcome_label"),
                "label_confidence": rec.get("label_confidence", 0.0),
                "label_model": rec.get("model_name", "mimo"),
                "label_source": "mimo_batch_or_fast",
                "pre_decision_text": pre,
                "pre_decision_chars": len(pre),
                "split_anchor": anchor,
                "decision_basis_span": rec.get("decision_basis_span", ""),
                "needs_review": rec.get("needs_review", 0),
                "conflict_flag": rec.get("conflict_flag", 0),
            }
        )
    return pd.DataFrame(records)


def build_master(v2_label_dir: Path) -> pd.DataFrame:
    frames = [rows_from_frozen_v1()]
    label_paths = [
        FAST_DIR / "mimo_fast_lora_usable_labels.csv",
        BATCH_DIR / "mimo_batch_lora_usable_labels.csv",
        v2_label_dir / "mimo_batch_lora_usable_labels.csv",
    ]
    frames.append(rows_from_mimo_labels(label_paths))
    master = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if master.empty:
        return master
    master["outcome_label"] = master["outcome_label"].map(normalize_label)
    master = master[master["outcome_label"].isin(VALID_EXPORT_LABELS)].copy()
    master["pre_decision_chars"] = pd.to_numeric(master["pre_decision_chars"], errors="coerce").fillna(0).astype(int)
    master = master[master["pre_decision_chars"] >= 500].copy()
    master["text_key"] = master["text_sha256"].fillna("").astype(str)
    master.loc[master["text_key"].eq(""), "text_key"] = master.loc[master["text_key"].eq(""), "pre_decision_text"].map(sha256_text)
    master = master.sort_values(["label_confidence", "pre_decision_chars"], ascending=[False, False])
    master = master.drop_duplicates("text_key", keep="first")
    master = master.drop_duplicates("case_id", keep="first")
    master = master.drop(columns=["text_key"], errors="ignore")
    return master.reset_index(drop=True)


def stratified_split(master: pd.DataFrame, dev_ratio: float = 0.12, seed: int = 2026) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    train_parts: List[pd.DataFrame] = []
    dev_parts: List[pd.DataFrame] = []
    for _, group in master.groupby("outcome_label"):
        idx = list(group.index)
        rng.shuffle(idx)
        n_dev = max(1, round(len(idx) * dev_ratio)) if len(idx) >= 10 else max(0, round(len(idx) * dev_ratio))
        dev_idx = set(idx[:n_dev])
        dev_parts.append(group.loc[list(dev_idx)])
        train_parts.append(group.loc[[i for i in idx if i not in dev_idx]])
    train = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    dev = pd.concat(dev_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, dev


def export_package(master: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train, dev = stratified_split(master)
    master.to_csv(out_dir / "strong_label_master_v2.csv", index=False, encoding="utf-8-sig")
    train.to_csv(out_dir / "lora_train_manifest.csv", index=False, encoding="utf-8-sig")
    dev.to_csv(out_dir / "lora_dev_manifest.csv", index=False, encoding="utf-8-sig")

    write_jsonl([alpaca_record(r) for r in train.to_dict("records")], out_dir / "lora_train_alpaca.jsonl")
    write_jsonl([alpaca_record(r) for r in dev.to_dict("records")], out_dir / "lora_dev_alpaca.jsonl")
    (out_dir / "lora_train_raw.txt").write_text("\n\n".join(raw_record(r) for r in train.to_dict("records")), encoding="utf-8")
    (out_dir / "lora_dev_raw.txt").write_text("\n\n".join(raw_record(r) for r in dev.to_dict("records")), encoding="utf-8")
    write_jsonl([alpaca_record(r, evidence_conditioned=True) for r in train.to_dict("records")], out_dir / "lora_train_evidence_conditioned_alpaca.jsonl")
    write_jsonl([alpaca_record(r, evidence_conditioned=True) for r in dev.to_dict("records")], out_dir / "lora_dev_evidence_conditioned_alpaca.jsonl")

    input_only = [
        {"case_id": r["case_id"], "instruction": INSTRUCTION, "input": build_input(r), "system": SYSTEM}
        for r in master.to_dict("records")
    ]
    write_jsonl(input_only, out_dir / "usable_cases_input_only.jsonl")

    dist_rows: List[Dict[str, Any]] = []
    for split, df in [("master", master), ("train", train), ("dev", dev)]:
        total = len(df)
        for label in sorted(VALID_EXPORT_LABELS):
            count = int((df["outcome_label"] == label).sum())
            dist_rows.append({"split": split, "label": label, "count": count, "ratio": count / total if total else 0.0})
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(out_dir / "label_distribution.csv", index=False, encoding="utf-8-sig")

    summary_rows = [
        {"metric": "master_total", "value": len(master)},
        {"metric": "train_total", "value": len(train)},
        {"metric": "dev_total", "value": len(dev)},
        {"metric": "source_frozen_v1_2384", "value": int(master["label_source"].astype(str).eq("frozen_v1_2384").sum())},
        {"metric": "source_mimo_batch_or_fast", "value": int(master["label_source"].astype(str).eq("mimo_batch_or_fast").sum())},
        {"metric": "min_pre_decision_chars", "value": int(master["pre_decision_chars"].min()) if not master.empty else 0},
        {"metric": "median_pre_decision_chars", "value": float(master["pre_decision_chars"].median()) if not master.empty else 0.0},
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "strong_label_quality_report.csv", index=False, encoding="utf-8-sig")

    readme = f"""# DelayDispute Copilot LoRA v2 data package

Purpose: supervised fine-tuning for one-label outcome prediction.

Primary files for training:
- `lora_train_alpaca.jsonl`
- `lora_dev_alpaca.jsonl`

Alternative format:
- `lora_train_raw.txt`
- `lora_dev_raw.txt`

Optional evidence-conditioned experiment:
- `lora_train_evidence_conditioned_alpaca.jsonl`
- `lora_dev_evidence_conditioned_alpaca.jsonl`

Leakage discipline:
- Training inputs use pre-decision-style factual information only.
- Post-decision/adjudicated reasoning is used for label extraction only and is not included in LoRA prompts.
- Frozen test labels are not included in this external zip.

Current package counts:
- master: {len(master)}
- train: {len(train)}
- dev: {len(dev)}

Labels are machine-assisted labels prepared for fine-tuning, not human-gold annotations.
"""
    (out_dir / "lora_data_readme.md").write_text(readme, encoding="utf-8")

    with pd.ExcelWriter(out_dir / "core_dataset_summary.xlsx") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        dist.to_excel(writer, sheet_name="label_distribution", index=False)
        master[["case_id", "pageid", "title", "outcome_label", "label_confidence", "label_source", "pre_decision_chars", "split_anchor"]].to_excel(writer, sheet_name="master_preview", index=False)

    manifest_rows = []
    for path in sorted(out_dir.glob("*")):
        if path.is_file() and path.name != "dataset_manifest.csv":
            manifest_rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(manifest_rows).to_csv(out_dir / "dataset_manifest.csv", index=False, encoding="utf-8-sig")

    zip_path = out_dir / "data_package_for_external_lora_finetuning_v2.zip"
    include = [
        "lora_train_alpaca.jsonl",
        "lora_dev_alpaca.jsonl",
        "lora_train_raw.txt",
        "lora_dev_raw.txt",
        "lora_train_evidence_conditioned_alpaca.jsonl",
        "lora_dev_evidence_conditioned_alpaca.jsonl",
        "lora_data_readme.md",
        "label_distribution.csv",
        "strong_label_quality_report.csv",
        "dataset_manifest.csv",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in include:
            path = out_dir / name
            if path.exists():
                zf.write(path, arcname=name)

    # Refresh manifest with the zip hash included.
    manifest_rows = []
    for path in sorted(out_dir.glob("*")):
        if path.is_file() and path.name != "dataset_manifest.csv":
            manifest_rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(manifest_rows).to_csv(out_dir / "dataset_manifest.csv", index=False, encoding="utf-8-sig")

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "master_total": int(len(master)),
        "train_total": int(len(train)),
        "dev_total": int(len(dev)),
        "zip_path": str(zip_path),
        "labels": sorted(VALID_EXPORT_LABELS),
        "input_rule": "pre_decision_style_text_only",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_manifest


def write_archive_index(archive_dir: Path) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(CORPUS_DIR.glob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md"}:
            rows.append({"file": str(path.relative_to(PROJECT_ROOT)), "bytes": path.stat().st_size, "last_write": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")})
    pd.DataFrame(rows).to_csv(archive_dir / "archive_index.csv", index=False, encoding="utf-8-sig")
    (archive_dir / "README.md").write_text(
        "This directory indexes intermediate Mimo collection and screening files. "
        "Files were not deleted; the index records what should be treated as intermediate rather than external LoRA delivery artifacts.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-queue", action="store_true")
    parser.add_argument("--export-current", action="store_true")
    parser.add_argument("--archive-index", action="store_true")
    parser.add_argument("--queue-path", default=str(V2_LABEL_DIR / "strict49k_label_queue.csv"))
    parser.add_argument("--skip-report-path", default=str(V2_LABEL_DIR / "strict49k_queue_skip_report.csv"))
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "config/research_mimo_batch_outcome_labeling_strict49k_v2.json"))
    parser.add_argument("--label-dir", default=str(V2_LABEL_DIR))
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    v2_label_dir = resolve_path(args.label_dir)
    summary: Dict[str, Any] = {}
    if args.build_queue:
        summary["queue"] = build_label_queue(resolve_path(args.queue_path), resolve_path(args.skip_report_path), resolve_path(args.config_path), v2_label_dir)
    if args.export_current:
        out_dir = resolve_path(args.out_dir) if args.out_dir else V2_EXPORT_ROOT / f"lora_strict49k_mimo_v2_current_{stamp()}"
        master = build_master(v2_label_dir)
        summary["export"] = export_package(master, out_dir)
    if args.archive_index:
        archive_dir = PROJECT_ROOT / f"data/archive/mimo_collection_{stamp()}"
        write_archive_index(archive_dir)
        summary["archive_index"] = str(archive_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
