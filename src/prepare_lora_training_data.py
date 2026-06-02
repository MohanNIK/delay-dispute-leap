# -*- coding: utf-8 -*-
"""Prepare leakage-safe LoRA training packages for delay-dispute outcome labels.

The script exports training/dev data for a roommate-managed LoRA pipeline. It
does not train LoRA locally. Post-decision text may be used only for machine
assisted label generation; exported model input is pre-decision-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_candidate_gold_v2_qwen import (  # noqa: E402
    build_case_text,
    compact_text,
    extract_json_object,
    normalize_label_v2,
    sha256_text,
)


INTERNAL_LABELS = ["support", "partial", "not_support"]
EXPORT_LABEL_MAP = {"support": "support", "partial": "partial_support", "not_support": "not_support"}
INSTRUCTION = (
    "Based only on the pre-decision information, predict the outcome label of "
    "the delay-related construction claim. Output only one label from: support, "
    "partial_support, not_support."
)
SYSTEM = "You are a construction schedule-delay dispute analysis assistant. Use only pre-decision information for prediction."

DEFAULT_CFG: Dict[str, Any] = {
    "paths": {
        "structured_index": "data/meta/structured_case_index.csv",
        "structured_case_dir": "data/3_structured_cases",
        "raw_docx_dir": "data/0_raw_docx",
        "test_label_file": "data/gold/candidate_gold_extended_v2.csv",
        "existing_train_label_records": "results/train1000_augmented_precision_20260521_153425/train_label_records.csv",
        "output_root": "data/lora_exports",
    },
    "quality": {
        "min_confidence": 0.80,
        "min_pre_chars": 600,
        "require_needs_review_zero": True,
        "require_api_available": True,
        "exclude_potential_leakage": True,
        "train_ratio": 0.90,
        "max_input_chars": 6500,
        "seed": 2026,
    },
    "labeling": {
        "provider": "openai",
        "model_name": "gpt-5.5",
        "openai_base_url": "https://api.openai.com/v1",
        "openai_api_key_env": "OPENAI_API_KEY",
        "dashscope_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "dashscope_api_key_env": "DASHSCOPE_API_KEY",
        "dashscope_model_name": "qwen-max",
        "temperature": 0.0,
        "timeout": 120,
        "retries": 1,
        "workers": 4,
        "max_label_cases": 0,
        "max_chars_pre": 4500,
        "max_chars_post": 5500,
    },
}


def deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(base, ensure_ascii=False))
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DEFAULT_CFG, ensure_ascii=False, indent=2), encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    return deep_update(DEFAULT_CFG, data)


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_internal_label(value: Any) -> str:
    lab = normalize_label_v2(value)
    if lab == "partial_support":
        lab = "partial"
    return lab if lab in INTERNAL_LABELS else "unknown"


def provider_settings(cfg: Dict[str, Any], provider: str) -> Tuple[str, str, str, str]:
    label_cfg = cfg["labeling"]
    if provider == "dashscope":
        env_name = str(label_cfg.get("dashscope_api_key_env", "DASHSCOPE_API_KEY"))
        return (
            str(label_cfg.get("dashscope_base_url", "")).rstrip("/"),
            str(label_cfg.get("dashscope_model_name", "qwen-max")),
            os.getenv(env_name, "").strip(),
            f"env:{env_name}",
        )
    env_name = str(label_cfg.get("openai_api_key_env", "OPENAI_API_KEY"))
    return (
        str(label_cfg.get("openai_base_url", "")).rstrip("/"),
        str(label_cfg.get("model_name", "gpt-5.5")),
        os.getenv(env_name, "").strip(),
        f"env:{env_name}",
    )


def chat_completion(base_url: str, api_key: str, model_name: str, messages: List[Dict[str, str]], cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": float(cfg["labeling"].get("temperature", 0.0)),
        "max_tokens": 900,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=int(cfg["labeling"].get("timeout", 120)))
    response.raise_for_status()
    obj = response.json()
    return obj["choices"][0]["message"]["content"], obj.get("usage", {})


def build_label_prompt(case: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    payload = {
        "case_id": case["case_id"],
        "task": "machine_assisted_training_label_for_lora",
        "important_note": "This is a machine-assisted training label, not human gold.",
        "label_schema": ["support", "partial_support", "not_support"],
        "rules": [
            "Use post_decision_text only to derive the training label.",
            "Use pre_decision_text only to summarize the model input.",
            "support means the delay-related claim is substantially supported.",
            "partial_support means mixed, partly supported, or partly rejected.",
            "not_support means rejected or evidence-insufficient.",
        ],
        "required_json": {
            "outcome_label": "support|partial_support|not_support",
            "confidence": "float 0-1",
            "needs_review": "boolean",
            "invalid_case_flag": "boolean",
            "invalid_reason": "empty if valid",
            "evidence_anchor": "short post-decision anchor excerpt",
            "note": "short Chinese audit note",
        },
        "pre_decision_text": compact_text(case.get("pre_decision_text", ""), int(cfg["labeling"].get("max_chars_pre", 4500))),
        "post_decision_text": compact_text(case.get("post_decision_text", ""), int(cfg["labeling"].get("max_chars_post", 5500))),
    }
    prompt = json.dumps(payload, ensure_ascii=False)
    return [
        {"role": "system", "content": "Return exactly one valid JSON object. Do not use Markdown."},
        {"role": "user", "content": prompt},
    ], prompt


def label_one_case(row: Dict[str, Any], cfg: Dict[str, Any], provider: str) -> Dict[str, Any]:
    base_url, model_name, api_key, key_source = provider_settings(cfg, provider)
    started = time.time()
    if not api_key:
        return {"case_id": row["case_id"], "api_status": "api_unavailable", "error": f"missing {key_source}", "model_name": model_name, "provider": provider}
    case = build_case_text(pd.Series(row), cfg)
    messages, prompt = build_label_prompt(case, cfg)
    last_error = ""
    for attempt in range(int(cfg["labeling"].get("retries", 1)) + 1):
        try:
            raw, usage = chat_completion(base_url, api_key, model_name, messages, cfg)
            parsed = extract_json_object(raw)
            label = normalize_internal_label(parsed.get("outcome_label"))
            return {
                "case_id": row["case_id"],
                "source_file": row.get("source_file", ""),
                "provider": provider,
                "model_name": model_name,
                "api_key_source": key_source,
                "api_status": "api_available",
                "attempts": attempt + 1,
                "latency_sec": round(time.time() - started, 4),
                "prompt_sha256": sha256_text(prompt),
                "prompt_chars": len(prompt),
                "outcome_label": label,
                "confidence": float(parsed.get("confidence", 0.0) or 0.0),
                "needs_review": int(bool(parsed.get("needs_review", True))),
                "invalid_case_flag": int(bool(parsed.get("invalid_case_flag", False))),
                "invalid_reason": str(parsed.get("invalid_reason", ""))[:500],
                "evidence_anchor": str(parsed.get("evidence_anchor", ""))[:1000],
                "note": str(parsed.get("note", ""))[:500],
                "pre_decision_text": case.get("pre_decision_text", ""),
                "usage_json": json.dumps(usage, ensure_ascii=False),
                "raw_response": raw,
            }
        except Exception as exc:
            last_error = str(exc)[:1000]
            if attempt < int(cfg["labeling"].get("retries", 1)):
                time.sleep(1 + attempt)
    return {"case_id": row["case_id"], "source_file": row.get("source_file", ""), "provider": provider, "model_name": model_name, "api_status": "api_error", "error": last_error, "latency_sec": round(time.time() - started, 4)}


def load_case_pool(cfg: Dict[str, Any]) -> pd.DataFrame:
    index = pd.read_csv(PROJECT_ROOT / cfg["paths"]["structured_index"], encoding="utf-8-sig")
    index["case_id"] = index["case_id"].astype(str)
    return index.drop_duplicates("case_id").reset_index(drop=True)


def load_existing_labels(cfg: Dict[str, Any]) -> pd.DataFrame:
    path = PROJECT_ROOT / cfg["paths"]["existing_train_label_records"]
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["case_id"] = df["case_id"].astype(str)
    df["outcome_label"] = df["outcome_label"].map(normalize_internal_label)
    return df


def build_quality_filtered_train(
    label_records: pd.DataFrame,
    test_ids: Set[str],
    min_confidence: float,
    min_pre_chars: int,
    require_needs_review_zero: bool = True,
    require_api_available: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = label_records.copy()
    if df.empty:
        return df, pd.DataFrame()
    for col, default in [("api_status", "api_available"), ("needs_review", 1), ("confidence", 0.0), ("pre_decision_text", ""), ("invalid_case_flag", 0)]:
        if col not in df:
            df[col] = default
    df["case_id"] = df["case_id"].astype(str)
    df["outcome_label"] = df["outcome_label"].map(normalize_internal_label)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["needs_review"] = pd.to_numeric(df["needs_review"], errors="coerce").fillna(1).astype(int)
    df["invalid_case_flag"] = pd.to_numeric(df["invalid_case_flag"], errors="coerce").fillna(0).astype(int)
    df["pre_chars"] = df["pre_decision_text"].fillna("").astype(str).str.len()
    kept = []
    audit = []
    seen: Set[str] = set()
    for _, row in df.iterrows():
        cid = str(row["case_id"])
        reason = ""
        if cid in seen:
            reason = "duplicate_case_id"
        elif cid in test_ids:
            reason = "test_excluded"
        elif require_api_available and str(row.get("api_status", "")) != "api_available":
            reason = "api_not_available"
        elif row["outcome_label"] not in INTERNAL_LABELS:
            reason = "invalid_label"
        elif float(row["confidence"]) < min_confidence:
            reason = "low_confidence"
        elif require_needs_review_zero and int(row["needs_review"]) != 0:
            reason = "needs_review"
        elif int(row["invalid_case_flag"]) != 0:
            reason = "invalid_case_flag"
        elif int(row["pre_chars"]) < int(min_pre_chars):
            reason = "short_pre_decision_text"
        if reason:
            audit.append({"case_id": cid, "filter_reason": reason, "outcome_label": row.get("outcome_label", ""), "confidence": row.get("confidence", "")})
            seen.add(cid)
            continue
        kept.append(row.to_dict())
        audit.append({"case_id": cid, "filter_reason": "kept", "outcome_label": row.get("outcome_label", ""), "confidence": row.get("confidence", "")})
        seen.add(cid)
    return pd.DataFrame(kept), pd.DataFrame(audit)


def train_dev_split(df: pd.DataFrame, ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()
    parts_train = []
    parts_dev = []
    rng = random.Random(seed)
    for _, group in df.groupby("outcome_label"):
        idx = list(group.index)
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * ratio)) if len(idx) > 1 else len(idx)
        parts_train.append(df.loc[idx[:cut]])
        parts_dev.append(df.loc[idx[cut:]])
    train = pd.concat(parts_train).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    dev = pd.concat(parts_dev).sample(frac=1.0, random_state=seed).reset_index(drop=True) if parts_dev else df.iloc[0:0].copy()
    return train, dev


def format_input(row: Dict[str, Any], max_input_chars: int) -> str:
    text = compact_text(str(row.get("pre_decision_text", "")), max_input_chars)
    return f"Case ID: {row.get('case_id', '')}\nPre-decision information:\n{text}"


def alpaca_record(row: Dict[str, Any], max_input_chars: int) -> Dict[str, str]:
    label = EXPORT_LABEL_MAP.get(normalize_internal_label(row.get("outcome_label")), "not_support")
    return {"instruction": INSTRUCTION, "input": format_input(row, max_input_chars), "output": label, "system": SYSTEM}


def raw_record(row: Dict[str, Any], max_input_chars: int) -> str:
    label = EXPORT_LABEL_MAP.get(normalize_internal_label(row.get("outcome_label")), "not_support")
    return f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{format_input(row, max_input_chars)}\n\n### Response:\n{label}\n"


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_lora_package(train: pd.DataFrame, dev: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path, cfg: Dict[str, Any]) -> None:
    max_chars = int(cfg["quality"].get("max_input_chars", 6500))
    write_jsonl([alpaca_record(r, max_chars) for r in train.to_dict("records")], out_dir / "lora_train_alpaca.jsonl")
    write_jsonl([alpaca_record(r, max_chars) for r in dev.to_dict("records")], out_dir / "lora_dev_alpaca.jsonl")
    (out_dir / "lora_train_raw.txt").write_text("\n".join(raw_record(r, max_chars) for r in train.to_dict("records")), encoding="utf-8")
    (out_dir / "lora_dev_raw.txt").write_text("\n".join(raw_record(r, max_chars) for r in dev.to_dict("records")), encoding="utf-8")

    frozen_inputs = []
    labels = []
    for _, row in test_df.iterrows():
        cid = str(row["case_id"])
        case = build_case_text(row, cfg)
        input_row = {"case_id": cid, "pre_decision_text": case.get("pre_decision_text", "")}
        frozen_inputs.append({"instruction": INSTRUCTION, "input": format_input(input_row, max_chars), "system": SYSTEM, "case_id": cid})
        labels.append({"case_id": cid, "private_label": EXPORT_LABEL_MAP.get(normalize_internal_label(row.get("candidate_outcome_label_v2")), "unknown")})
    write_jsonl(frozen_inputs, out_dir / "frozen_test_input_only.jsonl")
    pd.DataFrame(labels).to_csv(out_dir / "frozen_test_labels_private.csv", index=False, encoding="utf-8-sig")


def label_missing_cases(pool: pd.DataFrame, existing_ids: Set[str], test_ids: Set[str], cfg: Dict[str, Any], provider: str, out_dir: Path) -> pd.DataFrame:
    candidates = pool[~pool["case_id"].isin(existing_ids | test_ids)].copy()
    candidates = candidates.sort_values(["pre_post_split_confidence", "case_year"], ascending=[False, False])
    max_n = int(cfg["labeling"].get("max_label_cases", 0))
    if max_n <= 0:
        candidates.to_csv(out_dir / "unlabeled_pool.csv", index=False, encoding="utf-8-sig")
        return pd.DataFrame()
    candidates = candidates.head(max_n).copy()
    workers = max(1, int(cfg["labeling"].get("workers", 4)))
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool_exec:
        futs = [pool_exec.submit(label_one_case, r, cfg, provider) for r in candidates.to_dict("records")]
        for fut in as_completed(futs):
            rows.append(fut.result())
            if len(rows) % 25 == 0:
                pd.DataFrame(rows).to_csv(out_dir / "new_label_records_partial.csv", index=False, encoding="utf-8-sig")
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "new_label_records.csv", index=False, encoding="utf-8-sig")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/research_lora_data_prep.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--label-missing", action="store_true")
    parser.add_argument("--provider", choices=["openai", "dashscope"], default="")
    parser.add_argument("--max-label-cases", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(PROJECT_ROOT / args.config)
    if args.provider:
        cfg["labeling"]["provider"] = args.provider
    if args.max_label_cases is not None:
        cfg["labeling"]["max_label_cases"] = int(args.max_label_cases)
    out_dir = PROJECT_ROOT / (args.out_dir or f"{cfg['paths']['output_root']}/lora_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = load_case_pool(cfg)
    test_df = pd.read_csv(PROJECT_ROOT / cfg["paths"]["test_label_file"], encoding="utf-8-sig")
    test_df["case_id"] = test_df["case_id"].astype(str)
    test_ids = set(test_df["case_id"])
    existing = load_existing_labels(cfg)
    existing_ids = set(existing["case_id"]) if not existing.empty else set()

    new_labels = pd.DataFrame()
    provider = str(cfg["labeling"].get("provider", "openai"))
    if args.label_missing:
        new_labels = label_missing_cases(pool, existing_ids, test_ids, cfg, provider, out_dir)
    else:
        pool[~pool["case_id"].isin(existing_ids | test_ids)].to_csv(out_dir / "unlabeled_pool.csv", index=False, encoding="utf-8-sig")

    combined = pd.concat([existing, new_labels], ignore_index=True, sort=False) if not new_labels.empty else existing.copy()
    filtered, audit = build_quality_filtered_train(
        combined,
        test_ids=test_ids,
        min_confidence=float(cfg["quality"].get("min_confidence", 0.80)),
        min_pre_chars=int(cfg["quality"].get("min_pre_chars", 600)),
        require_needs_review_zero=bool(cfg["quality"].get("require_needs_review_zero", True)),
        require_api_available=bool(cfg["quality"].get("require_api_available", True)),
    )
    train, dev = train_dev_split(filtered, float(cfg["quality"].get("train_ratio", 0.90)), int(cfg["quality"].get("seed", 2026)))
    export_lora_package(train, dev, test_df, out_dir, cfg)

    filtered.to_csv(out_dir / "lora_training_pool_filtered.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out_dir / "quality_filter_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "provider_requested": provider,
        "openai_api_present": bool(os.getenv(str(cfg["labeling"].get("openai_api_key_env", "OPENAI_API_KEY")), "").strip()),
        "dashscope_api_present": bool(os.getenv(str(cfg["labeling"].get("dashscope_api_key_env", "DASHSCOPE_API_KEY")), "").strip()),
        "test500_frozen_n": len(test_df),
        "existing_label_records_n": len(existing),
        "new_label_records_n": len(new_labels),
        "filtered_train_pool_n": len(filtered),
        "train_n": len(train),
        "dev_n": len(dev),
        "train_label_distribution": train["outcome_label"].value_counts().to_dict() if not train.empty else {},
        "dev_label_distribution": dev["outcome_label"].value_counts().to_dict() if not dev.empty else {},
        "unlabeled_pool_n": int(pd.read_csv(out_dir / "unlabeled_pool.csv").shape[0]) if (out_dir / "unlabeled_pool.csv").exists() else None,
        "artifact_hashes": {
            "test_label_file": sha256_file(PROJECT_ROOT / cfg["paths"]["test_label_file"]),
            "existing_train_label_records": sha256_file(PROJECT_ROOT / cfg["paths"]["existing_train_label_records"]),
        },
    }
    (out_dir / "lora_package_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"LORA_PACKAGE_DIR={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
