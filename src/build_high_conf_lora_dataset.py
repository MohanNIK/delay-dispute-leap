# -*- coding: utf-8 -*-
"""Build high-confidence LoRA datasets for DelayDispute outcome prediction.

This script prepares data only. It does not train a LoRA model. Machine labels
from Qwen/Qwen-Flash are kept as machine-assisted labels, not human gold.

Leakage rule:
    - post_decision_text may be used for label extraction and audit only;
    - exported LoRA inputs contain pre_decision_text and pre-decision evidence
      summaries only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import requests

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


INTERNAL_LABELS = ["support", "partial", "not_support"]
EXPORT_LABEL_MAP = {"support": "support", "partial": "partial_support", "not_support": "not_support"}
EXPORT_LABELS = ["support", "partial_support", "not_support"]
INSTRUCTION = (
    "Based only on the pre-decision information, predict the outcome label of "
    "the delay-related construction claim. Output only one label from: support, "
    "partial_support, not_support."
)
SYSTEM = "You are a construction schedule-delay dispute analysis assistant. Use only pre-decision information for prediction."

ROLE_KEYS = {
    "ENT": "ent_evidence",
    "entitlement": "ent_evidence",
    "NOT": "not_evidence",
    "notice_substantiation": "not_evidence",
    "CAU": "cau_evidence",
    "causality": "cau_evidence",
    "IMP": "imp_evidence",
    "impact_schedule_relevance": "imp_evidence",
    "DOC": "doc_evidence",
    "documentation_integrity": "doc_evidence",
}

DEFAULT_CFG: Dict[str, Any] = {
    "paths": {
        "structured_index": "data/meta/structured_case_index.csv",
        "structured_case_dir": "data/3_structured_cases",
        "existing_train_label_records": "results/train1000_augmented_precision_20260521_153425/train_label_records.csv",
        "candidate_gold_strict_v1": "data/gold/candidate_gold_strict_v1.csv",
        "candidate_gold_extended_v1": "data/gold/candidate_gold_extended_v1.csv",
        "frozen_test500": "data/gold/candidate_gold_extended_v2.csv",
        "output_root": "data/lora_exports",
    },
    "quality": {
        "min_confidence_existing": 0.85,
        "min_confidence_new": 0.85,
        "min_pre_chars": 600,
        "min_post_chars": 120,
        "min_delay_relevance": 0.80,
        "strong_score_threshold": 0.78,
        "weak_score_threshold": 0.50,
        "train_ratio": 0.88,
        "max_input_chars": 6500,
        "seed": 2026,
    },
    "qwen_flash": {
        "enabled": False,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "model_name": "qwen-flash",
        "temperature": 0.0,
        "max_tokens_screen": 700,
        "max_tokens_label": 900,
        "timeout": 90,
        "retries": 1,
        "workers": 6,
        "max_screen_cases": 300,
        "max_label_cases": 200,
        "max_chars_pre_screen": 3200,
        "max_chars_post_screen": 2200,
        "max_chars_pre_label": 4200,
        "max_chars_post_label": 4200,
    },
}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
        text = yaml.safe_dump(DEFAULT_CFG, allow_unicode=True, sort_keys=False) if yaml else json.dumps(DEFAULT_CFG, ensure_ascii=False, indent=2)
        path.write_text(text, encoding="utf-8")
    raw = path.read_text(encoding="utf-8")
    if yaml:
        data = yaml.safe_load(raw) or {}
    else:
        data = json.loads(raw)
    return deep_update(DEFAULT_CFG, data)


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compact_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    keywords = ["工期", "延期", "延误", "停工", "签证", "通知", "索赔", "关键线路", "进度", "证据", "违约", "鉴定"]
    head = text[: int(max_chars * 0.38)]
    hits: List[str] = []
    for sent in re.split(r"(?<=[。！？；])", text):
        if any(k in sent for k in keywords):
            hits.append(sent.strip())
        if sum(len(x) for x in hits) >= int(max_chars * 0.42):
            break
    tail = text[-int(max_chars * 0.18) :]
    return (head + "\n[Relevant delay/evidence spans]\n" + "\n".join(hits) + "\n[Document tail]\n" + tail)[:max_chars]


def normalize_label(value: Any) -> str:
    raw = str(value or "").strip().lower()
    mapping = {
        "support": "support",
        "supported": "support",
        "支持": "support",
        "全部支持": "support",
        "partial": "partial",
        "partial_support": "partial",
        "partially_support": "partial",
        "partially supported": "partial",
        "部分支持": "partial",
        "酌情支持": "partial",
        "not_support": "not_support",
        "not-support": "not_support",
        "not supported": "not_support",
        "reject": "not_support",
        "rejected": "not_support",
        "不支持": "not_support",
        "不予支持": "not_support",
        "驳回": "not_support",
        "unknown": "unknown",
        "nan": "unknown",
        "": "unknown",
    }
    if raw in mapping:
        return mapping[raw]
    if "partial" in raw or "部分" in raw or "酌情" in raw:
        return "partial"
    if "not" in raw or "reject" in raw or "驳回" in raw or "不予" in raw or "不支持" in raw:
        return "not_support"
    if "support" in raw or "支持" in raw:
        return "support"
    return "unknown"


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "case_id" in df.columns:
        df["case_id"] = df["case_id"].astype(str)
    return df


def load_structured_case(case_id: str, structured_dir: Path) -> Dict[str, Any]:
    path = structured_dir / f"{case_id}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def enrich_with_structured_text(df: pd.DataFrame, structured_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rec in df.to_dict("records"):
        cid = str(rec.get("case_id", ""))
        obj = load_structured_case(cid, structured_dir)
        out = dict(rec)
        for key in [
            "source_file",
            "pre_decision_text",
            "post_decision_text",
            "case_year",
            "is_domain_case",
            "pre_post_split_confidence",
            "potential_leakage_flag",
        ]:
            if not str(out.get(key, "") or "").strip() and key in obj:
                out[key] = obj.get(key)
        if not out.get("pre_decision_text"):
            out["pre_decision_text"] = obj.get("pre_decision_text", "")
        if not out.get("post_decision_text"):
            out["post_decision_text"] = obj.get("post_decision_text", "")
        if "structured_case_json" not in out:
            out["structured_case_json"] = json.dumps(obj, ensure_ascii=False)[:50000]
        rows.append(out)
    return pd.DataFrame(rows)


def load_forbidden_ids(cfg: Dict[str, Any]) -> Tuple[Set[str], Dict[str, int]]:
    paths = cfg["paths"]
    files = {
        "candidate_gold_strict_v1": PROJECT_ROOT / paths["candidate_gold_strict_v1"],
        "candidate_gold_extended_v1": PROJECT_ROOT / paths["candidate_gold_extended_v1"],
        "frozen_test500": PROJECT_ROOT / paths["frozen_test500"],
    }
    ids: Set[str] = set()
    counts: Dict[str, int] = {}
    for name, path in files.items():
        df = load_csv_if_exists(path)
        values = set(df["case_id"].astype(str)) if "case_id" in df else set()
        counts[name] = len(values)
        ids |= values
    return ids, counts


def delay_relevance_score(pre_text: str, source_file: str = "", is_domain_case: Any = 1) -> float:
    text = f"{source_file} {pre_text}"
    construction = ["建设工程", "施工合同", "工程款", "承包", "发包", "分包", "竣工", "签证", "工程", "construction", "contract", "project"]
    delay = ["工期", "延误", "延期", "停工", "窝工", "顺延", "进度", "关键线路", "逾期", "delay", "schedule", "extension"]
    c_hits = sum(1 for k in construction if k in text)
    d_hits = sum(1 for k in delay if k in text)
    score = 0.15 * min(c_hits, 4) + 0.18 * min(d_hits, 4)
    if str(is_domain_case) in {"1", "1.0", "True", "true"}:
        score += 0.20
    return round(min(1.0, score), 4)


def substantive_decision_flag(post_text: str) -> bool:
    text = str(post_text or "")
    return len(text) >= 120 and any(k in text for k in ["本院认为", "法院认为", "判决如下", "裁判如下", "予以支持", "不予支持", "驳回", "court finds", "judgment", "supported", "rejected"])


def procedural_only_flag(pre_text: str, post_text: str, source_file: str = "") -> bool:
    text = f"{source_file} {pre_text} {post_text}"
    procedural_terms = ["管辖权异议", "撤诉", "准许撤回", "移送管辖", "驳回起诉", "中止诉讼", "程序"]
    substantive_terms = ["工期", "延误", "延期", "工程款", "违约金", "签证", "索赔", "质量", "竣工"]
    return any(k in text for k in procedural_terms) and sum(1 for k in substantive_terms if k in text) <= 1


def label_extractability_score(label: str, post_text: str, anchor: str = "") -> float:
    if label not in INTERNAL_LABELS:
        return 0.0
    score = 0.45
    if anchor:
        score += 0.30
    if any(k in str(post_text or "") for k in ["支持", "不予支持", "驳回", "判决如下", "本院认为"]):
        score += 0.25
    return round(min(score, 1.0), 4)


def infer_label_consistency(row: Dict[str, Any]) -> int:
    label = normalize_label(row.get("outcome_label"))
    conf = float(row.get("confidence", 0.0) or 0.0)
    needs_review = int(float(row.get("needs_review", 1) or 0))
    raw = str(row.get("raw_response", "") or "")
    if label not in INTERNAL_LABELS:
        return 0
    if needs_review != 0:
        return 0
    if conf >= 0.85:
        return 1
    if label == "support" and ("支持" in raw or "support" in raw.lower()):
        return 1
    if label == "not_support" and ("不予支持" in raw or "驳回" in raw or "not_support" in raw.lower()):
        return 1
    if label == "partial" and ("部分" in raw or "partial" in raw.lower()):
        return 1
    return 0


def build_duplicate_ids(rows: pd.DataFrame) -> Set[str]:
    seen: Dict[str, str] = {}
    dupes: Set[str] = set()
    for _, row in rows.iterrows():
        text = re.sub(r"\s+", "", str(row.get("pre_decision_text", "") or ""))[:5000]
        h = sha256_text(text)
        cid = str(row.get("case_id", ""))
        if h in seen:
            dupes.add(cid)
        else:
            seen[h] = cid
    return dupes


def bucket_existing_label(
    row: Dict[str, Any],
    frozen_ids: Set[str],
    duplicate_ids: Set[str],
    min_confidence: float,
    min_pre_chars: int,
) -> Dict[str, Any]:
    cid = str(row.get("case_id", ""))
    label = normalize_label(row.get("outcome_label"))
    pre_text = str(row.get("pre_decision_text", "") or "")
    post_text = str(row.get("post_decision_text", "") or "")
    conf = float(row.get("confidence", 0.0) or 0.0)
    needs_review = int(float(row.get("needs_review", 1) or 0))
    api_status = str(row.get("api_status", "api_available") or "")
    rel = delay_relevance_score(pre_text, str(row.get("source_file", "")), row.get("is_domain_case", 1))
    substantive = substantive_decision_flag(post_text)
    procedural = procedural_only_flag(pre_text, post_text, str(row.get("source_file", "")))
    dup = cid in duplicate_ids
    role_cov = float(row.get("role_coverage_rate", 0.0) or 0.0)
    valid_span = float(row.get("valid_span_rate", 0.0) or 0.0)
    anchor = str(row.get("outcome_anchor_text", "") or row.get("evidence_anchor", "") or "")
    extractability = label_extractability_score(label, post_text, anchor)
    consistency = infer_label_consistency(row)
    pre_len = len(pre_text)
    post_len = len(post_text)

    score = (
        0.18 * rel
        + 0.15 * float(substantive)
        + 0.14 * min(conf, 1.0)
        + 0.15 * extractability
        + 0.12 * consistency
        + 0.10 * min(role_cov, 1.0)
        + 0.06 * min(valid_span, 1.0)
        + 0.10 * min(pre_len / 3000.0, 1.0)
    )
    hard_fail = ""
    if cid in frozen_ids:
        bucket = "frozen_test"
        hard_fail = "candidate_or_frozen_test_excluded"
    elif dup:
        bucket = "discarded"
        hard_fail = "duplicate_case"
    elif api_status and api_status != "api_available":
        bucket = "discarded"
        hard_fail = "api_not_available"
    elif label not in INTERNAL_LABELS:
        bucket = "discarded"
        hard_fail = "invalid_or_unknown_label"
    elif pre_len < min_pre_chars:
        bucket = "discarded"
        hard_fail = "short_pre_decision_text"
    elif not substantive:
        bucket = "rag_only" if rel >= 0.8 else "discarded"
        hard_fail = "no_substantive_decision_content"
    elif procedural:
        bucket = "discarded"
        hard_fail = "procedural_only"
    elif conf >= min_confidence and needs_review == 0 and rel >= 0.80 and extractability >= 0.70 and score >= 0.78:
        bucket = "strong_label_train_candidate"
    elif rel >= 0.70 and label in INTERNAL_LABELS and pre_len >= min_pre_chars:
        bucket = "weak_label_candidate"
    else:
        bucket = "rag_only" if rel >= 0.55 else "discarded"

    return {
        "case_id": cid,
        "source_file": row.get("source_file", ""),
        "existing_outcome_label": label,
        "existing_responsibility_label": row.get("responsibility_label", row.get("candidate_responsibility_label", "")),
        "existing_label_model": row.get("model_name", ""),
        "existing_label_confidence": conf,
        "api_status": api_status,
        "needs_review": needs_review,
        "pre_decision_text_length": pre_len,
        "post_decision_text_length": post_len,
        "delay_dispute_relevance": rel,
        "substantive_decision_flag": int(substantive),
        "procedural_only_flag": int(procedural),
        "duplicate_flag": int(dup),
        "label_extractability": extractability,
        "label_consistency_flag": consistency,
        "evidence_role_coverage": role_cov,
        "valid_span_rate": valid_span,
        "final_quality_score": round(score, 4),
        "final_bucket": bucket,
        "hard_fail_reason": hard_fail,
        "outcome_label": label,
        "label_confidence": conf,
        "label_model": row.get("model_name", ""),
        "label_source": "existing_qwen_machine_assisted",
        "pre_decision_text": pre_text,
        "post_decision_text": post_text,
        "outcome_anchor_text": anchor,
        "evidence_chain_json": row.get("evidence_chain_json", ""),
        "text_hash": sha256_text(pre_text),
        "label_hash": sha256_text(f"{cid}|{label}|{conf}|existing_qwen_machine_assisted"),
    }


def audit_existing_labels(existing: pd.DataFrame, frozen_ids: Set[str], min_confidence: float, min_pre_chars: int) -> pd.DataFrame:
    duplicate_ids = build_duplicate_ids(existing)
    rows = [
        bucket_existing_label(row, frozen_ids=frozen_ids, duplicate_ids=duplicate_ids, min_confidence=min_confidence, min_pre_chars=min_pre_chars)
        for row in existing.to_dict("records")
    ]
    return pd.DataFrame(rows)


def resolve_dashscope_key(cfg: Dict[str, Any]) -> Tuple[str, str]:
    env_name = str(cfg["qwen_flash"].get("api_key_env", "DASHSCOPE_API_KEY"))
    key = os.getenv(env_name, "").strip()
    return key, f"env:{env_name}" if key else f"missing:{env_name}"


def extract_json_object(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(text[start : end + 1])


def chat_completion_qwen(messages: List[Dict[str, str]], cfg: Dict[str, Any], max_tokens_key: str) -> Tuple[str, Dict[str, Any]]:
    key, _ = resolve_dashscope_key(cfg)
    if not key:
        raise RuntimeError("missing DASHSCOPE_API_KEY")
    qcfg = cfg["qwen_flash"]
    payload = {
        "model": str(qcfg.get("model_name", "qwen-flash")),
        "messages": messages,
        "temperature": float(qcfg.get("temperature", 0.0)),
        "max_tokens": int(qcfg.get(max_tokens_key, 700)),
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    resp = requests.post(f"{str(qcfg.get('base_url')).rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=int(qcfg.get("timeout", 90)))
    resp.raise_for_status()
    obj = resp.json()
    return obj["choices"][0]["message"]["content"], obj.get("usage", {})


def build_screening_prompt(row: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    payload = {
        "case_id": row["case_id"],
        "task": "screen whether this case is suitable for high-confidence SFT label extraction",
        "rules": [
            "Use both pre_decision_text and post_decision_text for screening only.",
            "Do not invent missing facts.",
            "A suitable case must be a construction schedule delay dispute with substantive facts and an extractable outcome label.",
            "Return exactly one JSON object.",
        ],
        "required_json": {
            "case_id": row["case_id"],
            "delay_dispute_relevance": "float 0-1",
            "substantive_decision_flag": "boolean",
            "procedural_only_flag": "boolean",
            "can_extract_outcome_label": "boolean",
            "pre_decision_sufficiency": "float 0-1",
            "post_decision_label_availability": "float 0-1",
            "evidence_role_coverage_estimate": "float 0-1",
            "screening_confidence": "float 0-1",
            "screening_bucket": "label_now|weak_or_rag|discard",
        },
        "pre_decision_text": compact_text(row.get("pre_decision_text", ""), int(cfg["qwen_flash"].get("max_chars_pre_screen", 3200))),
        "post_decision_text": compact_text(row.get("post_decision_text", ""), int(cfg["qwen_flash"].get("max_chars_post_screen", 2200))),
    }
    return [
        {"role": "system", "content": "Return exactly one valid JSON object. No Markdown."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def screen_one_case(row: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    started = time.time()
    model = str(cfg["qwen_flash"].get("model_name", "qwen-flash"))
    try:
        raw, usage = chat_completion_qwen(build_screening_prompt(row, cfg), cfg, "max_tokens_screen")
        obj = extract_json_object(raw)
        return {
            "case_id": row["case_id"],
            "source_file": row.get("source_file", ""),
            "model_name": model,
            "api_status": "api_available",
            "delay_dispute_relevance": float(obj.get("delay_dispute_relevance", 0.0) or 0.0),
            "substantive_decision_flag": int(bool(obj.get("substantive_decision_flag", False))),
            "procedural_only_flag": int(bool(obj.get("procedural_only_flag", True))),
            "can_extract_outcome_label": int(bool(obj.get("can_extract_outcome_label", False))),
            "pre_decision_sufficiency": float(obj.get("pre_decision_sufficiency", 0.0) or 0.0),
            "post_decision_label_availability": float(obj.get("post_decision_label_availability", 0.0) or 0.0),
            "evidence_role_coverage_estimate": float(obj.get("evidence_role_coverage_estimate", 0.0) or 0.0),
            "screening_confidence": float(obj.get("screening_confidence", 0.0) or 0.0),
            "screening_bucket": str(obj.get("screening_bucket", "weak_or_rag")),
            "latency_sec": round(time.time() - started, 4),
            "usage_json": json.dumps(usage, ensure_ascii=False),
            "raw_response": raw,
        }
    except Exception as exc:
        return {
            "case_id": row["case_id"],
            "source_file": row.get("source_file", ""),
            "model_name": model,
            "api_status": "api_error",
            "error": str(exc)[:1000],
            "screening_bucket": "weak_or_rag",
            "latency_sec": round(time.time() - started, 4),
        }


def build_label_prompt(row: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    payload = {
        "case_id": row["case_id"],
        "task": "extract one machine-assisted training label for a construction schedule delay dispute",
        "important_note": "This is machine-assisted labeling, not human gold.",
        "rules": [
            "Use post_decision_text only to extract the training label.",
            "Do not put post_decision_text into future LoRA input.",
            "If the outcome is unclear, output unknown.",
            "support means the delay-related claim is substantially supported.",
            "partial_support means mixed, partly supported, or partly rejected.",
            "not_support means rejected or evidence-insufficient.",
            "Set conflict_flag=true only when the post-decision basis gives contradictory outcome signals or the label cannot be reconciled.",
            "Do not set conflict_flag=true merely because the litigating parties disagree.",
            "Return exactly one JSON object.",
        ],
        "required_json": {
            "case_id": row["case_id"],
            "outcome_label": "support|partial_support|not_support|unknown",
            "label_confidence": "float 0-1",
            "decision_basis_span": "short exact or near-exact post-decision span",
            "decision_basis_summary": "short summary",
            "label_source_section": "post_decision",
            "responsibility_folded": "owner|contractor|shared|unclear",
            "responsibility_confidence": "float 0-1",
            "conflict_flag": "boolean",
            "needs_review": "boolean",
            "reason_short": "short Chinese reason",
        },
        "pre_decision_text": compact_text(row.get("pre_decision_text", ""), int(cfg["qwen_flash"].get("max_chars_pre_label", 4200))),
        "post_decision_text": compact_text(row.get("post_decision_text", ""), int(cfg["qwen_flash"].get("max_chars_post_label", 4200))),
    }
    return [
        {"role": "system", "content": "Return exactly one valid JSON object. No Markdown."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def label_one_case(row: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    started = time.time()
    model = str(cfg["qwen_flash"].get("model_name", "qwen-flash"))
    try:
        raw, usage = chat_completion_qwen(build_label_prompt(row, cfg), cfg, "max_tokens_label")
        obj = extract_json_object(raw)
        return {
            "case_id": row["case_id"],
            "source_file": row.get("source_file", ""),
            "model_name": model,
            "api_status": "api_available",
            "outcome_label": normalize_label(obj.get("outcome_label")),
            "label_confidence": float(obj.get("label_confidence", 0.0) or 0.0),
            "decision_basis_span": str(obj.get("decision_basis_span", ""))[:1000],
            "decision_basis_summary": str(obj.get("decision_basis_summary", ""))[:1000],
            "label_source_section": str(obj.get("label_source_section", "post_decision")),
            "responsibility_folded": str(obj.get("responsibility_folded", "unclear")),
            "responsibility_confidence": float(obj.get("responsibility_confidence", 0.0) or 0.0),
            "conflict_flag": int(bool(obj.get("conflict_flag", True))),
            "needs_review": int(bool(obj.get("needs_review", True))),
            "reason_short": str(obj.get("reason_short", ""))[:1000],
            "latency_sec": round(time.time() - started, 4),
            "usage_json": json.dumps(usage, ensure_ascii=False),
            "raw_response": raw,
        }
    except Exception as exc:
        return {
            "case_id": row["case_id"],
            "source_file": row.get("source_file", ""),
            "model_name": model,
            "api_status": "api_error",
            "error": str(exc)[:1000],
            "outcome_label": "unknown",
            "label_confidence": 0.0,
            "conflict_flag": 1,
            "needs_review": 1,
            "latency_sec": round(time.time() - started, 4),
        }


def run_parallel(rows: List[Dict[str, Any]], fn: Any, cfg: Dict[str, Any], partial_path: Path, every: int = 25) -> pd.DataFrame:
    workers = max(1, int(cfg["qwen_flash"].get("workers", 6)))
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, row, cfg) for row in rows]
        for fut in as_completed(futs):
            results.append(fut.result())
            if len(results) % every == 0:
                pd.DataFrame(results).to_csv(partial_path, index=False, encoding="utf-8-sig")
                print(f"progress {partial_path.name}: {len(results)}/{len(rows)}")
    return pd.DataFrame(results)


def heuristic_screening_queue(pool: pd.DataFrame, forbidden_ids: Set[str], existing_ids: Set[str], cfg: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for row in pool.to_dict("records"):
        cid = str(row.get("case_id", ""))
        if cid in forbidden_ids or cid in existing_ids:
            continue
        pre = str(row.get("pre_decision_text", "") or "")
        post = str(row.get("post_decision_text", "") or "")
        rel = delay_relevance_score(pre, str(row.get("source_file", "")), row.get("is_domain_case", 1))
        proc = procedural_only_flag(pre, post, str(row.get("source_file", "")))
        substantive = substantive_decision_flag(post)
        score = 0.45 * rel + 0.20 * min(len(pre) / 3500.0, 1.0) + 0.20 * substantive + 0.15 * min(len(post) / 1800.0, 1.0) - 0.25 * proc
        out = dict(row)
        out.update({"heuristic_screen_score": round(score, 4), "delay_dispute_relevance_heuristic": rel, "procedural_only_heuristic": int(proc), "substantive_decision_heuristic": int(substantive)})
        rows.append(out)
    q = pd.DataFrame(rows)
    if q.empty:
        return q
    return q.sort_values(["heuristic_screen_score", "pre_post_split_confidence", "case_year"], ascending=[False, False, False]).reset_index(drop=True)


def convert_new_labels_to_audit(label_df: pd.DataFrame, screening_df: pd.DataFrame, pool_text: pd.DataFrame, forbidden_ids: Set[str], cfg: Dict[str, Any]) -> pd.DataFrame:
    if label_df.empty:
        return pd.DataFrame()
    merged = label_df.merge(screening_df[["case_id", "delay_dispute_relevance", "substantive_decision_flag", "procedural_only_flag", "evidence_role_coverage_estimate"]], on="case_id", how="left")
    merged = merged.merge(pool_text[["case_id", "pre_decision_text", "post_decision_text", "source_file"]], on="case_id", how="left", suffixes=("", "_pool"))
    rows = []
    for row in merged.to_dict("records"):
        label = normalize_label(row.get("outcome_label"))
        conf = float(row.get("label_confidence", 0.0) or 0.0)
        cid = str(row.get("case_id", ""))
        pre = str(row.get("pre_decision_text", "") or "")
        post = str(row.get("post_decision_text", "") or "")
        rel = float(row.get("delay_dispute_relevance", delay_relevance_score(pre)) or 0.0)
        substantive = int(float(row.get("substantive_decision_flag", substantive_decision_flag(post)) or 0))
        procedural = int(float(row.get("procedural_only_flag", procedural_only_flag(pre, post)) or 0))
        role_cov = float(row.get("evidence_role_coverage_estimate", 0.0) or 0.0)
        conflict = int(float(row.get("conflict_flag", 1) or 0))
        needs_review = int(float(row.get("needs_review", 1) or 0))
        extract = label_extractability_score(label, post, str(row.get("decision_basis_span", "") or ""))
        score = 0.20 * rel + 0.18 * substantive + 0.18 * conf + 0.18 * extract + 0.10 * (1 - conflict) + 0.08 * (1 - needs_review) + 0.08 * role_cov
        if cid in forbidden_ids:
            bucket = "frozen_test"
        elif label in INTERNAL_LABELS and conf >= float(cfg["quality"]["min_confidence_new"]) and conflict == 0 and needs_review == 0 and rel >= 0.8 and substantive and not procedural and len(pre) >= int(cfg["quality"]["min_pre_chars"]):
            bucket = "strong_label_train_candidate"
        elif label in INTERNAL_LABELS and rel >= 0.65:
            bucket = "weak_label_candidate"
        else:
            bucket = "discarded"
        rows.append(
            {
                "case_id": cid,
                "source_file": row.get("source_file", row.get("source_file_pool", "")),
                "outcome_label": label,
                "label_confidence": conf,
                "label_model": row.get("model_name", "qwen-flash"),
                "label_source": "qwen_flash_machine_assisted",
                "delay_dispute_relevance": rel,
                "substantive_decision_flag": substantive,
                "procedural_only_flag": procedural,
                "duplicate_flag": 0,
                "label_extractability": extract,
                "label_consistency_flag": int(conf >= 0.85 and conflict == 0 and needs_review == 0),
                "evidence_role_coverage": role_cov,
                "final_quality_score": round(score, 4),
                "final_bucket": bucket,
                "conflict_flag": conflict,
                "needs_review": needs_review,
                "decision_basis_span": row.get("decision_basis_span", ""),
                "decision_basis_summary": row.get("decision_basis_summary", ""),
                "responsibility_folded": row.get("responsibility_folded", ""),
                "pre_decision_text": pre,
                "post_decision_text": post,
                "text_hash": sha256_text(pre),
                "label_hash": sha256_text(f"{cid}|{label}|{conf}|qwen_flash_machine_assisted"),
            }
        )
    return pd.DataFrame(rows)


def parse_evidence_roles(row: Dict[str, Any]) -> Dict[str, str]:
    roles = {"ent_evidence": "", "not_evidence": "", "cau_evidence": "", "imp_evidence": "", "doc_evidence": ""}
    raw = row.get("evidence_chain_json", "")
    try:
        chain = json.loads(raw) if isinstance(raw, str) and raw.strip() else raw
    except Exception:
        chain = []
    if isinstance(chain, list):
        for item in chain:
            if not isinstance(item, dict):
                continue
            key = ROLE_KEYS.get(str(item.get("role_label", "")).strip())
            span = str(item.get("span_text", "") or "").strip()
            if key and span and span not in roles[key]:
                roles[key] = (roles[key] + " | " + span).strip(" |")
    for key in roles:
        if not roles[key] and row.get(key):
            roles[key] = str(row.get(key, ""))
    return roles


def format_input(row: Dict[str, Any], max_input_chars: int) -> str:
    text = compact_text(str(row.get("pre_decision_text", "") or ""), max_input_chars)
    return f"Case ID: {row.get('case_id', '')}\nPre-decision information:\n{text}"


def alpaca_record(row: Dict[str, Any], max_input_chars: int) -> Dict[str, str]:
    label = EXPORT_LABEL_MAP.get(normalize_label(row.get("outcome_label")), "not_support")
    return {"instruction": INSTRUCTION, "input": format_input(row, max_input_chars), "output": label, "system": SYSTEM}


def raw_record(row: Dict[str, Any], max_input_chars: int) -> str:
    label = EXPORT_LABEL_MAP.get(normalize_label(row.get("outcome_label")), "not_support")
    return f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{format_input(row, max_input_chars)}\n\n### Response:\n{label}\n"


def evidence_conditioned_input(row: Dict[str, Any], max_input_chars: int) -> str:
    roles = parse_evidence_roles(row)
    base = format_input(row, max_input_chars)
    evidence = (
        "\n\nEvidence roles extracted from pre-decision materials:\n"
        f"ENT evidence: {roles['ent_evidence']}\n"
        f"NOT evidence: {roles['not_evidence']}\n"
        f"CAU evidence: {roles['cau_evidence']}\n"
        f"IMP evidence: {roles['imp_evidence']}\n"
        f"DOC evidence: {roles['doc_evidence']}"
    )
    return compact_text(base + evidence, max_input_chars + 1800)


def evidence_conditioned_alpaca_record(row: Dict[str, Any], max_input_chars: int) -> Dict[str, str]:
    label = EXPORT_LABEL_MAP.get(normalize_label(row.get("outcome_label")), "not_support")
    return {"instruction": INSTRUCTION, "input": evidence_conditioned_input(row, max_input_chars), "output": label, "system": SYSTEM}


def evidence_conditioned_raw_record(row: Dict[str, Any], max_input_chars: int) -> str:
    label = EXPORT_LABEL_MAP.get(normalize_label(row.get("outcome_label")), "not_support")
    return f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{evidence_conditioned_input(row, max_input_chars)}\n\n### Response:\n{label}\n"


def split_train_dev(df: pd.DataFrame, train_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()
    rng = random.Random(seed)
    train_parts: List[pd.DataFrame] = []
    dev_parts: List[pd.DataFrame] = []
    for _, group in df.groupby("outcome_label"):
        idx = list(group.index)
        rng.shuffle(idx)
        if len(idx) <= 1:
            cut = len(idx)
        else:
            cut = max(1, min(len(idx) - 1, int(round(len(idx) * train_ratio))))
        train_parts.append(df.loc[idx[:cut]])
        dev_parts.append(df.loc[idx[cut:]])
    train = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    dev = pd.concat(dev_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, dev


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_lora_files(train: pd.DataFrame, dev: pd.DataFrame, frozen_test: pd.DataFrame, out_dir: Path, cfg: Dict[str, Any]) -> None:
    max_chars = int(cfg["quality"]["max_input_chars"])
    write_jsonl([alpaca_record(r, max_chars) for r in train.to_dict("records")], out_dir / "lora_train_alpaca.jsonl")
    write_jsonl([alpaca_record(r, max_chars) for r in dev.to_dict("records")], out_dir / "lora_dev_alpaca.jsonl")
    (out_dir / "lora_train_raw.txt").write_text("\n".join(raw_record(r, max_chars) for r in train.to_dict("records")), encoding="utf-8")
    (out_dir / "lora_dev_raw.txt").write_text("\n".join(raw_record(r, max_chars) for r in dev.to_dict("records")), encoding="utf-8")

    write_jsonl([evidence_conditioned_alpaca_record(r, max_chars) for r in train.to_dict("records")], out_dir / "lora_train_evidence_conditioned_alpaca.jsonl")
    write_jsonl([evidence_conditioned_alpaca_record(r, max_chars) for r in dev.to_dict("records")], out_dir / "lora_dev_evidence_conditioned_alpaca.jsonl")
    (out_dir / "lora_train_evidence_conditioned_raw.txt").write_text("\n".join(evidence_conditioned_raw_record(r, max_chars) for r in train.to_dict("records")), encoding="utf-8")
    (out_dir / "lora_dev_evidence_conditioned_raw.txt").write_text("\n".join(evidence_conditioned_raw_record(r, max_chars) for r in dev.to_dict("records")), encoding="utf-8")

    frozen_inputs = []
    private_labels = []
    for rec in frozen_test.to_dict("records"):
        label = normalize_label(rec.get("candidate_outcome_label_v2", rec.get("candidate_outcome_label", "")))
        frozen_inputs.append({"case_id": rec["case_id"], "instruction": INSTRUCTION, "input": format_input(rec, max_chars), "system": SYSTEM})
        private_labels.append({"case_id": rec["case_id"], "private_label": EXPORT_LABEL_MAP.get(label, "unknown")})
    write_jsonl(frozen_inputs, out_dir / "frozen_test_input_only.jsonl")
    pd.DataFrame(private_labels).to_csv(out_dir / "frozen_test_labels_private.csv", index=False, encoding="utf-8-sig")


def write_reports(out_dir: Path, train: pd.DataFrame, dev: pd.DataFrame, master: pd.DataFrame, frozen_ids: Set[str], cfg: Dict[str, Any]) -> None:
    train_ids, dev_ids = set(train["case_id"]), set(dev["case_id"])
    overlap_lines = [
        f"train_dev_overlap={len(train_ids & dev_ids)}",
        f"train_frozen_overlap={len(train_ids & frozen_ids)}",
        f"dev_frozen_overlap={len(dev_ids & frozen_ids)}",
    ]
    (out_dir / "overlap_check_report.txt").write_text("\n".join(overlap_lines) + "\n", encoding="utf-8")

    leakage_hits = []
    for fname in [
        "lora_train_alpaca.jsonl",
        "lora_dev_alpaca.jsonl",
        "lora_train_raw.txt",
        "lora_dev_raw.txt",
        "lora_train_evidence_conditioned_alpaca.jsonl",
        "lora_dev_evidence_conditioned_alpaca.jsonl",
    ]:
        text = (out_dir / fname).read_text(encoding="utf-8")
        leakage_hits.append({"file": fname, "contains_post_decision_field_name": int("post_decision_text" in text), "contains_private_label_field": int("private_label" in text)})
    pd.DataFrame(leakage_hits).to_csv(out_dir / "leakage_check_report.csv", index=False, encoding="utf-8-sig")
    (out_dir / "leakage_check_report.txt").write_text(
        "\n".join(f"{r['file']}: post_decision_field={r['contains_post_decision_field_name']}, private_label_field={r['contains_private_label_field']}" for r in leakage_hits) + "\n",
        encoding="utf-8",
    )

    output_values = set()
    for path in [out_dir / "lora_train_alpaca.jsonl", out_dir / "lora_dev_alpaca.jsonl"]:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    output_values.add(json.loads(line)["output"])
    validation = [
        f"train_n={len(train)}",
        f"dev_n={len(dev)}",
        f"master_n={len(master)}",
        f"outputs={sorted(output_values)}",
        f"outputs_valid={int(output_values <= set(EXPORT_LABELS))}",
        f"duplicate_case_ids_in_master={int(master['case_id'].duplicated().sum()) if not master.empty else 0}",
    ]
    (out_dir / "lora_data_validation_report.txt").write_text("\n".join(validation) + "\n", encoding="utf-8")

    dist = master["outcome_label"].value_counts().rename_axis("outcome_label").reset_index(name="count") if not master.empty else pd.DataFrame(columns=["outcome_label", "count"])
    if not dist.empty:
        dist["ratio"] = dist["count"] / dist["count"].sum()
    dist.to_csv(out_dir / "strong_label_distribution.csv", index=False, encoding="utf-8-sig")
    if not dist.empty:
        total = int(dist["count"].sum())
        n_classes = int(dist.shape[0])
        weights = dist.copy()
        weights["class_weight"] = weights["count"].map(lambda c: round(total / (n_classes * int(c)), 6))
        weights.to_csv(out_dir / "class_weight_metadata.csv", index=False, encoding="utf-8-sig")

    quality_rows = [
        {"metric": "total_strong_labels", "value": len(master)},
        {"metric": "train_size", "value": len(train)},
        {"metric": "dev_size", "value": len(dev)},
        {"metric": "mean_quality_score", "value": round(float(master["final_quality_score"].mean()), 4) if not master.empty else 0},
        {"metric": "min_quality_score", "value": round(float(master["final_quality_score"].min()), 4) if not master.empty else 0},
    ]
    for label, count in (master["outcome_label"].value_counts().to_dict() if not master.empty else {}).items():
        quality_rows.append({"metric": f"class_count_{label}", "value": count})
    pd.DataFrame(quality_rows).to_csv(out_dir / "strong_label_quality_report.csv", index=False, encoding="utf-8-sig")
    (out_dir / "strong_label_quality_report.txt").write_text("\n".join(f"{r['metric']}: {r['value']}" for r in quality_rows) + "\n", encoding="utf-8")

    readme = f"""# DelayDispute Copilot high-confidence LoRA data package

This package is for supervised fine-tuning of one-label outcome prediction.

Label status:
- Labels are machine-assisted strong labels, not human gold.
- Existing labels are from prior Qwen-assisted labeling.
- Optional new labels are from Qwen-Flash screening/label extraction.

Leakage boundary:
- LoRA inputs contain pre-decision information only.
- Post-decision text was used only to extract labels and audit label strength.
- Frozen test labels are not included in the external zip package.

Files:
- lora_train_alpaca.jsonl
- lora_dev_alpaca.jsonl
- lora_train_raw.txt
- lora_dev_raw.txt
- evidence-conditioned variants
- lora_split_report.csv
- strong_label_quality_report.csv
- dataset_manifest.csv

Train/dev sizes:
- train: {len(train)}
- dev: {len(dev)}
"""
    (out_dir / "lora_data_readme.md").write_text(readme, encoding="utf-8")


def make_zip(out_dir: Path) -> Path:
    zip_path = out_dir / "data_package_for_external_lora_finetuning.zip"
    include = [
        "lora_train_alpaca.jsonl",
        "lora_dev_alpaca.jsonl",
        "lora_train_raw.txt",
        "lora_dev_raw.txt",
        "lora_train_evidence_conditioned_alpaca.jsonl",
        "lora_dev_evidence_conditioned_alpaca.jsonl",
        "lora_train_evidence_conditioned_raw.txt",
        "lora_dev_evidence_conditioned_raw.txt",
        "lora_data_readme.md",
        "lora_split_report.csv",
        "strong_label_quality_report.csv",
        "class_weight_metadata.csv",
        "dataset_manifest.csv",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in include:
            p = out_dir / name
            if p.exists():
                zf.write(p, arcname=name)
    return zip_path


def build_dataset(cfg: Dict[str, Any], out_dir: Path, run_qwen_flash: bool, resume_from: Optional[Path] = None) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    structured_dir = PROJECT_ROOT / cfg["paths"]["structured_case_dir"]
    pool = load_csv_if_exists(PROJECT_ROOT / cfg["paths"]["structured_index"])
    pool = enrich_with_structured_text(pool, structured_dir)

    existing = load_csv_if_exists(PROJECT_ROOT / cfg["paths"]["existing_train_label_records"])
    existing = enrich_with_structured_text(existing, structured_dir)
    frozen_test = load_csv_if_exists(PROJECT_ROOT / cfg["paths"]["frozen_test500"])
    frozen_test = enrich_with_structured_text(frozen_test, structured_dir)
    forbidden_ids, forbidden_counts = load_forbidden_ids(cfg)

    existing_audit = audit_existing_labels(
        existing,
        frozen_ids=forbidden_ids,
        min_confidence=float(cfg["quality"]["min_confidence_existing"]),
        min_pre_chars=int(cfg["quality"]["min_pre_chars"]),
    )
    existing_audit.to_csv(out_dir / "existing_qwen_label_audit.csv", index=False, encoding="utf-8-sig")
    existing_audit[existing_audit["final_bucket"] == "strong_label_train_candidate"].to_csv(out_dir / "existing_qwen_strong_labels.csv", index=False, encoding="utf-8-sig")
    existing_audit[existing_audit["final_bucket"] == "weak_label_candidate"].to_csv(out_dir / "existing_qwen_weak_labels.csv", index=False, encoding="utf-8-sig")
    existing_audit[existing_audit["final_bucket"].isin(["discarded", "rag_only", "frozen_test"])].to_csv(out_dir / "existing_qwen_discarded.csv", index=False, encoding="utf-8-sig")

    existing_ids = set(existing["case_id"]) if "case_id" in existing else set()
    queue = heuristic_screening_queue(pool, forbidden_ids=forbidden_ids, existing_ids=existing_ids, cfg=cfg)
    queue.to_csv(out_dir / "qwen_flash_screening_queue.csv", index=False, encoding="utf-8-sig")

    screening = pd.DataFrame()
    labels = pd.DataFrame()
    resumed_screening = pd.DataFrame()
    resumed_labels = pd.DataFrame()
    if resume_from is not None and resume_from.exists():
        resumed_screening = load_csv_if_exists(resume_from / "qwen_flash_screening_results.csv")
        resumed_labels = load_csv_if_exists(resume_from / "qwen_flash_label_results.csv")
    if run_qwen_flash:
        key, _ = resolve_dashscope_key(cfg)
        if key:
            max_screen = int(cfg["qwen_flash"].get("max_screen_cases", 0))
            screen_queue = queue.head(max_screen) if max_screen > 0 else queue
            if not resumed_screening.empty and "case_id" in resumed_screening:
                done_screen_ids = set(resumed_screening["case_id"].astype(str))
                screen_queue = screen_queue[~screen_queue["case_id"].astype(str).isin(done_screen_ids)].copy()
            new_screening = run_parallel(screen_queue.to_dict("records"), screen_one_case, cfg, out_dir / "qwen_flash_screening_partial.csv") if not screen_queue.empty else pd.DataFrame()
            screening = pd.concat([resumed_screening, new_screening], ignore_index=True, sort=False) if not resumed_screening.empty else new_screening
            if not screening.empty:
                screening = screening.drop_duplicates("case_id", keep="last").reset_index(drop=True)
            screening.to_csv(out_dir / "qwen_flash_screening_results.csv", index=False, encoding="utf-8-sig")
            screening[screening["screening_bucket"] == "discard"].to_csv(out_dir / "qwen_flash_screening_discarded.csv", index=False, encoding="utf-8-sig")
            label_now = screening[screening["screening_bucket"] == "label_now"].copy()
            label_now = label_now.sort_values(["screening_confidence", "delay_dispute_relevance"], ascending=[False, False])
            max_label = int(cfg["qwen_flash"].get("max_label_cases", 0))
            if max_label > 0:
                label_now = label_now.head(max_label)
            if not resumed_labels.empty and "case_id" in resumed_labels:
                done_label_ids = set(resumed_labels["case_id"].astype(str))
                label_now = label_now[~label_now["case_id"].astype(str).isin(done_label_ids)].copy()
            label_now.to_csv(out_dir / "qwen_flash_screening_label_now_queue.csv", index=False, encoding="utf-8-sig")
            screening[screening["screening_bucket"] == "weak_or_rag"].to_csv(out_dir / "qwen_flash_screening_weak_or_rag.csv", index=False, encoding="utf-8-sig")
            label_cases = label_now.merge(pool, on="case_id", how="left", suffixes=("", "_pool"))
            new_labels = run_parallel(label_cases.to_dict("records"), label_one_case, cfg, out_dir / "qwen_flash_label_partial.csv") if not label_cases.empty else pd.DataFrame()
            labels = pd.concat([resumed_labels, new_labels], ignore_index=True, sort=False) if not resumed_labels.empty else new_labels
            if not labels.empty:
                labels = labels.drop_duplicates("case_id", keep="last").reset_index(drop=True)
            labels.to_csv(out_dir / "qwen_flash_label_results.csv", index=False, encoding="utf-8-sig")
        else:
            (out_dir / "qwen_flash_api_unavailable.txt").write_text("DASHSCOPE_API_KEY missing; skipped Qwen-Flash screening and labeling.\n", encoding="utf-8")

    if screening.empty:
        for name in ["qwen_flash_screening_results.csv", "qwen_flash_screening_discarded.csv", "qwen_flash_screening_label_now_queue.csv", "qwen_flash_screening_weak_or_rag.csv"]:
            pd.DataFrame().to_csv(out_dir / name, index=False, encoding="utf-8-sig")
    if labels.empty:
        for name in ["qwen_flash_label_results.csv", "qwen_flash_strong_labels.csv", "qwen_flash_weak_labels.csv", "qwen_flash_unknown_or_discarded.csv"]:
            pd.DataFrame().to_csv(out_dir / name, index=False, encoding="utf-8-sig")

    new_audit = convert_new_labels_to_audit(labels, screening, pool, forbidden_ids, cfg) if not labels.empty else pd.DataFrame()
    if not new_audit.empty:
        new_audit[new_audit["final_bucket"] == "strong_label_train_candidate"].to_csv(out_dir / "qwen_flash_strong_labels.csv", index=False, encoding="utf-8-sig")
        new_audit[new_audit["final_bucket"] == "weak_label_candidate"].to_csv(out_dir / "qwen_flash_weak_labels.csv", index=False, encoding="utf-8-sig")
        new_audit[new_audit["final_bucket"].isin(["discarded", "frozen_test"]) | (new_audit["outcome_label"] == "unknown")].to_csv(out_dir / "qwen_flash_unknown_or_discarded.csv", index=False, encoding="utf-8-sig")

    strong_existing = existing_audit[existing_audit["final_bucket"] == "strong_label_train_candidate"].copy()
    strong_new = new_audit[new_audit["final_bucket"] == "strong_label_train_candidate"].copy() if not new_audit.empty else pd.DataFrame()
    master = pd.concat([strong_existing, strong_new], ignore_index=True, sort=False)
    if not master.empty:
        master = master[~master["case_id"].isin(forbidden_ids)].copy()
        master = master[master["outcome_label"].isin(INTERNAL_LABELS)].copy()
        master = master[master["pre_decision_text"].fillna("").astype(str).str.len() >= int(cfg["quality"]["min_pre_chars"])].copy()
        master = master.drop_duplicates("case_id").reset_index(drop=True)
        master["text_hash"] = master["pre_decision_text"].fillna("").map(sha256_text)
        master["label_hash"] = master.apply(lambda r: sha256_text(f"{r['case_id']}|{r['outcome_label']}|{r.get('label_confidence', '')}|{r.get('label_source', '')}"), axis=1)
    master.to_csv(out_dir / "strong_label_master.csv", index=False, encoding="utf-8-sig")

    train, dev = split_train_dev(master, float(cfg["quality"]["train_ratio"]), int(cfg["quality"]["seed"]))
    train.to_csv(out_dir / "lora_train_manifest.csv", index=False, encoding="utf-8-sig")
    dev.to_csv(out_dir / "lora_dev_manifest.csv", index=False, encoding="utf-8-sig")

    split_rows = []
    for name, df in [("train", train), ("dev", dev), ("master", master)]:
        counts = df["outcome_label"].value_counts().to_dict() if not df.empty else {}
        split_rows.append({"split": name, "n": len(df), **{f"n_{label}": counts.get(label, 0) for label in INTERNAL_LABELS}})
    split_report = pd.DataFrame(split_rows)
    split_report.to_csv(out_dir / "lora_split_report.csv", index=False, encoding="utf-8-sig")

    export_lora_files(train, dev, frozen_test, out_dir, cfg)
    write_reports(out_dir, train, dev, master, forbidden_ids, cfg)

    dataset_manifest_rows = []
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            dataset_manifest_rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(dataset_manifest_rows).to_csv(out_dir / "dataset_manifest.csv", index=False, encoding="utf-8-sig")
    zip_path = make_zip(out_dir)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "output_dir": str(out_dir),
        "qwen_flash_enabled": bool(run_qwen_flash),
        "qwen_flash_model": cfg["qwen_flash"]["model_name"],
        "full_structured_corpus_n": len(pool),
        "existing_labels_audited_n": len(existing_audit),
        "existing_strong_labels_n": len(strong_existing),
        "remaining_screening_queue_n": len(queue),
        "remaining_cases_screened_by_qwen_flash_n": len(screening),
        "new_qwen_flash_label_results_n": len(labels),
        "new_qwen_flash_strong_labels_n": len(strong_new),
        "resumed_screening_n": len(resumed_screening),
        "resumed_label_results_n": len(resumed_labels),
        "total_strong_labels_n": len(master),
        "train_n": len(train),
        "dev_n": len(dev),
        "strong_label_distribution": master["outcome_label"].value_counts().to_dict() if not master.empty else {},
        "train_label_distribution": train["outcome_label"].value_counts().to_dict() if not train.empty else {},
        "dev_label_distribution": dev["outcome_label"].value_counts().to_dict() if not dev.empty else {},
        "forbidden_id_counts": forbidden_counts,
        "external_zip": str(zip_path),
        "artifact_hashes": {
            "structured_index": sha256_file(PROJECT_ROOT / cfg["paths"]["structured_index"]),
            "existing_train_label_records": sha256_file(PROJECT_ROOT / cfg["paths"]["existing_train_label_records"]),
            "frozen_test500": sha256_file(PROJECT_ROOT / cfg["paths"]["frozen_test500"]),
        },
    }
    (out_dir / "high_conf_lora_run_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/research_high_conf_lora.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-qwen-flash", action="store_true")
    parser.add_argument("--max-screen-cases", type=int, default=None)
    parser.add_argument("--max-label-cases", type=int, default=None)
    parser.add_argument("--resume-from", default="")
    args = parser.parse_args(argv)

    cfg = load_config(PROJECT_ROOT / args.config)
    if args.max_screen_cases is not None:
        cfg["qwen_flash"]["max_screen_cases"] = int(args.max_screen_cases)
    if args.max_label_cases is not None:
        cfg["qwen_flash"]["max_label_cases"] = int(args.max_label_cases)
    out_dir = PROJECT_ROOT / (args.out_dir or f"{cfg['paths']['output_root']}/high_conf_lora_{now_stamp()}")
    resume_from = PROJECT_ROOT / args.resume_from if args.resume_from else None
    build_dataset(cfg, out_dir, run_qwen_flash=bool(args.run_qwen_flash or cfg["qwen_flash"].get("enabled", False)), resume_from=resume_from)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
