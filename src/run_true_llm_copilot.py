# -*- coding: utf-8 -*-
"""True API-backed DelayDispute Copilot for candidate benchmark evaluation.

This entrypoint calls a real OpenAI-compatible DashScope/Qwen endpoint. It does
not use rule fallback predictions for headline metrics. If the API is
unavailable or a case fails, the failure is recorded and excluded from
prediction-level metric recomputation with an explicit audit status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from tqdm import tqdm

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit_utils import build_run_manifest, synthetic_file_diff, write_manifest, write_synthetic_git_summary  # noqa: E402

try:  # Keep compatibility with the existing research utilities when sklearn is available.
    from src.research_support import build_evidence_chain, normalize_label, normalize_resp, read_csv_flexible  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - fallback is exercised in lightweight environments.
    SENT_SPLIT_FALLBACK = re.compile(r"(?<=[。！？；\n])")
    ROLE_KEYWORDS_FALLBACK = {
        "entitlement": ["合同约定", "合同条款", "约定", "应予顺延", "享有", "工期", "索赔依据"],
        "notice_substantiation": ["通知", "签证", "报审", "申请", "备案", "报告", "监理", "纪要"],
        "causality": ["导致", "原因", "因", "造成", "影响", "阻碍", "拖延"],
        "impact_schedule_relevance": ["关键线路", "工期", "竣工", "进度", "停工", "顺延", "延期", "交付"],
        "documentation_integrity": ["证据", "资料", "日志", "函件", "纪要", "签证", "证明", "记录"],
    }
    RESP_ZH_TO_EN_FALLBACK = {
        "业主": "owner",
        "发包人": "owner",
        "建设单位": "owner",
        "甲方": "owner",
        "承包商": "contractor",
        "承包人": "contractor",
        "施工单位": "contractor",
        "乙方": "contractor",
        "分包商": "subcontractor",
        "分包": "subcontractor",
        "设计/监理": "designer_supervisor",
        "设计": "designer_supervisor",
        "监理": "designer_supervisor",
        "双方": "both",
        "不可抗力/政策": "force_majeure_policy",
        "不可抗力": "force_majeure_policy",
        "政策": "force_majeure_policy",
        "unknown": "unknown",
        "": "unknown",
    }

    def read_csv_flexible(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="gbk")

    def normalize_label(x: object) -> str:
        x = str(x or "").strip().lower()
        mapping = {
            "support": "support",
            "partial": "partial",
            "partial_support": "partial",
            "partially_support": "partial",
            "not_support": "not_support",
            "not-support": "not_support",
            "notsupport": "not_support",
            "支持": "support",
            "部分支持": "partial",
            "不支持": "not_support",
            "驳回": "not_support",
            "unknown": "unknown",
            "": "unknown",
            "nan": "unknown",
        }
        return mapping.get(x, "unknown")

    def normalize_resp(x: object) -> str:
        x = str(x or "").strip()
        if x in RESP_ZH_TO_EN_FALLBACK.values():
            return x
        return RESP_ZH_TO_EN_FALLBACK.get(x, "unknown")

    def split_sentences_fallback(text: str) -> List[Tuple[str, int, int]]:
        text = str(text or "")
        spans: List[Tuple[str, int, int]] = []
        start = 0
        for match in SENT_SPLIT_FALLBACK.finditer(text):
            end = match.end()
            sent = text[start:end].strip()
            if sent:
                sent_start = text.find(sent, start)
                spans.append((sent, sent_start, sent_start + len(sent)))
            start = end
        tail = text[start:].strip()
        if tail:
            tail_start = text.find(tail, start)
            spans.append((tail, tail_start, tail_start + len(tail)))
        return spans

    def build_evidence_chain(pre_text: str) -> List[Dict[str, object]]:
        chain: List[Dict[str, object]] = []
        seen = set()
        for role, keywords in ROLE_KEYWORDS_FALLBACK.items():
            picked = None
            for sent, start, end in split_sentences_fallback(pre_text):
                if any(k in sent for k in keywords):
                    picked = {
                        "role_label": role,
                        "text": sent,
                        "span_start": start,
                        "span_end": end,
                        "pre_decision_flag": 1,
                        "duplicate_flag": int(sent in seen),
                    }
                    seen.add(sent)
                    break
            chain.append(picked or {"role_label": role, "text": "", "span_start": -1, "span_end": -1, "pre_decision_flag": 0, "duplicate_flag": 0})
        return chain


LABELS = ["support", "partial", "not_support"]
RESP_LABELS = [
    "owner",
    "contractor",
    "subcontractor",
    "designer_supervisor",
    "both",
    "force_majeure_policy",
    "unknown",
]
FOLDED_RESP_LABELS = ["owner", "contractor", "other_external", "shared_or_uncertain"]
ROLE_ALIASES = {
    "ENT": "entitlement",
    "NOT": "notice_substantiation",
    "CAU": "causality",
    "IMP": "impact_schedule_relevance",
    "DOC": "documentation_integrity",
    "entitlement": "entitlement",
    "notice_substantiation": "notice_substantiation",
    "causality": "causality",
    "impact_schedule_relevance": "impact_schedule_relevance",
    "documentation_integrity": "documentation_integrity",
}
LEAKAGE_TERMS = ["判决如下", "裁定如下", "本院认为", "综上所述", "驳回", "判令", "予以支持", "不予支持"]
DEFAULT_TRUE_LLM_CFG: Dict[str, Any] = {
    "project": {"name": "delay_dispute_madra", "branch": "true_llm_copilot_qwen"},
    "paths": {
        "structured_case_dir": "data/3_structured_cases",
        "candidate_gold_strict_csv": "data/gold/candidate_gold_strict_v1.csv",
        "candidate_gold_extended_csv": "data/gold/candidate_gold_extended_v1.csv",
        "legacy_key_source_py": "src/llm_step2_fast_qc.py",
        "final_eval_reference_run": "results/final_eval_20260409_194025",
        "output_root": "results",
    },
    "llm": {
        "provider": "dashscope_qwen",
        "api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "debug_api_key": "",
        "reuse_legacy_debug_key": False,
        "model_name": "qwen-plus",
        "prompt_template_version": "true_qwen_delay_copilot_v1",
        "temperature": 0.0,
        "max_tokens": 1800,
        "timeout": 120,
        "retries": 2,
        "workers": 1,
        "sleep_seconds": 0.25,
        "max_chars": 6500,
        "allow_rule_fallback": False,
        "text_mode": "pre_decision_only",
    },
    "eval": {
        "labels": LABELS,
        "responsibility_labels": RESP_LABELS,
        "bootstrap_rounds": 300,
        "seed": 2026,
        "min_success_rate_for_claimable_metric": 0.95,
    },
    "run": {"preflight_cases": 5, "candidate_scope": "strict_plus_extended_deduplicated", "run_name_prefix": "true_llm_copilot"},
}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        return json.loads(json.dumps(DEFAULT_TRUE_LLM_CFG, ensure_ascii=False))
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    cfg = json.loads(json.dumps(DEFAULT_TRUE_LLM_CFG, ensure_ascii=False))

    def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> None:
        for key, value in (updates or {}).items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                deep_update(base[key], value)
            else:
                base[key] = value

    deep_update(cfg, loaded or {})
    return cfg


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def legacy_debug_key_from_script(script_path: Path) -> str:
    if not script_path.exists():
        return ""
    text = script_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r'DEBUG_API_KEY\s*=\s*["\']([^"\']+)["\']', text)
    if not match:
        return ""
    candidate = match.group(1).strip()
    if not candidate or candidate.startswith("sk-xxxx"):
        return ""
    return candidate


def resolve_api_key(cfg: Dict[str, Any]) -> Tuple[str, str]:
    llm_cfg = cfg["llm"]
    env_name = str(llm_cfg.get("api_key_env", "DASHSCOPE_API_KEY"))
    api_key = os.getenv(env_name, "").strip()
    if api_key:
        return api_key, f"env:{env_name}"
    debug_key = str(llm_cfg.get("debug_api_key", "")).strip()
    if debug_key:
        return debug_key, "config:debug_api_key"
    if bool(llm_cfg.get("reuse_legacy_debug_key", True)):
        script_path = PROJECT_ROOT / cfg["paths"].get("legacy_key_source_py", "src/llm_step2_fast_qc.py")
        legacy_key = legacy_debug_key_from_script(script_path)
        if legacy_key:
            return legacy_key, f"legacy_script:{rel(script_path)}"
    return "", "missing"


def compact_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.45)]
    tail = text[-int(max_chars * 0.20) :]
    middle_hits = []
    for sent in re.split(r"(?<=[。！？；])", text):
        if any(k in sent for k in ["工期", "延误", "延期", "逾期", "停工", "签证", "通知", "索赔", "关键线路", "证据"]):
            middle_hits.append(sent.strip())
        if sum(len(x) for x in middle_hits) >= int(max_chars * 0.35):
            break
    return (head + "\n【延误争议相关片段】\n" + "\n".join(middle_hits) + "\n【尾部事实片段】\n" + tail)[:max_chars]


def compact_evidence_pool(structured: Dict[str, Any], limit: int = 12) -> List[Dict[str, Any]]:
    pool = []
    for item in structured.get("source_span_pointers") or build_evidence_chain(structured.get("pre_decision_text", "")):
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        pool.append(
            {
                "role_label": item.get("role_label", ""),
                "span_text": text[:260],
                "span_start": item.get("span_start", -1),
                "span_end": item.get("span_end", -1),
            }
        )
        if len(pool) >= limit:
            break
    return pool


def build_prompt(case_id: str, structured: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    max_chars = int(cfg["llm"].get("max_chars", 6500))
    pre_text = compact_text(structured.get("pre_decision_text", ""), max_chars)
    evidence_pool = compact_evidence_pool(structured)
    payload = {
        "case_id": case_id,
        "task": "construction_schedule_delay_dispute_copilot",
        "hard_constraints": [
            "Use only pre_decision_text and candidate_evidence_pool.",
            "Do not cite or infer from judgment/ruling/post-decision language.",
            "Return one valid JSON object only. No Markdown.",
            "If evidence is insufficient, set uncertainty_flag=true and explain the missing management evidence.",
        ],
        "label_schema": {
            "outcome_label": ["support", "partial", "not_support"],
            "primary_responsible_party": RESP_LABELS,
            "secondary_responsible_party": RESP_LABELS,
            "evidence_roles": ["ENT", "NOT", "CAU", "IMP", "DOC"],
        },
        "required_json_schema": {
            "outcome_label": "support|partial|not_support",
            "outcome_confidence": "float 0-1",
            "primary_responsible_party": "owner|contractor|subcontractor|designer_supervisor|both|force_majeure_policy|unknown",
            "secondary_responsible_party": "same schema or unknown",
            "responsibility_type": "single_party|shared|external|uncertain",
            "responsibility_confidence": "float 0-1",
            "uncertainty_flag": "boolean",
            "evidence_chain": [
                {
                    "role_label": "ENT|NOT|CAU|IMP|DOC",
                    "span_text": "exact excerpt copied from pre_decision_text or candidate_evidence_pool",
                    "reason": "why this excerpt supports the role",
                }
            ],
            "delay_irac": {
                "issue": "short dispute issue",
                "rule": "contract/procedure/evidence rule stated without citing post-decision result",
                "application": "how facts, causality, procedure and evidence interact",
                "conclusion": "outcome and responsibility conclusion",
                "management_action": "evidence supplementation, negotiation or governance action",
            },
            "documentation_gap_index": "float 0-1",
            "procedural_compliance_risk": "float 0-1",
            "causality_ambiguity": "float 0-1",
            "concurrency_risk": "float 0-1",
            "critical_path_support": "float 0-1",
            "negotiation_readiness_score": "float 0-1",
            "managerial_failure_type": "short label",
            "recommended_management_action": "short action",
        },
        "candidate_evidence_pool": evidence_pool,
        "pre_decision_text": pre_text,
    }
    prompt_text = json.dumps(payload, ensure_ascii=False)
    messages = [
        {
            "role": "system",
            "content": (
                "你是工程管理与建设工程工期延误纠纷分析助手。"
                "你的任务是在人机混合框架中生成可审计的预测、责任诊断、证据链和管理建议。"
                "必须只输出 JSON。"
            ),
        },
        {"role": "user", "content": prompt_text},
    ]
    return messages, prompt_text


def extract_json_object(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(text[start : end + 1])


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 6)
    except Exception:
        return default


def fold_responsibility(label: Any) -> str:
    label = normalize_resp(label)
    if label in {"owner", "contractor"}:
        return label
    if label in {"subcontractor", "designer_supervisor", "force_majeure_policy"}:
        return "other_external"
    return "shared_or_uncertain"


def role_name(value: Any) -> str:
    return ROLE_ALIASES.get(str(value or "").strip(), str(value or "").strip() or "unknown")


def validate_evidence_chain(chain: Any, pre_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    pre_text = str(pre_text or "")
    rows: List[Dict[str, Any]] = []
    seen = set()
    if not isinstance(chain, list):
        chain = []
    for raw in chain:
        if not isinstance(raw, dict):
            continue
        span_text = str(raw.get("span_text", raw.get("text", "")) or "").strip()
        role = role_name(raw.get("role_label", "unknown"))
        start = pre_text.find(span_text) if span_text else -1
        end = start + len(span_text) if start >= 0 else -1
        duplicate = int(span_text in seen and bool(span_text))
        if span_text:
            seen.add(span_text)
        rows.append(
            {
                "role_label": role,
                "span_text": span_text,
                "span_start": start,
                "span_end": end,
                "pre_decision_flag": int(start >= 0),
                "valid_span_flag": int(start >= 0 and end <= len(pre_text)),
                "duplicate_flag": duplicate,
                "reason": str(raw.get("reason", ""))[:260],
            }
        )
    role_set = {r["role_label"] for r in rows if r["valid_span_flag"] == 1}
    expected_roles = {"entitlement", "notice_substantiation", "causality", "impact_schedule_relevance", "documentation_integrity"}
    denom = max(1, len(rows))
    metrics = {
        "valid_span_rate": round(sum(r["valid_span_flag"] for r in rows) / denom, 6),
        "pre_decision_span_rate": round(sum(r["pre_decision_flag"] for r in rows) / denom, 6),
        "duplicate_chain_rate": round(sum(r["duplicate_flag"] for r in rows) / denom, 6),
        "role_coverage_rate": round(len(role_set & expected_roles) / len(expected_roles), 6),
        "missing_role_rate": round(1.0 - len(role_set & expected_roles) / len(expected_roles), 6),
    }
    return rows, metrics


def normalize_llm_record(case_id: str, parsed: Dict[str, Any], pre_text: str) -> Dict[str, Any]:
    outcome = normalize_label(parsed.get("outcome_label", "unknown"))
    if outcome not in LABELS:
        outcome = "unknown"
    primary = normalize_resp(parsed.get("primary_responsible_party", "unknown"))
    secondary = normalize_resp(parsed.get("secondary_responsible_party", "unknown"))
    chain, chain_metrics = validate_evidence_chain(parsed.get("evidence_chain", []), pre_text)
    irac = parsed.get("delay_irac", {})
    if not isinstance(irac, dict):
        irac = {}
    return {
        "case_id": case_id,
        "api_status": "api_available",
        "parse_status": "parsed",
        "outcome_label": outcome,
        "outcome_confidence": clamp01(parsed.get("outcome_confidence", 0.0)),
        "primary_responsible_party": primary,
        "secondary_responsible_party": secondary,
        "responsibility_type": str(parsed.get("responsibility_type", "uncertain"))[:80],
        "responsibility_confidence": clamp01(parsed.get("responsibility_confidence", 0.0)),
        "uncertainty_flag": int(bool(parsed.get("uncertainty_flag", False)) or outcome == "unknown"),
        "evidence_chain_json": json.dumps(chain, ensure_ascii=False),
        "delay_irac_json": json.dumps(irac, ensure_ascii=False),
        "issue": str(irac.get("issue", ""))[:300],
        "rule": str(irac.get("rule", ""))[:500],
        "application": str(irac.get("application", ""))[:800],
        "conclusion": str(irac.get("conclusion", ""))[:400],
        "management_action": str(irac.get("management_action", parsed.get("recommended_management_action", "")))[:500],
        "documentation_gap_index": clamp01(parsed.get("documentation_gap_index", 0.0)),
        "procedural_compliance_risk": clamp01(parsed.get("procedural_compliance_risk", 0.0)),
        "causality_ambiguity": clamp01(parsed.get("causality_ambiguity", 0.0)),
        "concurrency_risk": clamp01(parsed.get("concurrency_risk", 0.0)),
        "critical_path_support": clamp01(parsed.get("critical_path_support", 0.0)),
        "negotiation_readiness_score": clamp01(parsed.get("negotiation_readiness_score", 0.0)),
        "managerial_failure_type": str(parsed.get("managerial_failure_type", ""))[:120],
        "recommended_management_action": str(parsed.get("recommended_management_action", ""))[:500],
        **chain_metrics,
    }


def recompute_outcome_metrics(rows: Iterable[Dict[str, Any]], labels: Sequence[str]) -> Dict[str, Any]:
    data = list(rows)
    y_true = [r["y_true"] for r in data]
    y_pred = [r["y_pred"] for r in data]
    n = len(data)
    cm = []
    per_class: Dict[str, Dict[str, float]] = {}
    for gold in labels:
        cm_row = []
        for pred in labels:
            cm_row.append(sum(1 for yt, yp in zip(y_true, y_pred) if yt == gold and yp == pred))
        cm.append(cm_row)
    total_correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    f1_values = []
    weighted_sum = 0.0
    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        support = sum(1 for yt in y_true if yt == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1-score": float(f1),
            "support": float(support),
        }
        f1_values.append(f1)
        weighted_sum += f1 * support
    return {
        "accuracy": float(total_correct / n) if n else 0.0,
        "macro_f1": float(sum(f1_values) / len(labels)) if labels else 0.0,
        "weighted_f1": float(weighted_sum / n) if n else 0.0,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


class QwenClient:
    def __init__(self, cfg: Dict[str, Any], api_key: str):
        llm_cfg = cfg["llm"]
        self.base_url = str(llm_cfg["api_base_url"]).rstrip("/")
        self.model = str(llm_cfg.get("model_name", "qwen-plus"))
        self.timeout = int(llm_cfg.get("timeout", 120))
        self.temperature = float(llm_cfg.get("temperature", 0.0))
        self.max_tokens = int(llm_cfg.get("max_tokens", 1800))
        self.api_key = api_key

    def chat(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        obj = response.json()
        content = obj["choices"][0]["message"]["content"]
        usage = obj.get("usage", {})
        return content, usage


def load_structured_cases(structured_dir: Path) -> Dict[str, Dict[str, Any]]:
    cases: Dict[str, Dict[str, Any]] = {}
    for fp in structured_dir.glob("*.json"):
        obj = json.loads(fp.read_text(encoding="utf-8"))
        cases[str(obj.get("case_id"))] = obj
    return cases


def load_candidate_rows(cfg: Dict[str, Any], max_cases: int = 0) -> pd.DataFrame:
    frames = []
    for key, dataset_name in [
        ("candidate_gold_strict_csv", "candidate_gold_strict_v1"),
        ("candidate_gold_extended_csv", "candidate_gold_extended_v1"),
    ]:
        fp = PROJECT_ROOT / cfg["paths"][key]
        df = read_csv_flexible(fp)
        df["case_id"] = df["case_id"].astype(str)
        df["dataset_name"] = df.get("dataset_name", dataset_name)
        df["_candidate_file"] = rel(fp)
        frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["candidate_outcome_label"] = all_rows["candidate_outcome_label"].map(normalize_label)
    all_rows["candidate_responsibility_label"] = all_rows["candidate_responsibility_label"].map(normalize_resp)
    all_rows = all_rows[all_rows["candidate_outcome_label"].isin(LABELS)].copy()
    if max_cases > 0:
        keep_ids = list(dict.fromkeys(all_rows["case_id"].tolist()))[:max_cases]
        all_rows = all_rows[all_rows["case_id"].isin(keep_ids)].copy()
    return all_rows


def one_api_case(case_id: str, structured: Dict[str, Any], cfg: Dict[str, Any], client: QwenClient) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    messages, prompt_text = build_prompt(case_id, structured, cfg)
    started = time.time()
    manifest = {
        "case_id": case_id,
        "model_name": cfg["llm"].get("model_name"),
        "prompt_template_version": cfg["llm"].get("prompt_template_version"),
        "prompt_sha256": sha256_text(prompt_text),
        "prompt_chars": len(prompt_text),
        "api_status": "not_started",
        "latency_sec": None,
        "error": "",
    }
    raw_row = {"case_id": case_id, "prompt_sha256": manifest["prompt_sha256"], "raw_response": "", "usage_json": "{}"}
    retries = int(cfg["llm"].get("retries", 2))
    for attempt in range(retries + 1):
        try:
            content, usage = client.chat(messages)
            parsed = extract_json_object(content)
            record = normalize_llm_record(case_id, parsed, structured.get("pre_decision_text", ""))
            manifest.update(
                {
                    "api_status": "api_available",
                    "parse_status": "parsed",
                    "latency_sec": round(time.time() - started, 4),
                    "response_chars": len(content),
                    "usage_json": json.dumps(usage, ensure_ascii=False),
                    "attempts": attempt + 1,
                }
            )
            raw_row.update({"raw_response": content, "usage_json": json.dumps(usage, ensure_ascii=False)})
            return record, raw_row, manifest
        except Exception as exc:
            manifest.update(
                {
                    "api_status": "api_error",
                    "parse_status": "failed",
                    "latency_sec": round(time.time() - started, 4),
                    "error": str(exc)[:1000],
                    "attempts": attempt + 1,
                }
            )
            if attempt < retries:
                time.sleep(1.0 + attempt * 1.5)
    return None, raw_row, manifest


def run_api_for_cases(case_ids: List[str], structured_cases: Dict[str, Dict[str, Any]], cfg: Dict[str, Any], out_dir: Path, api_key: str, resume: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_path = out_dir / "llm_raw_responses.jsonl"
    existing: Dict[str, Dict[str, Any]] = {}
    resume_path = out_dir / "llm_case_records.csv"
    if resume and not resume_path.exists() and (out_dir / "preflight_case_records.csv").exists():
        resume_path = out_dir / "preflight_case_records.csv"
    if resume and resume_path.exists():
        old = pd.read_csv(resume_path, encoding="utf-8-sig")
        if "case_id" in old.columns:
            existing = old.set_index("case_id").to_dict("index")

    client = QwenClient(cfg, api_key)
    records: List[Dict[str, Any]] = []
    manifests: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []

    pending = []
    for cid in case_ids:
        if cid in existing:
            records.append({"case_id": cid, **existing[cid]})
            manifests.append(
                {
                    "case_id": cid,
                    "model_name": cfg["llm"].get("model_name"),
                    "prompt_template_version": cfg["llm"].get("prompt_template_version"),
                    "api_status": "api_available",
                    "parse_status": "parsed_cached",
                    "latency_sec": 0.0,
                    "error": "",
                    "attempts": 0,
                    "resume_source": "cached_case_records",
                }
            )
        elif cid in structured_cases and str(structured_cases[cid].get("pre_decision_text", "")).strip():
            pending.append(cid)
        else:
            manifests.append({"case_id": cid, "api_status": "missing_pre_decision_text", "error": "structured case missing or empty pre_decision_text"})

    workers = int(cfg["llm"].get("workers", 1))
    sleep_seconds = float(cfg["llm"].get("sleep_seconds", 0.25))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for cid in pending:
            futures.append(executor.submit(one_api_case, cid, structured_cases[cid], cfg, client))
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        for fut in tqdm(as_completed(futures), total=len(futures), desc="True Qwen Copilot API"):
            record, raw_row, manifest = fut.result()
            manifests.append(manifest)
            raw_rows.append(raw_row)
            if record:
                records.append(record)

    if raw_rows:
        with raw_path.open("a", encoding="utf-8") as f:
            for row in raw_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rec_df = pd.DataFrame(records)
    api_df = pd.DataFrame(manifests)
    raw_df = pd.DataFrame(raw_rows)
    return rec_df, api_df, raw_df


def build_prediction_artifacts(records: pd.DataFrame, candidate_rows: pd.DataFrame, out_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = candidate_rows.merge(records, on="case_id", how="left")
    resolve_merged_record_columns(merged)
    merged["model_name"] = "true_qwen_direct"
    merged["eval_split"] = "external_candidate_eval"
    merged["y_true"] = merged["candidate_outcome_label"].map(normalize_label)
    merged["y_pred"] = merged["outcome_label"].map(normalize_label)
    merged["confidence"] = merged["outcome_confidence"].fillna(0.0)
    merged["api_status"] = merged["api_status"].fillna("missing_or_failed")
    valid_eval = merged[merged["api_status"].eq("api_available") & merged["y_pred"].isin(LABELS)].copy()

    pred_cols = [
        "case_id",
        "dataset_name",
        "eval_split",
        "model_name",
        "y_true",
        "y_pred",
        "confidence",
        "candidate_outcome_label",
        "candidate_responsibility_label",
        "primary_responsible_party",
        "secondary_responsible_party",
        "responsibility_type",
        "responsibility_confidence",
        "uncertainty_flag",
        "api_status",
        "valid_span_rate",
        "pre_decision_span_rate",
        "role_coverage_rate",
        "documentation_gap_index",
        "procedural_compliance_risk",
        "causality_ambiguity",
        "concurrency_risk",
        "critical_path_support",
        "negotiation_readiness_score",
        "managerial_failure_type",
        "recommended_management_action",
    ]
    for col in pred_cols:
        if col not in merged.columns:
            merged[col] = ""
    merged[pred_cols].to_csv(out_dir / "llm_predictions_main.csv", index=False, encoding="utf-8-sig")

    metrics: Dict[str, Any] = {
        "study_positioning": {
            "model": "true_qwen_direct",
            "input_constraint": "pre_decision_only",
            "candidate_benchmark_note": "Machine-assisted candidate benchmarks, not human gold.",
            "no_rule_fallback_for_headline": True,
        },
        "candidate_gold_evaluation": {},
    }
    per_class_rows = []
    cm_rows = []
    for dataset_name, sub in valid_eval.groupby("dataset_name"):
        rows = sub[["y_true", "y_pred"]].to_dict("records")
        metric = recompute_outcome_metrics(rows, LABELS)
        n_total = int((merged["dataset_name"] == dataset_name).sum())
        n_success = int(len(sub))
        success_rate = n_success / max(1, n_total)
        metric["n_total_candidate_rows"] = n_total
        metric["n_successful_api_rows"] = n_success
        metric["api_success_rate"] = float(success_rate)
        metric["audit_status"] = "complete" if success_rate >= float(cfg["eval"].get("min_success_rate_for_claimable_metric", 0.95)) else "partial_api_failures"
        metrics["candidate_gold_evaluation"][dataset_name] = {"true_qwen_direct": metric}

        for label in LABELS:
            item = metric["per_class"].get(label, {})
            per_class_rows.append(
                {
                    "dataset_name": dataset_name,
                    "model_name": "true_qwen_direct",
                    "class_label": label,
                    "precision": item.get("precision", 0.0),
                    "recall": item.get("recall", 0.0),
                    "f1": item.get("f1-score", 0.0),
                    "support": item.get("support", 0.0),
                }
            )
        cm = metric["confusion_matrix"]
        for i, gold in enumerate(LABELS):
            for j, pred in enumerate(LABELS):
                cm_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "model_name": "true_qwen_direct",
                        "gold_label": gold,
                        "pred_label": pred,
                        "count": cm[i][j] if cm else 0,
                    }
                )

    (out_dir / "metrics_main.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(per_class_rows).to_csv(out_dir / "per_class_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cm_rows).to_csv(out_dir / "confusion_matrix_data.csv", index=False, encoding="utf-8-sig")
    return metrics


def build_secondary_artifacts(records: pd.DataFrame, candidate_rows: pd.DataFrame, structured_cases: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    merged = candidate_rows.merge(records, on="case_id", how="left")
    resolve_merged_record_columns(merged)
    merged["api_status"] = merged["api_status"].fillna("missing_or_failed")
    resp_rows = []
    chain_rows = []
    mech_rows = []
    trace_rows = []
    error_rows = []

    for _, row in merged.iterrows():
        cid = str(row["case_id"])
        pred = normalize_label(row.get("outcome_label", "unknown"))
        gold = normalize_label(row.get("candidate_outcome_label", "unknown"))
        primary = normalize_resp(row.get("primary_responsible_party", "unknown"))
        gold_resp = normalize_resp(row.get("candidate_responsibility_label", "unknown"))
        folded_gold = fold_responsibility(gold_resp)
        folded_pred = fold_responsibility(primary)
        chain = []
        try:
            chain = json.loads(row.get("evidence_chain_json", "[]")) if isinstance(row.get("evidence_chain_json", ""), str) else []
        except Exception:
            chain = []
        valid_chain = [x for x in chain if x.get("valid_span_flag") == 1]
        evidence_consistency = int(primary != "unknown" and len(valid_chain) > 0)
        resp_rows.append(
            {
                "case_id": cid,
                "dataset_name": row.get("dataset_name"),
                "candidate_responsibility_label": gold_resp,
                "primary_responsible_party": primary,
                "secondary_responsible_party": normalize_resp(row.get("secondary_responsible_party", "unknown")),
                "folded_gold": folded_gold,
                "folded_pred": folded_pred,
                "responsibility_type": row.get("responsibility_type", ""),
                "responsibility_confidence": row.get("responsibility_confidence", 0.0),
                "uncertainty_flag": row.get("uncertainty_flag", 1),
                "evidence_consistency_rate": evidence_consistency,
                "api_status": row.get("api_status"),
            }
        )
        chain_rows.append(
            {
                "case_id": cid,
                "dataset_name": row.get("dataset_name"),
                "api_status": row.get("api_status"),
                "valid_span_rate": row.get("valid_span_rate", 0.0),
                "pre_decision_span_rate": row.get("pre_decision_span_rate", 0.0),
                "duplicate_chain_rate": row.get("duplicate_chain_rate", 0.0),
                "role_coverage_rate": row.get("role_coverage_rate", 0.0),
                "missing_role_rate": row.get("missing_role_rate", 1.0),
                "evidence_chain_json": row.get("evidence_chain_json", "[]"),
            }
        )
        mech_rows.append(
            {
                "case_id": cid,
                "dataset_name": row.get("dataset_name"),
                "documentation_gap_index": row.get("documentation_gap_index", 0.0),
                "procedural_compliance_risk": row.get("procedural_compliance_risk", 0.0),
                "causality_ambiguity": row.get("causality_ambiguity", 0.0),
                "concurrency_risk": row.get("concurrency_risk", 0.0),
                "critical_path_support": row.get("critical_path_support", 0.0),
                "negotiation_readiness_score": row.get("negotiation_readiness_score", 0.0),
                "managerial_failure_type": row.get("managerial_failure_type", ""),
                "recommended_management_action": row.get("recommended_management_action", ""),
                "api_status": row.get("api_status"),
            }
        )
        trace_rows.append(
            {
                "case_id": cid,
                "dataset_name": row.get("dataset_name"),
                "issue_focus": row.get("issue", ""),
                "rule_basis": row.get("rule", ""),
                "application_findings": row.get("application", ""),
                "conclusion": row.get("conclusion", ""),
                "management_action": row.get("management_action", ""),
                "evidence_citations": row.get("evidence_chain_json", "[]"),
                "responsibility_primary": primary,
                "outcome_label": pred,
                "high_dispute_flag": int(row.get("uncertainty_flag", 1) == 1 or row.get("outcome_confidence", 0.0) < 0.55),
                "api_status": row.get("api_status"),
            }
        )
        if row.get("api_status") == "api_available" and pred in LABELS and gold in LABELS and pred != gold:
            if gold == "partial" or pred == "partial":
                category = "partial_support_boundary_confusion"
            elif row.get("valid_span_rate", 0.0) < 0.5:
                category = "insufficient_evidence"
            elif row.get("causality_ambiguity", 0.0) >= 0.5:
                category = "ambiguous_causality"
            elif row.get("procedural_compliance_risk", 0.0) >= 0.5:
                category = "procedural_noncompliance_confusion"
            else:
                category = "model_label_mismatch"
            snippet = structured_cases.get(cid, {}).get("pre_decision_text", "")[:500]
            error_rows.append(
                {
                    "case_id": cid,
                    "dataset_name": row.get("dataset_name"),
                    "y_true": gold,
                    "y_pred": pred,
                    "error_category": category,
                    "case_snippet": snippet,
                    "api_status": row.get("api_status"),
                }
            )

    resp_df = pd.DataFrame(resp_rows)
    chain_df = pd.DataFrame(chain_rows)
    mech_df = pd.DataFrame(mech_rows)
    pd.DataFrame(trace_rows).to_json(out_dir / "reasoning_traces.jsonl", orient="records", lines=True, force_ascii=False)
    resp_df.to_csv(out_dir / "responsibility_eval.csv", index=False, encoding="utf-8-sig")
    chain_df.to_csv(out_dir / "evidence_chain_eval.csv", index=False, encoding="utf-8-sig")
    mech_df.to_csv(out_dir / "managerial_mechanisms.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(error_rows).to_csv(out_dir / "error_analysis.csv", index=False, encoding="utf-8-sig")

    resp_summary_rows = []
    valid_resp = resp_df[resp_df["api_status"].eq("api_available")]
    for dataset_name, sub in valid_resp.groupby("dataset_name"):
        fine = recompute_outcome_metrics(
            [{"y_true": r["candidate_responsibility_label"], "y_pred": r["primary_responsible_party"]} for _, r in sub.iterrows()],
            RESP_LABELS,
        )
        folded = recompute_outcome_metrics(
            [{"y_true": r["folded_gold"], "y_pred": r["folded_pred"]} for _, r in sub.iterrows()],
            FOLDED_RESP_LABELS,
        )
        resp_summary_rows.append(
            {
                "dataset_name": dataset_name,
                "fine_accuracy": fine["accuracy"],
                "fine_macro_f1": fine["macro_f1"],
                "folded_accuracy": folded["accuracy"],
                "folded_macro_f1": folded["macro_f1"],
                "uncertainty_rate": float(pd.to_numeric(sub["uncertainty_flag"], errors="coerce").fillna(1).mean()) if not sub.empty else 0.0,
                "evidence_consistency_rate": float(pd.to_numeric(sub["evidence_consistency_rate"], errors="coerce").fillna(0).mean()) if not sub.empty else 0.0,
            }
        )
    pd.DataFrame(resp_summary_rows).to_csv(out_dir / "responsibility_summary.csv", index=False, encoding="utf-8-sig")


def resolve_merged_record_columns(df: pd.DataFrame) -> None:
    """Prefer LLM record columns after merging with candidate labels.

    Candidate benchmark files already contain similarly named audit columns
    such as role_coverage_rate. Pandas suffixes these as _x/_y during merge,
    so paper-facing outputs must explicitly use the LLM record side.
    """

    record_cols = [
        "api_status",
        "parse_status",
        "outcome_label",
        "outcome_confidence",
        "primary_responsible_party",
        "secondary_responsible_party",
        "responsibility_type",
        "responsibility_confidence",
        "uncertainty_flag",
        "evidence_chain_json",
        "delay_irac_json",
        "issue",
        "rule",
        "application",
        "conclusion",
        "management_action",
        "documentation_gap_index",
        "procedural_compliance_risk",
        "causality_ambiguity",
        "concurrency_risk",
        "critical_path_support",
        "negotiation_readiness_score",
        "managerial_failure_type",
        "recommended_management_action",
        "valid_span_rate",
        "pre_decision_span_rate",
        "duplicate_chain_rate",
        "role_coverage_rate",
        "missing_role_rate",
    ]
    for col in record_cols:
        y_col = f"{col}_y"
        x_col = f"{col}_x"
        if y_col in df.columns:
            df[col] = df[y_col]
        elif col not in df.columns and x_col in df.columns:
            df[col] = df[x_col]


def build_model_comparison(out_dir: Path, metrics: Dict[str, Any]) -> None:
    ref_path = PROJECT_ROOT / "results" / "final_eval_20260409_194025" / "predictions_main.csv"
    rows = []
    if ref_path.exists():
        ref = pd.read_csv(ref_path, encoding="utf-8-sig")
        ref = ref[ref["dataset_name"].isin(["candidate_gold_strict_v1", "candidate_gold_extended_v1"])]
        for model in ["current_hybrid_baseline", "paesc_hybrid"]:
            sub_model = ref[ref["model_name"].eq(model)]
            for dataset_name, sub in sub_model.groupby("dataset_name"):
                if sub.empty:
                    continue
                metric = recompute_outcome_metrics(sub[["y_true", "y_pred"]].to_dict("records"), LABELS)
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "model_name": model,
                        "accuracy": metric["accuracy"],
                        "macro_f1": metric["macro_f1"],
                        "weighted_f1": metric["weighted_f1"],
                        "claim_note": "reference_recomputed_from_final_eval_20260409_194025",
                    }
                )
    for dataset_name, model_obj in metrics.get("candidate_gold_evaluation", {}).items():
        metric = model_obj.get("true_qwen_direct", {})
        rows.append(
            {
                "dataset_name": dataset_name,
                "model_name": "true_qwen_direct",
                "accuracy": metric.get("accuracy", 0.0),
                "macro_f1": metric.get("macro_f1", 0.0),
                "weighted_f1": metric.get("weighted_f1", 0.0),
                "claim_note": metric.get("audit_status", "unknown"),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")


def write_run_audit_files(out_dir: Path, cfg_path: Path, cfg: Dict[str, Any], api_key_source: str, command: str, audit_status: str) -> None:
    artifact_paths = [
        cfg_path,
        PROJECT_ROOT / "src" / "run_true_llm_copilot.py",
        PROJECT_ROOT / "src" / "llm_step2_fast_qc.py",
        PROJECT_ROOT / "src" / "research_support.py",
        PROJECT_ROOT / cfg["paths"]["candidate_gold_strict_csv"],
        PROJECT_ROOT / cfg["paths"]["candidate_gold_extended_csv"],
        out_dir / "llm_predictions_main.csv",
        out_dir / "metrics_main.json",
        out_dir / "responsibility_eval.csv",
        out_dir / "evidence_chain_eval.csv",
        out_dir / "managerial_mechanisms.csv",
        out_dir / "api_call_manifest.csv",
    ]
    manifest = build_run_manifest(
        PROJECT_ROOT,
        PROJECT_ROOT / "requirements.txt",
        artifact_paths,
        model_name=str(cfg["llm"].get("model_name", "qwen-plus")),
        prompt_template_version=str(cfg["llm"].get("prompt_template_version", "true_qwen_delay_copilot_v1")),
        embedding_model="none",
        label_schema_version="outcome_v1__responsibility_v1__candidate_gold_v1__true_llm_copilot_v1",
        command=command,
        seed=int(cfg["eval"].get("seed", 2026)),
        split_mode="external_candidate_eval",
        text_mode=str(cfg["llm"].get("text_mode", "pre_decision_only")),
        train_label_file=None,
        eval_label_file=PROJECT_ROOT / cfg["paths"]["candidate_gold_extended_csv"],
        metric_source_files=[out_dir / "llm_predictions_main.csv", out_dir / "responsibility_eval.csv", out_dir / "evidence_chain_eval.csv"],
        audit_status=audit_status,
        extra={
            "provider": cfg["llm"].get("provider", "dashscope_qwen"),
            "api_base_url": cfg["llm"].get("api_base_url"),
            "api_key_source": api_key_source,
            "allow_rule_fallback": bool(cfg["llm"].get("allow_rule_fallback", False)),
            "candidate_scope": cfg["run"].get("candidate_scope", "strict_plus_extended_deduplicated"),
        },
    )
    write_manifest(out_dir / "run_manifest.json", manifest)
    write_synthetic_git_summary(out_dir / "git_diff_summary.txt", "git unavailable in workspace; synthetic diff used")
    current_files = [p for p in artifact_paths if p.exists()]
    synthetic_file_diff([], current_files, PROJECT_ROOT, out_dir.name).to_csv(out_dir / "file_diff_summary.csv", index=False, encoding="utf-8-sig")


def preflight_gate(records: pd.DataFrame, api_manifest: pd.DataFrame, required_cases: int) -> Tuple[bool, str]:
    ok_api = int((api_manifest.get("api_status", pd.Series(dtype=str)) == "api_available").sum())
    if ok_api < required_cases:
        return False, f"preflight_api_success={ok_api}/{required_cases}"
    if records.empty or len(records) < required_cases:
        return False, f"preflight_parsed_records={len(records)}/{required_cases}"
    valid_span_rate = float(pd.to_numeric(records["valid_span_rate"], errors="coerce").fillna(0).mean())
    if valid_span_rate <= 0:
        return False, f"preflight_valid_span_rate={valid_span_rate:.4f}"
    return True, f"preflight_passed; valid_span_rate={valid_span_rate:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/research_true_llm.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--preflight-cases", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=0, help="Limit unique API cases for debugging. 0 means all candidate cases.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=0, help="Override API worker count.")
    parser.add_argument("--sleep-seconds", type=float, default=-1.0, help="Override submit delay between API calls.")
    parser.add_argument("--rebuild-from-records", action="store_true", help="Rebuild evaluation artifacts from existing llm_case_records.csv without API calls.")
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    cfg = load_yaml_config(cfg_path)
    if args.workers > 0:
        cfg["llm"]["workers"] = args.workers
    if args.sleep_seconds >= 0:
        cfg["llm"]["sleep_seconds"] = args.sleep_seconds
    api_key, api_key_source = resolve_api_key(cfg)
    out_dir = PROJECT_ROOT / args.out_dir if args.out_dir else PROJECT_ROOT / cfg["paths"].get("output_root", "results") / f"{cfg['run'].get('run_name_prefix', 'true_llm_copilot')}_{now_stamp()}"
    ensure_dir(out_dir)

    if not api_key:
        (out_dir / "api_unavailable.txt").write_text("No DashScope API key found from env/config/legacy script.\n", encoding="utf-8")
        write_run_audit_files(out_dir, cfg_path, cfg, api_key_source, " ".join(sys.argv), "api_unavailable")
        print(f"API unavailable. Wrote audit stub to {out_dir}")
        return 2

    structured_cases = load_structured_cases(PROJECT_ROOT / cfg["paths"]["structured_case_dir"])
    candidate_rows = load_candidate_rows(cfg, max_cases=args.max_cases)
    unique_case_ids = list(dict.fromkeys(candidate_rows["case_id"].astype(str).tolist()))

    if args.rebuild_from_records:
        records_path = out_dir / "llm_case_records.csv"
        if not records_path.exists():
            raise FileNotFoundError(f"Missing {records_path}; cannot rebuild artifacts without API records")
        records = pd.read_csv(records_path, encoding="utf-8-sig")
        if "api_call_manifest.csv" not in {p.name for p in out_dir.glob("*.csv")}:
            pd.DataFrame().to_csv(out_dir / "api_call_manifest.csv", index=False, encoding="utf-8-sig")
        metrics = build_prediction_artifacts(records, candidate_rows, out_dir, cfg)
        build_secondary_artifacts(records, candidate_rows, structured_cases, out_dir)
        build_model_comparison(out_dir, metrics)
        write_run_audit_files(out_dir, cfg_path, cfg, api_key_source, " ".join(sys.argv), "complete_rebuilt_from_true_api_records")
        print(f"Rebuilt artifacts from {records_path}")
        return 0

    preflight_n = args.preflight_cases if args.preflight_cases is not None else int(cfg["run"].get("preflight_cases", 5))
    preflight_ids = unique_case_ids[:preflight_n]

    print(f"Output dir: {out_dir}")
    print(f"API key source: {api_key_source}")
    print(f"Candidate rows: {len(candidate_rows)}; unique cases: {len(unique_case_ids)}")
    print(f"Running preflight cases: {len(preflight_ids)}")

    preflight_records, preflight_api, _ = run_api_for_cases(preflight_ids, structured_cases, cfg, out_dir, api_key, resume=args.resume)
    preflight_records.to_csv(out_dir / "preflight_case_records.csv", index=False, encoding="utf-8-sig")
    preflight_api.to_csv(out_dir / "preflight_api_manifest.csv", index=False, encoding="utf-8-sig")
    gate_ok, gate_msg = preflight_gate(preflight_records, preflight_api, len(preflight_ids))
    (out_dir / "preflight_report.txt").write_text(gate_msg + "\n", encoding="utf-8")
    if not gate_ok or args.preflight_only:
        audit_status = "preflight_only" if gate_ok else "preflight_failed"
        write_run_audit_files(out_dir, cfg_path, cfg, api_key_source, " ".join(sys.argv), audit_status)
        print(gate_msg)
        return 0 if gate_ok else 3

    print(gate_msg)
    print("Running full candidate benchmark API calls.")
    records, api_manifest, _ = run_api_for_cases(unique_case_ids, structured_cases, cfg, out_dir, api_key, resume=True)
    records.to_csv(out_dir / "llm_case_records.csv", index=False, encoding="utf-8-sig")
    api_manifest.to_csv(out_dir / "api_call_manifest.csv", index=False, encoding="utf-8-sig")
    failed = api_manifest[api_manifest["api_status"].ne("api_available")].copy() if not api_manifest.empty else pd.DataFrame()
    failed.to_csv(out_dir / "failed_cases.csv", index=False, encoding="utf-8-sig")

    metrics = build_prediction_artifacts(records, candidate_rows, out_dir, cfg)
    build_secondary_artifacts(records, candidate_rows, structured_cases, out_dir)
    build_model_comparison(out_dir, metrics)

    success_rates = [
        obj["true_qwen_direct"].get("api_success_rate", 0.0)
        for obj in metrics.get("candidate_gold_evaluation", {}).values()
        if "true_qwen_direct" in obj
    ]
    audit_status = "complete" if success_rates and min(success_rates) >= float(cfg["eval"].get("min_success_rate_for_claimable_metric", 0.95)) else "partial_api_failures"
    write_run_audit_files(out_dir, cfg_path, cfg, api_key_source, " ".join(sys.argv), audit_status)
    print(f"Done. Audit status: {audit_status}")
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
