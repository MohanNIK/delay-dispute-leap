# -*- coding: utf-8 -*-
"""Audit-first GPT-5.5/MMEC mechanism extraction for DelayDispute Copilot.

The script never treats local fallback outputs as real GPT-5.5 outputs. If the
API is unavailable, rows are marked as ``api_unavailable_rule_proxy`` so later
evaluation and manuscript text can keep the claim boundary honest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_support import (  # noqa: E402
    build_evidence_chain,
    diagnose_responsibility_from_pre,
    evidence_chain_metrics,
    evidence_sufficiency_score,
    json_dump,
    load_cfg,
    normalize_label,
    normalize_resp,
    read_csv_flexible,
    rule_baseline_prediction,
)
from src.audit_utils import build_run_manifest, write_manifest  # noqa: E402


LABELS = ["support", "partial", "not_support"]

DOC_GAP_KWS = ["缺乏证据", "未提交", "未提供", "资料缺失", "举证不足", "无证据", "不能证明", "证据不足"]
PROCEDURE_RISK_KWS = ["未通知", "未报审", "未备案", "未签证", "未验收", "未履行", "逾期申报", "未经确认"]
CAUSAL_AMBIGUITY_KWS = ["无法认定", "不能认定", "原因不明", "难以区分", "缺乏因果", "不能证明因果", "无法证明"]
CONCURRENCY_KWS = ["双方", "共同", "均有责任", "均存在", "交叉", "并发", "多种原因", "各方"]
CRITICAL_PATH_KWS = ["关键线路", "总工期", "工期顺延", "节点工期", "竣工日期", "进度计划", "停工", "窝工"]
NEGOTIATION_KWS = ["会议纪要", "签证", "确认单", "函件", "通知", "往来", "变更单", "监理日志", "施工日志"]


def norm01(count: int, denom: float) -> float:
    return round(max(0.0, min(1.0, count / denom)), 4)


def count_hits(text: str, keywords: Sequence[str]) -> int:
    return sum(1 for kw in keywords if kw in text)


def compact_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:max_chars]


def api_client_call(cfg: Dict, text: str, chain: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    llm_cfg = cfg.get("llm55", {})
    api_key = os.getenv(str(llm_cfg.get("api_key_env", "OPENAI_API_KEY")), "").strip()
    if not api_key:
        api_key = str(llm_cfg.get("debug_api_key", "")).strip()
    if not api_key:
        return None

    prompt = {
        "task": "construction_delay_dispute_pre_decision_mechanism_extraction",
        "constraint": "Use only the provided pre-decision text. Return JSON only.",
        "schema": {
            "outcome_label": "support|partial|not_support|unknown",
            "primary_responsible_party": "owner|contractor|subcontractor|designer_supervisor|both|force_majeure_policy|unknown",
            "documentation_gap_index": "0-1",
            "procedural_compliance_risk": "0-1",
            "causality_ambiguity": "0-1",
            "concurrency_risk": "0-1",
            "critical_path_support": "0-1",
            "negotiation_readiness_score": "0-1",
            "managerial_failure_type": "short label",
            "recommended_management_action": "short action",
            "evidence_citations": [{"role_label": "string", "text": "exact excerpt"}],
        },
        "pre_decision_text": compact_text(text, int(llm_cfg.get("max_chars", 5000))),
        "candidate_chain": chain,
    }
    payload = {
        "model": llm_cfg.get("model_name", "gpt-5.5"),
        "messages": [
            {"role": "system", "content": "You are an engineering-management dispute analysis assistant. Output JSON only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": int(llm_cfg.get("max_tokens", 1200)),
    }
    try:
        resp = requests.post(
            str(llm_cfg.get("api_base_url", "https://api.openai.com/v1")).rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=int(llm_cfg.get("timeout", 90)),
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
    except Exception:
        return None
    return None


def local_mechanism_from_structured(structured: Dict, *, api_status: str = "api_unavailable_rule_proxy") -> Dict[str, object]:
    pre_text = str(structured.get("pre_decision_text", "") or "")
    chain = structured.get("source_span_pointers") or build_evidence_chain(pre_text)
    chain_metrics = evidence_chain_metrics(chain)
    resp = diagnose_responsibility_from_pre(pre_text, "unknown", chain)

    doc_gap = norm01(count_hits(pre_text, DOC_GAP_KWS), 3.0)
    procedure_risk = norm01(count_hits(pre_text, PROCEDURE_RISK_KWS), 3.0)
    causality_ambiguity = norm01(count_hits(pre_text, CAUSAL_AMBIGUITY_KWS), 2.0)
    concurrency_risk = norm01(count_hits(pre_text, CONCURRENCY_KWS), 4.0)
    critical_path_support = norm01(count_hits(pre_text, CRITICAL_PATH_KWS), 4.0)
    negotiation_readiness = norm01(count_hits(pre_text, NEGOTIATION_KWS), 4.0)

    if doc_gap >= 0.67 or procedure_risk >= 0.67:
        outcome_label = "not_support"
    elif causality_ambiguity >= 0.5 or concurrency_risk >= 0.5:
        outcome_label = "partial"
    elif critical_path_support >= 0.5 and doc_gap < 0.34:
        outcome_label = "support"
    else:
        outcome_label = rule_baseline_prediction(structured)
    outcome_label = normalize_label(outcome_label)
    if outcome_label not in LABELS:
        outcome_label = "not_support"

    if doc_gap >= 0.67:
        failure_type = "documentation_gap"
        action = "补强同期记录、签证、通知和因果证据后再推进索赔或谈判"
    elif procedure_risk >= 0.67:
        failure_type = "procedural_noncompliance"
        action = "优先修复通知、报审、签证、验收等程序履约缺口"
    elif causality_ambiguity >= 0.5:
        failure_type = "causality_ambiguity"
        action = "补充关键线路、事件影响和责任边界证据"
    elif concurrency_risk >= 0.5:
        failure_type = "concurrent_delay_risk"
        action = "分解并发延误责任，准备比例化谈判方案"
    else:
        failure_type = "review_ready"
        action = "维持证据归档，准备类案对照和管理复盘"

    evidence_citations = []
    for item in chain:
        if item.get("text"):
            evidence_citations.append(
                {
                    "role_label": item.get("role_label", ""),
                    "text": str(item.get("text", ""))[:180],
                    "span_start": item.get("span_start", -1),
                    "span_end": item.get("span_end", -1),
                    "pre_decision_flag": item.get("pre_decision_flag", 0),
                }
            )

    return {
        "case_id": structured.get("case_id"),
        "source_file": structured.get("source_file", ""),
        "api_status": api_status,
        "mechanism_source": "local_rule_proxy" if api_status != "api_available" else "gpt-5.5",
        "gpt55_outcome_label": outcome_label,
        "gpt55_responsibility_label": normalize_resp(resp.get("primary_responsible_party", "unknown")),
        "documentation_gap_index": doc_gap,
        "procedural_compliance_risk": procedure_risk,
        "causality_ambiguity": causality_ambiguity,
        "concurrency_risk": concurrency_risk,
        "critical_path_support": critical_path_support,
        "negotiation_readiness_score": negotiation_readiness,
        "managerial_failure_type": failure_type,
        "recommended_management_action": action,
        "evidence_sufficiency": evidence_sufficiency_score(structured),
        "valid_span_rate": chain_metrics["valid_span_rate"],
        "pre_decision_span_rate": chain_metrics["pre_decision_span_rate"],
        "duplicate_chain_rate": chain_metrics["duplicate_chain_rate"],
        "role_coverage_rate": chain_metrics["role_coverage_rate"],
        "missing_role_rate": chain_metrics["missing_role_rate"],
        "managerial_mechanism_coverage": 1.0,
        "evidence_citations": evidence_citations,
    }


def merge_api_result(local: Dict[str, object], api_result: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not api_result:
        return local
    out = dict(local)
    out["api_status"] = "api_available"
    out["mechanism_source"] = "gpt-5.5"
    mapping = {
        "outcome_label": "gpt55_outcome_label",
        "primary_responsible_party": "gpt55_responsibility_label",
    }
    for src, dst in mapping.items():
        if src in api_result:
            out[dst] = api_result[src]
    for key in [
        "documentation_gap_index",
        "procedural_compliance_risk",
        "causality_ambiguity",
        "concurrency_risk",
        "critical_path_support",
        "negotiation_readiness_score",
        "managerial_failure_type",
        "recommended_management_action",
        "evidence_citations",
    ]:
        if key in api_result:
            out[key] = api_result[key]
    out["gpt55_outcome_label"] = normalize_label(out.get("gpt55_outcome_label", "unknown"))
    if out["gpt55_outcome_label"] not in LABELS:
        out["gpt55_outcome_label"] = local["gpt55_outcome_label"]
    out["gpt55_responsibility_label"] = normalize_resp(out.get("gpt55_responsibility_label", "unknown"))
    return out


def load_structured_cases(structured_dir: Path) -> Dict[str, Dict]:
    cases = {}
    for fp in structured_dir.glob("*.json"):
        obj = json.loads(fp.read_text(encoding="utf-8"))
        cases[str(obj.get("case_id"))] = obj
    return cases


def target_case_ids(cfg: Dict) -> List[str]:
    p = cfg["paths"]
    ids: List[str] = []
    for key in ["candidate_gold_strict_csv", "candidate_gold_extended_csv"]:
        fp = PROJECT_ROOT / p[key]
        if fp.exists():
            df = read_csv_flexible(fp)
            ids.extend(df["case_id"].astype(str).tolist())
            if cfg.get("llm55", {}).get("include_conflict_cases", True) and "conflict_flag" in df.columns:
                ids.extend(df[df["conflict_flag"].astype(int) == 1]["case_id"].astype(str).tolist())
    review_fp = PROJECT_ROOT / p.get("review_dir", "data/review") / "audit_subset_cases.csv"
    if cfg.get("llm55", {}).get("include_audit_subset", True) and review_fp.exists():
        review_df = read_csv_flexible(review_fp)
        if "case_id" in review_df.columns:
            ids.extend(review_df["case_id"].astype(str).tolist())
    return list(dict.fromkeys(ids))


def flatten_chain_rows(rows: Iterable[Dict[str, object]]) -> pd.DataFrame:
    out = []
    for row in rows:
        for item in row.get("evidence_citations", []) or []:
            out.append(
                {
                    "case_id": row["case_id"],
                    "role_label": item.get("role_label"),
                    "text": item.get("text"),
                    "span_start": item.get("span_start"),
                    "span_end": item.get("span_end"),
                    "pre_decision_flag": item.get("pre_decision_flag"),
                    "api_status": row.get("api_status"),
                    "mechanism_source": row.get("mechanism_source"),
                }
            )
    return pd.DataFrame(out)


def process_case(case_id: str, structured: Dict, cfg: Dict, use_api: bool) -> Dict[str, object]:
    local = local_mechanism_from_structured(structured)
    api_result = None
    if use_api:
        api_result = api_client_call(cfg, structured.get("pre_decision_text", ""), structured.get("source_span_pointers", []))
    return merge_api_result(local, api_result)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/research_v2_55.yaml")
    ap.add_argument("--use-api", action="store_true")
    ap.add_argument("--max-cases", type=int, default=0)
    args = ap.parse_args()

    cfg = load_cfg(PROJECT_ROOT / args.config)
    p = cfg["paths"]
    structured_cases = load_structured_cases(PROJECT_ROOT / p["structured_case_dir"])
    ids = [cid for cid in target_case_ids(cfg) if cid in structured_cases]
    if args.max_cases > 0:
        ids = ids[: args.max_cases]

    out_json_dir = PROJECT_ROOT / p["llm55_case_json_dir"]
    out_json_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(cfg.get("llm55", {}).get("workers", 1)))
    rows: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_case, cid, structured_cases[cid], cfg, args.use_api): cid for cid in ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="llm55_mechanism"):
            row = fut.result()
            rows.append(row)
            json_dump(out_json_dir / f"{row['case_id']}.json", row)

    mech_df = pd.DataFrame(rows).sort_values("case_id")
    labels_df = mech_df[
        [
            "case_id",
            "gpt55_outcome_label",
            "gpt55_responsibility_label",
            "api_status",
            "mechanism_source",
            "managerial_failure_type",
            "recommended_management_action",
        ]
    ].rename(
        columns={
            "gpt55_outcome_label": "delay_money_label",
            "gpt55_responsibility_label": "responsibility_hint",
        }
    )
    labels_df["key_reason"] = labels_df["managerial_failure_type"] + ": " + labels_df["recommended_management_action"]

    mech_out = PROJECT_ROOT / p["llm55_managerial_mechanisms_csv"]
    labels_out = PROJECT_ROOT / p["llm55_labels_csv"]
    chain_out = PROJECT_ROOT / p["llm55_evidence_chain_csv"]
    mech_out.parent.mkdir(parents=True, exist_ok=True)
    mech_df.drop(columns=["evidence_citations"]).to_csv(mech_out, index=False, encoding="utf-8-sig")
    labels_df.to_csv(labels_out, index=False, encoding="utf-8-sig")
    flatten_chain_rows(rows).to_csv(chain_out, index=False, encoding="utf-8-sig")

    manifest = build_run_manifest(
        PROJECT_ROOT,
        PROJECT_ROOT / "requirements.txt",
        [
            PROJECT_ROOT / args.config,
            PROJECT_ROOT / "src" / "llm55_mechanism_extraction.py",
            mech_out,
            labels_out,
            chain_out,
        ],
        model_name=cfg.get("llm55", {}).get("model_name", "gpt-5.5"),
        prompt_template_version=cfg.get("llm55", {}).get("prompt_template_version", "llm55_delay_mechanism_v1"),
        embedding_model=None,
        label_schema_version=cfg.get("mmec", {}).get("label_schema_version", "mmec_v1"),
        command=f"python src/llm55_mechanism_extraction.py --config {args.config}" + (" --use-api" if args.use_api else ""),
        seed=int(cfg["random"]["seed"]),
        split_mode=cfg.get("llm55", {}).get("target_scope", "audit_first_candidate_benchmarks"),
        text_mode="pre_decision_only",
        train_label_file=None,
        eval_label_file=PROJECT_ROOT / p["candidate_gold_extended_csv"],
        metric_source_files=[mech_out, labels_out, chain_out],
        audit_status="complete" if (mech_df["api_status"] == "api_available").any() else "api_unavailable",
        extra={
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "api_status_counts": mech_df["api_status"].value_counts().to_dict(),
            "claim_note": "Rows marked api_unavailable_rule_proxy are local MMEC mechanism proxies and must not be described as GPT-5.5 outputs.",
        },
    )
    write_manifest(PROJECT_ROOT / "results" / "llm55_extraction_manifest.json", manifest)
    print(f"[DONE] mechanisms: {mech_out}")
    print(f"[DONE] labels: {labels_out}")
    print(f"[DONE] evidence chain: {chain_out}")


if __name__ == "__main__":
    main()
