# -*- coding: utf-8 -*-
"""Build transparent candidate-gold datasets and audit-ready review subsets.

Important scientific note:
- This script never claims human gold.
- Outputs are candidate-gold / machine-assisted labels with provenance and uncertainty flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_support import (
    LABELS,
    RESP_EN_TO_ZH,
    build_evidence_chain,
    derive_outcome_from_post,
    derive_responsibility_candidate,
    ensure_dir,
    evidence_chain_metrics,
    export_table,
    json_dump,
    load_cfg,
    normalize_label,
    normalize_resp,
    read_csv_flexible,
)


def load_structured_cases(structured_dir: Path) -> Dict[str, Dict]:
    cases = {}
    for fp in structured_dir.glob("*.json"):
        obj = json.loads(fp.read_text(encoding="utf-8"))
        cases[str(obj.get("case_id"))] = obj
    return cases


def build_record(case_id: str, structured: Dict, weak_row: Dict | None, llm_row: Dict | None, seed_row: Dict | None) -> Dict:
    weak_label = normalize_label((weak_row or {}).get("eot_label", "unknown"))
    llm_label = normalize_label((llm_row or {}).get("delay_money_label", "unknown"))
    post_label, post_evidence = derive_outcome_from_post(structured.get("post_decision_text", ""))
    candidate_outcome = "unknown"
    source_parts = []
    available = [x for x in [weak_label, llm_label, post_label] if x != "unknown"]
    if seed_row is not None:
        candidate_outcome = normalize_label(seed_row.get("gpt_label", "unknown"))
        source_parts.append("seed_reference")
    elif weak_label != "unknown" and weak_label == post_label:
        candidate_outcome = weak_label
        source_parts.append("weak+post_agree")
    elif llm_label != "unknown" and llm_label == post_label:
        candidate_outcome = llm_label
        source_parts.append("llm+post_agree")
    elif weak_label != "unknown" and llm_label != "unknown" and weak_label == llm_label:
        candidate_outcome = weak_label
        source_parts.append("weak+llm_agree")
    elif post_label != "unknown":
        candidate_outcome = post_label
        source_parts.append("post_only")
    elif weak_label != "unknown":
        candidate_outcome = weak_label
        source_parts.append("weak_only")
    elif llm_label != "unknown":
        candidate_outcome = llm_label
        source_parts.append("llm_only")
    conflict_flag = int(len(set(available)) > 1)

    resp_label, resp_conf, resp_source = derive_responsibility_candidate(structured.get("post_decision_text", ""), (llm_row or {}).get("responsibility_hint", ""))
    if seed_row is not None and resp_label == "unknown":
        resp_label = normalize_resp((llm_row or {}).get("responsibility_hint", "unknown"))
    chain = build_evidence_chain(structured.get("pre_decision_text", ""))
    chain_metrics = evidence_chain_metrics(chain)

    confidence = 0.2
    if seed_row is not None:
        confidence += 0.25
        try:
            confidence += min(float(seed_row.get("gpt_conf", 0.7)) * 0.3, 0.3)
        except Exception:
            confidence += 0.2
    if post_label != "unknown":
        confidence += 0.25
    if weak_label != "unknown" and weak_label == candidate_outcome:
        confidence += 0.15
    if llm_label != "unknown" and llm_label == candidate_outcome:
        confidence += 0.15
    if resp_label != "unknown":
        confidence += 0.08
    confidence += chain_metrics["role_coverage_rate"] * 0.12
    if conflict_flag:
        confidence -= 0.18
    confidence = round(max(0.05, min(confidence, 0.98)), 4)

    needs_review = int(conflict_flag == 1 or confidence < 0.78 or resp_label == "unknown")
    evidence_span = (seed_row or {}).get("gpt_evidence", "") if seed_row is not None else post_evidence
    if not evidence_span:
        evidence_span = next((x.get("text", "") for x in chain if x.get("text")), "")[:220]

    return {
        "case_id": case_id,
        "source_file": structured.get("source_file", ""),
        "candidate_outcome_label": candidate_outcome,
        "candidate_responsibility_label": resp_label,
        "evidence_span": evidence_span,
        "generation_source": "+".join(source_parts) if source_parts else "insufficient_signal",
        "confidence": confidence,
        "conflict_flag": conflict_flag,
        "needs_review": needs_review,
        "note": f"weak={weak_label}; llm={llm_label}; post={post_label}; resp_source={resp_source}",
        "weak_label": weak_label,
        "llm_label": llm_label,
        "post_label": post_label,
        "llm_responsibility_hint": normalize_resp((llm_row or {}).get("responsibility_hint", "unknown")),
        "role_coverage_rate": round(chain_metrics["role_coverage_rate"], 4),
        "missing_role_rate": round(chain_metrics["missing_role_rate"], 4),
        "pre_post_split_confidence": structured.get("pre_post_split_confidence", 0.0),
        "potential_leakage_flag": structured.get("potential_leakage_flag", 0),
        "case_year": structured.get("case_year"),
        "is_domain_case": structured.get("is_domain_case", 0),
    }


def balanced_select(df: pd.DataFrame, target: int, min_per_label: int) -> List[str]:
    selected: List[str] = []
    for lb in LABELS:
        part = df[df["candidate_outcome_label"] == lb].sort_values(["confidence", "role_coverage_rate"], ascending=False)
        selected.extend(part.head(min(min_per_label, len(part))).case_id.tolist())
    if len(selected) < target:
        rest = df[~df["case_id"].isin(selected)].sort_values(["confidence", "role_coverage_rate"], ascending=False)
        selected.extend(rest.head(max(0, target - len(selected))).case_id.tolist())
    return selected[:target]


def build_review_package(cfg: Dict, strict_df: pd.DataFrame, review_dir: Path) -> None:
    ensure_dir(review_dir)
    audit_size = int(cfg["candidate_gold"].get("audit_subset_size", 60))
    picks = []
    for lb in LABELS:
        part = strict_df[strict_df["candidate_outcome_label"] == lb].sort_values(["needs_review", "confidence"], ascending=[False, True])
        picks.append(part.head(max(5, audit_size // len(LABELS))))
    subset = pd.concat(picks, ignore_index=True).drop_duplicates(subset=["case_id"]).head(audit_size)
    subset.to_csv(review_dir / "audit_subset_cases.csv", index=False, encoding="utf-8-sig")

    reviewer_cols = [
        "reviewer_id",
        "case_id",
        "task_type",
        "model_output",
        "source_evidence",
        "relevance_score",
        "sufficiency_score",
        "traceability_score",
        "consistency_score",
        "managerial_usefulness_score",
        "reviewer_comment",
    ]
    template = pd.DataFrame(columns=reviewer_cols)
    with pd.ExcelWriter(review_dir / "reviewer_sheet_template.xlsx", engine="openpyxl") as writer:
        subset.to_excel(writer, sheet_name="audit_subset_cases", index=False)
        template.to_excel(writer, sheet_name="review_sheet_template", index=False)

    (review_dir / "review_protocol_v1.md").write_text(
        "# Review Protocol v1\n"
        "This package supports later human or expert audit. It does not contain human-reviewed scores yet.\n\n"
        "Review tasks:\n"
        "1. outcome label plausibility\n"
        "2. responsibility diagnosis plausibility\n"
        "3. evidence relevance\n"
        "4. evidence sufficiency\n"
        "5. evidence traceability\n"
        "6. responsibility-evidence consistency\n"
        "7. managerial usefulness / actionability\n\n"
        "Suggested scale: 1-5, where 1=very poor and 5=very strong.\n",
        encoding="utf-8",
    )
    (review_dir / "human_review_instructions.md").write_text(
        "# Human Review Instructions\n"
        "Use the audit subset and review sheet template. Reviewers should inspect the candidate output, cited evidence, and pre-decision narrative only.\n\n"
        "Do not score whether the court was legally correct. Score whether the candidate output is plausible, traceable, and useful for pre-adjudication dispute triage.\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/research_v1.yaml")
    args = ap.parse_args()

    cfg = load_cfg(PROJECT_ROOT / args.config)
    p = cfg["paths"]
    structured_dir = PROJECT_ROOT / p["structured_case_dir"]
    candidate_strict_path = PROJECT_ROOT / p["candidate_gold_strict_csv"]
    candidate_extended_path = PROJECT_ROOT / p["candidate_gold_extended_csv"]
    sampling_manifest_path = PROJECT_ROOT / p["gold_sampling_manifest_csv"]
    provenance_manifest_path = PROJECT_ROOT / p["gold_provenance_manifest_csv"]
    qc_report_path = PROJECT_ROOT / p["gold_qc_txt"]
    guideline_path = PROJECT_ROOT / p["gold_guideline_md"]
    review_dir = PROJECT_ROOT / p["review_dir"]

    weak = read_csv_flexible(PROJECT_ROOT / p["meta_labels_csv"])
    llm = read_csv_flexible(PROJECT_ROOT / p["llm_labels_csv"])
    seed = read_csv_flexible(PROJECT_ROOT / p["seed_reference_csv"])
    structured_cases = load_structured_cases(structured_dir)

    weak["case_id"] = weak.get("case_id", pd.Series(dtype=str)).astype(str)
    llm["case_id"] = llm.get("case_id", pd.Series(dtype=str)).astype(str)
    seed["case_id"] = seed.get("case_id", pd.Series(dtype=str)).astype(str)

    weak_map = {r["case_id"]: r for _, r in weak.to_dict("index").items()} if not weak.empty else {}
    llm_map = {r["case_id"]: r for _, r in llm.to_dict("index").items()} if not llm.empty else {}
    seed_map = {r["case_id"]: r for _, r in seed.to_dict("index").items()} if not seed.empty else {}

    records = []
    for case_id, structured in structured_cases.items():
        if int(structured.get("is_domain_case", 0)) != 1:
            continue
        rec = build_record(case_id, structured, weak_map.get(case_id), llm_map.get(case_id), seed_map.get(case_id))
        records.append(rec)

    prov = pd.DataFrame(records)
    prov = prov[prov["candidate_outcome_label"].isin(LABELS)].copy()
    prov.sort_values(["confidence", "role_coverage_rate"], ascending=False, inplace=True)

    strict_pool = prov[
        (prov["conflict_flag"] == 0)
        & (prov["evidence_span"].astype(str).str.len() > 0)
        & (prov["pre_post_split_confidence"] >= 0.55)
    ].copy()
    strict_ids = balanced_select(
        strict_pool,
        int(cfg["candidate_gold"]["strict_target_size"]),
        int(cfg["candidate_gold"]["strict_min_per_label"]),
    )
    strict_df = strict_pool[strict_pool["case_id"].isin(strict_ids)].copy()
    strict_df["dataset_name"] = "candidate_gold_strict_v1"

    extended_pool = prov.copy()
    extended_ids = balanced_select(
        extended_pool,
        int(cfg["candidate_gold"]["extended_target_size"]),
        int(cfg["candidate_gold"]["extended_min_per_label"]),
    )
    extended_df = extended_pool[extended_pool["case_id"].isin(extended_ids)].copy()
    extended_df["dataset_name"] = "candidate_gold_extended_v1"

    strict_df.to_csv(candidate_strict_path, index=False, encoding="utf-8-sig")
    extended_df.to_csv(candidate_extended_path, index=False, encoding="utf-8-sig")

    sampling_manifest = pd.concat([
        strict_df.assign(selection_dataset="candidate_gold_strict_v1"),
        extended_df.assign(selection_dataset="candidate_gold_extended_v1"),
    ], ignore_index=True)
    sampling_manifest.to_csv(sampling_manifest_path, index=False, encoding="utf-8-sig")
    prov.to_csv(provenance_manifest_path, index=False, encoding="utf-8-sig")

    qc_lines = []
    for name, df in [("candidate_gold_strict_v1", strict_df), ("candidate_gold_extended_v1", extended_df)]:
        qc_lines.append(f"[{name}]")
        qc_lines.append(f"n={len(df)}")
        qc_lines.append("outcome_distribution:")
        qc_lines.append(df["candidate_outcome_label"].value_counts(dropna=False).to_string())
        qc_lines.append("responsibility_distribution:")
        qc_lines.append(df["candidate_responsibility_label"].value_counts(dropna=False).to_string())
        qc_lines.append(f"conflict_rate={df['conflict_flag'].mean():.4f}")
        qc_lines.append(f"needs_review_rate={df['needs_review'].mean():.4f}")
        qc_lines.append(f"avg_confidence={df['confidence'].mean():.4f}")
        qc_lines.append("")
    qc_report_path.write_text("\n".join(qc_lines), encoding="utf-8")

    guideline_path.write_text(
        "# Annotation / Candidate-Gold Guideline v1\n"
        "This project currently uses machine-assisted candidate labels, not verified human gold labels.\n\n"
        "## Label sets\n"
        "- outcome: support / partial / not_support\n"
        "- responsibility: owner / contractor / subcontractor / designer_supervisor / both / force_majeure_policy / unknown\n\n"
        "## Record meaning\n"
        "- candidate_outcome_label: machine-assisted candidate target for evaluation\n"
        "- candidate_responsibility_label: machine-assisted candidate responsibility label\n"
        "- evidence_span: supporting text snippet used for provenance\n"
        "- generation_source: which sources agreed or contributed\n"
        "- confidence: automatic confidence score, not human validation\n"
        "- conflict_flag: disagreement among available weak/LLM/post-decision signals\n"
        "- needs_review: recommended for later human or expert audit\n",
        encoding="utf-8",
    )

    build_review_package(cfg, strict_df, review_dir)
    print("[DONE] candidate-gold and review package generated")


if __name__ == "__main__":
    main()
