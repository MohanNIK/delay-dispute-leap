# -*- coding: utf-8 -*-
"""Generate structured error analysis for candidate-gold evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_support import latest_run_dir, load_cfg  # noqa: E402


def load_structured_case(structured_dir: Path, case_id: str) -> Dict:
    fp = structured_dir / f"{case_id}.json"
    if not fp.exists():
        return {}
    return json.loads(fp.read_text(encoding="utf-8"))


def classify_error(row: pd.Series) -> str:
    if int(row.get("evidence_consistency_rate", 1)) == 0:
        return "responsibility_evidence_inconsistency"
    if float(row.get("valid_span_rate", 1.0)) < 0.70 or float(row.get("role_coverage_rate", 1.0)) < 0.45 or float(row.get("evidence_sufficiency", 1.0)) < 0.38:
        return "insufficient_evidence"
    if str(row.get("primary_responsible_party", "unknown")) == "both" or "双方" in str(row.get("causality_chain_summary", "")):
        return "concurrent_delay_conflict"
    if str(row.get("procedural_compliance_status", "uncertain")) == "noncompliant":
        return "procedural_noncompliance_confusion"
    if "partial" in {str(row.get("y_true", "")), str(row.get("y_pred", ""))}:
        return "partial_support_boundary_confusion"
    return "ambiguous_causality"


def build_case_snippet(structured: Dict, limit: int = 240) -> str:
    pre = structured.get("pre_decision_text", "") or ""
    return pre.replace("\n", " ").strip()[:limit]


def stringify_items(items: List[object], limit: int) -> str:
    parts: List[str] = []
    for item in items[:limit]:
        if isinstance(item, dict):
            text = item.get("event_text") or item.get("text") or item.get("summary") or json.dumps(item, ensure_ascii=False)
            parts.append(str(text))
        else:
            parts.append(str(item))
    return " | ".join([p for p in parts if p])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/research_v1.yaml")
    ap.add_argument("--run_dir", type=str, default="")
    args = ap.parse_args()

    cfg = load_cfg(PROJECT_ROOT / args.config)
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(PROJECT_ROOT / cfg["paths"]["final_eval_root"])
    if run_dir is None:
        raise FileNotFoundError("No final_eval_* run directory found.")
    run_dir = run_dir.resolve()

    pred = pd.read_csv(run_dir / "predictions_main.csv")
    resp = pd.read_csv(run_dir / "responsibility_eval.csv")
    chain = pd.read_csv(run_dir / "evidence_chain_eval.csv")
    structured_dir = PROJECT_ROOT / cfg["paths"]["structured_case_dir"]

    pred = pred[(pred["model_name"] == "paesc_hybrid") & (pred["dataset_name"].isin(["candidate_gold_strict_v1", "candidate_gold_extended_v1"]))]
    merged = pred.merge(resp, on=["case_id", "dataset_name", "eval_split"], how="left").merge(
        chain, on=["case_id", "dataset_name", "eval_split"], how="left", suffixes=("", "_chain")
    )
    if "confidence_x" in merged.columns and "confidence" not in merged.columns:
        merged = merged.rename(columns={"confidence_x": "confidence", "confidence_y": "responsibility_confidence"})
    merged = merged[merged["y_true"] != merged["y_pred"]].copy()

    rows: List[Dict[str, object]] = []
    for _, row in merged.iterrows():
        structured = load_structured_case(structured_dir, str(row["case_id"]))
        enriched = row.to_dict()
        enriched["evidence_sufficiency"] = structured.get("evidence_sufficiency", enriched.get("evidence_sufficiency", 0.0))
        enriched["error_category"] = classify_error(pd.Series(enriched))
        enriched["case_snippet"] = build_case_snippet(structured)
        enriched["delay_events_preview"] = stringify_items(structured.get("delay_events", []) or [], 3)
        enriched["claims_preview"] = stringify_items(structured.get("claims_defenses", {}).get("claims", []) or [], 2)
        enriched["defenses_preview"] = stringify_items(structured.get("claims_defenses", {}).get("defenses", []) or [], 2)
        enriched["source_file"] = structured.get("source_file", row.get("source_file", ""))
        rows.append(enriched)

    error_df = pd.DataFrame(rows)
    error_df = error_df.sort_values(["dataset_name", "error_category", "high_dispute_flag", "confidence"], ascending=[True, True, False, False])
    error_df.to_csv(run_dir / "error_analysis.csv", index=False, encoding="utf-8-sig")
    error_df.to_excel(run_dir / "error_analysis.xlsx", index=False)

    rep = (
        error_df.groupby(["dataset_name", "error_category"], group_keys=False)
        .head(4)
        .reset_index(drop=True)
    )
    rep.to_csv(run_dir / "representative_cases.csv", index=False, encoding="utf-8-sig")
    rep.to_excel(run_dir / "representative_cases.xlsx", index=False)
    print(f"[DONE] {run_dir / 'error_analysis.csv'}")


if __name__ == "__main__":
    main()
