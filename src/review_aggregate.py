# -*- coding: utf-8 -*-
"""Aggregate human review sheets once reviewers fill the audit template."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_support import load_cfg, maybe_kappa  # noqa: E402


def read_review_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/research_v1.yaml")
    ap.add_argument("--review_file", type=str, default="data/review/reviewer_sheet_template.xlsx")
    args = ap.parse_args()

    cfg = load_cfg(PROJECT_ROOT / args.config)
    review_path = PROJECT_ROOT / args.review_file
    df = read_review_file(review_path)

    score_cols = [
        "relevance_score",
        "sufficiency_score",
        "traceability_score",
        "consistency_score",
        "managerial_usefulness_score",
    ]
    summary = {
        "n_rows": int(len(df)),
        "n_cases": int(df["case_id"].nunique()) if "case_id" in df.columns else 0,
        "n_reviewers": int(df["reviewer_id"].nunique()) if "reviewer_id" in df.columns else 0,
    }
    for col in score_cols:
        if col in df.columns:
            summary[f"{col}_mean"] = float(pd.to_numeric(df[col], errors="coerce").mean())

    summary.update(maybe_kappa(df, score_col="consistency_score"))
    summary.update({f"traceability_{k}": v for k, v in maybe_kappa(df, score_col="traceability_score").items()})

    disagreement = pd.DataFrame()
    if {"case_id", "reviewer_id", "consistency_score"}.issubset(df.columns):
        pivot = df.pivot_table(index="case_id", columns="reviewer_id", values="consistency_score", aggfunc="first")
        disagreement = pivot[pivot.nunique(axis=1) > 1].reset_index()

    out_dir = PROJECT_ROOT / cfg["paths"]["review_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "review_aggregate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not disagreement.empty:
        disagreement.to_csv(out_dir / "review_disagreement_cases.csv", index=False, encoding="utf-8-sig")
    print(f"[DONE] {out_dir / 'review_aggregate_summary.json'}")


if __name__ == "__main__":
    main()
