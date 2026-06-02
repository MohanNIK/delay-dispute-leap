# -*- coding: utf-8 -*-
"""Merge parallel Mimo screening batches into a candidate-document corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="data/external_mimo_candidates/wikisource_30k_mimo_delay2_20260527")
    args = ap.parse_args()

    base = Path(args.base_dir)
    root = base / "parallel_batches"
    pool = read_csv(base / "candidate_pool_prefilter.csv")
    if pool.empty:
        raise SystemExit(f"missing candidate pool: {base / 'candidate_pool_prefilter.csv'}")
    pool["pageid"] = pool["pageid"].astype(str)

    frames: List[pd.DataFrame] = []
    seed = root / "seed_completed_mimo_screening_results.csv"
    if seed.exists():
        frames.append(read_csv(seed))
    for path in sorted(root.glob("batch_*/mimo_screening_results.csv")):
        frames.append(read_csv(path))
    screening = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if screening.empty:
        raise SystemExit("no screening results to merge yet")
    screening["pageid"] = screening["pageid"].astype(str)
    screening = screening.drop_duplicates("pageid", keep="last").reset_index(drop=True)

    merge_cols = [
        "pageid",
        "mimo_status",
        "model_name",
        "mimo_accept_flag",
        "case_related_to_construction",
        "schedule_delay_material_issue",
        "substantive_facts_available",
        "adjudicated_outcome_available",
        "procedural_only",
        "pre_decision_facts_sufficient",
        "evidence_completeness",
        "overall_completeness_score",
        "recommended_bucket",
        "project_management_relevance",
        "main_delay_terms",
        "evidence_types_found",
        "reason_short",
        "error",
    ]
    available = [c for c in merge_cols if c in screening.columns]
    merged = pool.merge(screening[available], on="pageid", how="left")
    merged["mimo_screened_flag"] = merged["mimo_status"].notna().astype(int)
    merged["mimo_accept_flag"] = pd.to_numeric(merged.get("mimo_accept_flag", 0), errors="coerce").fillna(0).astype(int)
    for col in ["overall_completeness_score", "evidence_completeness", "pre_decision_facts_sufficient"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
        else:
            merged[col] = 0.0
    for col in [
        "case_related_to_construction",
        "schedule_delay_material_issue",
        "substantive_facts_available",
        "adjudicated_outcome_available",
        "procedural_only",
    ]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)
        else:
            merged[col] = 0
    risk_text = merged.get("reason_short", pd.Series([""] * len(merged), index=merged.index)).astype(str)
    incomplete_risk = risk_text.str.contains("不完整|不足|缺失|不充分|不清楚|无法|不明确", regex=True, na=False)
    merged["strict_info_complete_flag"] = (
        merged["mimo_accept_flag"].eq(1)
        & merged["case_related_to_construction"].eq(1)
        & merged["schedule_delay_material_issue"].eq(1)
        & merged["substantive_facts_available"].eq(1)
        & merged["adjudicated_outcome_available"].eq(1)
        & merged["procedural_only"].eq(0)
        & merged["overall_completeness_score"].ge(0.75)
        & merged["evidence_completeness"].ge(0.65)
        & merged["pre_decision_facts_sufficient"].ge(0.65)
        & ~incomplete_risk
    ).astype(int)
    accepted = merged[merged["strict_info_complete_flag"].eq(1)].copy().reset_index(drop=True)

    out_all = base / "mimo_screening_results_merged.csv"
    out_accepted = base / "mimo_delay_dispute_candidates_merged.csv"
    out_manifest = base / "mimo_delay_dispute_candidates_manifest_merged.csv"
    merged.to_csv(out_all, index=False, encoding="utf-8-sig")
    accepted.to_csv(out_accepted, index=False, encoding="utf-8-sig")
    manifest_cols = [c for c in accepted.columns if c != "raw_text"]
    accepted[manifest_cols].to_csv(out_manifest, index=False, encoding="utf-8-sig")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(base),
        "candidate_pool_size": int(len(pool)),
        "screened_unique": int(screening["pageid"].nunique()),
        "mimo_accepted_raw": int(merged["mimo_accept_flag"].sum()),
        "accepted": int(len(accepted)),
        "acceptance_rule": "strict_info_complete_flag: Mimo accept plus construction/delay/substantive/adjudicated/nonprocedural plus completeness/evidence/pre-decision thresholds and no incomplete-risk wording",
        "api_error": int((screening.get("mimo_status", pd.Series(dtype=str)) == "api_error").sum()),
        "accept_rate_among_screened": round(float(len(accepted) / max(1, screening["pageid"].nunique())), 4),
        "outputs": {
            "merged_screening": str(out_all),
            "accepted_full_text": str(out_accepted),
            "accepted_manifest": str(out_manifest),
        },
        "artifact_hashes": {
            out_all.name: sha256_file(out_all),
            out_accepted.name: sha256_file(out_accepted),
            out_manifest.name: sha256_file(out_manifest),
        },
    }
    (base / "mimo_merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
