# -*- coding: utf-8 -*-
"""Build a staged usable-corpus manifest for the public 100k expansion."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect-dir", default="data/1_raw_text/mimo_public_100k_collect_20260527")
    ap.add_argument("--mimo-dir", default="data/1_raw_text/mimo_public_100k_collect_20260527/mimo_screening_high_quality")
    ap.add_argument("--out-dir", default="data/1_raw_text/public100k_usable_corpus_20260527")
    args = ap.parse_args()

    collect_dir = Path(args.collect_dir)
    mimo_dir = Path(args.mimo_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_csv(collect_dir / "raw_text_manifest.csv")
    audit = read_csv(collect_dir / "local_quality_audit.csv")
    mimo = read_csv(mimo_dir / "mimo_screening_results.csv")
    if manifest.empty or audit.empty:
        raise SystemExit("raw_text_manifest.csv/local_quality_audit.csv missing")
    manifest["pageid"] = manifest["pageid"].astype(str)
    audit["pageid"] = audit["pageid"].astype(str)
    if not mimo.empty and "pageid" in mimo.columns:
        mimo["pageid"] = mimo["pageid"].astype(str)

    for col in [
        "local_quality_score",
        "delay_hits",
        "construction_hits",
        "evidence_hits",
        "decision_hits",
        "text_chars",
        "procedural_only_flag",
        "judgment_title_flag",
        "substantive_decision_flag",
    ]:
        if col in audit.columns:
            audit[col] = pd.to_numeric(audit[col], errors="coerce").fillna(0)

    audit_cols = [
        "pageid",
        "case_year",
        "text_chars",
        "construction_hits",
        "delay_hits",
        "evidence_hits",
        "decision_hits",
        "procedural_hits",
        "judgment_title_flag",
        "substantive_decision_flag",
        "procedural_only_flag",
        "local_quality_score",
    ]
    df = manifest.merge(audit[[c for c in audit_cols if c in audit.columns]], on="pageid", how="left", suffixes=("", "_audit"))
    if not mimo.empty:
        keep = [
            "pageid",
            "mimo_status",
            "model_name",
            "mimo_accept_flag",
            "strict_info_complete_flag",
            "recommended_bucket",
            "overall_completeness_score",
            "evidence_completeness",
            "pre_decision_facts_sufficient",
            "project_management_relevance",
            "reason_short",
            "main_delay_terms",
            "evidence_types_found",
        ]
        df = df.merge(mimo[[c for c in keep if c in mimo.columns]], on="pageid", how="left")

    for col in ["mimo_accept_flag", "strict_info_complete_flag"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0
    for col in ["text_chars", "local_quality_score", "delay_hits", "construction_hits", "evidence_hits", "decision_hits", "procedural_only_flag", "judgment_title_flag", "substantive_decision_flag"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    local_candidate = (
        df["text_chars"].ge(2500)
        & df["local_quality_score"].ge(0.45)
        & df["delay_hits"].ge(1)
        & df["construction_hits"].ge(1)
        & df["procedural_only_flag"].eq(0)
    )
    high_quality = (
        df["text_chars"].ge(2500)
        & df["local_quality_score"].ge(0.55)
        & df["delay_hits"].ge(2)
        & df["construction_hits"].ge(1)
        & df["evidence_hits"].ge(1)
        & df["decision_hits"].ge(2)
        & df["procedural_only_flag"].eq(0)
        & df["judgment_title_flag"].eq(1)
        & df["substantive_decision_flag"].eq(1)
    )

    df["usable_tier"] = "excluded_or_pending"
    df.loc[local_candidate, "usable_tier"] = "Tier_B_local_delay_candidate"
    df.loc[high_quality, "usable_tier"] = "Tier_B_plus_high_quality_local"
    df.loc[df["mimo_accept_flag"].eq(1), "usable_tier"] = "Tier_A_mimo_strict_complete"
    df["usable_for_research_flag"] = df["usable_tier"].ne("excluded_or_pending").astype(int)
    df["usage_scope"] = df["usable_tier"].map(
        {
            "Tier_A_mimo_strict_complete": "strict Mimo-screened corpus; suitable for high-confidence RAG/corpus analysis, not human gold",
            "Tier_B_plus_high_quality_local": "high-quality local corpus; suitable for RAG/future Mimo screening",
            "Tier_B_local_delay_candidate": "broad local delay corpus; suitable for retrieval/corpus statistics",
            "excluded_or_pending": "not currently usable",
        }
    )

    usable = df[df["usable_for_research_flag"].eq(1)].drop_duplicates("text_sha256", keep="first").reset_index(drop=True)
    excluded = df[df["usable_for_research_flag"].eq(0)].drop_duplicates("text_sha256", keep="first").reset_index(drop=True)
    usable_manifest = out_dir / "public100k_usable_manifest.csv"
    excluded_manifest = out_dir / "public100k_excluded_or_pending_manifest.csv"
    usable.to_csv(usable_manifest, index=False, encoding="utf-8-sig")
    excluded.to_csv(excluded_manifest, index=False, encoding="utf-8-sig")
    tier = usable["usable_tier"].value_counts().rename_axis("usable_tier").reset_index(name="count")
    tier.to_csv(out_dir / "public100k_usable_tier_distribution.csv", index=False, encoding="utf-8-sig")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_text_manifest_rows": int(len(manifest)),
        "unique_raw_text_rows_after_dedup": int(df["text_sha256"].nunique()) if "text_sha256" in df.columns else int(len(df)),
        "local_candidate_delay_ge1": int(local_candidate.sum()),
        "strict_high_quality_local_pool": int(high_quality.sum()),
        "mimo_screened_rows": int(len(mimo)),
        "mimo_strict_accepted": int(df["mimo_accept_flag"].sum()),
        "usable_for_research_total": int(len(usable)),
        "excluded_or_pending_total": int(len(excluded)),
        "note": "Tier B is not a validated label benchmark. It is a usable corpus for retrieval, statistics, and future labeling.",
        "outputs": {
            "usable_manifest": str(usable_manifest),
            "excluded_manifest": str(excluded_manifest),
            "tier_distribution": str(out_dir / "public100k_usable_tier_distribution.csv"),
        },
    }
    (out_dir / "public100k_usable_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
