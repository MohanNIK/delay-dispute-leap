# -*- coding: utf-8 -*-
"""Build a tiered usable corpus from 30k public candidates and Mimo screening.

Tier A is strict Mimo-accepted complete construction delay dispute cases.
Tier B is complete construction-adjudication text with delay indicators, usable
for RAG, retrieval, corpus statistics, and future LLM labeling, but not as a
verified outcome-label benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

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
    ap.add_argument("--collect-dir", default="data/external_mimo_candidates/wikisource_30k_collect_20260527_122315")
    ap.add_argument("--mimo-dir", default="data/external_mimo_candidates/wikisource_30k_mimo_delay2_20260527")
    ap.add_argument("--out-dir", default="data/external_mimo_candidates/research_usable_corpus_30k_20260527")
    args = ap.parse_args()

    collect_dir = Path(args.collect_dir)
    mimo_dir = Path(args.mimo_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = read_csv(collect_dir / "local_quality_audit.csv")
    merged = read_csv(mimo_dir / "mimo_screening_results_merged.csv")
    if audit.empty:
        raise SystemExit("local_quality_audit.csv missing or empty")
    audit["pageid"] = audit["pageid"].astype(str)
    if not merged.empty:
        merged["pageid"] = merged["pageid"].astype(str)
    mimo_cols = [
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
    available = [c for c in mimo_cols if c in merged.columns]
    df = audit.merge(merged[available], on="pageid", how="left") if available else audit.copy()
    df = df.drop_duplicates("text_sha256", keep="first").reset_index(drop=True)

    for col in ["mimo_accept_flag", "strict_info_complete_flag"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0
    df["mimo_screened_flag"] = df.get("mimo_status", pd.Series([""] * len(df))).notna().astype(int)

    common_complete = (
        df["procedural_only_flag"].eq(0)
        & df["judgment_title_flag"].eq(1)
        & df["substantive_decision_flag"].eq(1)
        & df["text_chars"].ge(2500)
        & df["delay_hits"].ge(1)
    )
    tier_a = df["strict_info_complete_flag"].eq(1)
    tier_b = common_complete & ~tier_a & (~df["mimo_screened_flag"].eq(1) | df.get("recommended_bucket", "").astype(str).isin(["rag_only", "accept"]))

    df["usable_tier"] = "excluded"
    df.loc[tier_b, "usable_tier"] = "Tier_B_local_delay_candidate"
    df.loc[tier_a, "usable_tier"] = "Tier_A_mimo_strict_complete"
    df["usable_for_research_flag"] = df["usable_tier"].ne("excluded").astype(int)
    df["usage_scope"] = ""
    df.loc[df["usable_tier"].eq("Tier_A_mimo_strict_complete"), "usage_scope"] = "RAG; corpus statistics; downstream labeling; candidate supervised data after label extraction"
    df.loc[df["usable_tier"].eq("Tier_B_local_delay_candidate"), "usage_scope"] = "RAG; corpus statistics; future Mimo/human review; not headline benchmark label"

    usable = df[df["usable_for_research_flag"].eq(1)].copy().reset_index(drop=True)
    manifest_cols = [c for c in usable.columns if c != "raw_text"]
    all_out = out_dir / "research_usable_corpus_tiered_fulltext.csv"
    manifest_out = out_dir / "research_usable_corpus_tiered_manifest.csv"
    excluded_out = out_dir / "research_excluded_or_pending_manifest.csv"
    usable.to_csv(all_out, index=False, encoding="utf-8-sig")
    usable[manifest_cols].to_csv(manifest_out, index=False, encoding="utf-8-sig")
    df[df["usable_for_research_flag"].eq(0)][[c for c in df.columns if c != "raw_text"]].to_csv(excluded_out, index=False, encoding="utf-8-sig")

    tier_counts = usable["usable_tier"].value_counts().reset_index()
    tier_counts.columns = ["usable_tier", "count"]
    tier_counts.to_csv(out_dir / "usable_tier_distribution.csv", index=False, encoding="utf-8-sig")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "search_hits": int(len(read_csv(collect_dir / "search_hits.csv"))),
        "raw_public_cases": int(len(read_csv(collect_dir / "raw_public_cases.csv"))),
        "unique_text_after_dedup": int(df["text_sha256"].nunique()),
        "local_delay_candidate_delay_hits_ge_1": int(common_complete.sum()),
        "mimo_screened_unique": int(df["mimo_screened_flag"].sum()),
        "tier_a_mimo_strict_complete": int((usable["usable_tier"] == "Tier_A_mimo_strict_complete").sum()),
        "tier_b_local_delay_candidate": int((usable["usable_tier"] == "Tier_B_local_delay_candidate").sum()),
        "usable_for_research_total": int(len(usable)),
        "meets_10000_requirement": bool(len(usable) >= 10000),
        "note": "Tier B is usable for corpus/RAG/future labeling, not a validated gold-label benchmark.",
        "outputs": {
            "fulltext": str(all_out),
            "manifest": str(manifest_out),
            "tier_distribution": str(out_dir / "usable_tier_distribution.csv"),
            "excluded_manifest": str(excluded_out),
        },
        "artifact_hashes": {
            all_out.name: sha256_file(all_out),
            manifest_out.name: sha256_file(manifest_out),
            "usable_tier_distribution.csv": sha256_file(out_dir / "usable_tier_distribution.csv"),
            excluded_out.name: sha256_file(excluded_out),
        },
    }
    (out_dir / "usable_corpus_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
