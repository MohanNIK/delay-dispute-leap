# -*- coding: utf-8 -*-
"""Export tiered research-pool views from the combined raw-text corpus.

The strict pool keeps the current delay-dispute usability definition. The broad
pool is for supporting retrieval/pretraining/labeling only and must not be
reported as the strict schedule-delay benchmark.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--combined-dir",
        default="data/1_raw_text/combined_delay_dispute_corpus_20260527",
    )
    args = ap.parse_args()

    combined_dir = resolve(args.combined_dir)
    manifest_path = combined_dir / "combined_raw_text_manifest_dedup.csv"
    df = read_csv(manifest_path)

    strict = df[numeric(df, "usable_for_research_flag").eq(1)].copy()
    broad_delay = df[
        numeric(df, "text_chars").ge(1200)
        & numeric(df, "construction_hits").ge(1)
        & numeric(df, "delay_hits").ge(1)
        & numeric(df, "procedural_only_flag").eq(0)
    ].copy()
    broad_construction = df[
        numeric(df, "text_chars").ge(1500)
        & numeric(df, "construction_hits").ge(1)
        & (numeric(df, "delay_hits").ge(1) | numeric(df, "evidence_hits").ge(1))
        & numeric(df, "procedural_only_flag").eq(0)
    ].copy()

    outputs = {
        "strict_delay_usable_manifest": combined_dir / "strict_delay_usable_manifest.csv",
        "broad_delay_candidate_manifest": combined_dir / "broad_delay_candidate_manifest.csv",
        "broad_construction_dispute_support_pool": combined_dir / "broad_construction_dispute_support_pool.csv",
        "research_pool_views_summary": combined_dir / "research_pool_views_summary.json",
    }
    strict.to_csv(outputs["strict_delay_usable_manifest"], index=False, encoding="utf-8-sig")
    broad_delay.to_csv(outputs["broad_delay_candidate_manifest"], index=False, encoding="utf-8-sig")
    broad_construction.to_csv(outputs["broad_construction_dispute_support_pool"], index=False, encoding="utf-8-sig")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifest": str(manifest_path),
        "dedup_rows": int(len(df)),
        "strict_delay_usable": int(len(strict)),
        "broad_delay_candidate": int(len(broad_delay)),
        "broad_construction_dispute_support_pool": int(len(broad_construction)),
        "important_boundary": (
            "Only strict_delay_usable should be treated as the current strict delay-dispute research pool. "
            "The broad construction-dispute support pool is for retrieval, pretraining, and further LLM screening."
        ),
        "outputs": {k: str(v) for k, v in outputs.items() if k != "research_pool_views_summary"},
    }
    outputs["research_pool_views_summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
