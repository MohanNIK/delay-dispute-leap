# -*- coding: utf-8 -*-
"""Prepare de-duplicated MediaWiki page-id queues for raw text fetching."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def normalize_pageid(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        return str(int(float(text)))
    except Exception:
        return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-hits", required=True)
    ap.add_argument("--existing-manifest", default="data/1_raw_text/combined_delay_dispute_corpus_20260527/combined_raw_text_manifest_dedup.csv")
    ap.add_argument(
        "--exclude-queue-dir",
        action="append",
        default=[],
        help="Additional in-flight fetch directories whose pageids should be excluded.",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-pages", type=int, default=30000)
    ap.add_argument("--min-search-size", type=int, default=2500)
    args = ap.parse_args()

    hits = read_csv(resolve(args.search_hits))
    if hits.empty or "pageid" not in hits.columns:
        raise SystemExit(f"missing or invalid search hits: {args.search_hits}")
    existing = read_csv(resolve(args.existing_manifest))
    existing_pageids = set()
    if not existing.empty and "pageid" in existing.columns:
        existing_pageids = set(existing["pageid"].dropna().map(normalize_pageid))
    excluded_queue_pageids = set()
    for queue_dir_arg in args.exclude_queue_dir:
        queue_dir = resolve(queue_dir_arg)
        for file_name in ["search_hits.csv", "raw_text_manifest.csv", "raw_text_manifest_incremental.csv"]:
            queue_df = read_csv(queue_dir / file_name)
            if not queue_df.empty and "pageid" in queue_df.columns:
                excluded_queue_pageids.update(queue_df["pageid"].dropna().map(normalize_pageid))

    hits = hits.copy()
    hits["pageid"] = hits["pageid"].map(normalize_pageid)
    hits = hits[hits["pageid"].ne("")]
    if "api_status" in hits.columns:
        hits = hits[hits["api_status"].astype(str).eq("search_hit")]
    if "search_size" in hits.columns:
        hits = hits[pd.to_numeric(hits["search_size"], errors="coerce").fillna(0).ge(args.min_search_size)]
    hits = hits[~hits["pageid"].isin(existing_pageids | excluded_queue_pageids)]
    hits = hits.drop_duplicates("pageid", keep="first")
    if args.max_pages > 0:
        hits = hits.head(args.max_pages)

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "search_hits.csv"
    hits.to_csv(out_path, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(
        [
            {
                "search_hits_source": str(resolve(args.search_hits)),
                "existing_manifest": str(resolve(args.existing_manifest)),
                "existing_pageids": len(existing_pageids),
                "excluded_inflight_pageids": len(excluded_queue_pageids),
                "queue_rows": len(hits),
                "out_path": str(out_path),
            }
        ]
    )
    summary.to_csv(out_dir / "fetch_queue_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_json(orient="records", force_ascii=False))


if __name__ == "__main__":
    main()
