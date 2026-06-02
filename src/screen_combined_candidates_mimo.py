# -*- coding: utf-8 -*-
"""Run Mimo screening on a wider combined-corpus candidate pool.

This targets documents that look like substantive construction disputes but
were not accepted by the stricter local delay-keyword filter. Mimo is used only
for corpus screening; accepted rows are machine-screened candidates, not human
gold labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from collect_public_cases_mimo import load_json_config, run_mimo_screening


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def normalize_pageid(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        return str(int(float(text)))
    except Exception:
        return text


def resolve_path(text: str) -> Path:
    p = Path(str(text or ""))
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_raw_text(path_text: str) -> str:
    path = resolve_path(path_text)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/research_public_case_collection_mimo_100k_rawtext.json")
    ap.add_argument("--combined-dir", default="data/1_raw_text/combined_delay_dispute_corpus_20260527")
    ap.add_argument("--out-dir", default="data/1_raw_text/combined_delay_dispute_corpus_20260527/mimo_screening_wide")
    ap.add_argument("--max-screen-cases", type=int, default=40)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_json_config(PROJECT_ROOT / args.config)
    combined_dir = PROJECT_ROOT / args.combined_dir
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_csv(combined_dir / "combined_raw_text_manifest_dedup.csv")
    if raw.empty:
        raise SystemExit("combined_raw_text_manifest_dedup.csv is missing/empty")
    if "pageid" in raw.columns:
        raw["pageid"] = raw["pageid"].map(normalize_pageid)

    # Exclude cases already Mimo-screened in the main active branch and this branch.
    done = set()
    for p in [
        PROJECT_ROOT / "data/1_raw_text/mimo_public_100k_collect_20260527/mimo_screening_high_quality/mimo_screening_results.csv",
        out_dir / "mimo_screening_results.csv",
    ]:
        d = read_csv(p)
        if not d.empty and "pageid" in d.columns:
            done.update(d["pageid"].dropna().map(normalize_pageid).tolist())

    for col in [
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
    ]:
        if col not in raw.columns:
            raw[col] = 0
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0)
    if "usable_tier" not in raw.columns:
        raw["usable_tier"] = ""

    # Wider than the local delay corpus: strong construction/adjudication signal,
    # but not already usable under the strict local delay filter.
    wide = (
        raw["fetch_status"].astype(str).eq("ok")
        & raw["pageid"].astype(str).ne("")
        & ~raw["pageid"].astype(str).isin(done)
        & raw["text_chars"].ge(2500)
        & raw["construction_hits"].ge(1)
        & raw["decision_hits"].ge(2)
        & raw["evidence_hits"].ge(1)
        & raw["local_quality_score"].ge(0.55)
        & raw["procedural_only_flag"].eq(0)
        & raw["judgment_title_flag"].eq(1)
        & raw["substantive_decision_flag"].eq(1)
        & raw["usable_tier"].astype(str).eq("excluded_or_pending")
    )
    pool = raw[wide].drop_duplicates("text_sha256", keep="first").copy()
    # Prioritize records with some weak delay/process signal, then longer text.
    pool["wide_rank_score"] = (
        pool["delay_hits"] * 5
        + pool["procedural_hits"] * 2
        + pool["evidence_hits"]
        + (pool["text_chars"] / 5000.0).clip(0, 5)
    )
    pool = pool.sort_values(["wide_rank_score", "text_chars"], ascending=[False, False]).reset_index(drop=True)
    pool.to_csv(out_dir / "candidate_pool_prefilter.csv", index=False, encoding="utf-8-sig")

    # Load raw text only for the next unscreened cases in this invocation.
    # Do not pass the full 39k pool with raw_text loaded into memory.
    existing_done = set()
    existing_results = read_csv(out_dir / "mimo_screening_results.csv")
    if not existing_results.empty and "pageid" in existing_results.columns:
        existing_done = set(existing_results["pageid"].dropna().map(normalize_pageid).tolist())
    unscreened = pool[~pool["pageid"].astype(str).isin(existing_done)].copy()
    invoke_pool = unscreened.head(args.max_screen_cases).copy() if args.max_screen_cases > 0 else unscreened.copy()
    invoke_pool["raw_text"] = invoke_pool["raw_text_path"].map(load_raw_text)
    invoke_pool = invoke_pool[invoke_pool["raw_text"].astype(str).str.len().ge(1000)].copy()
    screening = run_mimo_screening(cfg, invoke_pool, out_dir, 0, resume=args.resume)

    status = {
        "candidate_pool_prefilter": int(len(pool)),
        "already_screened_before_invocation": int(len(existing_done)),
        "remaining_before_invocation": int(len(unscreened)),
        "screened_this_invocation_pool": int(len(invoke_pool)),
        "mimo_screened_rows": int(len(screening)) if not screening.empty else 0,
        "mimo_accept_sum": int(pd.to_numeric(screening.get("mimo_accept_flag", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not screening.empty else 0,
        "note": "Accepted rows are Mimo-screened delay-dispute candidates, not human-reviewed labels.",
        "out_dir": str(out_dir),
    }
    (out_dir / "screening_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
