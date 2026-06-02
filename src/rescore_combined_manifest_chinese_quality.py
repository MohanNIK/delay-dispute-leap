# -*- coding: utf-8 -*-
"""Rescore the de-duplicated combined manifest using corrected Chinese terms."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

from collect_public_cases_mimo import local_quality_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path_value: object) -> Path:
    p = Path(str(path_value or ""))
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False) if path.exists() else pd.DataFrame()


def tier_rank(tier: str) -> int:
    return {
        "Tier_A_mimo_strict_complete": 3,
        "Tier_B_plus_high_quality_local": 2,
        "Tier_B_local_delay_candidate": 1,
        "excluded_or_pending": 0,
        "": 0,
    }.get(str(tier), 0)


def classify(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
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
        "mimo_accept_flag",
    ]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    local_candidate = (
        out["text_chars"].ge(2500)
        & out["delay_hits"].ge(1)
        & out["construction_hits"].ge(1)
        & out["decision_hits"].ge(1)
        & out["procedural_only_flag"].eq(0)
    )
    high_quality = (
        out["text_chars"].ge(2500)
        & out["delay_hits"].ge(2)
        & out["construction_hits"].ge(1)
        & out["evidence_hits"].ge(1)
        & out["decision_hits"].ge(2)
        & out["procedural_only_flag"].eq(0)
        & out["substantive_decision_flag"].eq(1)
    )
    old_tier = out.get("usable_tier", pd.Series([""] * len(out), index=out.index)).astype(str)
    out["usable_tier"] = "excluded_or_pending"
    out.loc[local_candidate, "usable_tier"] = "Tier_B_local_delay_candidate"
    out.loc[high_quality, "usable_tier"] = "Tier_B_plus_high_quality_local"
    out.loc[old_tier.eq("Tier_A_mimo_strict_complete") | out["mimo_accept_flag"].eq(1), "usable_tier"] = "Tier_A_mimo_strict_complete"
    # Do not downgrade previous strong tiers if rescoring misses a term.
    for idx, tier in old_tier.items():
        if tier_rank(tier) > tier_rank(out.at[idx, "usable_tier"]):
            out.at[idx, "usable_tier"] = tier
    out["usable_for_research_flag"] = out["usable_tier"].ne("excluded_or_pending").astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-dir", default="data/1_raw_text/combined_delay_dispute_corpus_20260527")
    ap.add_argument("--save-every", type=int, default=1000)
    args = ap.parse_args()

    combined_dir = resolve(args.combined_dir)
    raw_path = combined_dir / "combined_raw_text_manifest_dedup.csv"
    df = read_csv(raw_path)
    if df.empty:
        raise SystemExit(f"missing/empty {raw_path}")
    progress_path = combined_dir / "combined_raw_text_manifest_rescored_progress.csv"
    done = read_csv(progress_path)
    done_hashes = set(done.get("text_sha256", pd.Series(dtype=str)).dropna().astype(str).tolist()) if not done.empty else set()
    rows = done.to_dict("records") if not done.empty else []

    cache: Dict[str, Dict[str, object]] = {}
    pending = df[~df.get("text_sha256", pd.Series(dtype=str)).astype(str).isin(done_hashes)].copy()
    for i, (_, row) in enumerate(pending.iterrows(), 1):
        raw_text_path = resolve(row.get("raw_text_path", ""))
        try:
            key = str(raw_text_path)
            quality = cache.get(key)
            if quality is None:
                text = raw_text_path.read_text(encoding="utf-8", errors="ignore")
                quality = local_quality_score(str(row.get("title", "")), text, 2500)
                cache[key] = quality
            for k, v in quality.items():
                row[k] = v
        except Exception:
            pass
        rows.append(row.to_dict())
        if i % args.save_every == 0:
            pd.DataFrame(rows).to_csv(progress_path, index=False, encoding="utf-8-sig")
            print(json.dumps({"rescored_progress": len(rows), "total": len(df)}, ensure_ascii=False), flush=True)

    rescored = pd.DataFrame(rows)
    rescored = classify(rescored)
    rescored["tier_rank"] = rescored["usable_tier"].map(tier_rank)
    if "text_sha256" in rescored.columns:
        rescored = rescored.sort_values(["tier_rank", "text_chars"], ascending=[False, False]).drop_duplicates("text_sha256", keep="first")
    usable = rescored[rescored["usable_tier"].ne("excluded_or_pending")].copy()
    excluded = rescored[rescored["usable_tier"].eq("excluded_or_pending")].copy()

    rescored.drop(columns=["tier_rank"], errors="ignore").to_csv(raw_path, index=False, encoding="utf-8-sig")
    usable.drop(columns=["tier_rank"], errors="ignore").to_csv(combined_dir / "combined_usable_manifest.csv", index=False, encoding="utf-8-sig")
    excluded.drop(columns=["tier_rank"], errors="ignore").to_csv(combined_dir / "combined_excluded_or_pending_manifest.csv", index=False, encoding="utf-8-sig")
    usable["usable_tier"].value_counts().rename_axis("usable_tier").reset_index(name="count").to_csv(
        combined_dir / "combined_usable_tier_distribution.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_rows_after_text_dedup": int(len(rescored)),
        "usable_for_research_total": int(len(usable)),
        "excluded_or_pending_total": int(len(excluded)),
        "tier_counts": usable["usable_tier"].value_counts().to_dict(),
        "note": "Rescored with corrected Chinese construction/delay/evidence terms. Tier labels are automated corpus-quality tiers.",
    }
    (combined_dir / "combined_corpus_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
