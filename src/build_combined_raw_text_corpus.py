# -*- coding: utf-8 -*-
"""Build a deduplicated combined corpus manifest under data/1_raw_text.

The output separates raw collected documents from usable research candidates.
Tier labels are corpus-quality tiers, not human validation labels.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from collect_public_cases_mimo import local_quality_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALITY_CACHE: Dict[str, Dict[str, object]] = {}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def to_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def normalize_pageid(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        return str(int(float(text)))
    except Exception:
        return text


def resolve_raw_path(path_value: object) -> Path:
    p = Path(str(path_value or ""))
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def recompute_quality_fields(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "raw_text_path" not in df.columns:
        return df
    rows = []
    for _, row in df.iterrows():
        raw_path = resolve_raw_path(row.get("raw_text_path", ""))
        if not raw_path.exists():
            rows.append(row)
            continue
        try:
            cache_key = str(raw_path)
            quality = QUALITY_CACHE.get(cache_key)
            if quality is None:
                text = raw_path.read_text(encoding="utf-8", errors="ignore")
                quality = local_quality_score(str(row.get("title", "")), text, 2500)
                QUALITY_CACHE[cache_key] = quality
            for key, value in quality.items():
                row[key] = value
        except Exception:
            pass
        rows.append(row)
    return pd.DataFrame(rows)


def load_manifest(source_name: str, manifest_path: Path, mode: str, recompute_quality: bool = False) -> pd.DataFrame:
    df = read_csv(manifest_path)
    if df.empty:
        return df
    df = df.copy()
    if recompute_quality:
        df = recompute_quality_fields(df)
    df["source_batch"] = source_name
    df["source_manifest"] = str(manifest_path)
    if "pageid" in df.columns:
        df["pageid"] = df["pageid"].map(normalize_pageid)
    if "fetch_status" not in df.columns:
        df["fetch_status"] = "ok"
    if "usable_tier" not in df.columns:
        df["usable_tier"] = ""
    if "usage_scope" not in df.columns:
        df["usage_scope"] = ""

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
        "strict_info_complete_flag",
    ]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if mode == "pre_tiered":
        df["usable_for_research_flag"] = df["usable_tier"].astype(str).ne("").astype(int)
        return df

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
    df.loc[to_num(df, "mimo_accept_flag").eq(1), "usable_tier"] = "Tier_A_mimo_strict_complete"
    df["usable_for_research_flag"] = df["usable_tier"].ne("excluded_or_pending").astype(int)
    return df


def tier_rank(tier: str) -> int:
    return {
        "Tier_A_mimo_strict_complete": 3,
        "Tier_B_plus_high_quality_local": 2,
        "Tier_B_local_delay_candidate": 1,
        "excluded_or_pending": 0,
        "": 0,
    }.get(str(tier), 0)


def load_accepted_pageids(paths: List[Path]) -> set[str]:
    accepted: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        df = read_csv(path)
        if df.empty or "pageid" not in df.columns or "mimo_accept_flag" not in df.columns:
            continue
        flag = pd.to_numeric(df["mimo_accept_flag"], errors="coerce").fillna(0).astype(int)
        accepted.update(df.loc[flag.eq(1), "pageid"].dropna().map(normalize_pageid).tolist())
    return accepted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/1_raw_text")
    ap.add_argument("--out-dir", default="data/1_raw_text/combined_delay_dispute_corpus_20260527")
    ap.add_argument("--recompute-local-quality", action="store_true")
    args = ap.parse_args()

    root = PROJECT_ROOT / args.root
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_specs: List[Dict[str, str]] = [
        {
            "source_name": "old_mimo_usable_30k",
            "path": "mimo_public_usable_30k_20260527/raw_text_manifest.csv",
            "mode": "pre_tiered",
        },
        {
            "source_name": "public_100k_first_batch",
            "path": "mimo_public_100k_collect_20260527/raw_text_manifest.csv",
            "mode": "local",
        },
        {
            "source_name": "extra_broad_fetch1",
            "path": "mimo_public_extra_broad_fetch_20260527/raw_text_manifest.csv",
            "mode": "local",
        },
        {
            "source_name": "extra_broad_fetch2",
            "path": "mimo_public_extra_broad_fetch2_20260527/raw_text_manifest.csv",
            "mode": "local",
        },
        {
            "source_name": "extra_broad_fetch2_incremental",
            "path": "mimo_public_extra_broad_fetch2_20260527/raw_text_manifest_incremental.csv",
            "mode": "local",
        },
        {
            "source_name": "extra_broad_fetch3",
            "path": "mimo_public_extra_broad_fetch3_20260527/raw_text_manifest.csv",
            "mode": "local",
        },
        {
            "source_name": "extra_broad_fetch3_incremental",
            "path": "mimo_public_extra_broad_fetch3_20260527/raw_text_manifest_incremental.csv",
            "mode": "local",
        },
        {
            "source_name": "delay_deep_fetch",
            "path": "mimo_public_delay_deep_fetch_20260527/raw_text_manifest.csv",
            "mode": "local",
        },
        {
            "source_name": "delay_deep_fetch_incremental",
            "path": "mimo_public_delay_deep_fetch_20260527/raw_text_manifest_incremental.csv",
            "mode": "local",
        },
        {
            "source_name": "delay_generic_fetch",
            "path": "mimo_public_delay_generic_fetch_20260527/raw_text_manifest.csv",
            "mode": "local",
        },
        {
            "source_name": "delay_generic_fetch_incremental",
            "path": "mimo_public_delay_generic_fetch_20260527/raw_text_manifest_incremental.csv",
            "mode": "local",
        },
        {
            "source_name": "extra_broad_fetch4",
            "path": "mimo_public_extra_broad_fetch4_20260527/raw_text_manifest.csv",
            "mode": "local",
        },
        {
            "source_name": "extra_broad_fetch4_incremental",
            "path": "mimo_public_extra_broad_fetch4_20260527/raw_text_manifest_incremental.csv",
            "mode": "local",
        },
        {
            "source_name": "delay_deep_fetch2",
            "path": "mimo_public_delay_deep_fetch2_20260527/raw_text_manifest.csv",
            "mode": "local",
        },
        {
            "source_name": "delay_deep_fetch2_incremental",
            "path": "mimo_public_delay_deep_fetch2_20260527/raw_text_manifest_incremental.csv",
            "mode": "local",
        },
        {
            "source_name": "delay_generic_fetch2",
            "path": "mimo_public_delay_generic_fetch2_20260527/raw_text_manifest.csv",
            "mode": "local",
        },
        {
            "source_name": "delay_generic_fetch2_incremental",
            "path": "mimo_public_delay_generic_fetch2_20260527/raw_text_manifest_incremental.csv",
            "mode": "local",
        },
    ]
    seen_spec_paths = {spec["path"] for spec in manifest_specs}
    for child in sorted(root.glob("mimo_public_*")):
        if not child.is_dir():
            continue
        for manifest_name in ["raw_text_manifest.csv", "raw_text_manifest_incremental.csv"]:
            rel = f"{child.name}/{manifest_name}"
            if rel in seen_spec_paths:
                continue
            if (root / rel).exists():
                manifest_specs.append(
                    {
                        "source_name": f"{child.name}_{manifest_name.replace('.csv', '')}",
                        "path": rel,
                        "mode": "local",
                    }
                )
                seen_spec_paths.add(rel)

    frames: List[pd.DataFrame] = []
    source_rows: List[Dict[str, object]] = []
    for spec in manifest_specs:
        p = root / spec["path"]
        if not p.exists():
            source_rows.append({"source_batch": spec["source_name"], "path": str(p), "rows": 0, "status": "missing"})
            continue
        df = load_manifest(spec["source_name"], p, spec["mode"], recompute_quality=args.recompute_local_quality)
        source_rows.append({"source_batch": spec["source_name"], "path": str(p), "rows": len(df), "status": "loaded"})
        if not df.empty:
            frames.append(df)

    if not frames:
        raise SystemExit("no manifests loaded")

    raw = pd.concat(frames, ignore_index=True, sort=False)
    if "text_sha256" not in raw.columns:
        raise SystemExit("text_sha256 is required for deduplication")
    raw["tier_rank"] = raw["usable_tier"].map(tier_rank)
    raw["text_sha256"] = raw["text_sha256"].astype(str)
    raw = raw[raw["text_sha256"].ne("") & raw["text_sha256"].ne("nan")].copy()
    raw = raw.sort_values(["tier_rank", "text_chars"], ascending=[False, False])
    dedup = raw.drop_duplicates("text_sha256", keep="first").reset_index(drop=True)
    if "pageid" in dedup.columns:
        dedup["pageid"] = dedup["pageid"].map(normalize_pageid)
    accepted_pageids = load_accepted_pageids(
        [
            root / "mimo_public_100k_collect_20260527/mimo_screening_high_quality/mimo_screening_results.csv",
            root / "combined_delay_dispute_corpus_20260527/mimo_screening_wide/mimo_screening_results.csv",
        ]
    )
    if accepted_pageids and "pageid" in dedup.columns:
        accepted_mask = dedup["pageid"].astype(str).isin(accepted_pageids)
        dedup.loc[accepted_mask, "usable_tier"] = "Tier_A_mimo_strict_complete"
        dedup.loc[accepted_mask, "usable_for_research_flag"] = 1
    usable = dedup[dedup["usable_tier"].ne("excluded_or_pending") & dedup["usable_tier"].astype(str).ne("")].copy()
    excluded = dedup[~dedup.index.isin(usable.index)].copy()

    raw_out = out_dir / "combined_raw_text_manifest_dedup.csv"
    usable_out = out_dir / "combined_usable_manifest.csv"
    excluded_out = out_dir / "combined_excluded_or_pending_manifest.csv"
    tier_out = out_dir / "combined_usable_tier_distribution.csv"
    source_out = out_dir / "combined_source_manifest.csv"
    dedup.drop(columns=["tier_rank"], errors="ignore").to_csv(raw_out, index=False, encoding="utf-8-sig")
    usable.drop(columns=["tier_rank"], errors="ignore").to_csv(usable_out, index=False, encoding="utf-8-sig")
    excluded.drop(columns=["tier_rank"], errors="ignore").to_csv(excluded_out, index=False, encoding="utf-8-sig")
    usable["usable_tier"].value_counts().rename_axis("usable_tier").reset_index(name="count").to_csv(
        tier_out, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(source_rows).to_csv(source_out, index=False, encoding="utf-8-sig")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_rows_before_dedup": int(len(raw)),
        "raw_rows_after_text_dedup": int(len(dedup)),
        "usable_for_research_total": int(len(usable)),
        "excluded_or_pending_total": int(len(excluded)),
        "tier_counts": usable["usable_tier"].value_counts().to_dict(),
        "mimo_accepted_pageids_merged": int(len(accepted_pageids)),
        "note": "Tier labels are automated corpus-quality tiers, not human-reviewed gold labels.",
        "outputs": {
            "raw_manifest": str(raw_out),
            "usable_manifest": str(usable_out),
            "excluded_manifest": str(excluded_out),
            "tier_distribution": str(tier_out),
            "source_manifest": str(source_out),
        },
    }
    (out_dir / "combined_corpus_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
