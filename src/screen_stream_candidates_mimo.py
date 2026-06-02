# -*- coding: utf-8 -*-
"""Screen stream-fetched public cases with Mimo.

This consumes the streaming 100k expansion directory after some raw texts have
been downloaded. It builds a high-quality construction-delay candidate pool and
then reuses the same Mimo screening function as the earlier 30k batch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from collect_public_cases_mimo import load_json_config, run_mimo_screening


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/research_public_case_collection_mimo_100k_rawtext.json")
    ap.add_argument("--collect-dir", default="data/1_raw_text/mimo_public_100k_collect_20260527")
    ap.add_argument("--out-dir", default="data/1_raw_text/mimo_public_100k_collect_20260527/mimo_screening_high_quality")
    ap.add_argument("--max-screen-cases", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_json_config(PROJECT_ROOT / args.config)
    collect_dir = PROJECT_ROOT / args.collect_dir
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_csv(collect_dir / "raw_public_cases_incremental.csv")
    audit = read_csv(collect_dir / "local_quality_audit_incremental.csv")
    if raw.empty or audit.empty:
        raise SystemExit("raw_public_cases_incremental.csv or local_quality_audit_incremental.csv is missing/empty")

    raw["pageid"] = raw["pageid"].astype(str)
    audit["pageid"] = audit["pageid"].astype(str)
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

    meta_cols = [
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
    pool = raw.merge(audit[[c for c in meta_cols if c in audit.columns]], on="pageid", how="left", suffixes=("", "_audit"))
    for col in ["text_chars", "local_quality_score", "delay_hits", "construction_hits", "evidence_hits", "decision_hits", "procedural_only_flag", "judgment_title_flag", "substantive_decision_flag"]:
        if col in pool.columns:
            pool[col] = pd.to_numeric(pool[col], errors="coerce").fillna(0)

    strict = (
        pool["text_chars"].ge(2500)
        & pool["local_quality_score"].ge(0.55)
        & pool["delay_hits"].ge(2)
        & pool["construction_hits"].ge(1)
        & pool["evidence_hits"].ge(1)
        & pool["decision_hits"].ge(2)
        & pool["procedural_only_flag"].eq(0)
        & pool["judgment_title_flag"].eq(1)
        & pool["substantive_decision_flag"].eq(1)
    )
    pool = pool[strict].drop_duplicates("text_sha256", keep="first").reset_index(drop=True)
    pool.to_csv(out_dir / "candidate_pool_prefilter.csv", index=False, encoding="utf-8-sig")

    screening = run_mimo_screening(cfg, pool, out_dir, args.max_screen_cases, resume=args.resume)
    status = {
        "candidate_pool_prefilter": int(len(pool)),
        "mimo_screened_rows": int(len(screening)) if not screening.empty else 0,
        "mimo_accept_sum": int(pd.to_numeric(screening.get("mimo_accept_flag", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not screening.empty else 0,
        "out_dir": str(out_dir),
    }
    (out_dir / "screening_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
