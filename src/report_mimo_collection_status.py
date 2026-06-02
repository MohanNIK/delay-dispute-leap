# -*- coding: utf-8 -*-
"""Report Mimo collection progress and token usage."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def token_usage_from_df(df: pd.DataFrame) -> dict:
    prompt = completion = total = 0
    if "usage_json" not in df.columns:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for s in df["usage_json"].dropna().astype(str):
        try:
            u = json.loads(s)
            prompt += int(u.get("prompt_tokens", 0) or 0)
            completion += int(u.get("completion_tokens", 0) or 0)
            total += int(u.get("total_tokens", 0) or 0)
        except Exception:
            continue
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def count_csv(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(read_csv(path))
    except Exception:
        return 0


def relaxed_lora_usable_count(df: pd.DataFrame) -> int:
    """Count clear one-label rows under a relaxed training-data view."""
    if df.empty:
        return 0
    required = {"outcome_label", "label_confidence", "pre_decision_sufficiency"}
    if not required.issubset(set(df.columns)):
        return 0
    valid = df[df["outcome_label"].astype(str).isin(["support", "partial_support", "not_support"])].copy()
    if valid.empty:
        return 0
    conf = pd.to_numeric(valid["label_confidence"], errors="coerce").fillna(0)
    suff = pd.to_numeric(valid["pre_decision_sufficiency"], errors="coerce").fillna(0)
    conflict = valid.get("conflict_flag", pd.Series(False, index=valid.index)).astype(str).str.lower().isin(["true", "1", "yes"])
    review = valid.get("needs_review", pd.Series(False, index=valid.index)).astype(str).str.lower().isin(["true", "1", "yes"])
    mask = conf.ge(0.75) & suff.ge(0.30) & ~conflict & ~review
    return int(mask.sum())


def dynamic_raw_text_dir_stats(root: Path) -> dict:
    dirs = []
    manifest_rows_total = 0
    for child in sorted(root.glob("mimo_public_*")):
        if not child.is_dir():
            continue
        inc_rows = count_csv(child / "raw_text_manifest_incremental.csv")
        final_rows = count_csv(child / "raw_text_manifest.csv")
        if inc_rows == 0 and final_rows == 0:
            continue
        manifest_rows_total += max(inc_rows, final_rows)
        dirs.append(
            {
                "dir": str(child),
                "raw_text_manifest_incremental": inc_rows,
                "raw_text_manifest_final": final_rows,
                "last_write": max(
                    [
                        p.stat().st_mtime
                        for p in [child / "raw_text_manifest_incremental.csv", child / "raw_text_manifest.csv"]
                        if p.exists()
                    ]
                    or [0]
                ),
            }
        )
    return {
        "dynamic_raw_text_dirs_with_manifests": len(dirs),
        "dynamic_raw_text_manifest_rows_total_before_dedup": manifest_rows_total,
        "dynamic_recent_raw_text_dirs": dirs[-8:],
    }


def active_collect_processes() -> list[dict]:
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.Name -match 'python' -and ($_.CommandLine -match 'collect_public_cases_mimo' -or $_.CommandLine -match 'stream_fetch_public_cases_to_raw_text' -or $_.CommandLine -match 'screen_stream_candidates_mimo' -or $_.CommandLine -match 'screen_combined_candidates_mimo' -or $_.CommandLine -match 'run_mimo_fast_strong_labeling' -or $_.CommandLine -match 'run_mimo_batch_outcome_labeling' -or $_.CommandLine -match 'rescore_combined_manifest_chinese_quality') } | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
        ]
        out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="ignore", timeout=10).strip()
        if not out:
            return []
        data = json.loads(out)
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect-dir", default="data/external_mimo_candidates/wikisource_30k_collect_20260527_122315")
    ap.add_argument("--mimo-dir", default="data/external_mimo_candidates/wikisource_30k_mimo_delay2_20260527")
    ap.add_argument("--usable-dir", default="data/external_mimo_candidates/research_usable_corpus_30k_20260527")
    ap.add_argument("--active-100k-dir", default="data/1_raw_text/mimo_public_100k_collect_20260527")
    ap.add_argument("--extra-search-dir", default="data/1_raw_text/mimo_public_extra_broad_search_20260527")
    ap.add_argument("--extra-fetch-dir", default="data/1_raw_text/mimo_public_extra_broad_fetch_20260527")
    ap.add_argument("--extra-fetch2-dir", default="data/1_raw_text/mimo_public_extra_broad_fetch2_20260527")
    ap.add_argument("--extra-fetch3-dir", default="data/1_raw_text/mimo_public_extra_broad_fetch3_20260527")
    ap.add_argument("--delay-deep-search-dir", default="data/1_raw_text/mimo_public_delay_deep_search_20260527")
    ap.add_argument("--delay-deep-fetch-dir", default="data/1_raw_text/mimo_public_delay_deep_fetch_20260527")
    ap.add_argument("--delay-generic-search-dir", default="data/1_raw_text/mimo_public_delay_generic_search_20260527")
    ap.add_argument("--delay-generic-fetch-dir", default="data/1_raw_text/mimo_public_delay_generic_fetch_20260527")
    ap.add_argument("--extra-fetch4-dir", default="data/1_raw_text/mimo_public_extra_broad_fetch4_20260527")
    ap.add_argument("--delay-deep-fetch2-dir", default="data/1_raw_text/mimo_public_delay_deep_fetch2_20260527")
    ap.add_argument("--delay-generic-fetch2-dir", default="data/1_raw_text/mimo_public_delay_generic_fetch2_20260527")
    ap.add_argument("--combined-corpus-dir", default="data/1_raw_text/combined_delay_dispute_corpus_20260527")
    ap.add_argument("--wide-mimo-dir", default="data/1_raw_text/combined_delay_dispute_corpus_20260527/mimo_screening_wide")
    ap.add_argument("--fast-label-dir", default="data/1_raw_text/combined_delay_dispute_corpus_20260527/mimo_fast_strong_labels")
    ap.add_argument("--batch-label-dir", default="data/1_raw_text/combined_delay_dispute_corpus_20260527/mimo_batch_outcome_labels")
    ap.add_argument("--batch-label-v2-recall-dir", default="data/1_raw_text/combined_delay_dispute_corpus_20260527/mimo_batch_outcome_labels_v2_recall")
    ap.add_argument("--raw-text-root", default="data/1_raw_text")
    args = ap.parse_args()

    collect_dir = Path(args.collect_dir)
    mimo_dir = Path(args.mimo_dir)
    usable_dir = Path(args.usable_dir)
    active_dir = Path(args.active_100k_dir)
    extra_search_dir = Path(args.extra_search_dir)
    extra_fetch_dir = Path(args.extra_fetch_dir)
    extra_fetch2_dir = Path(args.extra_fetch2_dir)
    extra_fetch3_dir = Path(args.extra_fetch3_dir)
    delay_deep_search_dir = Path(args.delay_deep_search_dir)
    delay_deep_fetch_dir = Path(args.delay_deep_fetch_dir)
    delay_generic_search_dir = Path(args.delay_generic_search_dir)
    delay_generic_fetch_dir = Path(args.delay_generic_fetch_dir)
    extra_fetch4_dir = Path(args.extra_fetch4_dir)
    delay_deep_fetch2_dir = Path(args.delay_deep_fetch2_dir)
    delay_generic_fetch2_dir = Path(args.delay_generic_fetch2_dir)
    combined_corpus_dir = Path(args.combined_corpus_dir)
    wide_mimo_dir = Path(args.wide_mimo_dir)
    fast_label_dir = Path(args.fast_label_dir)
    batch_label_dir = Path(args.batch_label_dir)
    batch_label_v2_recall_dir = Path(args.batch_label_v2_recall_dir)
    active_mimo_dir = active_dir / "mimo_screening_high_quality"
    raw_text_root = Path(args.raw_text_root)
    root = mimo_dir / "parallel_batches"

    frames = []
    for p in [root / "seed_completed_mimo_screening_results.csv", *sorted(root.glob("batch_*/mimo_screening_results.csv"))]:
        if p.exists():
            frames.append(read_csv(p))
    screening = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not screening.empty and "pageid" in screening.columns:
        screening = screening.drop_duplicates("pageid", keep="last")
    usage = token_usage_from_df(screening)
    active_mimo = read_csv(active_mimo_dir / "mimo_screening_results.csv")
    active_mimo_usage = token_usage_from_df(active_mimo)
    wide_mimo = read_csv(wide_mimo_dir / "mimo_screening_results.csv")
    wide_mimo_usage = token_usage_from_df(wide_mimo)
    fast_label = read_csv(fast_label_dir / "mimo_fast_label_results.csv")
    fast_label_usage = token_usage_from_df(fast_label)
    fast_label_status_path = fast_label_dir / "strong_label_status.json"
    fast_label_status = json.loads(fast_label_status_path.read_text(encoding="utf-8")) if fast_label_status_path.exists() else {}
    batch_label = read_csv(batch_label_dir / "mimo_batch_label_results.csv")
    batch_label_usage = token_usage_from_df(batch_label)
    batch_label_status_path = batch_label_dir / "batch_label_status.json"
    batch_label_status = json.loads(batch_label_status_path.read_text(encoding="utf-8")) if batch_label_status_path.exists() else {}
    batch_label_v2_recall = read_csv(batch_label_v2_recall_dir / "mimo_batch_label_results.csv")
    batch_label_v2_recall_usage = token_usage_from_df(batch_label_v2_recall)
    batch_label_v2_recall_status_path = batch_label_v2_recall_dir / "batch_label_status.json"
    batch_label_v2_recall_status = json.loads(batch_label_v2_recall_status_path.read_text(encoding="utf-8")) if batch_label_v2_recall_status_path.exists() else {}

    usable_summary_path = usable_dir / "usable_corpus_summary.json"
    usable_summary = json.loads(usable_summary_path.read_text(encoding="utf-8")) if usable_summary_path.exists() else {}
    combined_summary_path = combined_corpus_dir / "combined_corpus_summary.json"
    combined_summary = json.loads(combined_summary_path.read_text(encoding="utf-8")) if combined_summary_path.exists() else {}
    combined_tiers = combined_summary.get("tier_counts", {}) if isinstance(combined_summary.get("tier_counts", {}), dict) else {}
    pool_views_path = combined_corpus_dir / "research_pool_views_summary.json"
    pool_views = json.loads(pool_views_path.read_text(encoding="utf-8")) if pool_views_path.exists() else {}
    chinese_delay_pool_path = combined_corpus_dir / "chinese_delay_pool_summary.json"
    chinese_delay_pool = json.loads(chinese_delay_pool_path.read_text(encoding="utf-8")) if chinese_delay_pool_path.exists() else {}
    status = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **dynamic_raw_text_dir_stats(raw_text_root),
        "search_hits": len(read_csv(collect_dir / "search_hits.csv")),
        "raw_public_cases": len(read_csv(collect_dir / "raw_public_cases.csv")),
        "high_delay_candidate_pool": len(read_csv(mimo_dir / "candidate_pool_prefilter.csv")),
        "active_100k_search_hits_incremental": count_csv(active_dir / "search_hits_incremental.csv"),
        "active_100k_search_hits_final": count_csv(active_dir / "search_hits.csv"),
        "active_100k_raw_public_cases_incremental": count_csv(active_dir / "raw_public_cases_incremental.csv"),
        "active_100k_raw_public_cases_final": count_csv(active_dir / "raw_public_cases.csv"),
        "active_100k_raw_text_manifest_incremental": count_csv(active_dir / "raw_text_manifest_incremental.csv"),
        "active_100k_raw_text_manifest_final": count_csv(active_dir / "raw_text_manifest.csv"),
        "active_100k_prefilter_candidates": count_csv(active_dir / "candidate_pool_prefilter.csv"),
        "active_100k_mimo_candidate_pool": count_csv(active_mimo_dir / "candidate_pool_prefilter.csv"),
        "active_100k_mimo_screened": int(active_mimo["pageid"].nunique()) if not active_mimo.empty and "pageid" in active_mimo.columns else 0,
        "active_100k_mimo_accepted": int(pd.to_numeric(active_mimo.get("mimo_accept_flag", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not active_mimo.empty else 0,
        "active_100k_mimo_api_errors": int((active_mimo.get("mimo_status", pd.Series(dtype=str)) == "api_error").sum()) if not active_mimo.empty else 0,
        "active_100k_mimo_prompt_tokens": active_mimo_usage["prompt_tokens"],
        "active_100k_mimo_completion_tokens": active_mimo_usage["completion_tokens"],
        "active_100k_mimo_total_tokens": active_mimo_usage["total_tokens"],
        "wide_mimo_candidate_pool": count_csv(wide_mimo_dir / "candidate_pool_prefilter.csv"),
        "wide_mimo_screened": int(wide_mimo["pageid"].nunique()) if not wide_mimo.empty and "pageid" in wide_mimo.columns else 0,
        "wide_mimo_accepted": int(pd.to_numeric(wide_mimo.get("mimo_accept_flag", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not wide_mimo.empty else 0,
        "wide_mimo_api_errors": int((wide_mimo.get("mimo_status", pd.Series(dtype=str)) == "api_error").sum()) if not wide_mimo.empty else 0,
        "wide_mimo_prompt_tokens": wide_mimo_usage["prompt_tokens"],
        "wide_mimo_completion_tokens": wide_mimo_usage["completion_tokens"],
        "wide_mimo_total_tokens": wide_mimo_usage["total_tokens"],
        "fast_label_results_total": int(len(fast_label)),
        "fast_label_strong_labels_new": int(fast_label_status.get("strong_labels_new", 0) or 0),
        "fast_label_lora_usable_labels_new": int(fast_label_status.get("lora_usable_labels_new", 0) or 0),
        "fast_label_weak_or_unknown_or_error": int(fast_label_status.get("weak_or_unknown_or_error", 0) or 0),
        "fast_label_existing_frozen_strong_labels": int(fast_label_status.get("existing_frozen_strong_labels", 0) or 0),
        "fast_label_total_strong_counting_existing": int(fast_label_status.get("total_strong_labels_counting_existing", 0) or 0),
        "fast_label_total_lora_usable_counting_existing": int(fast_label_status.get("total_lora_usable_counting_existing", 0) or 0),
        "fast_label_target_total_strong_labels": int(fast_label_status.get("target_total_strong_labels", 20000) or 20000),
        "fast_label_target_reached": bool(fast_label_status.get("target_reached", False)),
        "fast_label_prompt_tokens": fast_label_usage["prompt_tokens"],
        "fast_label_completion_tokens": fast_label_usage["completion_tokens"],
        "fast_label_total_tokens": fast_label_usage["total_tokens"],
        "batch_label_results_total": int(len(batch_label)),
        "batch_label_strong_labels_new": int(batch_label_status.get("strong_labels_new", 0) or 0),
        "batch_label_lora_usable_labels_new": int(batch_label_status.get("lora_usable_labels_new", 0) or 0),
        "batch_label_relaxed_lora_usable_new": relaxed_lora_usable_count(batch_label),
        "batch_label_weak_or_unknown_or_error": int(batch_label_status.get("weak_or_unknown_or_error", 0) or 0),
        "batch_label_prompt_tokens": batch_label_usage["prompt_tokens"],
        "batch_label_completion_tokens": batch_label_usage["completion_tokens"],
        "batch_label_total_tokens": batch_label_usage["total_tokens"],
        "batch_label_v2_recall_results_total": int(len(batch_label_v2_recall)),
        "batch_label_v2_recall_strong_labels_new": int(batch_label_v2_recall_status.get("strong_labels_new", 0) or 0),
        "batch_label_v2_recall_lora_usable_labels_new": int(batch_label_v2_recall_status.get("lora_usable_labels_new", 0) or 0),
        "batch_label_v2_recall_weak_or_unknown_or_error": int(batch_label_v2_recall_status.get("weak_or_unknown_or_error", 0) or 0),
        "batch_label_v2_recall_prompt_tokens": batch_label_v2_recall_usage["prompt_tokens"],
        "batch_label_v2_recall_completion_tokens": batch_label_v2_recall_usage["completion_tokens"],
        "batch_label_v2_recall_total_tokens": batch_label_v2_recall_usage["total_tokens"],
        "total_lora_usable_counting_existing_fast_and_batch": int(fast_label_status.get("existing_frozen_strong_labels", 0) or 0)
        + int(fast_label_status.get("lora_usable_labels_new", 0) or 0)
        + int(batch_label_status.get("lora_usable_labels_new", 0) or 0)
        + int(batch_label_v2_recall_status.get("lora_usable_labels_new", 0) or 0),
        "total_relaxed_lora_usable_counting_existing_fast_and_batch": int(fast_label_status.get("existing_frozen_strong_labels", 0) or 0)
        + int(fast_label_status.get("lora_usable_labels_new", 0) or 0)
        + relaxed_lora_usable_count(batch_label)
        + int(batch_label_v2_recall_status.get("lora_usable_labels_new", 0) or 0),
        "active_100k_output_dir": str(active_dir),
        "extra_broad_search_hits_incremental": count_csv(extra_search_dir / "search_hits_incremental.csv"),
        "extra_broad_search_hits_final": count_csv(extra_search_dir / "search_hits.csv"),
        "extra_broad_fetch_search_hits": count_csv(extra_fetch_dir / "search_hits.csv"),
        "extra_broad_fetch_raw_cases_incremental": count_csv(extra_fetch_dir / "raw_public_cases_incremental.csv"),
        "extra_broad_fetch_raw_text_manifest_incremental": count_csv(extra_fetch_dir / "raw_text_manifest_incremental.csv"),
        "extra_broad_fetch_raw_text_manifest_final": count_csv(extra_fetch_dir / "raw_text_manifest.csv"),
        "extra_broad_fetch2_search_hits": count_csv(extra_fetch2_dir / "search_hits.csv"),
        "extra_broad_fetch2_raw_cases_incremental": count_csv(extra_fetch2_dir / "raw_public_cases_incremental.csv"),
        "extra_broad_fetch2_raw_text_manifest_incremental": count_csv(extra_fetch2_dir / "raw_text_manifest_incremental.csv"),
        "extra_broad_fetch2_raw_text_manifest_final": count_csv(extra_fetch2_dir / "raw_text_manifest.csv"),
        "extra_broad_fetch3_search_hits": count_csv(extra_fetch3_dir / "search_hits.csv"),
        "extra_broad_fetch3_raw_cases_incremental": count_csv(extra_fetch3_dir / "raw_public_cases_incremental.csv"),
        "extra_broad_fetch3_raw_text_manifest_incremental": count_csv(extra_fetch3_dir / "raw_text_manifest_incremental.csv"),
        "extra_broad_fetch3_raw_text_manifest_final": count_csv(extra_fetch3_dir / "raw_text_manifest.csv"),
        "delay_deep_search_hits_incremental": count_csv(delay_deep_search_dir / "search_hits_incremental.csv"),
        "delay_deep_search_hits_final": count_csv(delay_deep_search_dir / "search_hits.csv"),
        "delay_deep_fetch_search_hits": count_csv(delay_deep_fetch_dir / "search_hits.csv"),
        "delay_deep_fetch_raw_cases_incremental": count_csv(delay_deep_fetch_dir / "raw_public_cases_incremental.csv"),
        "delay_deep_fetch_raw_text_manifest_incremental": count_csv(delay_deep_fetch_dir / "raw_text_manifest_incremental.csv"),
        "delay_deep_fetch_raw_text_manifest_final": count_csv(delay_deep_fetch_dir / "raw_text_manifest.csv"),
        "delay_generic_search_hits_incremental": count_csv(delay_generic_search_dir / "search_hits_incremental.csv"),
        "delay_generic_search_hits_final": count_csv(delay_generic_search_dir / "search_hits.csv"),
        "delay_generic_fetch_search_hits": count_csv(delay_generic_fetch_dir / "search_hits.csv"),
        "delay_generic_fetch_raw_cases_incremental": count_csv(delay_generic_fetch_dir / "raw_public_cases_incremental.csv"),
        "delay_generic_fetch_raw_text_manifest_incremental": count_csv(delay_generic_fetch_dir / "raw_text_manifest_incremental.csv"),
        "delay_generic_fetch_raw_text_manifest_final": count_csv(delay_generic_fetch_dir / "raw_text_manifest.csv"),
        "extra_broad_fetch4_search_hits": count_csv(extra_fetch4_dir / "search_hits.csv"),
        "extra_broad_fetch4_raw_text_manifest_incremental": count_csv(extra_fetch4_dir / "raw_text_manifest_incremental.csv"),
        "extra_broad_fetch4_raw_text_manifest_final": count_csv(extra_fetch4_dir / "raw_text_manifest.csv"),
        "delay_deep_fetch2_search_hits": count_csv(delay_deep_fetch2_dir / "search_hits.csv"),
        "delay_deep_fetch2_raw_text_manifest_incremental": count_csv(delay_deep_fetch2_dir / "raw_text_manifest_incremental.csv"),
        "delay_deep_fetch2_raw_text_manifest_final": count_csv(delay_deep_fetch2_dir / "raw_text_manifest.csv"),
        "delay_generic_fetch2_search_hits": count_csv(delay_generic_fetch2_dir / "search_hits.csv"),
        "delay_generic_fetch2_raw_text_manifest_incremental": count_csv(delay_generic_fetch2_dir / "raw_text_manifest_incremental.csv"),
        "delay_generic_fetch2_raw_text_manifest_final": count_csv(delay_generic_fetch2_dir / "raw_text_manifest.csv"),
        "active_collect_processes": active_collect_processes(),
        "raw_text_original_root_txt_count": len(list(raw_text_root.glob("*.txt"))) if raw_text_root.exists() else 0,
        "raw_text_exported_mimo_usable_txt_count": len(list((raw_text_root / "mimo_public_usable_30k_20260527" / "txt").glob("*.txt"))) if (raw_text_root / "mimo_public_usable_30k_20260527" / "txt").exists() else 0,
        "mimo_screened_unique": int(screening["pageid"].nunique()) if not screening.empty and "pageid" in screening.columns else 0,
        "mimo_raw_accepted": int(pd.to_numeric(screening.get("mimo_accept_flag", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not screening.empty else 0,
        "api_errors": int((screening.get("mimo_status", pd.Series(dtype=str)) == "api_error").sum()) if not screening.empty else 0,
        "usable_for_research_total": usable_summary.get("usable_for_research_total"),
        "tier_a_mimo_strict_complete": usable_summary.get("tier_a_mimo_strict_complete"),
        "tier_b_local_delay_candidate": usable_summary.get("tier_b_local_delay_candidate"),
        "combined_raw_rows_after_text_dedup": combined_summary.get("raw_rows_after_text_dedup"),
        "combined_usable_for_research_total": combined_summary.get("usable_for_research_total"),
        "strict_delay_usable_pool": pool_views.get("strict_delay_usable"),
        "broad_delay_candidate_pool": pool_views.get("broad_delay_candidate"),
        "broad_construction_dispute_support_pool": pool_views.get("broad_construction_dispute_support_pool"),
        "chinese_delay_research_candidate_pool": chinese_delay_pool.get("chinese_delay_research_candidate_total"),
        "chinese_delay_usable_relaxed_pool": chinese_delay_pool.get("chinese_delay_usable_relaxed_total"),
        "chinese_delay_usable_pool": chinese_delay_pool.get("chinese_delay_usable_total"),
        "chinese_delay_strong_pool": chinese_delay_pool.get("chinese_delay_strong_total"),
        "chinese_delay_pool_note": chinese_delay_pool.get("note"),
        "combined_tier_a_mimo_strict_complete": combined_tiers.get("Tier_A_mimo_strict_complete"),
        "combined_tier_b_plus_high_quality_local": combined_tiers.get("Tier_B_plus_high_quality_local"),
        "combined_tier_b_local_delay_candidate": combined_tiers.get("Tier_B_local_delay_candidate"),
        **usage,
        "combined_prompt_tokens": usage["prompt_tokens"] + active_mimo_usage["prompt_tokens"] + wide_mimo_usage["prompt_tokens"] + fast_label_usage["prompt_tokens"] + batch_label_usage["prompt_tokens"] + batch_label_v2_recall_usage["prompt_tokens"],
        "combined_completion_tokens": usage["completion_tokens"] + active_mimo_usage["completion_tokens"] + wide_mimo_usage["completion_tokens"] + fast_label_usage["completion_tokens"] + batch_label_usage["completion_tokens"] + batch_label_v2_recall_usage["completion_tokens"],
        "combined_total_tokens": usage["total_tokens"] + active_mimo_usage["total_tokens"] + wide_mimo_usage["total_tokens"] + fast_label_usage["total_tokens"] + batch_label_usage["total_tokens"] + batch_label_v2_recall_usage["total_tokens"],
    }
    out = mimo_dir / "mimo_token_status_latest.json"
    out.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    hist = mimo_dir / "mimo_token_usage_history.csv"
    row = pd.DataFrame([status])
    if hist.exists():
        old = read_csv(hist)
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(hist, index=False, encoding="utf-8-sig")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
