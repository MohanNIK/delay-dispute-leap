# -*- coding: utf-8 -*-
"""Batch Mimo outcome labeling for LoRA-usable construction-delay cases.

This is a throughput-oriented labeler. It extracts only the outcome label and
minimal quality fields for multiple downloaded documents per API call. Labels
remain machine-assisted and are intended for downstream fine-tuning after
quality filtering.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from collect_public_cases_mimo import call_mimo, clean_ws, extract_json_object, sha256_text
from run_mimo_fast_strong_labeling import compact_for_labeling, resolve_project_path, usage_sum

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_LABELS = {"support", "partial_support", "not_support"}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower().replace("-", "_")
    if label in {"partial", "partially_support", "partial support"}:
        return "partial_support"
    if label in {"not support", "not-supported", "reject", "rejected", "unsupported"}:
        return "not_support"
    return label


def get_mimo_settings(cfg: Dict[str, Any]) -> Tuple[str, str, str]:
    mc = cfg["mimo"]
    base_url = os.getenv(str(mc.get("base_url_env", "MIMO_OPENAI_BASE_URL")), "").strip().rstrip("/")
    api_key = os.getenv(str(mc.get("api_key_env", "MIMO_API_KEY")), "").strip()
    model = str(mc.get("model_name", "mimo-v2.5-pro"))
    return base_url, api_key, model


def load_text(row: Dict[str, Any], max_chars: int) -> str:
    path = resolve_project_path(row.get("raw_text_path", ""))
    try:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return compact_for_labeling(str(row.get("title", "")), raw_text, max_chars)


def build_messages(batch_rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    max_chars = int(cfg["labeling"].get("max_chars_per_case", 1800))
    relaxed_lora_mode = bool(cfg["labeling"].get("relaxed_lora_mode", False))
    force_three_label_output = bool(cfg["labeling"].get("force_three_label_output", False))
    cases = []
    for row in batch_rows:
        text = load_text(row, max_chars)
        cases.append(
            {
                "case_id": row.get("case_id", ""),
                "pageid": str(row.get("pageid", "")),
                "title": row.get("title", ""),
                "usable_tier": row.get("usable_tier", ""),
                "document_excerpt": text,
            }
        )
    rules = [
        "Return exactly one JSON object with key results, no Markdown.",
        "Use the adjudicated document excerpt to extract the delay-related outcome label.",
        "Do not guess. If the delay issue is not material or outcome is unclear, use unknown.",
        "Filter procedural-only, incomplete, non-construction, non-delay, or label-conflicted cases.",
        "These are machine-assisted labels for fine-tuning preparation.",
    ]
    if relaxed_lora_mode:
        rules = [
            "Return exactly one JSON object with key results, no Markdown.",
            "This is a high-recall LoRA labeling pass for construction schedule-delay disputes.",
            "If the case contains a substantive delay-related construction claim and the adjudicated disposition can be reasonably inferred, prefer one of support, partial_support, or not_support.",
            "Use unknown only when the case is non-construction, non-delay, purely procedural, lacks substantive adjudication, or the outcome is truly impossible to infer.",
            "A noisy but inferable case may set needs_review=true while still returning the most defensible label and confidence.",
            "Set conflict_flag=true only when multiple incompatible labels are equally supported.",
            "These are machine-assisted labels for fine-tuning preparation.",
        ]
    if force_three_label_output:
        rules = [
            "Return exactly one JSON object with key results, no Markdown.",
            "This pool has already been screened as construction schedule-delay related.",
            "For every case, output exactly one of support, partial_support, or not_support.",
            "Do not output unknown. If evidence is mixed, choose partial_support. If delay claim is mostly rejected, choose not_support. If delay claim is materially accepted, choose support.",
            "Use label_confidence to express uncertainty instead of returning unknown.",
            "Set needs_review=true for noisy or borderline cases, but still provide the best label.",
            "Set conflict_flag=true only when the document contains directly contradictory outcome signals.",
            "These are machine-assisted labels for LoRA fine-tuning preparation.",
        ]
    payload = {
        "task": "batch_outcome_label_extraction_for_construction_schedule_delay_disputes",
        "labels": ["support", "partial_support", "not_support"] if force_three_label_output else ["support", "partial_support", "not_support", "unknown"],
        "rules": rules,
        "required_result_item": {
            "pageid": "string",
            "outcome_label": "support|partial_support|not_support" if force_three_label_output else "support|partial_support|not_support|unknown",
            "label_confidence": "float 0-1",
            "delay_claim_material": "boolean",
            "substantive_decision_flag": "boolean",
            "procedural_only_flag": "boolean",
            "pre_decision_sufficiency": "float 0-1",
            "decision_basis_span": "short span from decision text",
            "conflict_flag": "boolean",
            "needs_review": "boolean",
            "reason_short": "short Chinese reason",
        },
        "cases": cases,
    }
    prompt = json.dumps(payload, ensure_ascii=False)
    return [
        {"role": "system", "content": "Return exactly one valid JSON object: {\"results\":[...]}. No Markdown."},
        {"role": "user", "content": prompt},
    ], prompt


def parse_results(raw: str) -> List[Dict[str, Any]]:
    obj = extract_json_object(raw)
    results = obj.get("results", [])
    return results if isinstance(results, list) else []


def classify_item(item: Dict[str, Any], row: Dict[str, Any], cfg: Dict[str, Any], usage: Dict[str, Any], raw: str, prompt_sha: str, prompt_chars: int, model: str, latency: float) -> Dict[str, Any]:
    label = normalize_label(item.get("outcome_label"))
    confidence = float(item.get("label_confidence", 0.0) or 0.0)
    pre_suff = float(item.get("pre_decision_sufficiency", 0.0) or 0.0)
    basis_span = str(item.get("decision_basis_span", "") or "").strip()
    delay_material = bool(item.get("delay_claim_material", False))
    substantive = bool(item.get("substantive_decision_flag", False))
    procedural = bool(item.get("procedural_only_flag", True))
    conflict = bool(item.get("conflict_flag", True))
    needs_review = bool(item.get("needs_review", True))
    strong = (
        label in VALID_LABELS
        and confidence >= float(cfg["labeling"].get("min_strong_confidence", 0.80))
        and pre_suff >= float(cfg["labeling"].get("min_pre_decision_sufficiency", 0.50))
        and delay_material
        and substantive
        and not procedural
        and not conflict
        and not needs_review
        and len(basis_span) >= 4
    )
    usable = (
        label in VALID_LABELS
        and confidence >= float(cfg["labeling"].get("min_usable_confidence", 0.72))
        and pre_suff >= float(cfg["labeling"].get("min_usable_pre_decision_sufficiency", 0.40))
        and delay_material
        and substantive
        and not procedural
        and not conflict
        and len(basis_span) >= 4
    )
    relaxed_min_conf = float(cfg["labeling"].get("min_relaxed_usable_confidence", cfg["labeling"].get("min_usable_confidence", 0.72)))
    relaxed_min_pre = float(cfg["labeling"].get("min_relaxed_usable_pre_decision_sufficiency", cfg["labeling"].get("min_usable_pre_decision_sufficiency", 0.40)))
    relaxed_min_basis = int(cfg["labeling"].get("min_relaxed_basis_span_chars", 2))
    allow_conflict_relaxed = bool(cfg["labeling"].get("allow_conflict_for_relaxed_lora", False))
    relaxed = (
        label in VALID_LABELS
        and confidence >= relaxed_min_conf
        and pre_suff >= relaxed_min_pre
        and delay_material
        and substantive
        and not procedural
        and (allow_conflict_relaxed or not conflict)
        and len(basis_span) >= relaxed_min_basis
    )
    if bool(cfg["labeling"].get("ultra_relaxed_lora_usable", False)):
        relaxed = (
            label in VALID_LABELS
            and confidence >= relaxed_min_conf
            and (len(basis_span) >= relaxed_min_basis or bool(cfg["labeling"].get("allow_empty_basis_for_ultra_relaxed", False)))
        )
    lora_usable = usable or (bool(cfg["labeling"].get("use_relaxed_as_lora_usable", False)) and relaxed)
    if label not in VALID_LABELS:
        bucket = "unknown_or_discarded"
    elif strong:
        bucket = "strong_label"
    elif usable:
        bucket = "lora_usable_label"
    elif relaxed:
        bucket = "relaxed_lora_usable_label"
    else:
        bucket = "weak_label"
    if strong:
        quality_tier = "strong"
    elif usable:
        quality_tier = "standard_lora"
    elif relaxed:
        quality_tier = "relaxed_lora"
    else:
        quality_tier = "weak_or_unknown"
    return {
        "case_id": row.get("case_id", ""),
        "pageid": row.get("pageid", ""),
        "title": row.get("title", ""),
        "source_url": row.get("source_url", ""),
        "raw_text_path": row.get("raw_text_path", ""),
        "text_sha256": row.get("text_sha256", ""),
        "usable_tier": row.get("usable_tier", ""),
        "label_status": "ok",
        "model_name": model,
        "latency_sec_batch": round(latency, 4),
        "prompt_sha256": prompt_sha,
        "prompt_chars": prompt_chars,
        "outcome_label": label,
        "label_confidence": confidence,
        "delay_claim_material": int(delay_material),
        "substantive_decision_flag": int(substantive),
        "procedural_only_flag": int(procedural),
        "pre_decision_sufficiency": pre_suff,
        "decision_basis_span": basis_span[:500],
        "conflict_flag": int(conflict),
        "needs_review": int(needs_review),
        "reason_short": str(item.get("reason_short", ""))[:500],
        "strong_label_flag": int(strong),
        "standard_lora_usable_label_flag": int(usable or strong),
        "relaxed_lora_usable_label_flag": int(relaxed or usable or strong),
        "lora_usable_label_flag": int(lora_usable or strong),
        "label_quality_tier": quality_tier,
        "final_bucket": bucket,
        "usage_json": json.dumps(usage, ensure_ascii=False),
        "raw_response": raw,
    }


def label_batch(rows: List[Dict[str, Any]], cfg: Dict[str, Any], base_url: str, api_key: str, model: str) -> List[Dict[str, Any]]:
    started = time.time()
    messages, prompt = build_messages(rows, cfg)
    prompt_sha = sha256_text(prompt)
    last_error = ""
    for attempt in range(int(cfg["mimo"].get("retries", 1)) + 1):
        try:
            raw, usage = call_mimo(base_url, api_key, model, messages, cfg)
            parsed_items = parse_results(raw)
            by_pageid = {str(x.get("pageid", "")): x for x in parsed_items if isinstance(x, dict)}
            out: List[Dict[str, Any]] = []
            latency = time.time() - started
            for row in rows:
                item = by_pageid.get(str(row.get("pageid", "")), {})
                if item:
                    out.append(classify_item(item, row, cfg, usage, raw, prompt_sha, len(prompt), model, latency))
                else:
                    out.append(
                        {
                            "case_id": row.get("case_id", ""),
                            "pageid": row.get("pageid", ""),
                            "title": row.get("title", ""),
                            "raw_text_path": row.get("raw_text_path", ""),
                            "label_status": "parse_missing_case",
                            "model_name": model,
                            "strong_label_flag": 0,
                            "lora_usable_label_flag": 0,
                            "final_bucket": "api_error",
                            "error": "case missing from batch response",
                            "usage_json": json.dumps(usage, ensure_ascii=False),
                            "raw_response": raw,
                        }
                    )
            return out
        except Exception as exc:
            last_error = str(exc)[:1000]
            if attempt < int(cfg["mimo"].get("retries", 1)):
                retry_sleep = float(cfg["mimo"].get("rate_limit_sleep_sec", 45.0)) if "429" in last_error else 1.5
                time.sleep(retry_sleep)
    return [
        {
            "case_id": row.get("case_id", ""),
            "pageid": row.get("pageid", ""),
            "title": row.get("title", ""),
            "raw_text_path": row.get("raw_text_path", ""),
            "label_status": "api_error",
            "model_name": model,
            "strong_label_flag": 0,
            "lora_usable_label_flag": 0,
            "final_bucket": "api_error",
            "error": last_error,
        }
        for row in rows
    ]


def make_batches(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def write_outputs(out_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    df = read_csv(out_dir / "mimo_batch_label_results.csv")
    raw_rows_total = int(len(df))
    if not df.empty and "pageid" in df.columns:
        df["_status_rank"] = df.get("label_status", pd.Series([""] * len(df))).astype(str).map(
            {
                "ok": 3,
                "parse_missing_case": 1,
                "api_error": 0,
            }
        ).fillna(2)
        df = df.sort_values(["pageid", "_status_rank"]).drop_duplicates("pageid", keep="last").drop(columns=["_status_rank"], errors="ignore")
    strong = df[df.get("strong_label_flag", pd.Series(dtype=int)).fillna(0).astype(int).eq(1)].copy() if not df.empty else pd.DataFrame()
    usable = df[df.get("lora_usable_label_flag", pd.Series(dtype=int)).fillna(0).astype(int).eq(1)].copy() if not df.empty else pd.DataFrame()
    weak = df[df.get("final_bucket", pd.Series(dtype=str)).isin(["weak_label", "unknown_or_discarded", "api_error"])].copy() if not df.empty else pd.DataFrame()
    strong.to_csv(out_dir / "mimo_batch_strong_labels.csv", index=False, encoding="utf-8-sig")
    usable.to_csv(out_dir / "mimo_batch_lora_usable_labels.csv", index=False, encoding="utf-8-sig")
    weak.to_csv(out_dir / "mimo_batch_weak_or_unknown.csv", index=False, encoding="utf-8-sig")
    status = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "raw_result_rows_total": raw_rows_total,
        "results_total": int(len(df)),
        "strong_labels_new": int(len(strong)),
        "lora_usable_labels_new": int(len(usable)),
        "weak_or_unknown_or_error": int(len(weak)),
        **usage_sum(df),
        "note": "Batch machine-assisted labels; not human gold.",
    }
    (out_dir / "batch_label_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/research_mimo_batch_outcome_labeling.json")
    ap.add_argument("--out-dir", default="data/1_raw_text/combined_delay_dispute_corpus_20260527/mimo_batch_outcome_labels")
    ap.add_argument("--max-label-cases", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--retry-api-errors", action="store_true")
    args = ap.parse_args()

    cfg = load_json(resolve_project_path(args.config))
    out_dir = ensure_dir(resolve_project_path(args.out_dir))
    base_url, api_key, model = get_mimo_settings(cfg)
    if not base_url or not api_key:
        pd.DataFrame([{"label_status": "api_unavailable", "error": "missing MIMO base_url or api_key"}]).to_csv(out_dir / "mimo_batch_label_results.csv", index=False, encoding="utf-8-sig")
        write_outputs(out_dir, cfg)
        return

    manifest = read_csv(resolve_project_path(cfg["input"]["combined_usable_manifest"]))
    if manifest.empty:
        raise SystemExit("missing combined usable manifest")
    done_pageids = set()
    for p in [
        out_dir / "mimo_batch_label_results.csv",
        resolve_project_path(cfg["input"].get("exclude_fast_label_results", "")),
    ]:
        d = read_csv(p)
        if not d.empty and "pageid" in d.columns:
            if args.retry_api_errors and p == out_dir / "mimo_batch_label_results.csv" and "label_status" in d.columns:
                status = d["label_status"].astype(str)
                done_pageids.update(d.loc[~status.isin(["api_error", "parse_missing_case"]), "pageid"].dropna().astype(str).tolist())
            else:
                done_pageids.update(d["pageid"].dropna().astype(str).tolist())
    manifest["pageid"] = manifest["pageid"].astype(str)
    pool = manifest[~manifest["pageid"].isin(done_pageids)].copy()
    tier_rank = {"Tier_A_mimo_strict_complete": 0, "Tier_B_plus_high_quality_local": 1, "Tier_B_local_delay_candidate": 2}
    pool["_tier_rank"] = pool.get("usable_tier", pd.Series([""] * len(pool))).map(tier_rank).fillna(9)
    for col in ["overall_completeness_score", "evidence_completeness", "pre_decision_facts_sufficient", "text_chars"]:
        if col in pool.columns:
            pool[col] = pd.to_numeric(pool[col], errors="coerce").fillna(0)
    sort_cols = ["_tier_rank"]
    ascending = [True]
    for col in ["overall_completeness_score", "evidence_completeness", "pre_decision_facts_sufficient", "chinese_evidence_hits", "chinese_decision_hits", "text_chars"]:
        if col in pool.columns:
            sort_cols.append(col)
            ascending.append(False)
    pool = pool.sort_values(sort_cols, ascending=ascending)
    if args.max_label_cases > 0:
        pool = pool.head(args.max_label_cases)
    pool.drop(columns=["_tier_rank"], errors="ignore").to_csv(out_dir / "candidate_pool_for_batch_labeling.csv", index=False, encoding="utf-8-sig")
    batches = make_batches(pool.drop(columns=["_tier_rank"], errors="ignore").to_dict("records"), int(cfg["labeling"].get("batch_size", 4)))
    existing = read_csv(out_dir / "mimo_batch_label_results.csv") if args.resume else pd.DataFrame()
    rows = existing.to_dict("records") if not existing.empty else []
    result_path = out_dir / "mimo_batch_label_results.csv"

    manifest_obj = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": model,
        "workers": int(cfg["mimo"].get("workers", 3)),
        "batch_size": int(cfg["labeling"].get("batch_size", 4)),
        "candidate_rows_this_run": int(len(pool)),
        "batches_this_run": int(len(batches)),
        "retry_api_errors": bool(args.retry_api_errors),
        "api_key_present": bool(api_key),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    if not batches:
        print(json.dumps(write_outputs(out_dir, cfg), ensure_ascii=False, indent=2))
        return

    save_every_batches = int(cfg["labeling"].get("save_every_batches", 5))
    with ThreadPoolExecutor(max_workers=int(cfg["mimo"].get("workers", 3))) as ex:
        futs = [ex.submit(label_batch, batch, cfg, base_url, api_key, model) for batch in batches]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.extend(fut.result())
            if i % save_every_batches == 0 or i == len(futs):
                pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
                status = write_outputs(out_dir, cfg)
                print(f"[mimo-batch-label] batches={i}/{len(futs)} rows={len(rows)} usable={status['lora_usable_labels_new']}", flush=True)
    print(json.dumps(write_outputs(out_dir, cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
