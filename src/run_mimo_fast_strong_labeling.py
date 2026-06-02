# -*- coding: utf-8 -*-
"""Fast Mimo-assisted strong-label extraction for downloaded public cases.

This script labels only documents that have already been downloaded into the
local raw-text corpus. It uses the full adjudicated document for machine-assisted
label extraction, but it does not create prediction inputs from post-decision
text. Downstream LoRA/export scripts must still build inputs from pre-decision
facts only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from collect_public_cases_mimo import call_mimo, clean_ws, extract_json_object, sha256_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALID_LABELS = {"support", "partial_support", "not_support"}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_project_path(path_value: Any) -> Path:
    p = Path(str(path_value or ""))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def usage_sum(df: pd.DataFrame) -> Dict[str, int]:
    prompt = completion = total = 0
    if "usage_json" not in df.columns:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for s in df["usage_json"].dropna().astype(str):
        try:
            obj = json.loads(s)
            prompt += int(obj.get("prompt_tokens", 0) or 0)
            completion += int(obj.get("completion_tokens", 0) or 0)
            total += int(obj.get("total_tokens", 0) or 0)
        except Exception:
            continue
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def compact_for_labeling(title: str, raw_text: str, max_chars: int) -> str:
    text = clean_ws(raw_text)
    if len(text) <= max_chars:
        return text

    chunks: List[str] = []
    # Facts and party claims are usually near the front.
    chunks.append(text[: int(max_chars * 0.34)])

    decision_markers = [
        "本院认为",
        "法院认为",
        "一审法院认为",
        "二审法院认为",
        "裁判结果",
        "判决如下",
        "综上",
        "依照",
    ]
    for marker in decision_markers:
        pos = text.find(marker)
        if pos >= 0:
            chunks.append(text[max(0, pos - 1600) : pos + 2600])
            break

    delay_terms = [
        "工期",
        "逾期",
        "延期",
        "延误",
        "停工",
        "窝工",
        "签证",
        "索赔",
        "开工",
        "竣工",
        "进度",
        "违约金",
        "关键线路",
        "鉴定",
    ]
    hit_chunks: List[str] = []
    for term in delay_terms:
        pos = text.find(term)
        if pos >= 0:
            hit_chunks.append(text[max(0, pos - 500) : pos + 1000])
        if sum(len(x) for x in hit_chunks) >= int(max_chars * 0.26):
            break
    chunks.extend(hit_chunks)
    chunks.append(text[-int(max_chars * 0.12) :])

    return clean_ws("\n".join(chunks))[:max_chars]


def build_label_messages(row: Dict[str, Any], raw_text: str, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    text = compact_for_labeling(
        str(row.get("title", "")),
        raw_text,
        int(cfg["labeling"].get("max_chars_for_labeling", 6500)),
    )
    payload = {
        "task": "machine_assisted_outcome_label_extraction_for_construction_schedule_delay_dispute",
        "label_definition": {
            "support": "The court substantially supports the delay-related construction claim, time-extension claim, delay compensation claim, or delay-liability position.",
            "partial_support": "The court partly supports the delay-related claim or accepts only part of the delay/liability/damages position.",
            "not_support": "The court rejects or does not support the delay-related claim or delay-liability position.",
            "unknown": "Use only when the outcome cannot be reliably extracted.",
        },
        "important_rules": [
            "Return exactly one JSON object and no Markdown.",
            "Do not fabricate facts or labels.",
            "Use the adjudicated decision section only to extract the training label.",
            "If the delay issue is not material, output unknown and needs_review=true.",
            "If the document is procedural-only, duplicated, incomplete, or label-conflicted, output unknown or needs_review=true.",
            "These are machine-assisted labels, not human gold labels.",
        ],
        "required_json": {
            "outcome_label": "support|partial_support|not_support|unknown",
            "label_confidence": "float 0-1",
            "delay_claim_material": "boolean",
            "substantive_decision_flag": "boolean",
            "procedural_only_flag": "boolean",
            "pre_decision_sufficiency": "float 0-1",
            "decision_basis_span": "short exact or near-exact span from decision text",
            "decision_basis_summary": "short Chinese summary",
            "responsibility_folded": "owner|contractor|shared|unclear",
            "responsibility_confidence": "float 0-1",
            "evidence_role_coverage": "float 0-1",
            "conflict_flag": "boolean",
            "needs_review": "boolean",
            "reason_short": "short Chinese reason",
        },
        "case_metadata": {
            "case_id": row.get("case_id", ""),
            "pageid": row.get("pageid", ""),
            "title": row.get("title", ""),
            "usable_tier": row.get("usable_tier", ""),
            "text_chars": row.get("text_chars", ""),
        },
        "document_excerpt": text,
    }
    prompt = json.dumps(payload, ensure_ascii=False)
    return [
        {"role": "system", "content": "You are a strict legal-data labeling assistant. Return exactly one valid JSON object. No Markdown."},
        {"role": "user", "content": prompt},
    ], prompt


def get_mimo_settings(cfg: Dict[str, Any]) -> Tuple[str, str, str, str]:
    mc = cfg["mimo"]
    base_url = os.getenv(str(mc.get("base_url_env", "MIMO_OPENAI_BASE_URL")), "").strip().rstrip("/")
    api_key = os.getenv(str(mc.get("api_key_env", "MIMO_API_KEY")), "").strip()
    model = str(mc.get("model_name", "mimo-v2.5-flash"))
    fallback_model = str(mc.get("fallback_model_name", "") or "")
    return base_url, api_key, model, fallback_model


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower().replace("-", "_")
    if label in {"partial", "partially_support", "partial support"}:
        return "partial_support"
    if label in {"not support", "not-supported", "reject", "rejected", "unsupported"}:
        return "not_support"
    return label


def call_with_fallback(
    base_url: str,
    api_key: str,
    model: str,
    fallback_model: str,
    messages: List[Dict[str, str]],
    cfg: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], str]:
    try:
        raw, usage = call_mimo(base_url, api_key, model, messages, cfg)
        return raw, usage, model
    except Exception as exc:
        if not fallback_model or fallback_model == model:
            raise
        msg = str(exc)
        # Fall back only for likely model availability errors. Rate-limit errors
        # should be retried by the caller rather than amplifying traffic.
        if any(x in msg.lower() for x in ["model", "404", "400", "not found", "invalid"]):
            raw, usage = call_mimo(base_url, api_key, fallback_model, messages, cfg)
            return raw, usage, fallback_model
        raise


def label_one(row: Dict[str, Any], cfg: Dict[str, Any], base_url: str, api_key: str, model: str, fallback_model: str) -> Dict[str, Any]:
    started = time.time()
    raw_path = resolve_project_path(row.get("raw_text_path", ""))
    try:
        raw_text = raw_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return {
            "case_id": row.get("case_id", ""),
            "pageid": row.get("pageid", ""),
            "title": row.get("title", ""),
            "raw_text_path": str(row.get("raw_text_path", "")),
            "label_status": "read_error",
            "error": str(exc)[:800],
        }

    messages, prompt = build_label_messages(row, raw_text, cfg)
    last_error = ""
    retries = int(cfg["mimo"].get("retries", 1))
    used_model = model
    for attempt in range(retries + 1):
        try:
            raw, usage, used_model = call_with_fallback(base_url, api_key, model, fallback_model, messages, cfg)
            parsed = extract_json_object(raw)
            label = normalize_label(parsed.get("outcome_label"))
            confidence = float(parsed.get("label_confidence", 0.0) or 0.0)
            pre_suff = float(parsed.get("pre_decision_sufficiency", 0.0) or 0.0)
            role_cov = float(parsed.get("evidence_role_coverage", 0.0) or 0.0)
            conflict = bool(parsed.get("conflict_flag", True))
            needs_review = bool(parsed.get("needs_review", True))
            delay_material = bool(parsed.get("delay_claim_material", False))
            substantive = bool(parsed.get("substantive_decision_flag", False))
            procedural_only = bool(parsed.get("procedural_only_flag", True))
            basis_span = str(parsed.get("decision_basis_span", "") or "").strip()
            min_conf = float(cfg["labeling"].get("min_label_confidence", 0.85))
            min_usable_conf = float(cfg["labeling"].get("min_usable_label_confidence", 0.75))
            min_pre_suff = float(cfg["labeling"].get("min_pre_decision_sufficiency", 0.60))
            strong = (
                label in VALID_LABELS
                and confidence >= min_conf
                and pre_suff >= min_pre_suff
                and delay_material
                and substantive
                and not procedural_only
                and not conflict
                and not needs_review
                and len(basis_span) >= 4
            )
            usable = (
                label in VALID_LABELS
                and confidence >= min_usable_conf
                and pre_suff >= max(0.45, min_pre_suff - 0.15)
                and delay_material
                and substantive
                and not procedural_only
                and not conflict
                and len(basis_span) >= 4
            )
            if label not in VALID_LABELS:
                bucket = "unknown_or_discarded"
            elif strong:
                bucket = "strong_label"
            elif usable:
                bucket = "lora_usable_label"
            else:
                bucket = "weak_label"
            return {
                "case_id": row.get("case_id", ""),
                "pageid": row.get("pageid", ""),
                "title": row.get("title", ""),
                "source_url": row.get("source_url", ""),
                "raw_text_path": str(row.get("raw_text_path", "")),
                "text_sha256": row.get("text_sha256", ""),
                "usable_tier": row.get("usable_tier", ""),
                "text_chars": row.get("text_chars", ""),
                "label_status": "ok",
                "model_name": used_model,
                "attempts": attempt + 1,
                "latency_sec": round(time.time() - started, 4),
                "prompt_sha256": sha256_text(prompt),
                "prompt_chars": len(prompt),
                "outcome_label": label,
                "label_confidence": confidence,
                "delay_claim_material": int(delay_material),
                "substantive_decision_flag": int(substantive),
                "procedural_only_flag": int(procedural_only),
                "pre_decision_sufficiency": pre_suff,
                "decision_basis_span": basis_span[:500],
                "decision_basis_summary": str(parsed.get("decision_basis_summary", ""))[:500],
                "responsibility_folded": str(parsed.get("responsibility_folded", "unclear")),
                "responsibility_confidence": parsed.get("responsibility_confidence", 0),
                "evidence_role_coverage": role_cov,
                "conflict_flag": int(conflict),
                "needs_review": int(needs_review),
                "reason_short": str(parsed.get("reason_short", ""))[:500],
                "strong_label_flag": int(strong),
                "lora_usable_label_flag": int(usable),
                "final_bucket": bucket,
                "usage_json": json.dumps(usage, ensure_ascii=False),
                "raw_response": raw,
            }
        except Exception as exc:
            last_error = str(exc)[:1000]
            if attempt < retries:
                if "429" in last_error or "Too Many Requests" in last_error:
                    time.sleep(min(45.0, 8.0 * (attempt + 1)))
                else:
                    time.sleep(1.0 + attempt)
    return {
        "case_id": row.get("case_id", ""),
        "pageid": row.get("pageid", ""),
        "title": row.get("title", ""),
        "source_url": row.get("source_url", ""),
        "raw_text_path": str(row.get("raw_text_path", "")),
        "text_sha256": row.get("text_sha256", ""),
        "usable_tier": row.get("usable_tier", ""),
        "label_status": "api_error",
        "model_name": used_model,
        "latency_sec": round(time.time() - started, 4),
        "strong_label_flag": 0,
        "lora_usable_label_flag": 0,
        "final_bucket": "api_error",
        "error": last_error,
    }


def build_candidate_pool(cfg: Dict[str, Any], out_dir: Path, resume: bool) -> pd.DataFrame:
    manifest = read_csv(resolve_project_path(cfg["input"]["combined_usable_manifest"]))
    if manifest.empty:
        raise FileNotFoundError("combined usable manifest is empty or missing")

    prefer_tiers = list(cfg["input"].get("prefer_tiers", []))
    if prefer_tiers and "usable_tier" in manifest.columns:
        manifest = manifest[manifest["usable_tier"].isin(prefer_tiers)].copy()
    manifest = manifest.drop_duplicates("text_sha256", keep="first") if "text_sha256" in manifest.columns else manifest
    tier_rank = {tier: i for i, tier in enumerate(prefer_tiers)}
    manifest["_tier_rank"] = manifest.get("usable_tier", pd.Series([""] * len(manifest))).map(tier_rank).fillna(999)
    for col in ["overall_completeness_score", "evidence_completeness", "pre_decision_facts_sufficient", "text_chars", "local_quality_score"]:
        if col in manifest.columns:
            manifest[col] = pd.to_numeric(manifest[col], errors="coerce").fillna(0)
    sort_cols = [c for c in ["_tier_rank", "overall_completeness_score", "evidence_completeness", "pre_decision_facts_sufficient", "local_quality_score", "text_chars"] if c in manifest.columns]
    ascending = [True] + [False] * (len(sort_cols) - 1)
    if sort_cols:
        manifest = manifest.sort_values(sort_cols, ascending=ascending)

    if resume:
        existing = read_csv(out_dir / "mimo_fast_label_results.csv")
        if not existing.empty:
            done_pageids = set(existing.get("pageid", pd.Series(dtype=str)).dropna().astype(str))
            done_hashes = set(existing.get("text_sha256", pd.Series(dtype=str)).dropna().astype(str))
            mask = ~manifest.get("pageid", pd.Series(dtype=str)).astype(str).isin(done_pageids)
            if "text_sha256" in manifest.columns:
                mask = mask & ~manifest["text_sha256"].astype(str).isin(done_hashes)
            manifest = manifest[mask].copy()

    manifest.drop(columns=["_tier_rank"], errors="ignore").to_csv(out_dir / "candidate_pool_for_labeling.csv", index=False, encoding="utf-8-sig")
    return manifest.drop(columns=["_tier_rank"], errors="ignore")


def write_derived_outputs(out_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    result_path = out_dir / "mimo_fast_label_results.csv"
    df = read_csv(result_path)
    strong = df[df.get("strong_label_flag", pd.Series(dtype=int)).fillna(0).astype(int) == 1].copy() if not df.empty else pd.DataFrame()
    if not df.empty:
        strong_flag = df.get("strong_label_flag", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int).eq(1)
        usable_flag = df.get("lora_usable_label_flag", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int).eq(1)
        usable = df[strong_flag | usable_flag].copy()
    else:
        usable = pd.DataFrame()
    weak = df[df.get("final_bucket", pd.Series(dtype=str)).isin(["weak_label", "unknown_or_discarded", "api_error"])].copy() if not df.empty else pd.DataFrame()
    strong.to_csv(out_dir / "mimo_fast_strong_labels.csv", index=False, encoding="utf-8-sig")
    usable.to_csv(out_dir / "mimo_fast_lora_usable_labels.csv", index=False, encoding="utf-8-sig")
    weak.to_csv(out_dir / "mimo_fast_weak_or_unknown.csv", index=False, encoding="utf-8-sig")

    existing_path = resolve_project_path(cfg["input"].get("existing_frozen_strong_labels", ""))
    existing_count = len(read_csv(existing_path)) if existing_path.exists() else 0
    label_dist = usable["outcome_label"].value_counts(dropna=False).reset_index() if not usable.empty and "outcome_label" in usable.columns else pd.DataFrame(columns=["outcome_label", "count"])
    label_dist.columns = ["outcome_label", "count"]
    label_dist.to_csv(out_dir / "mimo_fast_strong_label_distribution.csv", index=False, encoding="utf-8-sig")
    usage = usage_sum(df)
    status = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "results_total": int(len(df)),
        "strong_labels_new": int(len(strong)),
        "lora_usable_labels_new": int(len(usable)),
        "weak_or_unknown_or_error": int(len(weak)),
        "existing_frozen_strong_labels": int(existing_count),
        "total_strong_labels_counting_existing": int(existing_count + len(strong)),
        "total_lora_usable_counting_existing": int(existing_count + len(usable)),
        "target_total_strong_labels": int(cfg["labeling"].get("target_total_strong_labels", 20000)),
        "target_reached": bool(existing_count + len(usable) >= int(cfg["labeling"].get("target_total_strong_labels", 20000))),
        **usage,
        "note": "Machine-assisted strong labels only; not human gold. Label extraction may use full adjudicated text, while downstream prediction inputs must remain pre-decision only.",
    }
    (out_dir / "strong_label_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/research_mimo_fast_strong_labeling.json")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--max-label-cases", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_json(resolve_project_path(args.config))
    out_dir = ensure_dir(resolve_project_path(args.out_dir or cfg["output"]["out_dir"]))
    base_url, api_key, model, fallback_model = get_mimo_settings(cfg)
    if not base_url or not api_key:
        pd.DataFrame([{"label_status": "api_unavailable", "error": "missing MIMO base_url or api_key"}]).to_csv(
            out_dir / "mimo_fast_label_results.csv", index=False, encoding="utf-8-sig"
        )
        write_derived_outputs(out_dir, cfg)
        return

    pool = build_candidate_pool(cfg, out_dir, args.resume)
    if args.max_label_cases and args.max_label_cases > 0:
        pool = pool.head(args.max_label_cases).copy()

    result_path = out_dir / "mimo_fast_label_results.csv"
    existing = read_csv(result_path) if args.resume else pd.DataFrame()
    rows: List[Dict[str, Any]] = existing.to_dict("records") if not existing.empty else []
    workers = int(cfg["mimo"].get("workers", 4))
    save_every = int(cfg["labeling"].get("save_every", 10))
    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(resolve_project_path(args.config)),
        "out_dir": str(out_dir),
        "model_name": model,
        "fallback_model_name": fallback_model,
        "workers": workers,
        "max_label_cases": args.max_label_cases,
        "candidate_rows_this_run": int(len(pool)),
        "api_key_present": bool(api_key),
        "note": "API key value intentionally not recorded.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if pool.empty:
        write_derived_outputs(out_dir, cfg)
        return

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(label_one, row.to_dict(), cfg, base_url, api_key, model, fallback_model) for _, row in pool.iterrows()]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % save_every == 0 or i == len(futs):
                pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
                write_derived_outputs(out_dir, cfg)
                print(f"[mimo-fast-label] saved {len(rows)} rows; this_run={i}/{len(futs)}", flush=True)

    status = write_derived_outputs(out_dir, cfg)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
