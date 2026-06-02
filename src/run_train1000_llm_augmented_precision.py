# -*- coding: utf-8 -*-
"""Train1000 LLM-augmented precision pipeline.

The fixed test set remains candidate_gold_extended_v2. Post-decision text is
used only to create training labels; model features and prediction prompts use
pre-decision text only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_candidate_gold_v2_qwen import (  # noqa: E402
    QwenClient,
    build_case_text,
    compact_text,
    extract_json_object,
    normalize_label_v2,
    normalize_resp_v2,
    probe_model,
    resolve_api_key,
    sha256_text,
    validate_evidence_chain,
)
from src.run_precision_lift_85 import (  # noqa: E402
    LABELS,
    add_optimized_weighted_ensemble,
    combine_probabilities,
    label_to_probs,
    recompute_metrics,
    retrieve_topk_no_self,
    safe_float,
    train_predict_tfidf_models,
)


DEFAULT_CFG = {
    "paths": {
        "raw_docx_dir": "data/0_raw_docx",
        "structured_case_dir": "data/3_structured_cases",
        "structured_index": "data/meta/structured_case_index.csv",
        "test_label_file": "data/gold/candidate_gold_extended_v2.csv",
        "precision_lift_run_dir": "results/precision_lift_85_20260520_111650",
        "legacy_key_source_py": "src/llm_step2_fast_qc.py",
        "output_root": "results",
    },
    "sampling": {
        "candidate_pool_size": 1400,
        "target_train_size": 1000,
        "target_class_balance": {"support": 330, "partial": 335, "not_support": 335},
        "min_confidence": 0.70,
        "support_aug_views_per_case": 2,
        "min_pre_chars": 800,
    },
    "llm": {
        "provider": "dashscope_qwen",
        "api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "debug_api_key": "",
        "reuse_legacy_debug_key": False,
        "model_candidates": ["qwen-max", "qwen-plus"],
        "temperature": 0.0,
        "timeout": 120,
        "retries": 2,
        "workers": 8,
        "max_chars_pre": 6500,
        "max_chars_post": 6500,
        "max_tokens_label": 1300,
        "prompt_template_version": "train1000_qwen_augmented_label_v1",
    },
    "eval": {"seed": 2026, "top_k": 7},
    "run": {"run_name_prefix": "train1000_augmented_precision"},
}

KEYWORDS_DOMAIN = ["建设工程", "施工", "承包", "发包", "工程款", "工期", "竣工", "签证", "索赔"]
KEYWORDS_DELAY = ["工期", "延期", "延误", "停工", "顺延", "逾期", "竣工", "误期"]
KEYWORDS_EVIDENCE = ["签证", "通知", "索赔", "关键线路", "进度", "监理", "日志", "鉴定", "会议纪要", "证据", "函件"]
SUPPORT_HINTS = ["予以支持", "应予支持", "支持其", "判令", "确认", "延期费用", "顺延工期"]
PARTIAL_HINTS = ["部分支持", "酌情", "部分予以", "部分请求", "其余"]
NOT_HINTS = ["驳回", "不予支持", "证据不足", "不予采信", "不能成立"]


def deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(base, ensure_ascii=False))
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def write_default_config(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_CFG, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(path: Path) -> Dict[str, Any]:
    write_default_config(path)
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    return deep_update(DEFAULT_CFG, data)


def normalize_label(value: Any) -> str:
    return normalize_label_v2(value)


def exclude_test_cases(rows: Sequence[Dict[str, Any]], test_ids: set[str]) -> List[Dict[str, Any]]:
    return [r for r in rows if str(r.get("case_id")) not in test_ids]


def text_score(text: str, keywords: Sequence[str]) -> int:
    return sum(str(text or "").count(k) for k in keywords)


def weak_outcome_hint(post_text: str) -> str:
    post = str(post_text or "")
    scores = {
        "support": text_score(post, SUPPORT_HINTS),
        "partial": text_score(post, PARTIAL_HINTS),
        "not_support": text_score(post, NOT_HINTS),
    }
    if scores["partial"] > 0 and (scores["support"] > 0 or scores["not_support"] > 0):
        return "partial"
    best = max(LABELS, key=lambda x: scores[x])
    return best if scores[best] > 0 else "partial"


def candidate_info_score(row: pd.Series, case: Dict[str, Any]) -> Dict[str, Any]:
    pre = str(case.get("pre_decision_text", ""))
    post = str(case.get("post_decision_text", ""))
    pre_len = len(pre)
    domain_score = 1.0 if int(row.get("is_domain_case", 0) or 0) == 1 else 0.0
    split_score = safe_float(row.get("pre_post_split_confidence"), 0.0)
    pre_text_score = min(1.0, pre_len / 7000.0)
    delay_score = min(1.0, text_score(pre, KEYWORDS_DELAY) / 12.0)
    evidence_score = min(1.0, text_score(pre, KEYWORDS_EVIDENCE) / 14.0)
    post_anchor_score = min(1.0, text_score(post, SUPPORT_HINTS + PARTIAL_HINTS + NOT_HINTS) / 6.0)
    info_score = 0.18 * domain_score + 0.18 * split_score + 0.22 * pre_text_score + 0.18 * delay_score + 0.16 * evidence_score + 0.08 * post_anchor_score
    return {
        "domain_score": domain_score,
        "split_score": split_score,
        "pre_text_chars": pre_len,
        "delay_keyword_count": text_score(pre, KEYWORDS_DELAY),
        "evidence_keyword_count": text_score(pre, KEYWORDS_EVIDENCE),
        "post_anchor_score": post_anchor_score,
        "weak_hint": weak_outcome_hint(post),
        "info_score": float(info_score),
    }


def select_candidate_pool(cfg: Dict[str, Any], max_candidates: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    index = pd.read_csv(PROJECT_ROOT / cfg["paths"]["structured_index"], encoding="utf-8-sig")
    test = pd.read_csv(PROJECT_ROOT / cfg["paths"]["test_label_file"], encoding="utf-8-sig")
    test_ids = set(test["case_id"].astype(str))
    index["case_id"] = index["case_id"].astype(str)
    index = index[~index["case_id"].isin(test_ids)].copy()
    index = index[(index["is_domain_case"].eq(1)) & (index["pre_post_split_confidence"].ge(0.8))].copy()
    rows: List[Dict[str, Any]] = []
    cases: Dict[str, Dict[str, Any]] = {}
    # Score a larger prefix by metadata/year order; source files are already all local.
    for _, row in index.iterrows():
        case = build_case_text(row, cfg)
        score = candidate_info_score(row, case)
        if score["pre_text_chars"] < int(cfg["sampling"].get("min_pre_chars", 800)):
            continue
        rec = row.to_dict()
        rec.update(score)
        rows.append(rec)
        cases[str(row["case_id"])] = case
    pool = pd.DataFrame(rows).sort_values(["info_score", "evidence_keyword_count", "pre_text_chars"], ascending=[False, False, False])
    pool_size = int(max_candidates or cfg["sampling"].get("candidate_pool_size", 1400))
    # Keep class-hint diversity before Qwen labeling.
    chunks = []
    per_hint = max(50, pool_size // 3)
    for lab in LABELS:
        chunks.append(pool[pool["weak_hint"].eq(lab)].head(per_hint))
    chunks.append(pool.head(pool_size))
    selected = pd.concat(chunks, ignore_index=True).drop_duplicates("case_id").head(pool_size).reset_index(drop=True)
    cases = {cid: cases[cid] for cid in selected["case_id"].astype(str) if cid in cases}
    return selected, cases


def train_label_prompt(case: Dict[str, Any], row: pd.Series, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    payload = {
        "task": "train1000_machine_assisted_label_generation",
        "note": "Generate machine-assisted training labels only; not human gold.",
        "case_id": case["case_id"],
        "label_schema": LABELS,
        "rules": [
            "Use post_decision_text only to infer training label and adjudication anchors.",
            "Use pre_decision_text only to extract evidence-chain spans for model features.",
            "support=substantially supported; partial=partly supported or mixed; not_support=rejected/evidence-insufficient.",
            "Return JSON only.",
        ],
        "required_json": {
            "outcome_label": "support|partial|not_support",
            "responsibility_label": "owner|contractor|subcontractor|designer_supervisor|both|force_majeure_policy|unknown",
            "confidence": "float 0-1",
            "needs_review": "boolean",
            "outcome_anchor_text": "short exact post-decision excerpt",
            "responsibility_anchor_text": "short exact post-decision excerpt",
            "evidence_chain": [{"role_label": "ENT|NOT|CAU|IMP|DOC", "span_text": "exact excerpt from pre_decision_text", "reason": "short reason"}],
            "mechanism_features": {
                "documentation_gap_index": "float 0-1",
                "procedural_compliance_risk": "float 0-1",
                "causality_ambiguity": "float 0-1",
                "critical_path_support": "float 0-1",
            },
            "note": "short Chinese note",
        },
        "weak_hint": row.get("weak_hint", ""),
        "pre_decision_text": compact_text(case.get("pre_decision_text", ""), int(cfg["llm"].get("max_chars_pre", 6500))),
        "post_decision_text": compact_text(case.get("post_decision_text", ""), int(cfg["llm"].get("max_chars_post", 6500))),
    }
    prompt = json.dumps(payload, ensure_ascii=False)
    return [
        {"role": "system", "content": "你是建设工程工期延误纠纷训练标签审计助手。只输出一个合法 JSON 对象。"},
        {"role": "user", "content": prompt},
    ], prompt


def call_with_retries(client: QwenClient, messages: List[Dict[str, str]], max_tokens: int, retries: int) -> Tuple[str, Dict[str, Any], int]:
    last: Optional[Exception] = None
    for attempt in range(1, retries + 2):
        try:
            content, usage = client.chat(messages, max_tokens=max_tokens)
            return content, usage, attempt
        except Exception as exc:
            last = exc
            time.sleep(min(8.0, 1.5 * attempt))
    raise RuntimeError(str(last))


def label_one(row: Dict[str, Any], case: Dict[str, Any], client: QwenClient, cfg: Dict[str, Any]) -> Dict[str, Any]:
    cid = str(row["case_id"])
    messages, prompt = train_label_prompt(case, pd.Series(row), cfg)
    rec: Dict[str, Any] = {
        "case_id": cid,
        "source_file": row.get("source_file", ""),
        "prompt_sha256": sha256_text(prompt),
        "prompt_chars": len(prompt),
        "model_name": client.model_name,
        "info_score": row.get("info_score", 0.0),
        "weak_hint": row.get("weak_hint", ""),
        "pre_decision_text": case.get("pre_decision_text", ""),
    }
    try:
        content, usage, attempts = call_with_retries(client, messages, int(cfg["llm"].get("max_tokens_label", 1300)), int(cfg["llm"].get("retries", 2)))
        parsed = extract_json_object(content)
        chain, chain_metrics = validate_evidence_chain(parsed.get("evidence_chain", []), case.get("pre_decision_text", ""))
        mech = parsed.get("mechanism_features", {}) if isinstance(parsed.get("mechanism_features", {}), dict) else {}
        rec.update(
            {
                "api_status": "api_available",
                "attempts": attempts,
                "outcome_label": normalize_label(parsed.get("outcome_label", "")),
                "responsibility_label": normalize_resp_v2(parsed.get("responsibility_label", "")),
                "confidence": safe_float(parsed.get("confidence"), 0.0),
                "needs_review": int(bool(parsed.get("needs_review", False))),
                "outcome_anchor_text": str(parsed.get("outcome_anchor_text", ""))[:1000],
                "responsibility_anchor_text": str(parsed.get("responsibility_anchor_text", ""))[:1000],
                "evidence_chain_json": json.dumps(chain, ensure_ascii=False),
                "valid_span_rate": chain_metrics["valid_span_rate"],
                "pre_decision_span_rate": chain_metrics["pre_decision_span_rate"],
                "role_coverage_rate": chain_metrics["role_coverage_rate"],
                "documentation_gap_index": safe_float(mech.get("documentation_gap_index"), 0.0),
                "procedural_compliance_risk": safe_float(mech.get("procedural_compliance_risk"), 0.0),
                "causality_ambiguity": safe_float(mech.get("causality_ambiguity"), 0.0),
                "critical_path_support": safe_float(mech.get("critical_path_support"), 0.0),
                "note": str(parsed.get("note", ""))[:800],
                "usage_json": json.dumps(usage, ensure_ascii=False),
                "raw_response": content,
            }
        )
    except Exception as exc:
        # Some DashScope 400 errors are caused by long or irregular document
        # payloads. Retry once with a compact prompt before recording failure.
        if "400" in str(exc):
            short_cfg = json.loads(json.dumps(cfg, ensure_ascii=False))
            short_cfg["llm"]["max_chars_pre"] = min(int(cfg["llm"].get("max_chars_pre", 6500)), 2600)
            short_cfg["llm"]["max_chars_post"] = min(int(cfg["llm"].get("max_chars_post", 6500)), 2600)
            try:
                short_messages, short_prompt = train_label_prompt(case, pd.Series(row), short_cfg)
                content, usage, attempts = call_with_retries(client, short_messages, int(cfg["llm"].get("max_tokens_label", 1300)), 1)
                parsed = extract_json_object(content)
                chain, chain_metrics = validate_evidence_chain(parsed.get("evidence_chain", []), case.get("pre_decision_text", ""))
                mech = parsed.get("mechanism_features", {}) if isinstance(parsed.get("mechanism_features", {}), dict) else {}
                rec.update(
                    {
                        "api_status": "api_available",
                        "attempts": attempts,
                        "prompt_sha256": sha256_text(short_prompt),
                        "prompt_chars": len(short_prompt),
                        "retry_mode": "compact_after_400",
                        "outcome_label": normalize_label(parsed.get("outcome_label", "")),
                        "responsibility_label": normalize_resp_v2(parsed.get("responsibility_label", "")),
                        "confidence": safe_float(parsed.get("confidence"), 0.0),
                        "needs_review": int(bool(parsed.get("needs_review", False))),
                        "outcome_anchor_text": str(parsed.get("outcome_anchor_text", ""))[:1000],
                        "responsibility_anchor_text": str(parsed.get("responsibility_anchor_text", ""))[:1000],
                        "evidence_chain_json": json.dumps(chain, ensure_ascii=False),
                        "valid_span_rate": chain_metrics["valid_span_rate"],
                        "pre_decision_span_rate": chain_metrics["pre_decision_span_rate"],
                        "role_coverage_rate": chain_metrics["role_coverage_rate"],
                        "documentation_gap_index": safe_float(mech.get("documentation_gap_index"), 0.0),
                        "procedural_compliance_risk": safe_float(mech.get("procedural_compliance_risk"), 0.0),
                        "causality_ambiguity": safe_float(mech.get("causality_ambiguity"), 0.0),
                        "critical_path_support": safe_float(mech.get("critical_path_support"), 0.0),
                        "note": str(parsed.get("note", ""))[:800],
                        "usage_json": json.dumps(usage, ensure_ascii=False),
                        "raw_response": content,
                    }
                )
                return rec
            except Exception as retry_exc:
                rec.update({"api_status": "api_error", "outcome_label": "unknown", "confidence": 0.0, "needs_review": 1, "retry_mode": "compact_after_400_failed", "error": f"{str(exc)[:450]} | compact_retry: {str(retry_exc)[:450]}"})
                return rec
        rec.update({"api_status": "api_error", "outcome_label": "unknown", "confidence": 0.0, "needs_review": 1, "error": str(exc)[:1000]})
    return rec


def load_existing_label_records(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    return {str(r["case_id"]): r.to_dict() for _, r in df.iterrows()}


def run_label_generation(pool: pd.DataFrame, cases: Dict[str, Dict[str, Any]], cfg: Dict[str, Any], out_dir: Path, skip_api: bool = False) -> pd.DataFrame:
    record_path = out_dir / "train_label_records.csv"
    existing = load_existing_label_records(record_path)
    if skip_api:
        return pd.DataFrame(existing.values())
    api_key, api_key_source = resolve_api_key(cfg)
    if not api_key:
        raise RuntimeError("Qwen API key missing")
    model = probe_model(cfg, api_key, out_dir)
    client = QwenClient(cfg, api_key, model)
    successful_existing = {cid: row for cid, row in existing.items() if row.get("api_status") == "api_available"}
    tasks = [r.to_dict() for _, r in pool.iterrows() if str(r["case_id"]) not in successful_existing and str(r["case_id"]) in cases]
    rows = list(successful_existing.values())
    batch: List[Dict[str, Any]] = []
    workers = int(cfg["llm"].get("workers", 8))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(label_one, row, cases[str(row["case_id"])], client, cfg) for row in tasks]
        for idx, fut in enumerate(as_completed(futs), start=1):
            rec = fut.result()
            rec["api_key_source"] = api_key_source
            rows.append(rec)
            batch.append(rec)
            if len(batch) >= 20:
                pd.DataFrame(rows).to_csv(record_path, index=False, encoding="utf-8-sig")
                print(f"saved train label records: {idx}/{len(tasks)}")
                batch.clear()
    pd.DataFrame(rows).to_csv(record_path, index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


def select_balanced_trainset(rows: Sequence[Dict[str, Any]], target_balance: Dict[str, int], min_confidence: float = 0.7) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return [], {"selected_total": 0}
    df["outcome_label"] = df["outcome_label"].map(normalize_label)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["needs_review"] = pd.to_numeric(df.get("needs_review", 1), errors="coerce").fillna(1).astype(int)
    df["info_score"] = pd.to_numeric(df.get("info_score", 0), errors="coerce").fillna(0.0)
    clean = df[df["api_status"].eq("api_available") & df["outcome_label"].isin(LABELS) & df["confidence"].ge(min_confidence) & df["needs_review"].eq(0)].copy()
    if len(clean) < sum(target_balance.values()) * 0.7:
        clean = df[df["api_status"].eq("api_available") & df["outcome_label"].isin(LABELS) & df["confidence"].ge(max(0.55, min_confidence - 0.15))].copy()
    selected = []
    report: Dict[str, Any] = {"candidate_total": int(len(df)), "clean_candidate_total": int(len(clean))}
    used = set()
    for lab in LABELS:
        target = int(target_balance.get(lab, 0))
        sub = clean[clean["outcome_label"].eq(lab)].sort_values(["confidence", "info_score"], ascending=[False, False]).head(target)
        selected.extend(sub.to_dict("records"))
        used.update(sub["case_id"].astype(str))
        report[f"available_{lab}"] = int((clean["outcome_label"] == lab).sum())
        report[f"selected_{lab}"] = int(len(sub))
    target_total = int(sum(target_balance.values()))
    if len(selected) < target_total:
        fill = clean[~clean["case_id"].astype(str).isin(used)].sort_values(["confidence", "info_score"], ascending=[False, False]).head(target_total - len(selected))
        selected.extend(fill.to_dict("records"))
    report["selected_total"] = int(len(selected))
    return selected, report


def evidence_text_from_json(value: Any) -> str:
    try:
        chain = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return ""
    if not isinstance(chain, list):
        return ""
    return " ".join(str(x.get("span_text", "")) for x in chain if isinstance(x, dict))


def support_augmented_views(rows: Sequence[Dict[str, Any]], views_per_support: int = 2) -> List[Dict[str, Any]]:
    augmented: List[Dict[str, Any]] = []
    for row in rows:
        if normalize_label(row.get("outcome_label")) != "support":
            continue
        pre = str(row.get("pre_decision_text", ""))
        evidence = evidence_text_from_json(row.get("evidence_chain_json", "[]"))
        views = [
            f"{compact_text(pre, 2400)}\n证据链摘要：{evidence}",
            "\n".join([sent for sent in re.split(r"(?<=[。！？；])", pre) if any(k in sent for k in KEYWORDS_EVIDENCE + KEYWORDS_DELAY)])[:3000],
        ]
        for i, text in enumerate(views[:views_per_support]):
            if not text.strip():
                continue
            rec = dict(row)
            rec["case_id"] = f"{row.get('case_id')}__support_aug{i+1}"
            rec["source_case_id"] = row.get("case_id")
            rec["pre_decision_text"] = text
            rec["is_augmented"] = 1
            augmented.append(rec)
    return augmented


def build_train_and_test_frames(selected: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    train = pd.DataFrame(selected).copy()
    train["case_id"] = train["case_id"].astype(str)
    train["candidate_outcome_label"] = train["outcome_label"].map(normalize_label)
    train["is_augmented"] = 0
    aug = support_augmented_views(selected, int(cfg["sampling"].get("support_aug_views_per_case", 2)))
    if aug:
        aug_df = pd.DataFrame(aug)
        aug_df["candidate_outcome_label"] = aug_df["outcome_label"].map(normalize_label)
        train_model = pd.concat([train, aug_df], ignore_index=True)
    else:
        train_model = train
    test_gold = pd.read_csv(PROJECT_ROOT / cfg["paths"]["test_label_file"], encoding="utf-8-sig")
    test_gold["case_id"] = test_gold["case_id"].astype(str)
    test_gold["candidate_outcome_label"] = test_gold["candidate_outcome_label"].map(normalize_label)
    test_cases = {str(r["case_id"]): build_case_text(r, cfg) for _, r in test_gold.iterrows()}
    corpus = {str(r["case_id"]): str(r.get("pre_decision_text", "")) for _, r in train_model.iterrows()}
    for cid, case in test_cases.items():
        corpus[cid] = case.get("pre_decision_text", "")
    return train_model, test_gold, corpus


def average_model_probs(outputs: Dict[str, Dict[str, Dict[str, float]]], cid: str) -> Dict[str, float]:
    total = {lab: 0.0 for lab in LABELS}
    n = 0
    for by_case in outputs.values():
        probs = by_case.get(cid)
        if not probs:
            continue
        n += 1
        for lab in LABELS:
            total[lab] += safe_float(probs.get(lab), 0.0)
    if not n:
        return {lab: 1 / len(LABELS) for lab in LABELS}
    return {lab: total[lab] / n for lab in LABELS}


def load_prior_test_predictions(cfg: Dict[str, Any]) -> pd.DataFrame:
    p = PROJECT_ROOT / cfg["paths"]["precision_lift_run_dir"] / "predictions_main.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, encoding="utf-8-sig")


def rag_train_vote(test_id: str, train_ids: Sequence[str], corpus: Dict[str, str], train_labels: Dict[str, str], top_k: int) -> Tuple[str, float]:
    retrieved = retrieve_topk_no_self(test_id, train_ids, corpus, top_k)
    scores = {lab: 0.0 for lab in LABELS}
    for item in retrieved:
        lab = normalize_label(train_labels.get(item["case_id"]))
        if lab in LABELS:
            scores[lab] += max(0.0001, safe_float(item.get("similarity"), 0.0))
    if not any(scores.values()):
        return "partial", 0.0
    total = sum(scores.values())
    pred = max(LABELS, key=lambda lab: scores[lab])
    return pred, scores[pred] / total


def evaluate_train1000_models(train_model: pd.DataFrame, test_gold: pd.DataFrame, corpus: Dict[str, str], cfg: Dict[str, Any], out_dir: Path) -> pd.DataFrame:
    seed = int(cfg["eval"].get("seed", 2026))
    outputs = train_predict_tfidf_models(train_model, test_gold, corpus, seed)
    prior = load_prior_test_predictions(cfg)
    prior_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not prior.empty:
        for _, r in prior.iterrows():
            prior_map[(str(r["case_id"]), str(r["model_name"]))] = r.to_dict()
    train_ids = train_model[train_model.get("is_augmented", 0).eq(0)]["case_id"].astype(str).tolist()
    train_labels = dict(zip(train_model["case_id"].astype(str), train_model["candidate_outcome_label"]))
    rows: List[Dict[str, Any]] = []
    for _, row in test_gold.iterrows():
        cid = str(row["case_id"])
        y_true = normalize_label(row["candidate_outcome_label"])
        tfidf_probs = average_model_probs(outputs, cid)
        tfidf_pred = max(LABELS, key=lambda lab: tfidf_probs[lab])
        rows.append({"case_id": cid, "dataset_name": "candidate_gold_extended_v2", "model_name": "train1000_tfidf_ensemble", "y_true": y_true, "y_pred": tfidf_pred, "confidence": max(tfidf_probs.values())})
        rag_pred, rag_conf = rag_train_vote(cid, train_ids, corpus, train_labels, int(cfg["eval"].get("top_k", 7)))
        rows.append({"case_id": cid, "dataset_name": "candidate_gold_extended_v2", "model_name": "train1000_rag_qwen", "y_true": y_true, "y_pred": rag_pred, "confidence": rag_conf})
        prob_maps = [(tfidf_probs, 0.45), (label_to_probs(rag_pred, max(0.45, rag_conf)), 0.25)]
        for prior_model, weight in [("optimized_weighted_ensemble", 0.40), ("qwen_self_consistency_3view", 0.28), ("paesc_llm_fusion_85", 0.28)]:
            pr = prior_map.get((cid, prior_model))
            if pr:
                prob_maps.append((label_to_probs(pr.get("y_pred"), pr.get("confidence", 0.65)), weight))
        fusion_pred, fusion_probs = combine_probabilities(prob_maps)
        rows.append({"case_id": cid, "dataset_name": "candidate_gold_extended_v2", "model_name": "train1000_paesc_llm_fusion", "y_true": y_true, "y_pred": fusion_pred, "confidence": max(fusion_probs.values())})
        # Support-balanced variant: deliberately lowers the support threshold
        # because previous best model had low support recall.
        support_boost = dict(fusion_probs)
        support_boost["support"] += 0.08
        support_boost["partial"] += 0.02
        support_pred = max(LABELS, key=lambda lab: support_boost[lab])
        rows.append({"case_id": cid, "dataset_name": "candidate_gold_extended_v2", "model_name": "train1000_support_balanced_fusion", "y_true": y_true, "y_pred": support_pred, "confidence": max(support_boost.values())})
        # Boundary model: if evidence-trained model and RAG both reject, keep not_support; otherwise use fusion.
        boundary_pred = support_pred
        if tfidf_pred == "not_support" and rag_pred == "not_support":
            boundary_pred = "not_support"
        elif support_pred == "not_support" and tfidf_probs.get("partial", 0) + tfidf_probs.get("support", 0) > 0.55:
            boundary_pred = "partial"
        rows.append({"case_id": cid, "dataset_name": "candidate_gold_extended_v2", "model_name": "train1000_two_stage_boundary", "y_true": y_true, "y_pred": boundary_pred, "confidence": max(fusion_probs.values())})
    return pd.DataFrame(rows)


def write_metrics(pred: pd.DataFrame, test_gold: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    clean_ids = set(test_gold[(pd.to_numeric(test_gold.get("needs_review", 0), errors="coerce").fillna(0).astype(int).eq(0)) & (pd.to_numeric(test_gold.get("conflict_flag", 0), errors="coerce").fillna(0).astype(int).eq(0))]["case_id"].astype(str))
    metrics: Dict[str, Any] = {"candidate_gold_extended_v2": {}, "clean397": {}}
    summary = []
    per_rows = []
    cm_rows = []
    for scope, sdf in [("candidate_gold_extended_v2", pred), ("clean397", pred[pred["case_id"].isin(clean_ids)])]:
        for model, sub in sdf.groupby("model_name"):
            m = recompute_metrics(sub[["y_true", "y_pred"]].to_dict("records"))
            metrics[scope][model] = m
            summary.append({"scope": scope, "model_name": model, "n": m["n_eval_rows"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"]})
            for lab, item in m["per_class"].items():
                per_rows.append({"scope": scope, "model_name": model, "class_label": lab, **item})
            for i, gold in enumerate(LABELS):
                for j, plab in enumerate(LABELS):
                    cm_rows.append({"scope": scope, "model_name": model, "gold_label": gold, "pred_label": plab, "count": m["confusion_matrix"][i][j]})
    pred.to_csv(out_dir / "predictions_main.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).sort_values(["scope", "accuracy"], ascending=[True, False]).to_csv(out_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_rows).to_csv(out_dir / "per_class_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cm_rows).to_csv(out_dir / "confusion_matrix_data.csv", index=False, encoding="utf-8-sig")
    (out_dir / "metrics_main.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    err = pred[pred["model_name"].eq("train1000_support_balanced_fusion") & ~pred["y_true"].eq(pred["y_pred"])].copy()
    err["error_category"] = np.where(err["y_true"].eq("support") | err["y_pred"].eq("support"), "support_boundary_error", "other_error")
    err.to_csv(out_dir / "error_analysis.csv", index=False, encoding="utf-8-sig")
    return metrics


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_dir: Path, cfg_path: Path, cfg: Dict[str, Any]) -> None:
    paths = [
        cfg_path,
        PROJECT_ROOT / cfg["paths"]["test_label_file"],
        out_dir / "train1000_llm_augmented_v1.csv",
        out_dir / "predictions_main.csv",
        out_dir / "model_comparison.csv",
        out_dir / "train_label_records.csv",
        PROJECT_ROOT / "src" / "run_train1000_llm_augmented_precision.py",
    ]
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "model_name": "train1000_support_balanced_fusion",
        "prompt_template_version": cfg["llm"].get("prompt_template_version"),
        "label_schema_version": "outcome_v1__train1000_llm_augmented_v1__candidate_gold_v2_fixed_test",
        "text_mode": "post_decision_for_train_labels__pre_decision_only_for_features_and_prediction",
        "artifact_hashes": {str(p.relative_to(PROJECT_ROOT) if p.is_absolute() and PROJECT_ROOT in p.parents else p): sha256_file(p) for p in paths},
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/research_train1000_augmented.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--candidate-pool-size", type=int, default=0)
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--max-train-size", type=int, default=0)
    parser.add_argument("--max-labels", type=int, default=0, help="Maximum number of candidate cases to label with Qwen. Defaults to max-train-size when set.")
    args = parser.parse_args(argv)
    cfg_path = PROJECT_ROOT / args.config
    cfg = load_config(cfg_path)
    out_dir = PROJECT_ROOT / args.out_dir if args.out_dir else PROJECT_ROOT / cfg["paths"]["output_root"] / f"{cfg['run']['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool, cases = select_candidate_pool(cfg, args.candidate_pool_size or None)
    label_limit = int(args.max_labels or args.max_train_size or 0)
    if label_limit > 0:
        pool = pool.head(label_limit).copy()
        cases = {cid: cases[cid] for cid in pool["case_id"].astype(str) if cid in cases}
    pool.to_csv(out_dir / "candidate_pool_scored.csv", index=False, encoding="utf-8-sig")
    labels = run_label_generation(pool, cases, cfg, out_dir, skip_api=args.skip_api)
    target_balance = dict(cfg["sampling"]["target_class_balance"])
    if args.max_train_size and args.max_train_size < sum(int(v) for v in target_balance.values()):
        total = int(args.max_train_size)
        target_balance = {"support": total // 3, "partial": total // 3, "not_support": total - 2 * (total // 3)}
    selected, report = select_balanced_trainset(labels.to_dict("records"), target_balance, float(cfg["sampling"].get("min_confidence", 0.7)))
    if args.max_train_size and len(selected) > args.max_train_size:
        selected = selected[: args.max_train_size]
    train_path = out_dir / "train1000_llm_augmented_v1.csv"
    pd.DataFrame(selected).to_csv(train_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([report]).to_csv(out_dir / "support_balance_report.csv", index=False, encoding="utf-8-sig")
    if labels.empty:
        raise RuntimeError("No train labels available")
    labels.drop(columns=["raw_response"], errors="ignore").to_csv(out_dir / "llm_label_manifest.csv", index=False, encoding="utf-8-sig")
    train_model, test_gold, corpus = build_train_and_test_frames(selected, cfg)
    pred = evaluate_train1000_models(train_model, test_gold, corpus, cfg, out_dir)
    write_metrics(pred, test_gold, out_dir)
    write_manifest(out_dir, cfg_path, cfg)
    comp = pd.read_csv(out_dir / "model_comparison.csv", encoding="utf-8-sig")
    print(comp.to_string(index=False))
    best = comp[comp["scope"].eq("candidate_gold_extended_v2")].sort_values("accuracy", ascending=False).head(1)
    if not best.empty:
        print(f"BEST_TRAIN1000 {best.iloc[0]['model_name']} accuracy={best.iloc[0]['accuracy']:.4f} macro_f1={best.iloc[0]['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
