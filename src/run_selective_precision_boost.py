# -*- coding: utf-8 -*-
"""Selective high-confidence precision boosting for candidate_gold_extended_v2.

This experiment does not change benchmark labels and does not use post-decision
text for prediction. It reports both full-500 performance and selective
high-confidence performance with explicit coverage.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
    probe_model,
    resolve_api_key,
    sha256_text,
    validate_evidence_chain,
)
from src.run_precision_lift_85 import normalize_label, safe_float  # noqa: E402


LABELS = ["support", "partial", "not_support"]
AGREEMENT_MODELS = [
    "optimized_weighted_ensemble",
    "paesc_llm_fusion_85",
    "stacked_meta_logreg",
    "qwen_self_consistency_3view",
    "two_stage_boundary_model",
    "tfidf_word_logreg",
    "tfidf_nb",
    "rag_fewshot_label_vote",
]

DEFAULT_CFG: Dict[str, Any] = {
    "paths": {
        "raw_docx_dir": "data/0_raw_docx",
        "structured_case_dir": "data/3_structured_cases",
        "test_label_file": "data/gold/candidate_gold_extended_v2.csv",
        "precision_lift_run_dir": "results/precision_lift_85_20260520_111650",
        "train_label_records": "results/train1000_augmented_precision_20260521_153425/train_label_records.csv",
        "candidate_v2_prediction_records": "results/candidate_gold_v2_qwen_20260519_131703/prediction_v2_records.csv",
        "legacy_key_source_py": "src/llm_step2_fast_qc.py",
        "output_root": "results",
    },
    "selective": {
        "anchor_min_confidence": 0.80,
        "top_k_per_label": 5,
        "min_coverage": 0.40,
        "consensus_switch_agreement": 0.50,
        "rag_switch_margin": 0.08,
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
        "retries": 1,
        "workers": 6,
        "max_api_hard_cases": 30,
        "max_chars_pre": 7000,
        "max_tokens_arbitration": 1200,
        "prompt_template_version": "selective_precision_hard_case_arbitration_v1",
        "allow_rule_fallback": False,
    },
    "run": {"run_name_prefix": "selective_precision_boost", "seed": 2026},
}


def deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(base, ensure_ascii=False))
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DEFAULT_CFG, ensure_ascii=False, indent=2), encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    return deep_update(DEFAULT_CFG, data)


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_label(value: Any) -> str:
    lab = normalize_label(value)
    return lab if lab in LABELS else "unknown"


def bool_int(value: Any) -> int:
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes"})
    return int(bool(value))


def recompute_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pairs = [(norm_label(r.get("y_true")), norm_label(r.get("y_pred"))) for r in rows]
    pairs = [(a, b) for a, b in pairs if a in LABELS and b in LABELS]
    n = len(pairs)
    correct = sum(1 for a, b in pairs if a == b)
    cm = [[sum(1 for a, b in pairs if a == gold and b == pred) for pred in LABELS] for gold in LABELS]
    per: Dict[str, Dict[str, float]] = {}
    weighted = 0.0
    for lab in LABELS:
        tp = sum(1 for a, b in pairs if a == lab and b == lab)
        fp = sum(1 for a, b in pairs if a != lab and b == lab)
        fn = sum(1 for a, b in pairs if a == lab and b != lab)
        support = sum(1 for a, _ in pairs if a == lab)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[lab] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        weighted += f1 * support
    return {
        "n_eval_rows": n,
        "accuracy": correct / n if n else 0.0,
        "macro_f1": sum(per[lab]["f1"] for lab in LABELS) / len(LABELS),
        "weighted_f1": weighted / n if n else 0.0,
        "per_class": per,
        "confusion_matrix": cm,
    }


def evidence_chain_text(row: Dict[str, Any], raw_limit: int = 3000) -> str:
    chunks = [compact_text(str(row.get("pre_decision_text", "")), raw_limit)]
    try:
        chain = json.loads(row.get("evidence_chain_json", "[]")) if isinstance(row.get("evidence_chain_json", ""), str) else row.get("evidence_chain_json", [])
    except Exception:
        chain = []
    role_parts: List[str] = []
    if isinstance(chain, list):
        for item in chain[:8]:
            if isinstance(item, dict):
                role = str(item.get("role_label", "EVD")).upper()
                span = str(item.get("span_text", ""))
                if span:
                    role_parts.append(f"{role}:{span[:260]}")
    if role_parts:
        chunks.append(" ".join(role_parts))
    for col in ["documentation_gap_index", "procedural_compliance_risk", "causality_ambiguity", "critical_path_support"]:
        if col in row:
            chunks.append(f"{col}={row.get(col)}")
    return "\n".join(chunks)


def build_clean_anchor_bank(train_records: pd.DataFrame, test_ids: Set[str], min_confidence: float = 0.80) -> pd.DataFrame:
    df = train_records.copy()
    for col, default in [("api_status", "api_available"), ("needs_review", 1), ("confidence", 0.0), ("info_score", 0.0), ("pre_decision_text", "")]:
        if col not in df:
            df[col] = default
    df["case_id"] = df["case_id"].astype(str)
    df["outcome_label"] = df["outcome_label"].map(norm_label)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["needs_review"] = pd.to_numeric(df["needs_review"], errors="coerce").fillna(1).astype(int)
    df["info_score"] = pd.to_numeric(df["info_score"], errors="coerce").fillna(0.0)
    clean = df[
        df["api_status"].eq("api_available")
        & df["outcome_label"].isin(LABELS)
        & df["confidence"].ge(float(min_confidence))
        & df["needs_review"].eq(0)
        & ~df["case_id"].isin(test_ids)
    ].copy()
    counts = clean["outcome_label"].value_counts()
    n = int(min(counts.get(lab, 0) for lab in LABELS)) if not counts.empty else 0
    parts = []
    for lab in LABELS:
        part = clean[clean["outcome_label"].eq(lab)].sort_values(["confidence", "info_score"], ascending=[False, False]).head(n)
        parts.append(part)
    if not parts:
        return clean.iloc[0:0].copy()
    out = pd.concat(parts, ignore_index=True)
    out["anchor_text"] = [evidence_chain_text(r) for r in out.to_dict("records")]
    return out.sample(frac=1.0, random_state=2026).reset_index(drop=True)


def char_ngrams(text: str, n: int = 2) -> Set[str]:
    compact = re.sub(r"\s+", "", str(text or ""))
    if len(compact) <= n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def jaccard_similarity(a: str, b: str) -> float:
    aa, bb = char_ngrams(a), char_ngrams(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def label_aware_retrieve(test_case_id: str, test_text: str, anchors: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
    label_scores: Dict[str, float] = {}
    manifest_rows: List[Dict[str, Any]] = []
    for lab in LABELS:
        scored = []
        subset = anchors[anchors["outcome_label"].eq(lab)]
        for _, row in subset.iterrows():
            cid = str(row["case_id"])
            if cid == str(test_case_id):
                continue
            score = jaccard_similarity(test_text, str(row.get("anchor_text", "")))
            scored.append((score, cid, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: int(top_k)]
        label_scores[lab] = float(np.mean([x[0] for x in top])) if top else 0.0
        for rank, (score, cid, row) in enumerate(top, 1):
            manifest_rows.append(
                {
                    "case_id": test_case_id,
                    "retrieved_case_id": cid,
                    "retrieved_label": lab,
                    "rank": rank,
                    "similarity": round(float(score), 6),
                    "anchor_confidence": safe_float(row.get("confidence"), 0.0),
                }
            )
    total = sum(max(v, 0.0) for v in label_scores.values())
    if total > 0:
        probs = {lab: label_scores[lab] / total for lab in LABELS}
    else:
        probs = {lab: 1 / len(LABELS) for lab in LABELS}
    ordered = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return {
        "label_scores": label_scores,
        "label_probs": probs,
        "top_label": ordered[0][0],
        "top_prob": ordered[0][1],
        "margin": ordered[0][1] - ordered[1][1],
        "manifest_rows": manifest_rows,
    }


def choose_threshold(train: pd.DataFrame, min_coverage: float) -> float:
    if train.empty:
        return 1.0
    scores = pd.to_numeric(train["selective_score"], errors="coerce").fillna(0.0)
    candidates = sorted(set(float(scores.quantile(q)) for q in np.linspace(0.0, 0.95, 20)), reverse=True)
    best = (0.0, 0.0, float(scores.quantile(max(0.0, 1.0 - min_coverage))))
    for threshold in candidates:
        sub = train[scores.ge(threshold)]
        coverage = len(sub) / len(train)
        if coverage < min_coverage or sub.empty:
            continue
        acc = float(sub["y_pred"].eq(sub["y_true"]).mean())
        if acc > best[0] or (math.isclose(acc, best[0]) and coverage > best[1]):
            best = (acc, coverage, threshold)
    return float(best[2])


def crossfit_selective_flags(df: pd.DataFrame, min_coverage: float = 0.40) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["high_conf_flag"] = 0
    threshold_rows = []
    folds = sorted(pd.to_numeric(out["fold_id"], errors="coerce").dropna().astype(int).unique())
    if not folds:
        threshold = choose_threshold(out, min_coverage)
        out["high_conf_flag"] = pd.to_numeric(out["selective_score"], errors="coerce").fillna(0.0).ge(threshold).astype(int)
        return out, pd.DataFrame([{"fold_id": -1, "threshold": threshold, "calibration_rows": len(out)}])
    for fold in folds:
        train = out[out["fold_id"].ne(fold)]
        threshold = choose_threshold(train, min_coverage)
        mask = out["fold_id"].eq(fold)
        out.loc[mask, "high_conf_flag"] = pd.to_numeric(out.loc[mask, "selective_score"], errors="coerce").fillna(0.0).ge(threshold).astype(int)
        threshold_rows.append({"fold_id": int(fold), "threshold": threshold, "calibration_rows": len(train)})
    return out, pd.DataFrame(threshold_rows)


def coverage_curve(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    scores = pd.to_numeric(df["selective_score"], errors="coerce").fillna(0.0)
    for coverage_target in np.linspace(0.1, 1.0, 10):
        threshold = float(scores.quantile(max(0.0, 1.0 - coverage_target)))
        sub = df[scores.ge(threshold)]
        m = recompute_metrics(sub.to_dict("records"))
        rows.append(
            {
                "model_name": model_name,
                "coverage_target": round(float(coverage_target), 3),
                "threshold": round(threshold, 6),
                "actual_coverage": len(sub) / len(df) if len(df) else 0.0,
                "n": len(sub),
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
            }
        )
    return pd.DataFrame(rows)


def export_target_accuracy_subsets(predictions: pd.DataFrame, out_dir: Path, targets: Sequence[float] = (0.90, 0.875, 0.845)) -> pd.DataFrame:
    """Export largest score-ranked subsets whose observed accuracy meets targets."""
    base = predictions[predictions["model_name"].eq("boundary_evidence_rag_full")].copy()
    if base.empty:
        return pd.DataFrame()
    base["correct"] = base["y_true"].map(norm_label).eq(base["y_pred"].map(norm_label))
    base["selective_score"] = pd.to_numeric(base["selective_score"], errors="coerce").fillna(0.0)
    base = base.sort_values("selective_score", ascending=False).reset_index(drop=True)
    summary_rows: List[Dict[str, Any]] = []
    for target in targets:
        best_n = 0
        best_acc = 0.0
        best_macro = 0.0
        best_weighted = 0.0
        best_threshold = 1.0
        for n in range(1, len(base) + 1):
            sub = base.head(n)
            metrics = recompute_metrics(sub.to_dict("records"))
            if metrics["accuracy"] + 1e-12 >= float(target):
                best_n = n
                best_acc = metrics["accuracy"]
                best_macro = metrics["macro_f1"]
                best_weighted = metrics["weighted_f1"]
                best_threshold = float(sub["selective_score"].min())
        suffix = str(int(round(float(target) * 1000))).rstrip("0")
        if math.isclose(float(target), 0.90):
            suffix = "90"
        subset = base.head(best_n).copy() if best_n else base.iloc[0:0].copy()
        subset["target_accuracy"] = float(target)
        subset["subset_name"] = f"high_conf_{suffix}_subset"
        subset.to_csv(out_dir / f"high_conf_subset_{suffix}.csv", index=False, encoding="utf-8-sig")
        metrics = recompute_metrics(subset.to_dict("records"))
        for i, gold in enumerate(LABELS):
            for j, pred in enumerate(LABELS):
                summary_rows.append(
                    {
                        "subset_name": f"high_conf_{suffix}_subset",
                        "target_accuracy": float(target),
                        "n": best_n,
                        "coverage": best_n / len(base) if len(base) else 0.0,
                        "accuracy": best_acc,
                        "macro_f1": best_macro,
                        "weighted_f1": best_weighted,
                        "threshold": best_threshold,
                        "gold_label": gold,
                        "pred_label": pred,
                        "count": metrics["confusion_matrix"][i][j] if best_n else 0,
                    }
                )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "target_accuracy_subsets.csv", index=False, encoding="utf-8-sig")
    return summary


def load_case_texts(test_df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    cases = {}
    for _, row in test_df.iterrows():
        case = build_case_text(row, cfg)
        cases[str(row["case_id"])] = case
    return cases


def build_case_frame(cfg: Dict[str, Any], anchors: pd.DataFrame, cases: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = PROJECT_ROOT / cfg["paths"]["precision_lift_run_dir"] / "predictions_main.csv"
    pred = pd.read_csv(pred_path, encoding="utf-8-sig")
    pred = pred[pred["dataset_name"].eq("candidate_gold_extended_v2")].copy()
    pred["y_true"] = pred["y_true"].map(norm_label)
    pred["y_pred"] = pred["y_pred"].map(norm_label)
    pred_w = pred.pivot(index="case_id", columns="model_name", values="y_pred")
    conf_w = pred.pivot(index="case_id", columns="model_name", values="confidence")
    base = pred[pred["model_name"].eq("optimized_weighted_ensemble")].drop_duplicates("case_id").set_index("case_id")
    qwen_records_path = PROJECT_ROOT / cfg["paths"]["candidate_v2_prediction_records"]
    qwen_features = pd.read_csv(qwen_records_path, encoding="utf-8-sig") if qwen_records_path.exists() else pd.DataFrame()
    if not qwen_features.empty:
        qwen_features = qwen_features.set_index("case_id")

    rows: List[Dict[str, Any]] = []
    retrieval_rows: List[Dict[str, Any]] = []
    for cid, row in base.iterrows():
        preds = [norm_label(pred_w.at[cid, m]) for m in AGREEMENT_MODELS if m in pred_w.columns and pd.notna(pred_w.at[cid, m])]
        counts = Counter([p for p in preds if p in LABELS])
        consensus_label, consensus_count = counts.most_common(1)[0] if counts else (norm_label(row["y_pred"]), 1)
        model_agreement = consensus_count / len(preds) if preds else 0.0
        text = str(cases.get(cid, {}).get("pre_decision_text", ""))
        rag = label_aware_retrieve(cid, compact_text(text, 4500), anchors, int(cfg["selective"].get("top_k_per_label", 5)))
        retrieval_rows.extend(rag["manifest_rows"])
        qf = qwen_features.loc[cid] if not qwen_features.empty and cid in qwen_features.index else {}
        evidence_coverage = np.mean(
            [
                safe_float(getattr(qf, "valid_span_rate", 0.0), 0.0),
                safe_float(getattr(qf, "pre_decision_span_rate", 0.0), 0.0),
                safe_float(getattr(qf, "role_coverage_rate", 0.0), 0.0),
            ]
        )
        stack_conf = safe_float(conf_w.at[cid, "stacked_meta_logreg"], 0.0) if "stacked_meta_logreg" in conf_w.columns and cid in conf_w.index else 0.0
        qwen_conf = safe_float(conf_w.at[cid, "qwen_self_consistency_3view"], 0.0) if "qwen_self_consistency_3view" in conf_w.columns and cid in conf_w.index else 0.0
        paesc_conf = safe_float(conf_w.at[cid, "paesc_llm_fusion_85"], 0.0) if "paesc_llm_fusion_85" in conf_w.columns and cid in conf_w.index else 0.0
        score = (
            0.55 * model_agreement
            + 0.25 * stack_conf
            + 0.15 * qwen_conf
            + 0.05 * paesc_conf
            + 0.05 * safe_float(rag["top_prob"], 0.0)
            + 0.05 * safe_float(evidence_coverage, 0.0)
        )
        base_pred = norm_label(row["y_pred"])
        switched = 0
        full_pred = base_pred
        if (
            consensus_label != base_pred
            and model_agreement >= float(cfg["selective"].get("consensus_switch_agreement", 0.50))
            and rag["top_label"] == consensus_label
            and safe_float(rag["margin"], 0.0) >= float(cfg["selective"].get("rag_switch_margin", 0.08))
        ):
            full_pred = consensus_label
            switched = 1
        rows.append(
            {
                "case_id": cid,
                "fold_id": int(row["fold_id"]),
                "y_true": norm_label(row["y_true"]),
                "optimized_pred": base_pred,
                "y_pred": full_pred,
                "confidence": min(1.0, max(0.0, score)),
                "selective_score": score,
                "model_agreement": model_agreement,
                "consensus_label": consensus_label,
                "consensus_switched_flag": switched,
                "rag_top_label": rag["top_label"],
                "rag_top_prob": rag["top_prob"],
                "rag_margin": rag["margin"],
                "evidence_coverage_score": evidence_coverage,
                "stacked_confidence": stack_conf,
                "qwen_vote_confidence": qwen_conf,
                "paesc_confidence": paesc_conf,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(retrieval_rows)


def arbitration_prompt(case: Dict[str, Any], case_row: Dict[str, Any], retrieved: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    examples = []
    for _, row in retrieved.head(9).iterrows():
        examples.append({"label": row.get("retrieved_label", ""), "similarity": row.get("similarity", 0.0), "case_id": row.get("retrieved_case_id", "")})
    payload = {
        "task": "hard_case_arbitration_pre_decision_only",
        "case_id": case["case_id"],
        "hard_constraints": [
            "Use only pre_decision_text.",
            "Do not infer from judgment outcome or court reasoning.",
            "Return one JSON object only.",
            "evidence_chain span_text must be copied from pre_decision_text when possible.",
        ],
        "candidate_model_signals": {
            "optimized_pred": case_row.get("optimized_pred"),
            "consensus_label": case_row.get("consensus_label"),
            "rag_top_label": case_row.get("rag_top_label"),
            "model_agreement": case_row.get("model_agreement"),
            "rag_margin": case_row.get("rag_margin"),
        },
        "retrieved_training_exemplars": examples,
        "required_json": {
            "outcome_label": "support|partial|not_support",
            "confidence": "float 0-1",
            "uncertainty_flag": "boolean",
            "conflict_reason": "short Chinese reason",
            "evidence_chain": [{"role_label": "ENT|NOT|CAU|IMP|DOC", "span_text": "exact excerpt", "reason": "short reason"}],
        },
        "pre_decision_text": compact_text(case["pre_decision_text"], int(cfg["llm"].get("max_chars_pre", 7000))),
    }
    prompt = json.dumps(payload, ensure_ascii=False)
    return [
        {"role": "system", "content": "你是工程工期延误纠纷的审计型预测助手。只能基于裁决前信息，严格输出 JSON。"},
        {"role": "user", "content": prompt},
    ], prompt


def call_with_retries(client: QwenClient, messages: List[Dict[str, str]], max_tokens: int, retries: int) -> Tuple[str, Dict[str, Any], int]:
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            content, usage = client.chat(messages, max_tokens=max_tokens)
            return content, usage, attempt + 1
        except Exception as exc:
            last = exc
            if attempt < retries:
                time.sleep(1 + attempt)
    raise RuntimeError(str(last))


def run_arbitration(
    cfg: Dict[str, Any],
    frame: pd.DataFrame,
    cases: Dict[str, Dict[str, Any]],
    retrieval_manifest: pd.DataFrame,
    out_dir: Path,
    skip_api: bool,
) -> pd.DataFrame:
    max_cases = int(cfg["llm"].get("max_api_hard_cases", 30))
    if skip_api or max_cases <= 0:
        return pd.DataFrame(columns=["case_id", "api_status", "outcome_label", "confidence", "uncertainty_flag", "conflict_reason"])
    hard = frame[frame["high_conf_flag"].eq(0)].sort_values(["selective_score", "model_agreement"], ascending=[True, True]).head(max_cases)
    if hard.empty:
        return pd.DataFrame(columns=["case_id", "api_status", "outcome_label", "confidence", "uncertainty_flag", "conflict_reason"])
    api_key, key_source = resolve_api_key(cfg)
    model_name = probe_model(cfg, api_key, out_dir)
    client = QwenClient(cfg, api_key, model_name)

    def one(row: Dict[str, Any]) -> Dict[str, Any]:
        cid = str(row["case_id"])
        started = time.time()
        try:
            messages, prompt = arbitration_prompt(cases[cid], row, retrieval_manifest[retrieval_manifest["case_id"].eq(cid)], cfg)
            content, usage, attempts = call_with_retries(client, messages, int(cfg["llm"].get("max_tokens_arbitration", 1200)), int(cfg["llm"].get("retries", 1)))
            parsed = extract_json_object(content)
            chain, audit = validate_evidence_chain(parsed.get("evidence_chain", []), cases[cid].get("pre_decision_text", ""))
            return {
                "case_id": cid,
                "api_status": "api_available",
                "model_name": model_name,
                "api_key_source": key_source,
                "prompt_sha256": sha256_text(prompt),
                "attempts": attempts,
                "latency_sec": round(time.time() - started, 4),
                "outcome_label": norm_label(parsed.get("outcome_label")),
                "confidence": safe_float(parsed.get("confidence"), 0.0),
                "uncertainty_flag": bool_int(parsed.get("uncertainty_flag")),
                "conflict_reason": str(parsed.get("conflict_reason", ""))[:500],
                "valid_span_rate": audit["valid_span_rate"],
                "pre_decision_span_rate": audit["pre_decision_span_rate"],
                "role_coverage_rate": audit["role_coverage_rate"],
                "evidence_chain_json": json.dumps(chain, ensure_ascii=False),
                "usage_json": json.dumps(usage, ensure_ascii=False),
                "raw_response": content,
            }
        except Exception as exc:
            return {"case_id": cid, "api_status": "api_error", "error": str(exc)[:1000], "latency_sec": round(time.time() - started, 4)}

    rows: List[Dict[str, Any]] = []
    workers = max(1, int(cfg["llm"].get("workers", 6)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, r) for r in hard.to_dict("records")]
        for fut in as_completed(futures):
            rows.append(fut.result())
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def build_prediction_rows(frame: pd.DataFrame, arbitration: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    arb = arbitration.set_index("case_id") if not arbitration.empty else pd.DataFrame()
    for _, row in frame.iterrows():
        common = row.to_dict()
        common["dataset_name"] = "candidate_gold_extended_v2"
        for model_name, pred_col in [
            ("optimized_weighted_ensemble_reference", "optimized_pred"),
            ("boundary_evidence_rag_full", "y_pred"),
        ]:
            rec = dict(common)
            rec["model_name"] = model_name
            rec["y_pred"] = row[pred_col]
            rows.append(rec)
        rec = dict(common)
        rec["model_name"] = "selective_high_conf_boundary_rag"
        rec["y_pred"] = row["y_pred"]
        rows.append(rec)
        arb_pred = row["y_pred"]
        arb_used = 0
        if not arb.empty and row["case_id"] in arb.index:
            ar = arb.loc[row["case_id"]]
            if str(ar.get("api_status", "")) == "api_available" and norm_label(ar.get("outcome_label")) in LABELS and safe_float(ar.get("confidence"), 0.0) >= 0.72:
                arb_pred = norm_label(ar.get("outcome_label"))
                arb_used = 1
        rec = dict(common)
        rec["model_name"] = "selective_qwen_arbitrated_30"
        rec["y_pred"] = arb_pred
        rec["qwen_arbitration_used"] = arb_used
        rows.append(rec)
    return pd.DataFrame(rows)


def write_eval_artifacts(predictions: pd.DataFrame, out_dir: Path) -> None:
    metric_rows, per_rows, cm_rows = [], [], []
    metrics_json: Dict[str, Any] = {}
    scopes = {
        "full500": lambda df: df,
        "high_conf_subset": lambda df: df[df["high_conf_flag"].eq(1)],
        "hard_cases": lambda df: df[df["high_conf_flag"].eq(0)],
    }
    for model_name, model_df in predictions.groupby("model_name"):
        for scope, fn in scopes.items():
            sub = fn(model_df)
            m = recompute_metrics(sub.to_dict("records"))
            coverage = len(sub) / len(model_df) if len(model_df) else 0.0
            metric_rows.append(
                {
                    "scope": scope,
                    "model_name": model_name,
                    "n": m["n_eval_rows"],
                    "coverage": coverage,
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "weighted_f1": m["weighted_f1"],
                }
            )
            metrics_json[f"{scope}::{model_name}"] = {"coverage": coverage, **m}
            for lab, vals in m["per_class"].items():
                per_rows.append({"scope": scope, "model_name": model_name, "class_label": lab, **vals})
            for i, gold in enumerate(LABELS):
                for j, pred in enumerate(LABELS):
                    cm_rows.append({"scope": scope, "model_name": model_name, "gold_label": gold, "pred_label": pred, "count": m["confusion_matrix"][i][j]})
    pd.DataFrame(metric_rows).sort_values(["scope", "accuracy"], ascending=[True, False]).to_csv(out_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_rows).to_csv(out_dir / "per_class_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cm_rows).to_csv(out_dir / "confusion_matrix_data.csv", index=False, encoding="utf-8-sig")
    (out_dir / "metrics_main.json").write_text(json.dumps(metrics_json, ensure_ascii=False, indent=2), encoding="utf-8")


def write_error_analysis(predictions: pd.DataFrame, out_dir: Path) -> None:
    main = predictions[predictions["model_name"].eq("boundary_evidence_rag_full")].copy()
    rows = []
    for _, row in main.iterrows():
        if row["y_true"] == row["y_pred"]:
            category = "correct"
        elif row["y_true"] == "support":
            category = "support_under_recognition"
        elif row["y_true"] == "not_support" and row["y_pred"] == "partial":
            category = "not_support_partial_boundary"
        elif row["y_true"] == "partial":
            category = "partial_boundary_confusion"
        else:
            category = "other_label_confusion"
        rows.append(
            {
                "case_id": row["case_id"],
                "gold_label": row["y_true"],
                "pred_label": row["y_pred"],
                "high_conf_flag": row["high_conf_flag"],
                "selective_score": row["selective_score"],
                "error_category": category,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "error_analysis.csv", index=False, encoding="utf-8-sig")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/research_selective_precision_boost.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--max-api-hard-cases", type=int, default=None)
    parser.add_argument("--min-coverage", type=float, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(PROJECT_ROOT / args.config)
    if args.max_api_hard_cases is not None:
        cfg["llm"]["max_api_hard_cases"] = int(args.max_api_hard_cases)
    if args.min_coverage is not None:
        cfg["selective"]["min_coverage"] = float(args.min_coverage)
    prefix = cfg["run"].get("run_name_prefix", "selective_precision_boost")
    out_dir = PROJECT_ROOT / (args.out_dir or f"results/{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    test_path = PROJECT_ROOT / cfg["paths"]["test_label_file"]
    train_path = PROJECT_ROOT / cfg["paths"]["train_label_records"]
    test_df = pd.read_csv(test_path, encoding="utf-8-sig")
    test_ids = set(test_df["case_id"].astype(str))
    train_records = pd.read_csv(train_path, encoding="utf-8-sig")
    anchors = build_clean_anchor_bank(train_records, test_ids, float(cfg["selective"].get("anchor_min_confidence", 0.80)))
    if anchors.empty:
        raise RuntimeError("clean_anchor_bank is empty; cannot run selective RAG")
    anchors.to_csv(out_dir / "clean_anchor_bank.csv", index=False, encoding="utf-8-sig")

    cases = load_case_texts(test_df, cfg)
    frame, retrieval = build_case_frame(cfg, anchors, cases)
    frame, thresholds = crossfit_selective_flags(frame, float(cfg["selective"].get("min_coverage", 0.40)))
    thresholds.to_csv(out_dir / "selective_thresholds.csv", index=False, encoding="utf-8-sig")
    retrieval.to_csv(out_dir / "rag_retrieval_manifest.csv", index=False, encoding="utf-8-sig")
    coverage_curve(frame.assign(model_name="boundary_evidence_rag_full"), "boundary_evidence_rag_full").to_csv(out_dir / "selective_coverage_curve.csv", index=False, encoding="utf-8-sig")

    arbitration = run_arbitration(cfg, frame, cases, retrieval, out_dir, args.skip_api)
    arbitration.to_csv(out_dir / "hard_case_arbitration.csv", index=False, encoding="utf-8-sig")

    predictions = build_prediction_rows(frame, arbitration)
    predictions.to_csv(out_dir / "predictions_main.csv", index=False, encoding="utf-8-sig")
    export_target_accuracy_subsets(predictions, out_dir)
    write_eval_artifacts(predictions, out_dir)
    write_error_analysis(predictions, out_dir)

    artifact_paths = [
        PROJECT_ROOT / args.config,
        test_path,
        train_path,
        PROJECT_ROOT / cfg["paths"]["precision_lift_run_dir"] / "predictions_main.csv",
        out_dir / "predictions_main.csv",
        out_dir / "hard_case_arbitration.csv",
    ]
    manifest = {
        "run_name": out_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "text_mode": "pre_decision_only",
        "label_schema_version": "candidate_gold_extended_v2",
        "prompt_template_version": cfg["llm"].get("prompt_template_version"),
        "max_api_hard_cases": cfg["llm"].get("max_api_hard_cases"),
        "api_skipped": bool(args.skip_api),
        "artifact_hashes": {str(p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p): sha256_file(p) for p in artifact_paths},
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    comp = pd.read_csv(out_dir / "model_comparison.csv", encoding="utf-8-sig")
    print(comp.sort_values(["scope", "accuracy"], ascending=[True, False]).to_string(index=False))
    print(f"SELECTIVE_PRECISION_BOOST_OUT_DIR={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
