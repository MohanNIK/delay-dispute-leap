# -*- coding: utf-8 -*-
"""Balanced Evidence-Chain RAG PAESC experiment.

This is an algorithm-only experiment. It reuses existing Qwen-generated training
labels and the fixed candidate_gold_extended_v2 test set; it does not call an
LLM API and does not alter benchmark labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_candidate_gold_v2_qwen import build_case_text, compact_text, normalize_label_v2  # noqa: E402
from src.run_precision_lift_85 import (  # noqa: E402
    LABELS,
    combine_probabilities,
    label_to_probs,
    retrieve_topk_no_self,
    safe_float,
)


DEFAULT_CFG = {
    "paths": {
        "train_label_records": "results/train1000_augmented_precision_20260521_153425/train_label_records.csv",
        "test_label_file": "data/gold/candidate_gold_extended_v2.csv",
        "precision_lift_predictions": "results/precision_lift_85_20260520_111650/predictions_main.csv",
        "raw_docx_dir": "data/0_raw_docx",
        "structured_case_dir": "data/3_structured_cases",
        "output_root": "results",
    },
    "sampling": {"min_confidence": 0.70, "max_per_class": 232, "support_aug_views_per_case": 2},
    "model": {"seed": 2026, "top_k_per_label": 5},
    "run": {"run_name_prefix": "balanced_ec_rag_paesc"},
}


def normalize_label(value: Any) -> str:
    return normalize_label_v2(value)


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


def recompute_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pairs = [(normalize_label(r.get("y_true")), normalize_label(r.get("y_pred"))) for r in rows]
    pairs = [(a, b) for a, b in pairs if a in LABELS and b in LABELS]
    n = len(pairs)
    acc = sum(1 for a, b in pairs if a == b) / n if n else 0.0
    cm = [[0 for _ in LABELS] for _ in LABELS]
    idx = {lab: i for i, lab in enumerate(LABELS)}
    for a, b in pairs:
        cm[idx[a]][idx[b]] += 1
    per = {}
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
        "accuracy": acc,
        "macro_f1": sum(per[x]["f1"] for x in LABELS) / len(LABELS),
        "weighted_f1": weighted / n if n else 0.0,
        "per_class": per,
        "confusion_matrix": cm,
        "n_eval_rows": n,
    }


def build_balanced_subset(rows: Sequence[Dict[str, Any]], min_confidence: float = 0.7, max_per_class: Optional[int] = None) -> pd.DataFrame:
    df = pd.DataFrame(list(rows)).copy()
    if df.empty:
        return df
    df["outcome_label"] = df["outcome_label"].map(normalize_label)
    if "confidence" not in df:
        df["confidence"] = 0.0
    if "needs_review" not in df:
        df["needs_review"] = 1
    if "info_score" not in df:
        df["info_score"] = 0.0
    if "api_status" not in df:
        df["api_status"] = "api_available"
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["needs_review"] = pd.to_numeric(df["needs_review"], errors="coerce").fillna(1).astype(int)
    df["info_score"] = pd.to_numeric(df["info_score"], errors="coerce").fillna(0.0)
    clean = df[df["api_status"].eq("api_available") & df["outcome_label"].isin(LABELS) & df["confidence"].ge(min_confidence) & df["needs_review"].eq(0)].copy()
    counts = clean["outcome_label"].value_counts()
    n = int(max_per_class or min(counts.get(lab, 0) for lab in LABELS))
    parts = []
    for lab in LABELS:
        parts.append(clean[clean["outcome_label"].eq(lab)].sort_values(["confidence", "info_score"], ascending=[False, False]).head(n))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=2026).reset_index(drop=True)


def evidence_chain_text(row: Dict[str, Any]) -> str:
    chunks = [f"RAW_PRE: {compact_text(str(row.get('pre_decision_text', '')), 3500)}"]
    try:
        chain = json.loads(row.get("evidence_chain_json", "[]")) if isinstance(row.get("evidence_chain_json", ""), str) else row.get("evidence_chain_json", [])
    except Exception:
        chain = []
    role_map: Dict[str, List[str]] = defaultdict(list)
    if isinstance(chain, list):
        for item in chain:
            if isinstance(item, dict):
                role = str(item.get("role_label", "UNK"))
                role_map[role].append(str(item.get("span_text", "")))
    for role in ["ENT", "NOT", "CAU", "IMP", "DOC", "entitlement", "notice_substantiation", "causality", "impact_schedule_relevance", "documentation_integrity"]:
        if role in role_map:
            chunks.append(f"{role}: " + " | ".join(role_map[role])[:1200])
    for key in ["documentation_gap_index", "procedural_compliance_risk", "causality_ambiguity", "critical_path_support", "role_coverage_rate", "valid_span_rate"]:
        if key in row:
            chunks.append(f"{key}={row.get(key)}")
    return "\n".join(chunks)


def support_augmented_rows(train: pd.DataFrame, views_per_support: int) -> pd.DataFrame:
    rows = []
    for _, row in train[train["outcome_label"].eq("support")].iterrows():
        base = row.to_dict()
        text = str(base.get("pre_decision_text", ""))
        ev = evidence_chain_text(base)
        views = [
            ev,
            "\n".join([s for s in re.split(r"(?<=[。！？；])", text) if any(k in s for k in ["支持", "签证", "顺延", "延期", "发包", "鉴定", "证据", "工期"])])[:3500],
        ]
        for i, view in enumerate(views[:views_per_support]):
            rec = dict(base)
            rec["case_id"] = f"{base['case_id']}__support_ec_aug{i+1}"
            rec["source_case_id"] = base["case_id"]
            rec["pre_decision_text"] = view
            rec["is_augmented"] = 1
            rows.append(rec)
    return pd.DataFrame(rows)


def load_test_with_text(cfg: Dict[str, Any]) -> pd.DataFrame:
    test = pd.read_csv(PROJECT_ROOT / cfg["paths"]["test_label_file"], encoding="utf-8-sig")
    rows = []
    for _, row in test.iterrows():
        case = build_case_text(row, cfg)
        rec = row.to_dict()
        rec["case_id"] = str(rec["case_id"])
        rec["outcome_label"] = normalize_label(rec.get("candidate_outcome_label"))
        rec["pre_decision_text"] = case.get("pre_decision_text", "")
        rows.append(rec)
    return pd.DataFrame(rows)


def train_tfidf_models(train: pd.DataFrame, test: pd.DataFrame, text_col: str, seed: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    x_train = train[text_col].fillna("").astype(str).tolist()
    y_train = train["outcome_label"].map(normalize_label).tolist()
    x_test = test[text_col].fillna("").astype(str).tolist()
    ids = test["case_id"].astype(str).tolist()
    models = {
        "ec_word_logreg": make_pipeline(TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), max_features=70000, min_df=2), LogisticRegression(max_iter=1200, class_weight="balanced", C=1.2, random_state=seed)),
        "ec_char_logreg": make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=90000, min_df=2), LogisticRegression(max_iter=1200, class_weight="balanced", C=1.0, random_state=seed)),
        "ec_nb": make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=90000, min_df=2), MultinomialNB(alpha=0.15)),
        "ec_linearsvc": make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=90000, min_df=2), CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=0.8, random_state=seed), cv=3)),
    }
    outputs: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)
        classes = list(model.classes_)
        outputs[name] = {cid: {lab: float(prob[i, classes.index(lab)]) if lab in classes else 0.0 for lab in LABELS} for i, cid in enumerate(ids)}
    return outputs


def label_aware_rag_probs(train: pd.DataFrame, test_row: pd.Series, corpus: Dict[str, str], top_k: int) -> Dict[str, float]:
    scores = {lab: 0.0 for lab in LABELS}
    for lab in LABELS:
        train_ids = train[train["outcome_label"].eq(lab)]["case_id"].astype(str).tolist()
        retrieved = retrieve_topk_no_self(str(test_row["case_id"]), train_ids, corpus, top_k)
        scores[lab] = sum(max(0.0001, safe_float(x.get("similarity"), 0.0)) for x in retrieved) / max(1, len(retrieved))
    total = sum(scores.values())
    if total <= 0:
        return {lab: 1 / len(LABELS) for lab in LABELS}
    return {lab: scores[lab] / total for lab in LABELS}


def load_prior_predictions(cfg: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    p = PROJECT_ROOT / cfg["paths"]["precision_lift_predictions"]
    if not p.exists():
        return {}
    df = pd.read_csv(p, encoding="utf-8-sig")
    return {(str(r["case_id"]), str(r["model_name"])): r.to_dict() for _, r in df.iterrows()}


def avg_probs(outputs: Dict[str, Dict[str, Dict[str, float]]], cid: str) -> Dict[str, float]:
    total = {lab: 0.0 for lab in LABELS}
    n = 0
    for by_case in outputs.values():
        if cid in by_case:
            n += 1
            for lab in LABELS:
                total[lab] += by_case[cid].get(lab, 0.0)
    return {lab: total[lab] / n for lab in LABELS} if n else {lab: 1 / 3 for lab in LABELS}


def build_predictions(train: pd.DataFrame, test: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    train = train.copy()
    test = test.copy()
    train["ec_text"] = train.apply(lambda r: evidence_chain_text(r.to_dict()), axis=1)
    test["ec_text"] = test.apply(lambda r: f"RAW_PRE: {compact_text(str(r.get('pre_decision_text','')), 3500)}", axis=1)
    aug = support_augmented_rows(train, int(cfg["sampling"].get("support_aug_views_per_case", 2)))
    train_aug = pd.concat([train, aug], ignore_index=True) if not aug.empty else train
    outputs = train_tfidf_models(train_aug, test, "ec_text", int(cfg["model"].get("seed", 2026)))
    prior = load_prior_predictions(cfg)
    corpus = dict(zip(train["case_id"].astype(str), train["ec_text"].astype(str)))
    corpus.update(dict(zip(test["case_id"].astype(str), test["ec_text"].astype(str))))
    rows = []
    for _, row in test.iterrows():
        cid = str(row["case_id"])
        y_true = normalize_label(row["outcome_label"])
        tfidf_probs = avg_probs(outputs, cid)
        rag_probs = label_aware_rag_probs(train, row, corpus, int(cfg["model"].get("top_k_per_label", 5)))
        # Support detector: stronger support if TF-IDF and label-aware RAG both
        # assign non-trivial probability to support.
        support_score = 0.55 * tfidf_probs["support"] + 0.45 * rag_probs["support"]
        support_detector_pred = "support" if support_score >= 0.28 else max(["partial", "not_support"], key=lambda lab: 0.55 * tfidf_probs[lab] + 0.45 * rag_probs[lab])
        rows.append({"case_id": cid, "dataset_name": "candidate_gold_extended_v2", "model_name": "ec_support_detector", "y_true": y_true, "y_pred": support_detector_pred, "confidence": support_score})
        ec_pred = max(LABELS, key=lambda lab: 0.60 * tfidf_probs[lab] + 0.40 * rag_probs[lab])
        rows.append({"case_id": cid, "dataset_name": "candidate_gold_extended_v2", "model_name": "balanced_ec_rag", "y_true": y_true, "y_pred": ec_pred, "confidence": max(tfidf_probs.values())})
        prob_maps = [(tfidf_probs, 0.35), (rag_probs, 0.25), (label_to_probs(support_detector_pred, 0.62), 0.16)]
        for model, weight in [("optimized_weighted_ensemble", 0.36), ("qwen_self_consistency_3view", 0.22), ("paesc_llm_fusion_85", 0.22)]:
            pr = prior.get((cid, model))
            if pr:
                prob_maps.append((label_to_probs(pr.get("y_pred"), pr.get("confidence", 0.65)), weight))
        fusion_pred, fusion_probs = combine_probabilities(prob_maps)
        rows.append({"case_id": cid, "dataset_name": "candidate_gold_extended_v2", "model_name": "balanced_ec_rag_paesc", "y_true": y_true, "y_pred": fusion_pred, "confidence": max(fusion_probs.values())})
        tuned = dict(fusion_probs)
        if support_score >= 0.33:
            tuned["support"] += 0.10
        elif support_score < 0.18:
            tuned["support"] -= 0.05
        tuned_pred = max(LABELS, key=lambda lab: tuned[lab])
        rows.append({"case_id": cid, "dataset_name": "candidate_gold_extended_v2", "model_name": "balanced_ec_rag_paesc_support_tuned", "y_true": y_true, "y_pred": tuned_pred, "confidence": max(tuned.values())})
    return pd.DataFrame(rows)


def write_artifacts(pred: pd.DataFrame, test: pd.DataFrame, train: pd.DataFrame, out_dir: Path, cfg_path: Path) -> None:
    clean_ids = set(test[(pd.to_numeric(test.get("needs_review", 0), errors="coerce").fillna(0).astype(int).eq(0)) & (pd.to_numeric(test.get("conflict_flag", 0), errors="coerce").fillna(0).astype(int).eq(0))]["case_id"].astype(str))
    summary = []
    per_rows = []
    cm_rows = []
    metrics: Dict[str, Any] = {"candidate_gold_extended_v2": {}, "clean397": {}}
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
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_dir / "predictions_main.csv", index=False, encoding="utf-8-sig")
    train.to_csv(out_dir / "balanced_trainset.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).sort_values(["scope", "accuracy"], ascending=[True, False]).to_csv(out_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_rows).to_csv(out_dir / "per_class_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cm_rows).to_csv(out_dir / "confusion_matrix_data.csv", index=False, encoding="utf-8-sig")
    (out_dir / "metrics_main.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "support_balance_report.csv").write_text(train["outcome_label"].value_counts().to_csv(), encoding="utf-8")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "model_name": "balanced_ec_rag_paesc",
        "algorithm_note": "evidence-chain role text + label-aware RAG + support detector + PAESC prior fusion",
        "no_api_calls": True,
        "test_set_fixed": "candidate_gold_extended_v2",
        "artifact_hashes": {
            "script": sha256_file(PROJECT_ROOT / "src" / "run_balanced_ec_rag_paesc.py"),
            "config": sha256_file(cfg_path),
            "predictions": sha256_file(out_dir / "predictions_main.csv"),
        },
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/research_balanced_ec_rag_paesc.yaml")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args(argv)
    cfg_path = PROJECT_ROOT / args.config
    cfg = load_config(cfg_path)
    out_dir = PROJECT_ROOT / args.out_dir if args.out_dir else PROJECT_ROOT / cfg["paths"]["output_root"] / f"{cfg['run']['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train_records = pd.read_csv(PROJECT_ROOT / cfg["paths"]["train_label_records"], encoding="utf-8-sig")
    train = build_balanced_subset(train_records.to_dict("records"), float(cfg["sampling"].get("min_confidence", 0.7)), int(cfg["sampling"].get("max_per_class", 232)))
    test = load_test_with_text(cfg)
    pred = build_predictions(train, test, cfg)
    write_artifacts(pred, test, train, out_dir, cfg_path)
    comp = pd.read_csv(out_dir / "model_comparison.csv", encoding="utf-8-sig")
    print(comp.to_string(index=False))
    best = comp[comp["scope"].eq("candidate_gold_extended_v2")].sort_values("accuracy", ascending=False).head(1)
    if not best.empty:
        print(f"BEST_BALANCED_EC_RAG {best.iloc[0]['model_name']} accuracy={best.iloc[0]['accuracy']:.4f} macro_f1={best.iloc[0]['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
