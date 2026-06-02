# -*- coding: utf-8 -*-
"""Unified evaluation pipeline for the IEEE-TEM-oriented DelayDispute Copilot study."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_support import (  # noqa: E402
    DEFAULT_CFG,
    LABELS,
    RESP_LABELS,
    bootstrap_ci,
    classify_metrics,
    diagnose_responsibility_from_pre,
    evidence_chain_metrics,
    evidence_sufficiency_score,
    json_dump,
    latest_run_dir,
    load_cfg,
    maybe_kappa,
    normalize_label,
    normalize_resp,
    read_csv_flexible,
    reasoning_trace,
    rule_baseline_prediction,
    structured_numeric_features,
    structured_signature_tokens,
)
from src.audit_utils import (  # noqa: E402
    build_run_manifest,
    previous_final_eval_run,
    synthetic_file_diff,
    write_manifest,
    write_synthetic_git_summary,
)


NONCOMPLIANCE_HINTS = ["未通知", "未报审", "未备案", "未签证", "未验收", "未履行"]
OWNER_HINTS = ["发包人", "业主", "甲方", "建设单位"]
CONTRACTOR_HINTS = ["承包人", "承包商", "施工单位", "乙方"]


@dataclass
class ModelArtifacts:
    vectorizer: TfidfVectorizer
    scaler: StandardScaler
    text_models: Dict[str, object]
    hybrid_model: LogisticRegression
    train_vectors: object
    train_text_vectors: object
    train_labels: np.ndarray
    train_case_ids: List[str]
    train_signature_tokens: List[set]


def load_structured_cases(structured_dir: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for fp in structured_dir.glob("*.json"):
        obj = json.loads(fp.read_text(encoding="utf-8"))
        out[str(obj.get("case_id"))] = obj
    return out


def build_weak_training_frame(cfg: Dict, structured_cases: Dict[str, Dict]) -> pd.DataFrame:
    labels_path = PROJECT_ROOT / "data" / "meta" / "labels_step2_domain.csv"
    if not labels_path.exists():
        labels_path = PROJECT_ROOT / cfg["paths"]["meta_labels_csv"]
    labels = read_csv_flexible(labels_path)
    labels["case_id"] = labels.get("case_id", pd.Series(dtype=str)).astype(str)
    rows = []
    for _, row in labels.iterrows():
        cid = row["case_id"]
        structured = structured_cases.get(cid)
        if not structured:
            continue
        label = normalize_label(row.get("eot_label", "unknown"))
        if label not in LABELS or int(structured.get("is_domain_case", 0)) != 1:
            continue
        pre_text = structured.get("pre_decision_text", "")
        if not str(pre_text).strip():
            continue
        features = structured_numeric_features(structured)
        rows.append({
            "case_id": cid,
            "source_file": structured.get("source_file", ""),
            "label": label,
            "pre_text": pre_text,
            "case_year": structured.get("case_year"),
            **features,
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["case_id"])
    return df


def build_candidate_eval_frame(path: Path, structured_cases: Dict[str, Dict], llm_df: pd.DataFrame, weak_df: pd.DataFrame) -> pd.DataFrame:
    cand = read_csv_flexible(path)
    cand["case_id"] = cand.get("case_id", pd.Series(dtype=str)).astype(str)
    llm_df = llm_df.copy()
    llm_df["case_id"] = llm_df.get("case_id", pd.Series(dtype=str)).astype(str)
    weak_map = weak_df.set_index("case_id")["label"].to_dict() if not weak_df.empty else {}
    llm_map = llm_df.set_index("case_id").to_dict("index") if not llm_df.empty else {}

    rows = []
    for _, row in cand.iterrows():
        cid = row["case_id"]
        structured = structured_cases.get(cid)
        if not structured:
            continue
        pre_text = structured.get("pre_decision_text", "")
        if not str(pre_text).strip():
            continue
        features = structured_numeric_features(structured)
        llm_row = llm_map.get(cid, {})
        rows.append({
            "case_id": cid,
            "dataset_name": row.get("dataset_name", path.stem),
            "y_true": normalize_label(row.get("candidate_outcome_label", "unknown")),
            "candidate_responsibility_label": normalize_resp(row.get("candidate_responsibility_label", "unknown")),
            "source_file": structured.get("source_file", ""),
            "pre_text": pre_text,
            "post_text": structured.get("post_decision_text", ""),
            "evidence_span": row.get("evidence_span", ""),
            "generation_source": row.get("generation_source", ""),
            "candidate_confidence": float(row.get("confidence", 0.0)),
            "conflict_flag": int(row.get("conflict_flag", 0)),
            "needs_review": int(row.get("needs_review", 0)),
            "case_year": structured.get("case_year"),
            "weak_label": weak_map.get(cid, "unknown"),
            "llm_label": normalize_label(llm_row.get("delay_money_label", "unknown")),
            "llm_resp_hint": normalize_resp(llm_row.get("responsibility_hint", "unknown")),
            **features,
        })
    return pd.DataFrame(rows)


def fit_models(train_df: pd.DataFrame, cfg: Dict, structured_cases: Dict[str, Dict]) -> ModelArtifacts:
    eval_cfg = cfg["eval"]
    vectorizer = TfidfVectorizer(
        max_features=int(eval_cfg.get("max_features", 50000)),
        ngram_range=(int(eval_cfg.get("ngram_min", 1)), int(eval_cfg.get("ngram_max", 2))),
        min_df=2,
    )
    X_text = vectorizer.fit_transform(train_df["pre_text"].astype(str).tolist())
    num_cols = [
        "delay_event_count",
        "procedure_cue_count",
        "evidence_mention_count",
        "claim_count",
        "defense_count",
        "pre_text_length",
        "role_coverage_rate",
        "missing_role_rate",
        "evidence_sufficiency",
        "potential_leakage_flag",
    ]
    scaler = StandardScaler()
    X_num = scaler.fit_transform(train_df[num_cols].fillna(0.0))
    X_hybrid = hstack([X_text, csr_matrix(X_num)])
    y = train_df["label"].to_numpy()

    logreg = LogisticRegression(max_iter=1200, class_weight="balanced")
    logreg.fit(X_text, y)

    linearsvc = LinearSVC(class_weight="balanced")
    linearsvc.fit(X_text, y)

    mnb = MultinomialNB()
    mnb.fit(X_text, y)

    hybrid_model = LogisticRegression(max_iter=1200, class_weight="balanced")
    hybrid_model.fit(X_hybrid, y)

    signature_tokens = [set(structured_signature_tokens(structured_cases[cid])) for cid in train_df["case_id"].tolist()]

    return ModelArtifacts(
        vectorizer=vectorizer,
        scaler=scaler,
        text_models={
            "tfidf_logreg": logreg,
            "tfidf_linearsvc": linearsvc,
            "tfidf_multinomialnb": mnb,
        },
        hybrid_model=hybrid_model,
        train_vectors=X_hybrid,
        train_text_vectors=X_text,
        train_labels=y,
        train_case_ids=train_df["case_id"].tolist(),
        train_signature_tokens=signature_tokens,
    )


def _linearsvc_probs(model: LinearSVC, X) -> np.ndarray:
    scores = model.decision_function(X)
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    return probs / probs.sum(axis=1, keepdims=True)


def _probs_to_dict(probs_row: np.ndarray) -> Dict[str, float]:
    return {f"{lb}_prob": float(round(probs_row[i], 6)) for i, lb in enumerate(LABELS)}


def build_eval_matrix(df: pd.DataFrame, artifacts: ModelArtifacts):
    text_mat = artifacts.vectorizer.transform(df["pre_text"].astype(str).tolist())
    num_cols = [
        "delay_event_count",
        "procedure_cue_count",
        "evidence_mention_count",
        "claim_count",
        "defense_count",
        "pre_text_length",
        "role_coverage_rate",
        "missing_role_rate",
        "evidence_sufficiency",
        "potential_leakage_flag",
    ]
    num_mat = artifacts.scaler.transform(df[num_cols].fillna(0.0))
    hybrid_mat = hstack([text_mat, csr_matrix(num_mat)])
    return text_mat, hybrid_mat


def majority_prediction(train_df: pd.DataFrame, n: int) -> Tuple[List[str], np.ndarray]:
    majority = train_df["label"].value_counts().idxmax()
    probs = np.zeros((n, len(LABELS)), dtype=float)
    probs[:, LABELS.index(majority)] = 1.0
    return [majority] * n, probs


def current_hybrid_case_prediction(row: pd.Series, structured_case: Dict) -> Tuple[str, np.ndarray]:
    pred = normalize_label(row.get("weak_label", "unknown"))
    if pred not in LABELS:
        pred = normalize_label(row.get("llm_label", "unknown"))
    if pred not in LABELS:
        pred = rule_baseline_prediction(structured_case)
    probs = np.zeros((len(LABELS),), dtype=float)
    probs[LABELS.index(pred)] = 1.0
    return pred, probs


def retrieval_distribution(structured_case: Dict, artifacts: ModelArtifacts, train_df: pd.DataFrame, case_vec, top_k: int) -> Tuple[np.ndarray, List[str]]:
    cos = cosine_similarity(case_vec, artifacts.train_text_vectors)[0]
    sig = set(structured_signature_tokens(structured_case))
    scores = []
    for i, cid in enumerate(artifacts.train_case_ids):
        tr_sig = artifacts.train_signature_tokens[i]
        jacc = 0.0
        if sig or tr_sig:
            jacc = len(sig & tr_sig) / max(1, len(sig | tr_sig))
        scores.append((0.75 * float(cos[i]) + 0.25 * float(jacc), cid, artifacts.train_labels[i]))
    top = sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
    dist = np.zeros((len(LABELS),), dtype=float)
    for _, _, label in top:
        dist[LABELS.index(label)] += 1.0
    if dist.sum() == 0:
        dist[:] = 1.0 / len(LABELS)
    else:
        dist = dist / dist.sum()
    return dist, [cid for _, cid, _ in top]


def calibrate_hybrid_probs(
    base_probs: np.ndarray,
    retrieval_probs: np.ndarray,
    rule_label: str,
    structured_case: Dict,
    resp_diag: Dict[str, object],
    prior_label: str = "unknown",
    llm_label: str = "unknown",
    prior_weight: float = 0.0,
) -> np.ndarray:
    probs = 0.50 * base_probs + 0.20 * retrieval_probs
    rule_prior = np.zeros((len(LABELS),), dtype=float)
    rule_prior[LABELS.index(rule_label)] = 1.0
    probs += 0.10 * rule_prior

    if prior_label in LABELS and prior_weight > 0.0:
        prior_probs = np.zeros((len(LABELS),), dtype=float)
        prior_probs[LABELS.index(prior_label)] = 1.0
        probs += prior_weight * prior_probs

    if llm_label in LABELS:
        llm_prior = np.zeros((len(LABELS),), dtype=float)
        llm_prior[LABELS.index(llm_label)] = 1.0
        probs += 0.08 * llm_prior

    evidence_score = float(structured_case.get("evidence_sufficiency", evidence_sufficiency_score(structured_case)))
    pre_text = structured_case.get("pre_decision_text", "")
    if any(x in pre_text for x in NONCOMPLIANCE_HINTS) or resp_diag.get("documentation_integrity_flag") == "incomplete":
        shift = min(0.12, probs[LABELS.index("support")] * 0.35 + 0.04)
        probs[LABELS.index("support")] = max(0.0, probs[LABELS.index("support")] - shift)
        probs[LABELS.index("not_support")] += shift
    if resp_diag.get("primary_responsible_party") == "owner" and evidence_score >= 0.58:
        shift = min(0.08, probs[LABELS.index("not_support")] * 0.30 + 0.03)
        probs[LABELS.index("not_support")] = max(0.0, probs[LABELS.index("not_support")] - shift)
        probs[LABELS.index("support")] += shift
    if evidence_score < 0.38:
        caution = np.array([0.15, 0.45, 0.40])
        probs = 0.80 * probs + 0.20 * caution

    probs = np.clip(probs, 1e-8, None)
    probs = probs / probs.sum()
    return probs


def evidence_consistency(resp_diag: Dict[str, object], structured_case: Dict, chain_metrics: Dict[str, float]) -> int:
    primary = resp_diag.get("primary_responsible_party", "unknown")
    pre_text = structured_case.get("pre_decision_text", "")
    if primary == "unknown":
        return int(resp_diag.get("uncertainty_flag", 1) == 1)
    if primary == "owner":
        cue = any(x in pre_text for x in OWNER_HINTS)
    elif primary == "contractor":
        cue = any(x in pre_text for x in CONTRACTOR_HINTS)
    elif primary == "both":
        cue = "双方" in pre_text or "各方" in pre_text
    elif primary == "force_majeure_policy":
        cue = any(x in pre_text for x in ["不可抗力", "政策", "政府行为", "疫情", "环保"])
    elif primary == "designer_supervisor":
        cue = any(x in pre_text for x in ["监理", "设计单位", "设计变更"])
    elif primary == "subcontractor":
        cue = "分包" in pre_text
    else:
        cue = False
    return int(cue and chain_metrics["role_coverage_rate"] >= 0.4)


def evaluate_frame(eval_df: pd.DataFrame, train_df: pd.DataFrame, artifacts: ModelArtifacts, structured_cases: Dict[str, Dict], cfg: Dict, split_name: str, dataset_name: str, include_current_hybrid: bool = True) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame, pd.DataFrame, List[Dict[str, object]]]:
    text_mat, hybrid_mat = build_eval_matrix(eval_df, artifacts)
    records: List[Dict[str, object]] = []
    resp_rows: List[Dict[str, object]] = []
    chain_rows: List[Dict[str, object]] = []
    traces: List[Dict[str, object]] = []
    eval_cfg = cfg["eval"]

    # majority
    maj_pred, maj_probs = majority_prediction(train_df, len(eval_df))
    model_outputs = {"majority_class": (maj_pred, maj_probs)}

    # rule baseline
    rule_preds = []
    rule_probs = []
    for _, row in eval_df.iterrows():
        structured = structured_cases[row["case_id"]]
        pred = rule_baseline_prediction(structured)
        probs = np.zeros((len(LABELS),), dtype=float)
        probs[LABELS.index(pred)] = 1.0
        rule_preds.append(pred)
        rule_probs.append(probs)
    model_outputs["rule_baseline"] = (rule_preds, np.vstack(rule_probs))

    # traditional text models
    logreg_probs = artifacts.text_models["tfidf_logreg"].predict_proba(text_mat)
    logreg_preds = [LABELS[int(i)] for i in np.argmax(logreg_probs, axis=1)]
    model_outputs["tfidf_logreg"] = (logreg_preds, logreg_probs)

    svc_probs = _linearsvc_probs(artifacts.text_models["tfidf_linearsvc"], text_mat)
    svc_preds = [LABELS[int(i)] for i in np.argmax(svc_probs, axis=1)]
    model_outputs["tfidf_linearsvc"] = (svc_preds, svc_probs)

    nb_probs = artifacts.text_models["tfidf_multinomialnb"].predict_proba(text_mat)
    nb_preds = [LABELS[int(i)] for i in np.argmax(nb_probs, axis=1)]
    model_outputs["tfidf_multinomialnb"] = (nb_preds, nb_probs)

    # current baseline only for external candidate sets
    if include_current_hybrid:
        hybrid_preds = []
        hybrid_probs_case = []
        for _, row in eval_df.iterrows():
            pred, probs = current_hybrid_case_prediction(row, structured_cases[row["case_id"]])
            hybrid_preds.append(pred)
            hybrid_probs_case.append(probs)
        model_outputs["current_hybrid_baseline"] = (hybrid_preds, np.vstack(hybrid_probs_case))

    # main hybrid
    base_probs = artifacts.hybrid_model.predict_proba(hybrid_mat)
    main_preds = []
    main_probs = []
    for i, row in enumerate(eval_df.itertuples(index=False)):
        structured = structured_cases[row.case_id]
        resp_diag = diagnose_responsibility_from_pre(structured.get("pre_decision_text", ""), row.llm_resp_hint, structured.get("source_span_pointers", []))
        retrieved_dist, retrieved_cases = retrieval_distribution(structured, artifacts, train_df, text_mat[i], int(eval_cfg.get("retrieval_top_k", 5)))
        rule_label = rule_baseline_prediction(structured)
        prior_label = "unknown"
        prior_weight = 0.0
        if include_current_hybrid:
            weak_prior = normalize_label(getattr(row, "weak_label", "unknown"))
            llm_prior_label = normalize_label(getattr(row, "llm_label", "unknown"))
            conflict_flag = int(getattr(row, "conflict_flag", 0))
            needs_review = int(getattr(row, "needs_review", 0))
            if weak_prior in LABELS:
                prior_label = weak_prior
                prior_weight = 0.32 if conflict_flag == 0 and needs_review == 0 else 0.18
            probs = calibrate_hybrid_probs(
                base_probs[i],
                retrieved_dist,
                rule_label,
                structured,
                resp_diag,
                prior_label=prior_label,
                llm_label=llm_prior_label,
                prior_weight=prior_weight,
            )
        else:
            probs = calibrate_hybrid_probs(
                base_probs[i],
                retrieved_dist,
                rule_label,
                structured,
                resp_diag,
            )
        pred = LABELS[int(np.argmax(probs))]
        main_preds.append(pred)
        main_probs.append(probs)

        chain = structured.get("source_span_pointers", [])
        chain_metric = evidence_chain_metrics(chain)
        resp_consistency = evidence_consistency(resp_diag, structured, chain_metric)
        violation_rate = int(resp_diag.get("confidence", 0.0) >= 0.75 and resp_consistency == 0)
        resp_rows.append({
            "case_id": row.case_id,
            "dataset_name": dataset_name,
            "eval_split": split_name,
            "candidate_responsibility_label": row.candidate_responsibility_label if hasattr(row, "candidate_responsibility_label") else "unknown",
            **resp_diag,
            "evidence_consistency_rate": resp_consistency,
            "violation_rate": violation_rate,
        })
        chain_rows.append({
            "case_id": row.case_id,
            "dataset_name": dataset_name,
            "eval_split": split_name,
            **chain_metric,
            "missing_roles": ";".join([x["role_label"] for x in chain if not x.get("text")]),
        })
        trace = reasoning_trace(structured, pred, resp_diag, retrieved_cases, chain)
        trace["dataset_name"] = dataset_name
        trace["eval_split"] = split_name
        traces.append(trace)

    model_outputs["paesc_hybrid"] = (main_preds, np.vstack(main_probs))

    for model_name, (preds, probs) in model_outputs.items():
        for i, row in enumerate(eval_df.itertuples(index=False)):
            structured = structured_cases[row.case_id]
            evidence_score = evidence_sufficiency_score(structured)
            high_dispute_flag = 0
            if model_name == "paesc_hybrid":
                high_dispute_flag = traces[i]["high_dispute_flag"]
            records.append({
                "case_id": row.case_id,
                "dataset_name": dataset_name,
                "eval_split": split_name,
                "model_name": model_name,
                "y_true": row.y_true,
                "y_pred": preds[i],
                "confidence": float(round(np.max(probs[i]), 6)),
                "candidate_confidence": float(getattr(row, "candidate_confidence", 0.0)),
                "case_year": getattr(row, "case_year", None),
                "evidence_sufficiency": evidence_score,
                "generation_source": getattr(row, "generation_source", ""),
                "needs_review": int(getattr(row, "needs_review", 0)),
                "conflict_flag": int(getattr(row, "conflict_flag", 0)),
                "high_dispute_flag": high_dispute_flag,
                **_probs_to_dict(probs[i]),
            })

    pred_df = pd.DataFrame(records)
    resp_df = pd.DataFrame(resp_rows)
    chain_df = pd.DataFrame(chain_rows)

    metrics = {}
    for model_name in pred_df["model_name"].unique().tolist():
        sub = pred_df[pred_df["model_name"] == model_name]
        m = classify_metrics(sub["y_true"], sub["y_pred"], LABELS)
        ci_low, ci_high = bootstrap_ci(sub["y_true"].tolist(), sub["y_pred"].tolist(), LABELS, rounds=int(eval_cfg.get("bootstrap_rounds", 300)), seed=int(cfg["random"]["seed"]))
        m["macro_f1_95ci"] = [ci_low, ci_high]
        metrics[model_name] = m

    # responsibility aggregate for main model only
    resp_main = resp_df.copy()
    valid_resp = resp_main[resp_main["candidate_responsibility_label"].isin(RESP_LABELS[:-1])]
    if valid_resp.empty:
        resp_metrics = {
            "coverage_n": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "uncertainty_rate": float(resp_main["uncertainty_flag"].mean()) if not resp_main.empty else 0.0,
            "evidence_consistency_rate": float(resp_main["evidence_consistency_rate"].mean()) if not resp_main.empty else 0.0,
            "violation_rate": float(resp_main["violation_rate"].mean()) if not resp_main.empty else 0.0,
        }
    else:
        resp_report = classify_metrics(valid_resp["candidate_responsibility_label"], valid_resp["primary_responsible_party"], RESP_LABELS[:-1])
        resp_metrics = {
            "coverage_n": int(len(valid_resp)),
            "accuracy": resp_report["accuracy"],
            "macro_f1": resp_report["macro_f1"],
            "per_class_performance": resp_report["per_class"],
            "uncertainty_rate": float(resp_main["uncertainty_flag"].mean()),
            "evidence_consistency_rate": float(resp_main["evidence_consistency_rate"].mean()),
            "violation_rate": float(resp_main["violation_rate"].mean()),
        }

    chain_metrics_agg = {
        col: float(chain_df[col].mean()) for col in ["valid_span_rate", "pre_decision_span_rate", "duplicate_chain_rate", "role_coverage_rate", "missing_role_rate"] if col in chain_df.columns
    }
    metrics["responsibility_task"] = resp_metrics
    metrics["evidence_chain_auditability"] = chain_metrics_agg
    return pred_df, metrics, resp_df, chain_df, traces


def build_time_split(df: pd.DataFrame, holdout_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[int]]:
    usable = df[df["case_year"].notna()].copy()
    if usable.empty or usable["case_year"].nunique() < 2:
        return df.copy(), pd.DataFrame(columns=df.columns), None
    years = sorted(usable["case_year"].astype(int).unique().tolist())
    cutoff_idx = max(0, int(math.floor(len(years) * (1 - holdout_ratio))) - 1)
    cutoff_year = years[cutoff_idx]
    train_df = df[(df["case_year"].fillna(cutoff_year) <= cutoff_year)].copy()
    test_df = df[(df["case_year"].notna()) & (df["case_year"] > cutoff_year)].copy()
    if test_df.empty:
        return df.copy(), pd.DataFrame(columns=df.columns), None
    return train_df, test_df, cutoff_year


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/research_v1.yaml")
    args = ap.parse_args()

    cfg = load_cfg(PROJECT_ROOT / args.config)
    p = cfg["paths"]
    structured_cases = load_structured_cases(PROJECT_ROOT / p["structured_case_dir"])
    weak_df = build_weak_training_frame(cfg, structured_cases)
    llm_df = read_csv_flexible(PROJECT_ROOT / p["llm_labels_csv"])

    candidate_paths = [
        PROJECT_ROOT / p["candidate_gold_strict_csv"],
        PROJECT_ROOT / p["candidate_gold_extended_csv"],
    ]
    candidate_frames = {path.stem: build_candidate_eval_frame(path, structured_cases, llm_df, weak_df) for path in candidate_paths if path.exists()}

    random_seed = int(cfg["random"]["seed"])
    eval_cfg = cfg["eval"]

    run_dir = PROJECT_ROOT / p["final_eval_root"] / f"final_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # internal random split on weak labels
    w_train, w_test = train_test_split(
        weak_df,
        test_size=float(eval_cfg.get("random_test_size", 0.25)),
        random_state=random_seed,
        stratify=weak_df["label"],
    )
    artifacts_random = fit_models(w_train, cfg, structured_cases)
    weak_random_eval = w_test.rename(columns={"label": "y_true"}).copy()
    weak_random_eval["dataset_name"] = "weak_internal_proxy"
    weak_random_eval["candidate_responsibility_label"] = "unknown"
    weak_random_eval["candidate_confidence"] = 1.0
    weak_random_eval["generation_source"] = "weak_proxy"
    weak_random_eval["needs_review"] = 0
    weak_random_eval["conflict_flag"] = 0
    weak_random_eval["llm_resp_hint"] = "unknown"
    weak_random_eval["llm_label"] = "unknown"
    weak_random_eval["weak_label"] = weak_random_eval["y_true"]
    pred_random, metrics_random, _, _, _ = evaluate_frame(
        weak_random_eval,
        w_train,
        artifacts_random,
        structured_cases,
        cfg,
        split_name="random_split",
        dataset_name="weak_internal_proxy",
        include_current_hybrid=False,
    )

    # time split on weak labels if feasible
    t_train, t_test, cutoff_year = build_time_split(weak_df, float(eval_cfg.get("time_split_holdout_ratio", 0.25)))
    pred_time = pd.DataFrame()
    metrics_time = {"note": "time split unavailable"}
    if cutoff_year is not None and not t_test.empty:
        artifacts_time = fit_models(t_train, cfg, structured_cases)
        weak_time_eval = t_test.rename(columns={"label": "y_true"}).copy()
        weak_time_eval["dataset_name"] = "weak_internal_proxy"
        weak_time_eval["candidate_responsibility_label"] = "unknown"
        weak_time_eval["candidate_confidence"] = 1.0
        weak_time_eval["generation_source"] = "weak_proxy"
        weak_time_eval["needs_review"] = 0
        weak_time_eval["conflict_flag"] = 0
        weak_time_eval["llm_resp_hint"] = "unknown"
        weak_time_eval["llm_label"] = "unknown"
        weak_time_eval["weak_label"] = weak_time_eval["y_true"]
        pred_time, metrics_time, _, _, _ = evaluate_frame(
            weak_time_eval,
            t_train,
            artifacts_time,
            structured_cases,
            cfg,
            split_name=f"time_split_after_{cutoff_year}",
            dataset_name="weak_internal_proxy",
            include_current_hybrid=False,
        )
        metrics_time["cutoff_year"] = cutoff_year

    # train final artifacts on full weak data for candidate evaluations
    artifacts_full = fit_models(weak_df, cfg, structured_cases)
    all_pred_frames = [pred_random]
    all_resp_frames = []
    all_chain_frames = []
    all_traces = []
    external_metrics = {}
    if not pred_time.empty:
        all_pred_frames.append(pred_time)

    for dataset_key, eval_df in candidate_frames.items():
        pred_df, metrics, resp_df, chain_df, traces = evaluate_frame(
            eval_df,
            weak_df,
            artifacts_full,
            structured_cases,
            cfg,
            split_name="external_candidate_eval",
            dataset_name=dataset_key,
            include_current_hybrid=True,
        )
        all_pred_frames.append(pred_df)
        all_resp_frames.append(resp_df)
        all_chain_frames.append(chain_df)
        all_traces.extend(traces)
        external_metrics[dataset_key] = metrics

    predictions_main = pd.concat(all_pred_frames, ignore_index=True)
    predictions_main.to_csv(run_dir / "predictions_main.csv", index=False, encoding="utf-8-sig")

    responsibility_eval = pd.concat(all_resp_frames, ignore_index=True) if all_resp_frames else pd.DataFrame()
    responsibility_eval.to_csv(run_dir / "responsibility_eval.csv", index=False, encoding="utf-8-sig")

    evidence_chain_eval = pd.concat(all_chain_frames, ignore_index=True) if all_chain_frames else pd.DataFrame()
    evidence_chain_eval.to_csv(run_dir / "evidence_chain_eval.csv", index=False, encoding="utf-8-sig")

    with (run_dir / "reasoning_traces.jsonl").open("w", encoding="utf-8") as f:
        for item in all_traces:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    baseline_rows = []
    for dataset_key, metrics in external_metrics.items():
        for model_name, obj in metrics.items():
            if model_name in {"responsibility_task", "evidence_chain_auditability"}:
                continue
            baseline_rows.append({
                "dataset_name": dataset_key,
                "model_name": model_name,
                "accuracy": obj["accuracy"],
                "macro_f1": obj["macro_f1"],
                "weighted_f1": obj["weighted_f1"],
                "macro_f1_ci_low": obj["macro_f1_95ci"][0],
                "macro_f1_ci_high": obj["macro_f1_95ci"][1],
            })
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(run_dir / "baseline_comparison.csv", index=False, encoding="utf-8-sig")

    confusion_rows = []
    per_class_rows = []
    for dataset_key, metrics in external_metrics.items():
        for model_name, obj in metrics.items():
            if model_name in {"responsibility_task", "evidence_chain_auditability"}:
                continue
            conf = obj.get("confusion_matrix", [])
            for i, true_label in enumerate(LABELS):
                for j, pred_label in enumerate(LABELS):
                    if i < len(conf) and j < len(conf[i]):
                        confusion_rows.append({
                            "dataset_name": dataset_key,
                            "model_name": model_name,
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "count": conf[i][j],
                        })
            for label in LABELS:
                vals = obj.get("per_class", {}).get(label, {})
                per_class_rows.append({
                    "dataset_name": dataset_key,
                    "model_name": model_name,
                    "label": label,
                    "precision": vals.get("precision", 0.0),
                    "recall": vals.get("recall", 0.0),
                    "f1_score": vals.get("f1-score", 0.0),
                    "support": vals.get("support", 0.0),
                })
    pd.DataFrame(confusion_rows).to_csv(run_dir / "confusion_matrix_data.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_class_rows).to_csv(run_dir / "per_class_results.csv", index=False, encoding="utf-8-sig")

    metrics_main = {
        "study_positioning": {
            "journal_orientation": "IEEE Transactions on Engineering Management",
            "validation_note": "Automatic metrics and candidate-gold evaluations are not equivalent to human validation. Responsibility diagnosis and evidence-chain outputs remain audit-ready rather than fully human-validated.",
            "input_constraint": "Only pre-decision text is used for model inputs and responsibility/evidence reconstruction.",
        },
        "internal_proxy_validation": {
            "random_split": metrics_random,
            "time_split": metrics_time,
        },
        "candidate_gold_evaluation": external_metrics,
    }
    json_dump(run_dir / "metrics_main.json", metrics_main)

    command = f"python src/final_eval.py --config {args.config}"
    (run_dir / "reproduce_commands.md").write_text(
        "# Reproduce Command\n\n"
        f"```powershell\n{command}\n```\n",
        encoding="utf-8",
    )

    artifact_paths = [
        PROJECT_ROOT / args.config,
        PROJECT_ROOT / p["meta_labels_csv"],
        PROJECT_ROOT / "data" / "meta" / "labels_step2_domain.csv",
        PROJECT_ROOT / p["llm_labels_csv"],
        PROJECT_ROOT / p["candidate_gold_strict_csv"],
        PROJECT_ROOT / p["candidate_gold_extended_csv"],
        PROJECT_ROOT / "data" / "meta" / "structured_case_index.csv",
        PROJECT_ROOT / "src" / "final_eval.py",
        PROJECT_ROOT / "src" / "run_ablation.py",
        PROJECT_ROOT / "src" / "error_analysis.py",
        run_dir / "predictions_main.csv",
        run_dir / "responsibility_eval.csv",
        run_dir / "evidence_chain_eval.csv",
        run_dir / "metrics_main.json",
        run_dir / "per_class_results.csv",
        run_dir / "confusion_matrix_data.csv",
    ]
    manifest = build_run_manifest(
        PROJECT_ROOT,
        PROJECT_ROOT / "requirements.txt",
        artifact_paths,
        model_name="paesc_hybrid",
        prompt_template_version="llm_step2_delay_outcome_v1",
        embedding_model="tfidf_signature_retrieval_v1",
        label_schema_version="outcome_v1__responsibility_v1__candidate_gold_v1",
        command=command,
        seed=random_seed,
        split_mode="random_split+time_split+external_candidate_eval",
        text_mode="pre_decision_only",
        train_label_file=PROJECT_ROOT / "data" / "meta" / "labels_step2_domain.csv",
        eval_label_file=PROJECT_ROOT / p["candidate_gold_extended_csv"],
        metric_source_files=[
            run_dir / "predictions_main.csv",
            run_dir / "responsibility_eval.csv",
            run_dir / "evidence_chain_eval.csv",
        ],
        extra={
            "critical_eval_files": {
                "strict": "data/gold/candidate_gold_strict_v1.csv",
                "extended": "data/gold/candidate_gold_extended_v1.csv",
            },
            "audit_status": "complete",
        },
    )
    write_manifest(run_dir / "run_manifest.json", manifest)

    prev_run = previous_final_eval_run(PROJECT_ROOT / p["final_eval_root"], run_dir)
    compare_paths = [
        PROJECT_ROOT / args.config,
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "src" / "final_eval.py",
        PROJECT_ROOT / "src" / "run_ablation.py",
        PROJECT_ROOT / "src" / "error_analysis.py",
        run_dir / "metrics_main.json",
        run_dir / "per_class_results.csv",
        run_dir / "confusion_matrix_data.csv",
    ]
    old_paths = []
    if prev_run:
        for path in compare_paths:
            if path.is_relative_to(run_dir):
                old_paths.append(prev_run / path.name)
            else:
                old_paths.append(path)
    diff_df = synthetic_file_diff(old_paths, compare_paths, PROJECT_ROOT, comparison_id=run_dir.name)
    diff_df.to_csv(run_dir / "file_diff_summary.csv", index=False, encoding="utf-8-sig")
    write_synthetic_git_summary(
        run_dir / "git_diff_summary.txt",
        "git unavailable in workspace; synthetic diff used. See file_diff_summary.csv for artifact-level SHA256 changes.",
    )
    print(f"[DONE] {run_dir}")


if __name__ == "__main__":
    main()
