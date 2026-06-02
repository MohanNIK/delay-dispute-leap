# -*- coding: utf-8 -*-
"""Controlled MMEC-PAESC 5.5 branch evaluation.

This evaluator reuses the existing PAESC pipeline, then appends audit-aware
MMEC variants. It does not rebuild candidate benchmarks and does not use
post-decision text for inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit_utils import (  # noqa: E402
    build_run_manifest,
    previous_final_eval_run,
    synthetic_file_diff,
    write_manifest,
    write_synthetic_git_summary,
)
from src.final_eval import (  # noqa: E402
    build_candidate_eval_frame,
    build_weak_training_frame,
    evaluate_frame,
    fit_models,
    load_structured_cases,
)
from src.llm55_mechanism_extraction import local_mechanism_from_structured  # noqa: E402
from src.research_support import (  # noqa: E402
    LABELS,
    RESP_LABELS,
    bootstrap_ci,
    classify_metrics,
    json_dump,
    load_cfg,
    normalize_label,
    normalize_resp,
    read_csv_flexible,
)


FOLDED_RESP = ["owner", "contractor", "other_external", "shared_or_uncertain"]


def fold_resp(label: str) -> str:
    label = normalize_resp(label)
    if label in {"owner", "contractor"}:
        return label
    if label in {"subcontractor", "designer_supervisor", "force_majeure_policy"}:
        return "other_external"
    return "shared_or_uncertain"


def probs_from_label(label: str, confidence: float = 0.58) -> np.ndarray:
    label = normalize_label(label)
    if label not in LABELS:
        label = "not_support"
    confidence = float(max(0.34, min(0.96, confidence)))
    probs = np.ones(len(LABELS), dtype=float) * ((1.0 - confidence) / (len(LABELS) - 1))
    probs[LABELS.index(label)] = confidence
    return probs / probs.sum()


def normalize_probs(probs: Sequence[float]) -> np.ndarray:
    arr = np.asarray(probs, dtype=float)
    arr = np.clip(arr, 1e-8, None)
    return arr / arr.sum()


def mechanism_confidence(row: pd.Series) -> float:
    coverage = float(row.get("role_coverage_rate", 0.0))
    evidence = float(row.get("evidence_sufficiency", 0.0))
    risk = max(float(row.get("documentation_gap_index", 0.0)), float(row.get("procedural_compliance_risk", 0.0)))
    return round(max(0.42, min(0.82, 0.48 + 0.18 * coverage + 0.14 * evidence - 0.10 * risk)), 4)


def apply_mmec_calibration(
    base_probs: np.ndarray,
    mech: pd.Series,
    *,
    weight: float,
    api_prior_weight: float,
    use_mechanism: bool = True,
    use_evidence_chain: bool = True,
) -> np.ndarray:
    probs = normalize_probs(base_probs)
    if not use_mechanism:
        return probs

    doc_gap = float(mech.get("documentation_gap_index", 0.0)) if use_evidence_chain else 0.0
    proc_risk = float(mech.get("procedural_compliance_risk", 0.0))
    ambiguity = float(mech.get("causality_ambiguity", 0.0))
    concurrency = float(mech.get("concurrency_risk", 0.0))
    critical_path = float(mech.get("critical_path_support", 0.0)) if use_evidence_chain else 0.0
    negotiation = float(mech.get("negotiation_readiness_score", 0.0)) if use_evidence_chain else 0.0

    risk_score = 0.36 * doc_gap + 0.26 * proc_risk + 0.24 * ambiguity + 0.14 * concurrency
    support_score = 0.36 * critical_path + 0.24 * (1.0 - doc_gap) + 0.18 * (1.0 - proc_risk) + 0.22 * negotiation
    partial_score = max(ambiguity, concurrency)

    adjusted = probs.copy()
    if risk_score > 0.42:
        shift = min(0.16, weight * (risk_score - 0.42))
        adjusted[LABELS.index("support")] = max(0.0, adjusted[LABELS.index("support")] - shift)
        adjusted[LABELS.index("not_support")] += shift
    if support_score > 0.62 and doc_gap < 0.55:
        shift = min(0.12, weight * (support_score - 0.62))
        adjusted[LABELS.index("not_support")] = max(0.0, adjusted[LABELS.index("not_support")] - shift)
        adjusted[LABELS.index("support")] += shift
    if partial_score > 0.50:
        shift = min(0.10, weight * 0.5 * partial_score)
        donor = "support" if adjusted[LABELS.index("support")] >= adjusted[LABELS.index("not_support")] else "not_support"
        adjusted[LABELS.index(donor)] = max(0.0, adjusted[LABELS.index(donor)] - shift)
        adjusted[LABELS.index("partial")] += shift

    if api_prior_weight > 0:
        direct = probs_from_label(mech.get("gpt55_outcome_label", "unknown"), mechanism_confidence(mech))
        adjusted = (1.0 - api_prior_weight) * adjusted + api_prior_weight * direct
    return normalize_probs(adjusted)


def load_mechanisms(path: Path) -> pd.DataFrame:
    df = read_csv_flexible(path)
    df["case_id"] = df["case_id"].astype(str)
    return df.drop_duplicates("case_id")


def ensure_mechanism_for_case(case_id: str, structured: Dict, mech_map: Dict[str, pd.Series]) -> pd.Series:
    if case_id in mech_map:
        return mech_map[case_id]
    return pd.Series(local_mechanism_from_structured(structured))


def base_probs_from_row(row: pd.Series) -> np.ndarray:
    return normalize_probs([row.get(f"{lb}_prob", 0.0) for lb in LABELS])


def tune_mmec_weight(
    weak_eval_df: pd.DataFrame,
    pred_random: pd.DataFrame,
    structured_cases: Dict[str, Dict],
    cfg: Dict,
) -> Tuple[float, pd.DataFrame]:
    grid = [float(x) for x in cfg.get("mmec", {}).get("weight_grid", [0.0, 0.08, 0.12])]
    paesc = pred_random[pred_random["model_name"] == "paesc_hybrid"].copy()
    truth = weak_eval_df.set_index("case_id")["y_true"].to_dict()
    rows = []
    for weight in grid:
        y_true, y_pred = [], []
        for _, pred_row in paesc.iterrows():
            cid = str(pred_row["case_id"])
            mech = pd.Series(local_mechanism_from_structured(structured_cases[cid]))
            probs = apply_mmec_calibration(base_probs_from_row(pred_row), mech, weight=weight, api_prior_weight=0.0)
            y_true.append(truth[cid])
            y_pred.append(LABELS[int(np.argmax(probs))])
        macro = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        rows.append({"weight": weight, "validation_macro_f1": float(macro)})
    tune_df = pd.DataFrame(rows).sort_values(["validation_macro_f1", "weight"], ascending=[False, True])
    return float(tune_df.iloc[0]["weight"]), tune_df


def build_mmec_prediction_rows(
    eval_df: pd.DataFrame,
    base_pred_df: pd.DataFrame,
    structured_cases: Dict[str, Dict],
    mechanisms: pd.DataFrame,
    *,
    dataset_name: str,
    split_name: str,
    weight: float,
    cfg: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, object]]]:
    mech_map = {str(row["case_id"]): row for _, row in mechanisms.iterrows()}
    paesc = base_pred_df[base_pred_df["model_name"] == "paesc_hybrid"].set_index("case_id")
    api_available = bool((mechanisms.get("api_status", pd.Series(dtype=str)) == "api_available").any())
    api_prior = float(
        cfg.get("mmec", {}).get(
            "api_prior_weight_if_available" if api_available else "api_prior_weight_if_unavailable",
            0.0,
        )
    )
    records: List[Dict[str, object]] = []
    resp_rows: List[Dict[str, object]] = []
    chain_rows: List[Dict[str, object]] = []
    traces: List[Dict[str, object]] = []

    eval_truth = eval_df.set_index("case_id").to_dict("index")
    for cid, pred_row in paesc.iterrows():
        cid = str(cid)
        structured = structured_cases[cid]
        truth_row = eval_truth[cid]
        mech = ensure_mechanism_for_case(cid, structured, mech_map)
        base_probs = base_probs_from_row(pred_row)
        variants = {
            "gpt55_direct": probs_from_label(mech.get("gpt55_outcome_label", "unknown"), mechanism_confidence(mech)),
            "mmec_paesc_55_no_mechanism": apply_mmec_calibration(
                base_probs,
                mech,
                weight=weight,
                api_prior_weight=api_prior,
                use_mechanism=False,
            ),
            "mmec_paesc_55_no_evidence_chain": apply_mmec_calibration(
                base_probs,
                mech,
                weight=weight,
                api_prior_weight=api_prior,
                use_mechanism=True,
                use_evidence_chain=False,
            ),
            "mmec_paesc_55": apply_mmec_calibration(
                base_probs,
                mech,
                weight=weight,
                api_prior_weight=api_prior,
                use_mechanism=True,
                use_evidence_chain=True,
            ),
        }
        for model_name, probs in variants.items():
            pred = LABELS[int(np.argmax(probs))]
            records.append(
                {
                    "case_id": cid,
                    "dataset_name": dataset_name,
                    "eval_split": split_name,
                    "model_name": model_name,
                    "y_true": truth_row["y_true"],
                    "y_pred": pred,
                    "confidence": float(round(np.max(probs), 6)),
                    "case_year": truth_row.get("case_year"),
                    "candidate_confidence": truth_row.get("candidate_confidence", 0.0),
                    "generation_source": truth_row.get("generation_source", ""),
                    "needs_review": truth_row.get("needs_review", 0),
                    "conflict_flag": truth_row.get("conflict_flag", 0),
                    "high_dispute_flag": int(
                        max(
                            float(mech.get("documentation_gap_index", 0.0)),
                            float(mech.get("procedural_compliance_risk", 0.0)),
                            float(mech.get("causality_ambiguity", 0.0)),
                        )
                        >= 0.67
                    ),
                    "mechanism_source": mech.get("mechanism_source", "local_rule_proxy"),
                    "api_status": mech.get("api_status", "api_unavailable_rule_proxy"),
                    **{f"{lb}_prob": float(round(probs[i], 6)) for i, lb in enumerate(LABELS)},
                }
            )
        primary = normalize_resp(mech.get("gpt55_responsibility_label", "unknown"))
        resp_rows.append(
            {
                "case_id": cid,
                "dataset_name": dataset_name,
                "eval_split": split_name,
                "model_name": "mmec_paesc_55",
                "candidate_responsibility_label": truth_row.get("candidate_responsibility_label", "unknown"),
                "primary_responsible_party": primary,
                "folded_candidate_responsibility_label": fold_resp(truth_row.get("candidate_responsibility_label", "unknown")),
                "folded_primary_responsible_party": fold_resp(primary),
                "documentation_gap_index": mech.get("documentation_gap_index", 0.0),
                "procedural_compliance_risk": mech.get("procedural_compliance_risk", 0.0),
                "causality_ambiguity": mech.get("causality_ambiguity", 0.0),
                "concurrency_risk": mech.get("concurrency_risk", 0.0),
                "critical_path_support": mech.get("critical_path_support", 0.0),
                "negotiation_readiness_score": mech.get("negotiation_readiness_score", 0.0),
                "managerial_failure_type": mech.get("managerial_failure_type", ""),
                "recommended_management_action": mech.get("recommended_management_action", ""),
                "uncertainty_flag": int(mechanism_confidence(mech) < 0.62),
                "confidence": mechanism_confidence(mech),
                "evidence_consistency_rate": int(float(mech.get("role_coverage_rate", 0.0)) >= 0.4),
                "violation_rate": int(float(mech.get("pre_decision_span_rate", 0.0)) < 1.0),
                "api_status": mech.get("api_status", "api_unavailable_rule_proxy"),
            }
        )
        chain_rows.append(
            {
                "case_id": cid,
                "dataset_name": dataset_name,
                "eval_split": split_name,
                "model_name": "mmec_paesc_55",
                "valid_span_rate": mech.get("valid_span_rate", 0.0),
                "pre_decision_span_rate": mech.get("pre_decision_span_rate", 0.0),
                "duplicate_chain_rate": mech.get("duplicate_chain_rate", 0.0),
                "role_coverage_rate": mech.get("role_coverage_rate", 0.0),
                "missing_role_rate": mech.get("missing_role_rate", 1.0),
                "managerial_mechanism_coverage": mech.get("managerial_mechanism_coverage", 1.0),
                "api_status": mech.get("api_status", "api_unavailable_rule_proxy"),
            }
        )
        traces.append(
            {
                "case_id": cid,
                "dataset_name": dataset_name,
                "eval_split": split_name,
                "model_name": "mmec_paesc_55",
                "outcome_label": records[-1]["y_pred"],
                "responsibility_primary": primary,
                "management_mechanism": {
                    "documentation_gap_index": mech.get("documentation_gap_index", 0.0),
                    "procedural_compliance_risk": mech.get("procedural_compliance_risk", 0.0),
                    "causality_ambiguity": mech.get("causality_ambiguity", 0.0),
                    "concurrency_risk": mech.get("concurrency_risk", 0.0),
                    "critical_path_support": mech.get("critical_path_support", 0.0),
                    "negotiation_readiness_score": mech.get("negotiation_readiness_score", 0.0),
                },
                "management_action": mech.get("recommended_management_action", ""),
                "api_status": mech.get("api_status", "api_unavailable_rule_proxy"),
            }
        )
    return pd.DataFrame(records), pd.DataFrame(resp_rows), pd.DataFrame(chain_rows), traces


def metric_rows(pred_df: pd.DataFrame, labels: Sequence[str], cfg: Dict) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    metrics: Dict[str, object] = {}
    per_class_rows: List[Dict[str, object]] = []
    confusion_rows: List[Dict[str, object]] = []
    for model_name, sub in pred_df.groupby("model_name"):
        m = classify_metrics(sub["y_true"], sub["y_pred"], labels)
        ci_low, ci_high = bootstrap_ci(
            sub["y_true"].tolist(),
            sub["y_pred"].tolist(),
            labels,
            rounds=int(cfg["eval"].get("bootstrap_rounds", 300)),
            seed=int(cfg["random"]["seed"]),
        )
        m["macro_f1_95ci"] = [ci_low, ci_high]
        metrics[model_name] = m
        conf = m.get("confusion_matrix", [])
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                confusion_rows.append(
                    {
                        "model_name": model_name,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": conf[i][j] if i < len(conf) and j < len(conf[i]) else 0,
                    }
                )
        for label in labels:
            vals = m.get("per_class", {}).get(label, {})
            per_class_rows.append(
                {
                    "model_name": model_name,
                    "label": label,
                    "precision": vals.get("precision", 0.0),
                    "recall": vals.get("recall", 0.0),
                    "f1_score": vals.get("f1-score", 0.0),
                    "support": vals.get("support", 0.0),
                }
            )
    return metrics, pd.DataFrame(per_class_rows), pd.DataFrame(confusion_rows)


def responsibility_metrics(resp_df: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {}
    valid = resp_df[resp_df["candidate_responsibility_label"].isin(RESP_LABELS[:-1])]
    if not valid.empty:
        fine = classify_metrics(valid["candidate_responsibility_label"], valid["primary_responsible_party"], RESP_LABELS[:-1])
    else:
        fine = {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "confusion_matrix": []}
    folded_valid = resp_df[resp_df["folded_candidate_responsibility_label"].isin(FOLDED_RESP)]
    if not folded_valid.empty:
        folded = classify_metrics(
            folded_valid["folded_candidate_responsibility_label"],
            folded_valid["folded_primary_responsible_party"],
            FOLDED_RESP,
        )
    else:
        folded = {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "confusion_matrix": []}
    out["fine_schema"] = fine
    out["folded_schema"] = folded
    out["uncertainty_rate"] = float(resp_df["uncertainty_flag"].mean()) if not resp_df.empty else 0.0
    out["evidence_consistency_rate"] = float(resp_df["evidence_consistency_rate"].mean()) if not resp_df.empty else 0.0
    out["violation_rate"] = float(resp_df["violation_rate"].mean()) if not resp_df.empty else 0.0
    return out


def error_category(row: pd.Series) -> str:
    if row["y_true"] == row["y_pred"]:
        return "correct"
    if float(row.get("confidence", 0.0)) < 0.55:
        return "low_confidence_boundary"
    if int(row.get("conflict_flag", 0)) == 1:
        return "candidate_label_conflict"
    if row["y_true"] == "partial" or row["y_pred"] == "partial":
        return "partial_support_boundary_confusion"
    if int(row.get("high_dispute_flag", 0)) == 1:
        return "management_mechanism_high_dispute"
    return "outcome_class_confusion"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/research_v2_55.yaml")
    args = ap.parse_args()

    cfg = load_cfg(PROJECT_ROOT / args.config)
    p = cfg["paths"]
    structured_cases = load_structured_cases(PROJECT_ROOT / p["structured_case_dir"])
    weak_df = build_weak_training_frame(cfg, structured_cases)
    llm_df = read_csv_flexible(PROJECT_ROOT / p["llm_labels_csv"])
    mechanisms = load_mechanisms(PROJECT_ROOT / p["llm55_managerial_mechanisms_csv"])

    run_dir = PROJECT_ROOT / p["final_eval_root"] / f"final_eval_55_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    random_seed = int(cfg["random"]["seed"])
    w_train, w_test = train_test_split(
        weak_df,
        test_size=float(cfg["eval"].get("random_test_size", 0.25)),
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
    best_weight, tuning_df = tune_mmec_weight(weak_random_eval, pred_random, structured_cases, cfg)
    tuning_df.to_csv(run_dir / "mmec_weight_tuning.csv", index=False, encoding="utf-8-sig")

    artifacts_full = fit_models(weak_df, cfg, structured_cases)
    all_pred_frames = [pred_random]
    all_resp_frames: List[pd.DataFrame] = []
    all_chain_frames: List[pd.DataFrame] = []
    all_trace_rows: List[Dict[str, object]] = []
    external_metrics: Dict[str, object] = {}
    per_class_frames: List[pd.DataFrame] = []
    confusion_frames: List[pd.DataFrame] = []

    for key in ["candidate_gold_strict_csv", "candidate_gold_extended_csv"]:
        path = PROJECT_ROOT / p[key]
        eval_df = build_candidate_eval_frame(path, structured_cases, llm_df, weak_df)
        dataset_name = path.stem
        base_pred, base_metrics, base_resp, base_chain, base_traces = evaluate_frame(
            eval_df,
            weak_df,
            artifacts_full,
            structured_cases,
            cfg,
            split_name="external_candidate_eval",
            dataset_name=dataset_name,
            include_current_hybrid=True,
        )
        mmec_pred, mmec_resp, mmec_chain, mmec_traces = build_mmec_prediction_rows(
            eval_df,
            base_pred,
            structured_cases,
            mechanisms,
            dataset_name=dataset_name,
            split_name="external_candidate_eval",
            weight=best_weight,
            cfg=cfg,
        )
        pred_df = pd.concat([base_pred, mmec_pred], ignore_index=True)
        dataset_metrics, per_class_df, confusion_df = metric_rows(
            pred_df[pred_df["eval_split"] == "external_candidate_eval"],
            LABELS,
            cfg,
        )
        dataset_metrics["responsibility_task"] = base_metrics["responsibility_task"]
        dataset_metrics["responsibility_task_mmec_55"] = responsibility_metrics(mmec_resp)
        dataset_metrics["evidence_chain_auditability"] = base_metrics["evidence_chain_auditability"]
        dataset_metrics["evidence_chain_auditability_mmec_55"] = {
            col: float(mmec_chain[col].mean())
            for col in [
                "valid_span_rate",
                "pre_decision_span_rate",
                "duplicate_chain_rate",
                "role_coverage_rate",
                "missing_role_rate",
                "managerial_mechanism_coverage",
            ]
            if col in mmec_chain.columns
        }
        external_metrics[dataset_name] = dataset_metrics
        per_class_df.insert(0, "dataset_name", dataset_name)
        confusion_df.insert(0, "dataset_name", dataset_name)
        per_class_frames.append(per_class_df)
        confusion_frames.append(confusion_df)
        all_pred_frames.append(pred_df)
        base_resp = base_resp.copy()
        base_resp["model_name"] = "paesc_hybrid"
        all_resp_frames.extend([base_resp, mmec_resp])
        base_chain = base_chain.copy()
        base_chain["model_name"] = "paesc_hybrid"
        all_chain_frames.extend([base_chain, mmec_chain])
        all_trace_rows.extend(base_traces)
        all_trace_rows.extend(mmec_traces)

    predictions_main = pd.concat(all_pred_frames, ignore_index=True)
    predictions_main.to_csv(run_dir / "predictions_main.csv", index=False, encoding="utf-8-sig")
    responsibility_eval = pd.concat(all_resp_frames, ignore_index=True)
    responsibility_eval.to_csv(run_dir / "responsibility_eval.csv", index=False, encoding="utf-8-sig")
    evidence_chain_eval = pd.concat(all_chain_frames, ignore_index=True)
    evidence_chain_eval.to_csv(run_dir / "evidence_chain_eval.csv", index=False, encoding="utf-8-sig")
    pd.concat(per_class_frames, ignore_index=True).to_csv(run_dir / "per_class_results.csv", index=False, encoding="utf-8-sig")
    pd.concat(confusion_frames, ignore_index=True).to_csv(run_dir / "confusion_matrix_data.csv", index=False, encoding="utf-8-sig")

    baseline_rows = []
    for dataset_name, metrics in external_metrics.items():
        for model_name, obj in metrics.items():
            if model_name.startswith("responsibility") or model_name.startswith("evidence_chain"):
                continue
            baseline_rows.append(
                {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "accuracy": obj["accuracy"],
                    "macro_f1": obj["macro_f1"],
                    "weighted_f1": obj["weighted_f1"],
                    "macro_f1_ci_low": obj["macro_f1_95ci"][0],
                    "macro_f1_ci_high": obj["macro_f1_95ci"][1],
                }
            )
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(run_dir / "baseline_comparison.csv", index=False, encoding="utf-8-sig")

    ablation_rows = baseline_df[baseline_df["model_name"].isin(["paesc_hybrid", "mmec_paesc_55_no_mechanism", "mmec_paesc_55_no_evidence_chain", "mmec_paesc_55"])].copy()
    ablation_rows["ablation_setting"] = ablation_rows["model_name"].map(
        {
            "paesc_hybrid": "paesc_reference",
            "mmec_paesc_55_no_mechanism": "remove_management_mechanism",
            "mmec_paesc_55_no_evidence_chain": "remove_evidence_chain_from_mmec",
            "mmec_paesc_55": "full_mmec_paesc_55",
        }
    )
    ablation_rows.to_csv(run_dir / "ablation_results.csv", index=False, encoding="utf-8-sig")

    err = predictions_main[
        (predictions_main["eval_split"] == "external_candidate_eval")
        & (predictions_main["model_name"].isin(["mmec_paesc_55", "paesc_hybrid", "current_hybrid_baseline"]))
    ].copy()
    err["error_category"] = err.apply(error_category, axis=1)
    err.to_csv(run_dir / "error_analysis.csv", index=False, encoding="utf-8-sig")
    rep_cases = err.sort_values(["high_dispute_flag", "confidence"], ascending=[False, True]).head(30)
    rep_cases.to_csv(run_dir / "representative_cases.csv", index=False, encoding="utf-8-sig")

    mech_eval = mechanisms.copy()
    mech_eval.to_csv(run_dir / "managerial_mechanisms.csv", index=False, encoding="utf-8-sig")
    with (run_dir / "reasoning_traces.jsonl").open("w", encoding="utf-8") as f:
        for item in all_trace_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    metrics_main = {
        "study_positioning": {
            "journal_orientation": "IEEE-TEM / AIC / AEI",
            "branch": "MMEC-PAESC 5.5 audit-first rerun",
            "claim_note": "Rows with api_unavailable_rule_proxy are local MMEC mechanism proxies and are not valid GPT-5.5 headline claims.",
            "input_constraint": "pre_decision_only",
        },
        "internal_proxy_validation": {
            "random_split": metrics_random,
            "mmec_weight_tuning": tuning_df.to_dict("records"),
            "selected_mmec_weight": best_weight,
        },
        "candidate_gold_evaluation": external_metrics,
    }
    json_dump(run_dir / "metrics_main.json", metrics_main)

    command = f"python src/final_eval_55.py --config {args.config}"
    (run_dir / "reproduce_commands.md").write_text(f"# Reproduce Command\n\n```powershell\n{command}\n```\n", encoding="utf-8")

    artifact_paths = [
        PROJECT_ROOT / args.config,
        PROJECT_ROOT / p["llm55_managerial_mechanisms_csv"],
        PROJECT_ROOT / p["llm55_labels_csv"],
        PROJECT_ROOT / p["candidate_gold_strict_csv"],
        PROJECT_ROOT / p["candidate_gold_extended_csv"],
        PROJECT_ROOT / "data" / "meta" / "structured_case_index.csv",
        PROJECT_ROOT / "src" / "final_eval_55.py",
        PROJECT_ROOT / "src" / "llm55_mechanism_extraction.py",
        PROJECT_ROOT / "src" / "final_eval.py",
        run_dir / "predictions_main.csv",
        run_dir / "responsibility_eval.csv",
        run_dir / "evidence_chain_eval.csv",
        run_dir / "metrics_main.json",
    ]
    api_status_counts = mechanisms["api_status"].value_counts().to_dict() if "api_status" in mechanisms.columns else {}
    manifest = build_run_manifest(
        PROJECT_ROOT,
        PROJECT_ROOT / "requirements.txt",
        artifact_paths,
        model_name=cfg.get("mmec", {}).get("model_name", "mmec_paesc_55"),
        prompt_template_version=cfg.get("llm55", {}).get("prompt_template_version", "llm55_delay_mechanism_v1"),
        embedding_model=cfg.get("mmec", {}).get("embedding_model", "tfidf_signature_retrieval_v1"),
        label_schema_version=cfg.get("mmec", {}).get("label_schema_version", "mmec_v1"),
        command=command,
        seed=int(cfg["random"]["seed"]),
        split_mode="random_split+external_candidate_eval",
        text_mode="pre_decision_only",
        train_label_file=PROJECT_ROOT / "data" / "meta" / "labels_step2_domain.csv",
        eval_label_file=PROJECT_ROOT / p["candidate_gold_extended_csv"],
        metric_source_files=[
            run_dir / "predictions_main.csv",
            run_dir / "responsibility_eval.csv",
            run_dir / "evidence_chain_eval.csv",
        ],
        audit_status="complete",
        extra={
            "api_status_counts": api_status_counts,
            "selected_mmec_weight": best_weight,
            "claim_note": "If api_status_counts lacks api_available, this is a local MMEC rerun, not a true GPT-5.5 result.",
        },
    )
    write_manifest(run_dir / "run_manifest.json", manifest)
    write_synthetic_git_summary(
        run_dir / "git_diff_summary.txt",
        "git unavailable in workspace; synthetic diff used. See file_diff_summary.csv for artifact-level SHA256 changes.",
    )
    prev_run = previous_final_eval_run(PROJECT_ROOT / p["final_eval_root"], run_dir)
    old_paths = [prev_run / name for name in ["metrics_main.json", "predictions_main.csv"] if prev_run and (prev_run / name).exists()]
    new_paths = [run_dir / "metrics_main.json", run_dir / "predictions_main.csv", PROJECT_ROOT / "src" / "final_eval_55.py"]
    synthetic_file_diff(old_paths, new_paths, PROJECT_ROOT, run_dir.name).to_csv(run_dir / "file_diff_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[DONE] final_eval_55: {run_dir}")


if __name__ == "__main__":
    main()
