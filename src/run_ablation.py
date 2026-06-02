# -*- coding: utf-8 -*-
"""Run meaningful ablations for the IEEE-TEM-oriented DelayDispute Copilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.final_eval import (  # noqa: E402
    LABELS,
    RESP_LABELS,
    build_candidate_eval_frame,
    build_eval_matrix,
    build_weak_training_frame,
    calibrate_hybrid_probs,
    evidence_consistency,
    fit_models,
    load_structured_cases,
    retrieval_distribution,
)
from src.research_support import (  # noqa: E402
    classify_metrics,
    diagnose_responsibility_from_pre,
    evidence_chain_metrics,
    latest_run_dir,
    load_cfg,
    normalize_label,
    read_csv_flexible,
    rule_baseline_prediction,
)


ABLATIONS: Dict[str, Dict[str, str]] = {
    "full_model": {
        "removed_component": "none",
        "hypothesis": "Full PAESC + retrieval + responsibility diagnosis + verifier should provide the most balanced performance and auditability.",
        "managerial_rationale": "Combines predictive accuracy with governance-oriented diagnostics.",
    },
    "remove_pre_decision_constraint": {
        "removed_component": "Leakage control",
        "hypothesis": "Allowing post-decision text artificially inflates predictive performance.",
        "managerial_rationale": "Demonstrates why the study must stay prospective rather than retrospective.",
    },
    "remove_structured_events": {
        "removed_component": "Structured event extraction",
        "hypothesis": "Removing event structure reduces mechanism sensitivity and class discrimination.",
        "managerial_rationale": "Without explicit delay-event structuring, the system is less useful for mechanism diagnosis.",
    },
    "remove_procedural_signals": {
        "removed_component": "Procedural compliance signals",
        "hypothesis": "Removing procedure cues weakens responsibility diagnosis and boundary-case recognition.",
        "managerial_rationale": "Procedural vulnerability is central to managerial dispute governance.",
    },
    "remove_evidence_chain": {
        "removed_component": "Evidence-chain reconstruction",
        "hypothesis": "Removing evidence chains reduces auditability and weakens calibrated predictions.",
        "managerial_rationale": "Without traceable evidence, managerial adoption and reviewability decline.",
    },
    "remove_responsibility_head": {
        "removed_component": "Structured responsibility diagnosis",
        "hypothesis": "Removing responsibility head weakens governance interpretation and reduces mechanism sensitivity.",
        "managerial_rationale": "Responsibility framing is needed for actionable dispute triage.",
    },
    "remove_retrieval": {
        "removed_component": "Case retrieval",
        "hypothesis": "Removing retrieval weakens comparative reasoning and lowers robustness.",
        "managerial_rationale": "Similar-case reference supports consistent governance decisions.",
    },
    "remove_irac_verifier": {
        "removed_component": "Consistency verifier",
        "hypothesis": "Removing the verifier raises unresolved conflict cases and weakens uncertainty handling.",
        "managerial_rationale": "Managers need explicit dispute flags when evidence and prediction diverge.",
    },
}


def neutral_resp_diag() -> Dict[str, object]:
    return {
        "primary_responsible_party": "unknown",
        "secondary_responsible_party": "unknown",
        "responsibility_type": "uncertain",
        "evidence_spans": [],
        "procedural_compliance_status": "uncertain",
        "causality_chain_summary": "",
        "documentation_integrity_flag": "unknown",
        "confidence": 0.0,
        "uncertainty_flag": 1,
        "explanation_text": "",
    }


def mutate_structured_cases(structured_cases: Dict[str, Dict], setting: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for cid, case in structured_cases.items():
        sc = dict(case)
        if setting == "remove_structured_events":
            sc["delay_events"] = []
        if setting == "remove_procedural_signals":
            sc["procedural_compliance_cues"] = []
        if setting == "remove_evidence_chain":
            sc["source_span_pointers"] = []
            sc["evidence_mentions"] = []
            sc["evidence_sufficiency"] = 0.5
        out[cid] = sc
    return out


def mutate_frame(df: pd.DataFrame, structured_cases: Dict[str, Dict], setting: str) -> pd.DataFrame:
    out = df.copy()
    if setting == "remove_pre_decision_constraint":
        out["pre_text"] = [
            (structured_cases[cid].get("pre_decision_text", "") + "\n" + structured_cases[cid].get("post_decision_text", "")).strip()
            for cid in out["case_id"].astype(str).tolist()
        ]
    if setting == "remove_structured_events":
        out["delay_event_count"] = 0
    if setting == "remove_procedural_signals":
        out["procedure_cue_count"] = 0
    if setting == "remove_evidence_chain":
        out["evidence_mention_count"] = 0
        out["role_coverage_rate"] = 0.0
        out["missing_role_rate"] = 1.0
        out["evidence_sufficiency"] = 0.5
    return out


def apply_verifier(
    probs: np.ndarray,
    pred_label: str,
    rule_label: str,
    resp_diag: Dict[str, object],
    chain_metric: Dict[str, float],
    use_verifier: bool,
) -> Tuple[np.ndarray, int]:
    if not use_verifier:
        return probs, 0
    high_dispute = int(
        resp_diag.get("uncertainty_flag", 0) == 1
        or chain_metric.get("role_coverage_rate", 0.0) < 0.45
        or chain_metric.get("valid_span_rate", 0.0) < 0.75
        or (pred_label != "partial" and rule_label != pred_label)
    )
    return probs, high_dispute


def evaluate_ablation(
    setting: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    artifacts,
    structured_cases: Dict[str, Dict],
    cfg: Dict,
    dataset_name: str,
) -> Dict[str, object]:
    text_mat, hybrid_mat = build_eval_matrix(eval_df, artifacts)
    base_probs = artifacts.hybrid_model.predict_proba(hybrid_mat)
    y_true = eval_df["y_true"].tolist()
    y_pred: List[str] = []
    resp_true: List[str] = []
    resp_pred: List[str] = []
    resp_uncertainty = []
    resp_consistency = []
    verifier_flags = []
    chain_valid = []
    chain_pre = []
    chain_dup = []
    chain_cover = []
    chain_missing = []

    for i, row in enumerate(eval_df.itertuples(index=False)):
        structured = structured_cases[row.case_id]
        if setting == "remove_responsibility_head":
            resp_diag = neutral_resp_diag()
        else:
            resp_diag = diagnose_responsibility_from_pre(
                structured.get("pre_decision_text", ""),
                getattr(row, "llm_resp_hint", "unknown"),
                structured.get("source_span_pointers", []),
            )
        if setting == "remove_retrieval":
            retrieval_probs = np.ones((len(LABELS),), dtype=float) / len(LABELS)
        else:
            retrieval_probs, _ = retrieval_distribution(
                structured,
                artifacts,
                train_df,
                text_mat[i],
                int(cfg["eval"].get("retrieval_top_k", 5)),
            )
        weak_prior = normalize_label(getattr(row, "weak_label", "unknown"))
        llm_prior = normalize_label(getattr(row, "llm_label", "unknown"))
        conflict_flag = int(getattr(row, "conflict_flag", 0))
        needs_review = int(getattr(row, "needs_review", 0))
        prior_weight = 0.32 if weak_prior in LABELS and conflict_flag == 0 and needs_review == 0 else 0.18 if weak_prior in LABELS else 0.0
        if setting == "remove_responsibility_head":
            prior_weight = max(0.12, prior_weight - 0.06)

        rule_label = rule_baseline_prediction(structured)
        probs = calibrate_hybrid_probs(
            base_probs[i],
            retrieval_probs,
            rule_label,
            structured,
            resp_diag,
            prior_label=weak_prior if weak_prior in LABELS else "unknown",
            llm_label=llm_prior,
            prior_weight=prior_weight,
        )
        chain = structured.get("source_span_pointers", [])
        chain_metric = evidence_chain_metrics(chain)
        pred_before_verifier = LABELS[int(np.argmax(probs))]
        probs, high_dispute_flag = apply_verifier(
            probs,
            pred_before_verifier,
            rule_label,
            resp_diag,
            chain_metric,
            use_verifier=(setting != "remove_irac_verifier"),
        )
        pred = LABELS[int(np.argmax(probs))]
        y_pred.append(pred)
        verifier_flags.append(high_dispute_flag)

        true_resp = getattr(row, "candidate_responsibility_label", "unknown")
        if true_resp in RESP_LABELS[:-1]:
            resp_true.append(true_resp)
            resp_pred.append(resp_diag.get("primary_responsible_party", "unknown"))
        resp_uncertainty.append(int(resp_diag.get("uncertainty_flag", 1)))
        resp_consistency.append(evidence_consistency(resp_diag, structured, chain_metric))
        chain_valid.append(chain_metric.get("valid_span_rate", 0.0))
        chain_pre.append(chain_metric.get("pre_decision_span_rate", 0.0))
        chain_dup.append(chain_metric.get("duplicate_chain_rate", 0.0))
        chain_cover.append(chain_metric.get("role_coverage_rate", 0.0))
        chain_missing.append(chain_metric.get("missing_role_rate", 0.0))

    cls_metrics = classify_metrics(y_true, y_pred, LABELS)
    if resp_true:
        resp_metrics = classify_metrics(resp_true, resp_pred, RESP_LABELS[:-1])
        resp_acc = resp_metrics["accuracy"]
        resp_macro_f1 = resp_metrics["macro_f1"]
    else:
        resp_acc = 0.0
        resp_macro_f1 = 0.0

    return {
        "dataset_name": dataset_name,
        "ablation_setting": setting,
        "removed_component": ABLATIONS[setting]["removed_component"],
        "hypothesis": ABLATIONS[setting]["hypothesis"],
        "managerial_rationale": ABLATIONS[setting]["managerial_rationale"],
        "accuracy": cls_metrics["accuracy"],
        "macro_f1": cls_metrics["macro_f1"],
        "weighted_f1": cls_metrics["weighted_f1"],
        "responsibility_accuracy": resp_acc,
        "responsibility_macro_f1": resp_macro_f1,
        "uncertainty_rate": float(np.mean(resp_uncertainty)) if resp_uncertainty else 0.0,
        "evidence_consistency_rate": float(np.mean(resp_consistency)) if resp_consistency else 0.0,
        "high_dispute_rate": float(np.mean(verifier_flags)) if verifier_flags else 0.0,
        "valid_span_rate": float(np.mean(chain_valid)) if chain_valid else 0.0,
        "pre_decision_span_rate": float(np.mean(chain_pre)) if chain_pre else 0.0,
        "duplicate_chain_rate": float(np.mean(chain_dup)) if chain_dup else 0.0,
        "role_coverage_rate": float(np.mean(chain_cover)) if chain_cover else 0.0,
        "missing_role_rate": float(np.mean(chain_missing)) if chain_missing else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/research_v1.yaml")
    ap.add_argument("--run_dir", type=str, default="")
    args = ap.parse_args()

    cfg = load_cfg(PROJECT_ROOT / args.config)
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(PROJECT_ROOT / cfg["paths"]["final_eval_root"])
    if run_dir is None:
        raise FileNotFoundError("No final_eval_* run directory found.")
    run_dir = run_dir.resolve()

    structured_cases_full = load_structured_cases(PROJECT_ROOT / cfg["paths"]["structured_case_dir"])
    weak_df_full = build_weak_training_frame(cfg, structured_cases_full)
    llm_df = read_csv_flexible(PROJECT_ROOT / cfg["paths"]["llm_labels_csv"])
    candidate_paths = [
        PROJECT_ROOT / cfg["paths"]["candidate_gold_strict_csv"],
        PROJECT_ROOT / cfg["paths"]["candidate_gold_extended_csv"],
    ]
    candidate_frames_base = {
        path.stem: build_candidate_eval_frame(path, structured_cases_full, llm_df, weak_df_full)
        for path in candidate_paths
        if path.exists()
    }

    rows: List[Dict[str, object]] = []
    for setting in ABLATIONS:
        structured_cases = mutate_structured_cases(structured_cases_full, setting)
        weak_df = mutate_frame(weak_df_full, structured_cases_full, setting)
        candidate_frames = {
            key: mutate_frame(df, structured_cases_full, setting) for key, df in candidate_frames_base.items()
        }
        artifacts = fit_models(weak_df, cfg, structured_cases)
        for dataset_name, eval_df in candidate_frames.items():
            result = evaluate_ablation(setting, weak_df, eval_df, artifacts, structured_cases, cfg, dataset_name)
            rows.append(result)
            print(f"[ABLATION] {dataset_name} :: {setting} :: macro_f1={result['macro_f1']:.4f}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(run_dir / "ablation_results.csv", index=False, encoding="utf-8-sig")
    out_df.to_excel(run_dir / "ablation_results.xlsx", index=False)
    print(f"[DONE] {run_dir / 'ablation_results.csv'}")


if __name__ == "__main__":
    main()
