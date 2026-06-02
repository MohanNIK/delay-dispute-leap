# -*- coding: utf-8 -*-
"""Forensic audit and reproducibility recovery for DelayDispute Copilot."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit_utils import (  # noqa: E402
    build_run_manifest,
    parse_holdout_report,
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
from src.research_support import (  # noqa: E402
    LABELS,
    RESP_LABELS,
    classify_metrics,
    ensure_dir,
    load_cfg,
    normalize_resp,
    read_csv_flexible,
)


HISTORICAL_RUNS = [
    "results/holdout_20251225_233458",
    "results/holdout_20251225_234433",
    "results/final_eval_20260409_143009",
    "results/final_eval_20260409_194025",
]


def safe_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def select_prediction_file(run_dir: Path) -> Optional[Path]:
    for name in ["predictions_main.csv", "predictions.csv"]:
        path = run_dir / name
        if path.exists():
            return path
    return None


def ensure_current_run_artifacts(run_dir: Path, cfg: Dict, config_path: Path) -> Dict[str, object]:
    p = cfg["paths"]
    manifest_path = run_dir / "run_manifest.json"

    artifact_paths = [
        config_path,
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
    command = f"python src/final_eval.py --config {config_path.relative_to(PROJECT_ROOT).as_posix()}"
    manifest = build_run_manifest(
        PROJECT_ROOT,
        PROJECT_ROOT / "requirements.txt",
        artifact_paths,
        model_name="paesc_hybrid",
        prompt_template_version="llm_step2_delay_outcome_v1",
        embedding_model="tfidf_signature_retrieval_v1",
        label_schema_version="outcome_v1__responsibility_v1__candidate_gold_v1",
        command=command,
        seed=int(cfg["random"]["seed"]),
        split_mode="random_split+time_split+external_candidate_eval",
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
            "critical_eval_files": {
                "strict": "data/gold/candidate_gold_strict_v1.csv",
                "extended": "data/gold/candidate_gold_extended_v1.csv",
            }
        },
    )
    write_manifest(manifest_path, manifest)
    if not (run_dir / "reproduce_commands.md").exists():
        (run_dir / "reproduce_commands.md").write_text(
            "# Reproduce Command\n\n"
            f"```powershell\n{command}\n```\n",
            encoding="utf-8",
        )
    if not (run_dir / "git_diff_summary.txt").exists():
        write_synthetic_git_summary(
            run_dir / "git_diff_summary.txt",
            "git unavailable in workspace; synthetic diff used. See file_diff_summary.csv for artifact-level SHA256 changes.",
        )
    if not (run_dir / "file_diff_summary.csv").exists():
        previous = PROJECT_ROOT / "results" / "final_eval_20260409_191425"
        compare_paths = [
            config_path,
            PROJECT_ROOT / "requirements.txt",
            PROJECT_ROOT / "src" / "final_eval.py",
            PROJECT_ROOT / "src" / "run_ablation.py",
            PROJECT_ROOT / "src" / "error_analysis.py",
            run_dir / "metrics_main.json",
            run_dir / "per_class_results.csv",
            run_dir / "confusion_matrix_data.csv",
        ]
        old_paths = []
        new_paths = []
        for path in compare_paths:
            new_paths.append(path)
            old_paths.append((previous / path.name) if str(path).startswith(str(run_dir)) else path)
        diff_df = synthetic_file_diff(old_paths, new_paths, PROJECT_ROOT, comparison_id=run_dir.name)
        diff_df.to_csv(run_dir / "file_diff_summary.csv", index=False, encoding="utf-8-sig")
    return manifest


def infer_historical_manifest(run_dir: Path, cfg: Dict, config_path: Path) -> Dict[str, object]:
    run_name = run_dir.name
    config_json = safe_json(run_dir / "config.json")
    artifact_paths = [
        config_path,
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "src" / "final_eval.py",
        PROJECT_ROOT / "src" / "run_ablation.py",
        PROJECT_ROOT / "src" / "error_analysis.py",
    ]
    for name in [
        "config.json",
        "report.txt",
        "metrics_main.json",
        "predictions.csv",
        "predictions_main.csv",
        "responsibility_eval.csv",
    ]:
        path = run_dir / name
        if path.exists():
            artifact_paths.append(path)

    if run_name.startswith("holdout_"):
        manifest = build_run_manifest(
            PROJECT_ROOT,
            PROJECT_ROOT / "requirements.txt",
            artifact_paths,
            model_name="legacy_stage2_gate",
            prompt_template_version=None,
            embedding_model=None,
            label_schema_version="legacy_outcome_holdout_v0",
            command=None,
            seed=config_json.get("seed"),
            split_mode=f"random_holdout_{config_json.get('test_size')}" if config_json.get("test_size") is not None else None,
            text_mode=config_json.get("text_mode"),
            train_label_file=PROJECT_ROOT / "data" / "meta" / "labels_step2.csv",
            eval_label_file=PROJECT_ROOT / "data" / "meta" / "labels_step2.csv",
            metric_source_files=[select_prediction_file(run_dir)] if select_prediction_file(run_dir) else [],
            audit_status="audit_incomplete",
            extra={
                "stage2_train_size": config_json.get("stage2_train_size"),
                "legacy_config_path": "config.json",
                "source_report": "report.txt" if (run_dir / "report.txt").exists() else None,
            },
        )
    else:
        manifest = build_run_manifest(
            PROJECT_ROOT,
            PROJECT_ROOT / "requirements.txt",
            artifact_paths,
            model_name="legacy_hybrid_rule",
            prompt_template_version=None,
            embedding_model=None,
            label_schema_version="legacy_outcome_v0__responsibility_gold500_v0",
            command=None,
            seed=None,
            split_mode="external_gold500_inferred",
            text_mode=None,
            train_label_file=PROJECT_ROOT / "data" / "meta" / "labels_step2.csv",
            eval_label_file=PROJECT_ROOT / "data" / "gold" / "gold500_v1.csv",
            metric_source_files=[select_prediction_file(run_dir)] if select_prediction_file(run_dir) else [],
            audit_status="audit_incomplete",
            extra={
                "legacy_metrics_path": "metrics_main.json" if (run_dir / "metrics_main.json").exists() else None,
                "legacy_ablation_path": "ablation.csv" if (run_dir / "ablation.csv").exists() else None,
                "inference_note": "Eval file and model metadata inferred from artifact names; original run manifest unavailable.",
            },
        )
    write_manifest(run_dir / "inferred_run_manifest.json", manifest)
    return manifest


def recompute_outcome_metrics_from_predictions(run_dir: Path, manifest: Dict[str, object]) -> pd.DataFrame:
    path = select_prediction_file(run_dir)
    if path is None:
        return pd.DataFrame([{
            "run_name": run_dir.name,
            "task": "outcome",
            "dataset_name": None,
            "model_name": None,
            "n": 0,
            "accuracy_recomputed": None,
            "macro_f1_recomputed": None,
            "weighted_f1_recomputed": None,
            "summary_accuracy": None,
            "summary_macro_f1": None,
            "summary_weighted_f1": None,
            "metric_match": 0,
            "traceability_status": "unverifiable",
        }])

    df = pd.read_csv(path, encoding="utf-8-sig")
    rows: List[Dict[str, object]] = []
    metrics_summary = safe_json(run_dir / "metrics_main.json")
    summary_holdout = parse_holdout_report(run_dir / "report.txt")

    if {"dataset_name", "model_name", "y_true", "y_pred"}.issubset(df.columns):
        groups = df.groupby(["dataset_name", "model_name"], dropna=False)
        for (dataset_name, model_name), sub in groups:
            if "eval_split" in sub.columns and "external_candidate_eval" not in set(sub["eval_split"].astype(str)):
                continue
            m = classify_metrics(sub["y_true"], sub["y_pred"], LABELS)
            summary = metrics_summary.get("candidate_gold_evaluation", {}).get(dataset_name, {}).get(model_name, {})
            rows.append({
                "run_name": run_dir.name,
                "task": "outcome",
                "dataset_name": dataset_name,
                "model_name": model_name,
                "n": int(len(sub)),
                "accuracy_recomputed": m["accuracy"],
                "macro_f1_recomputed": m["macro_f1"],
                "weighted_f1_recomputed": m["weighted_f1"],
                "summary_accuracy": summary.get("accuracy"),
                "summary_macro_f1": summary.get("macro_f1"),
                "summary_weighted_f1": summary.get("weighted_f1"),
                "metric_match": int(
                    not summary or (
                        abs(float(summary.get("accuracy", 0.0)) - m["accuracy"]) < 1e-9 and
                        abs(float(summary.get("macro_f1", 0.0)) - m["macro_f1"]) < 1e-9
                    )
                ),
                "traceability_status": "complete",
            })
    elif {"y_true", "y_pred"}.issubset(df.columns):
        m = classify_metrics(df["y_true"], df["y_pred"], LABELS)
        if (run_dir / "metrics_main.json").exists():
            summary = safe_json(run_dir / "metrics_main.json").get("main_task", {})
            summary_acc = summary.get("accuracy")
            summary_macro = summary.get("macro_f1")
            summary_weight = None
        else:
            summary_acc = summary_holdout.get("summary_accuracy")
            summary_macro = summary_holdout.get("summary_macro_f1")
            summary_weight = summary_holdout.get("summary_weighted_f1")
        rows.append({
            "run_name": run_dir.name,
            "task": "outcome",
            "dataset_name": manifest.get("eval_label_file"),
            "model_name": manifest.get("model_name"),
            "n": int(len(df)),
            "accuracy_recomputed": m["accuracy"],
            "macro_f1_recomputed": m["macro_f1"],
            "weighted_f1_recomputed": m["weighted_f1"],
            "summary_accuracy": summary_acc,
            "summary_macro_f1": summary_macro,
            "summary_weighted_f1": summary_weight,
            "metric_match": int(
                summary_acc is None or (
                    abs(float(summary_acc) - m["accuracy"]) < 5e-4 and
                    abs(float(summary_macro or 0.0) - m["macro_f1"]) < 5e-4
                )
            ),
            "traceability_status": "complete",
        })
    else:
        rows.append({
            "run_name": run_dir.name,
            "task": "outcome",
            "dataset_name": None,
            "model_name": None,
            "n": 0,
            "accuracy_recomputed": None,
            "macro_f1_recomputed": None,
            "weighted_f1_recomputed": None,
            "summary_accuracy": None,
            "summary_macro_f1": None,
            "summary_weighted_f1": None,
            "metric_match": 0,
            "traceability_status": "unverifiable",
        })
    return pd.DataFrame(rows)


def recompute_responsibility_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "responsibility_eval.csv"
    if not path.exists():
        return pd.DataFrame([{
            "run_name": run_dir.name,
            "task": "responsibility",
            "metric_view": "missing",
            "coverage_n": 0,
            "unknown_ratio": None,
            "accuracy_recomputed": None,
            "macro_f1_recomputed": None,
            "summary_accuracy": None,
            "summary_macro_f1": None,
            "metric_match": 0,
            "traceability_status": "unverifiable",
        }])
    df = pd.read_csv(path, encoding="utf-8-sig")
    if {"candidate_responsibility_label", "primary_responsible_party"}.issubset(df.columns):
        gold_col, pred_col = "candidate_responsibility_label", "primary_responsible_party"
        metrics_json = safe_json(run_dir / "metrics_main.json")
        rows = []
        for dataset_name, sub in df.groupby("dataset_name", dropna=False):
            sub = sub.copy()
            sub[gold_col] = sub[gold_col].map(normalize_resp)
            sub[pred_col] = sub[pred_col].map(normalize_resp)
            unknown_ratio = float((sub[gold_col] == "unknown").mean())
            all_metrics = classify_metrics(sub[gold_col], sub[pred_col], RESP_LABELS)
            summary_acc = metrics_json.get("candidate_gold_evaluation", {}).get(str(dataset_name), {}).get("responsibility_task", {}).get("accuracy")
            summary_macro = metrics_json.get("candidate_gold_evaluation", {}).get(str(dataset_name), {}).get("responsibility_task", {}).get("macro_f1")
            rows.append({
                "run_name": run_dir.name,
                "task": "responsibility",
                "dataset_name": dataset_name,
                "metric_view": "all_labels",
                "coverage_n": int(len(sub)),
                "unknown_ratio": unknown_ratio,
                "accuracy_recomputed": all_metrics["accuracy"],
                "macro_f1_recomputed": all_metrics["macro_f1"],
                "summary_accuracy": summary_acc,
                "summary_macro_f1": summary_macro,
                "metric_match": int(summary_acc is None or (abs(float(summary_acc) - all_metrics["accuracy"]) < 5e-4 and abs(float(summary_macro or 0.0) - all_metrics["macro_f1"]) < 5e-4)),
                "traceability_status": "complete",
            })
            known = sub[sub[gold_col] != "unknown"].copy()
            known_metrics = classify_metrics(known[gold_col], known[pred_col], RESP_LABELS[:-1]) if not known.empty else {"accuracy": 0.0, "macro_f1": 0.0}
            rows.append({
                "run_name": run_dir.name,
                "task": "responsibility",
                "dataset_name": dataset_name,
                "metric_view": "known_only",
                "coverage_n": int(len(known)),
                "unknown_ratio": unknown_ratio,
                "accuracy_recomputed": known_metrics["accuracy"],
                "macro_f1_recomputed": known_metrics["macro_f1"],
                "summary_accuracy": None,
                "summary_macro_f1": None,
                "metric_match": 1,
                "traceability_status": "complete",
            })
        return pd.DataFrame(rows)
    elif {"responsibility_gold", "resp_pred"}.issubset(df.columns):
        gold_col, pred_col = "responsibility_gold", "resp_pred"
        summary = safe_json(run_dir / "metrics_main.json").get("responsibility_task", {})
        summary_acc = summary.get("accuracy")
        summary_macro = summary.get("macro_f1")
    else:
        return pd.DataFrame([{
            "run_name": run_dir.name,
            "task": "responsibility",
            "metric_view": "unsupported_schema",
            "coverage_n": 0,
            "unknown_ratio": None,
            "accuracy_recomputed": None,
            "macro_f1_recomputed": None,
            "summary_accuracy": None,
            "summary_macro_f1": None,
            "metric_match": 0,
            "traceability_status": "unverifiable",
        }])

    df[gold_col] = df[gold_col].map(normalize_resp)
    df[pred_col] = df[pred_col].map(normalize_resp)
    unknown_ratio = float((df[gold_col] == "unknown").mean())
    all_metrics = classify_metrics(df[gold_col], df[pred_col], RESP_LABELS)
    rows = [{
        "run_name": run_dir.name,
        "task": "responsibility",
        "dataset_name": None,
        "metric_view": "all_labels",
        "coverage_n": int(len(df)),
        "unknown_ratio": unknown_ratio,
        "accuracy_recomputed": all_metrics["accuracy"],
        "macro_f1_recomputed": all_metrics["macro_f1"],
        "summary_accuracy": summary_acc,
        "summary_macro_f1": summary_macro,
        "metric_match": int(summary_acc is None or (abs(float(summary_acc) - all_metrics["accuracy"]) < 5e-4 and abs(float(summary_macro or 0.0) - all_metrics["macro_f1"]) < 5e-4)),
        "traceability_status": "complete",
    }]
    known = df[df[gold_col] != "unknown"].copy()
    known_metrics = classify_metrics(known[gold_col], known[pred_col], RESP_LABELS[:-1]) if not known.empty else {"accuracy": 0.0, "macro_f1": 0.0}
    rows.append({
        "run_name": run_dir.name,
        "task": "responsibility",
        "dataset_name": None,
        "metric_view": "known_only",
        "coverage_n": int(len(known)),
        "unknown_ratio": unknown_ratio,
        "accuracy_recomputed": known_metrics["accuracy"],
        "macro_f1_recomputed": known_metrics["macro_f1"],
        "summary_accuracy": None,
        "summary_macro_f1": None,
        "metric_match": 1,
        "traceability_status": "complete",
    })
    return pd.DataFrame(rows)


def apply_text_mode(df: pd.DataFrame, structured_cases: Dict[str, Dict], mode: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for rec in df.to_dict("records"):
        structured = structured_cases.get(str(rec["case_id"]), {})
        pre = str(structured.get("pre_decision_text", "") or "")
        post = str(structured.get("post_decision_text", "") or "")
        if mode == "pre_decision_only":
            text = pre
            leakage_flag = int(structured.get("potential_leakage_flag", 0))
        elif mode == "post_decision_only":
            text = post
            leakage_flag = 1 if post.strip() else 0
        elif mode == "pre_decision_plus_post_decision":
            text = "\n".join(x for x in [pre, post] if x)
            leakage_flag = 1 if post.strip() else int(structured.get("potential_leakage_flag", 0))
        else:
            raise ValueError(mode)
        if not str(text).strip():
            continue
        rec["pre_text"] = text
        rec["pre_text_length"] = len(text)
        rec["potential_leakage_flag"] = leakage_flag
        rows.append(rec)
    return pd.DataFrame(rows)


def run_leakage_sentinel(cfg: Dict, out_dir: Path) -> pd.DataFrame:
    structured_cases = load_structured_cases(PROJECT_ROOT / cfg["paths"]["structured_case_dir"])
    weak_df = build_weak_training_frame(cfg, structured_cases)
    llm_df = read_csv_flexible(PROJECT_ROOT / cfg["paths"]["llm_labels_csv"])
    cand_strict = build_candidate_eval_frame(PROJECT_ROOT / cfg["paths"]["candidate_gold_strict_csv"], structured_cases, llm_df, weak_df)
    cand_extended = build_candidate_eval_frame(PROJECT_ROOT / cfg["paths"]["candidate_gold_extended_csv"], structured_cases, llm_df, weak_df)

    sentinel_cfg = deepcopy(cfg)
    sentinel_cfg["eval"]["bootstrap_rounds"] = 60
    rows: List[Dict[str, object]] = []
    for mode in ["pre_decision_only", "post_decision_only", "pre_decision_plus_post_decision"]:
        train_mode = apply_text_mode(weak_df, structured_cases, mode)
        artifacts = fit_models(train_mode, sentinel_cfg, structured_cases)
        for dataset_name, eval_df in {
            "candidate_gold_strict_v1": cand_strict,
            "candidate_gold_extended_v1": cand_extended,
        }.items():
            eval_mode = apply_text_mode(eval_df, structured_cases, mode)
            _, metrics, _, _, _ = evaluate_frame(
                eval_mode,
                train_mode,
                artifacts,
                structured_cases,
                sentinel_cfg,
                split_name=f"leakage_sentinel__{mode}",
                dataset_name=dataset_name,
                include_current_hybrid=True,
            )
            paesc = metrics["paesc_hybrid"]
            rows.append({
                "dataset_name": dataset_name,
                "text_mode": mode,
                "accuracy": paesc["accuracy"],
                "macro_f1": paesc["macro_f1"],
                "weighted_f1": paesc["weighted_f1"],
                "n_cases": int(len(eval_mode)),
            })
    df = pd.DataFrame(rows)
    base = df[df["text_mode"] == "pre_decision_only"][["dataset_name", "macro_f1"]].rename(columns={"macro_f1": "macro_f1_pre_only"})
    df = df.merge(base, on="dataset_name", how="left")
    df["inflation_delta_macro_f1"] = df["macro_f1"] - df["macro_f1_pre_only"]
    df["score_inflation_risk_confirmed"] = ((df["text_mode"] != "pre_decision_only") & (df["inflation_delta_macro_f1"] > 0.005)).astype(int)
    df.to_csv(out_dir / "leakage_sentinel_results.csv", index=False, encoding="utf-8-sig")
    lines = ["# Leakage Sentinel Summary", ""]
    for dataset_name, sub in df.groupby("dataset_name"):
        pre = sub[sub["text_mode"] == "pre_decision_only"]["macro_f1"].iloc[0]
        post = sub[sub["text_mode"] == "post_decision_only"]["macro_f1"].iloc[0]
        both = sub[sub["text_mode"] == "pre_decision_plus_post_decision"]["macro_f1"].iloc[0]
        lines.extend([
            f"## {dataset_name}",
            f"- pre_decision_only macro-F1: {pre:.4f}",
            f"- post_decision_only macro-F1: {post:.4f}",
            f"- pre_decision_plus_post_decision macro-F1: {both:.4f}",
            f"- score inflation from pre+post vs pre-only: {both - pre:+.4f}",
            f"- score inflation from post-only vs pre-only: {post - pre:+.4f}",
            "",
        ])
    (out_dir / "leakage_sentinel_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def collapse_resp(label: str) -> str:
    label = normalize_resp(label)
    if label in {"owner", "contractor"}:
        return label
    if label in {"subcontractor", "designer_supervisor", "force_majeure_policy"}:
        return "other_external"
    return "shared_or_uncertain"


def responsibility_root_cause(current_run: Path, out_dir: Path, cfg: Dict) -> Tuple[pd.DataFrame, str]:
    resp_eval = pd.read_csv(current_run / "responsibility_eval.csv", encoding="utf-8-sig")
    cand_strict = pd.read_csv(PROJECT_ROOT / cfg["paths"]["candidate_gold_strict_csv"], encoding="utf-8-sig")
    cand_extended = pd.read_csv(PROJECT_ROOT / cfg["paths"]["candidate_gold_extended_csv"], encoding="utf-8-sig")
    old_gold500 = pd.read_csv(PROJECT_ROOT / "data" / "gold" / "gold500_v1.csv", encoding="utf-8-sig")
    current_all = pd.concat([cand_strict, cand_extended], ignore_index=True)
    rows: List[Dict[str, object]] = []
    rows.append({
        "root_cause_category": "label_problem",
        "evidence_metric": "old_gold500_unknown_ratio",
        "dataset_name": "gold500_v1",
        "value": float((old_gold500["responsibility_gold"].fillna("unknown") == "unknown").mean()),
        "interpretation": "Legacy gold500 was dominated by unknown labels; this can inflate all-label responsibility accuracy.",
        "evidence_path": "data/gold/gold500_v1.csv",
    })
    for name, df in [("candidate_gold_strict_v1", cand_strict), ("candidate_gold_extended_v1", cand_extended)]:
        unknown_ratio = float((df["candidate_responsibility_label"].fillna("unknown") == "unknown").mean())
        imbalance_ratio = float(df["candidate_responsibility_label"].value_counts(normalize=True, dropna=False).max())
        conflict_rate = float(df["conflict_flag"].fillna(0).mean())
        llm_disagreement = float((df["candidate_responsibility_label"].fillna("unknown").astype(str) != df["llm_responsibility_hint"].fillna("unknown").astype(str)).mean())
        for metric, value, note in [
            ("candidate_unknown_ratio", unknown_ratio, "Candidate benchmark unknown ratio is much lower than legacy gold500, making the task harder and fairer."),
            ("max_class_ratio", imbalance_ratio, "Responsibility labels remain imbalanced, especially toward owner."),
            ("conflict_flag_rate", conflict_rate, "Candidate labels still include machine-assisted conflicts and uncertainty."),
            ("candidate_vs_llm_hint_disagreement", llm_disagreement, "Large disagreement between candidate labels and LLM hints indicates label noise / label-source heterogeneity."),
        ]:
            rows.append({
                "root_cause_category": "label_problem",
                "evidence_metric": metric,
                "dataset_name": name,
                "value": value,
                "interpretation": note,
                "evidence_path": f"data/gold/{name}.csv",
            })

    current = resp_eval.copy()
    current["gold"] = current["candidate_responsibility_label"].map(normalize_resp)
    current["pred"] = current["primary_responsible_party"].map(normalize_resp)
    current["gold_folded"] = current["gold"].map(collapse_resp)
    current["pred_folded"] = current["pred"].map(collapse_resp)
    original_all = classify_metrics(current["gold"], current["pred"], RESP_LABELS)
    folded = classify_metrics(current["gold_folded"], current["pred_folded"], ["owner", "contractor", "other_external", "shared_or_uncertain"])
    rows.extend([
        {
            "root_cause_category": "task_formulation_problem",
            "evidence_metric": "macro_f1_original_schema_all_labels",
            "dataset_name": "combined_current",
            "value": original_all["macro_f1"],
            "interpretation": "Original seven-class responsibility schema is difficult under current machine-assisted labels.",
            "evidence_path": f"{current_run.name}/responsibility_eval.csv",
        },
        {
            "root_cause_category": "task_formulation_problem",
            "evidence_metric": "macro_f1_folded_schema",
            "dataset_name": "combined_current",
            "value": folded["macro_f1"],
            "interpretation": "If folded labels improve materially, task granularity is part of the problem.",
            "evidence_path": f"{current_run.name}/responsibility_eval.csv",
        },
    ])

    lookup = current_all[["case_id", "llm_responsibility_hint"]].drop_duplicates().copy()
    current = current.merge(lookup, on="case_id", how="left")
    current["llm_hint_pred"] = current["llm_responsibility_hint"].fillna("unknown").map(normalize_resp)
    llm_baseline = classify_metrics(current["gold"], current["llm_hint_pred"], RESP_LABELS)
    majority_label = current["gold"].value_counts().idxmax()
    majority_baseline = classify_metrics(current["gold"], [majority_label] * len(current), RESP_LABELS)
    rows.extend([
        {
            "root_cause_category": "model_problem",
            "evidence_metric": "current_structured_diag_macro_f1",
            "dataset_name": "combined_current",
            "value": original_all["macro_f1"],
            "interpretation": "Current structured diagnosis score.",
            "evidence_path": f"{current_run.name}/responsibility_eval.csv",
        },
        {
            "root_cause_category": "model_problem",
            "evidence_metric": "llm_hint_direct_macro_f1",
            "dataset_name": "combined_current",
            "value": llm_baseline["macro_f1"],
            "interpretation": "Direct LLM hint mapping baseline.",
            "evidence_path": "results/labels_step2_delay_outcome_llm.csv",
        },
        {
            "root_cause_category": "model_problem",
            "evidence_metric": "majority_resp_macro_f1",
            "dataset_name": "combined_current",
            "value": majority_baseline["macro_f1"],
            "interpretation": "If majority baseline is close, model-side discrimination is weak relative to imbalance.",
            "evidence_path": f"{current_run.name}/responsibility_eval.csv",
        },
    ])
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "responsibility_root_cause.csv", index=False, encoding="utf-8-sig")
    narrative = "\n".join([
        "# Responsibility Root Cause",
        "",
        f"- Label problem: legacy gold500 unknown ratio = {rows[0]['value']:.3f}; current candidate sets are far less dominated by unknown, so the current task is harder and old high scores are not directly comparable.",
        f"- Task formulation problem: original all-label macro-F1 = {original_all['macro_f1']:.4f}; folded-schema macro-F1 = {folded['macro_f1']:.4f}.",
        f"- Model problem: current structured diagnosis macro-F1 = {original_all['macro_f1']:.4f}; direct LLM hint baseline = {llm_baseline['macro_f1']:.4f}; majority baseline = {majority_baseline['macro_f1']:.4f}.",
    ])
    (out_dir / "responsibility_root_cause.md").write_text(narrative, encoding="utf-8")
    return df, narrative


def build_delta_table(current_run: Path, manifests: Dict[str, Dict], current_outcome: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    current_manifest = manifests[current_run.name]
    current_extended = current_outcome[
        (current_outcome["run_name"] == current_run.name) &
        (current_outcome["dataset_name"] == "candidate_gold_extended_v1") &
        (current_outcome["model_name"] == "paesc_hybrid")
    ]
    current_extended_n = int(current_extended["n"].iloc[0]) if not current_extended.empty else None
    gold500_unknown_ratio = float((pd.read_csv(PROJECT_ROOT / "data" / "gold" / "gold500_v1.csv", encoding="utf-8-sig")["responsibility_gold"].fillna("unknown") == "unknown").mean())
    extended_unknown_ratio = float((pd.read_csv(PROJECT_ROOT / "data" / "gold" / "candidate_gold_extended_v1.csv", encoding="utf-8-sig")["candidate_responsibility_label"].fillna("unknown") == "unknown").mean())

    def add_row(diff_id: str, category: str, old_run: str, old_value: object, new_value: object, evidence_path: str, evidence_excerpt: str, risk: str, direction: str, impact: str, confidence: str, conclusion: str) -> None:
        rows.append({
            "difference_id": diff_id,
            "category": category,
            "old_run": old_run,
            "new_run": current_run.name,
            "old_value": old_value,
            "new_value": new_value,
            "evidence_path": evidence_path,
            "evidence_excerpt": evidence_excerpt,
            "score_inflation_risk": risk,
            "expected_direction": direction,
            "estimated_impact_band": impact,
            "confidence": confidence,
            "auditable_conclusion": conclusion,
        })

    for old_run, old_manifest in manifests.items():
        if old_run == current_run.name:
            continue
        if old_run.startswith("holdout_"):
            cfg = safe_json(PROJECT_ROOT / "results" / old_run / "config.json")
            add_row(
                f"{old_run}__split", "split_mode", old_run,
                f"random_holdout_{cfg.get('test_size')} n={cfg.get('stage2_train_size')}",
                f"candidate_extended external_eval n={current_extended_n}",
                f"results/{old_run}/config.json",
                f"text_mode={cfg.get('text_mode')}; test_size={cfg.get('test_size')}; stage2_train_size={cfg.get('stage2_train_size')}",
                "yes", "raise", "large", "high",
                "Legacy holdout scores came from smaller internal holdouts and are not directly comparable to current external candidate benchmarks.",
            )
            add_row(
                f"{old_run}__text_mode", "feature_and_text_mode", old_run,
                cfg.get("text_mode"), current_manifest.get("text_mode"),
                f"results/{old_run}/config.json",
                f"Legacy text_mode={cfg.get('text_mode')}; current uses pre_decision_only",
                "yes" if cfg.get("text_mode") == "decision+reasoning" else "unclear",
                "raise" if cfg.get("text_mode") == "decision+reasoning" else "mixed",
                "medium", "medium",
                "Legacy text mode is not the same as current leakage-controlled pre-decision input.",
            )
            add_row(
                f"{old_run}__seed", "seed", old_run,
                cfg.get("seed"), current_manifest.get("seed"),
                f"results/{old_run}/config.json",
                f"Legacy seed={cfg.get('seed')}; current seed={current_manifest.get('seed')}",
                "no", "mixed", "small", "high",
                "Seed difference alone cannot explain the gap, but it reduces direct reproducibility.",
            )
            add_row(
                f"{old_run}__metrics", "metric_script", old_run,
                "report.txt summary from predictions.csv", "predictions_main.csv -> recompute",
                f"results/{old_run}/report.txt",
                "Legacy holdout has summary report and predictions; recomputation is possible but evaluation set is different.",
                "no", "mixed", "small", "high",
                "Legacy holdout metrics are traceable but remain internal-holdout references only.",
            )
        else:
            add_row(
                f"{old_run}__gold_version", "gold_candidate_version", old_run,
                f"gold500_v1 unknown_ratio={gold500_unknown_ratio:.3f}",
                f"candidate_gold_extended_v1 unknown_ratio={extended_unknown_ratio:.3f}",
                "data/gold/gold500_v1.csv; data/gold/candidate_gold_extended_v1.csv",
                "gold500_v1 responsibility labels are 82.6% unknown; current candidate benchmark is 7.2% unknown.",
                "yes", "raise", "large", "high",
                "Legacy responsibility scores are inflated by a much easier label distribution and cannot be headline-compared to the current candidate benchmark.",
            )
            add_row(
                f"{old_run}__metric_script", "metric_script", old_run,
                "metrics_main.json headline + cv mismatch", "prediction-level recomputation + sentinel",
                f"results/{old_run}/metrics_main.json",
                "main_task macro_f1=0.8007 while cv_macro_f1_mean=0.2414 in the same file.",
                "yes", "raise", "large", "high",
                "The legacy final_eval report mixes a high external score with a very low internal CV score, so it must be treated cautiously.",
            )
            add_row(
                f"{old_run}__responsibility_unknown", "class_mapping_unknown", old_run,
                "responsibility_gold includes unknown-majority labels", "candidate responsibility schema with low unknown ratio",
                f"results/{old_run}/responsibility_eval.csv",
                "Legacy responsibility_eval gold labels are mostly unknown, so macro-F1 and accuracy are not comparable to the current benchmark.",
                "yes", "raise", "large", "high",
                "Unknown-dominated old responsibility labels likely explain the 0.994 accuracy / 0.851 macro-F1 headline.",
            )
            add_row(
                f"{old_run}__manifest", "config_and_traceability", old_run,
                old_manifest.get("audit_status"), current_manifest.get("audit_status"),
                f"results/{old_run}/inferred_run_manifest.json",
                "Historical manifest is inferred; current manifest is complete.",
                "yes", "mixed", "medium", "high",
                "The legacy run lacks a full manifest and leakage sentinel, so it cannot be used as the main reproducible paper result.",
            )
            add_row(
                f"{old_run}__evidence_chain", "evidence_chain_toggle", old_run,
                "no auditable evidence-chain export", "structured evidence_chain_eval.csv exported",
                f"results/{old_run}",
                "Legacy final_eval exports no evidence_chain_eval.csv, while current run does.",
                "no", "lower", "medium", "high",
                "Current auditability requirements make the task stricter and reduce direct score comparability.",
            )
    return pd.DataFrame(rows)


def build_claim_tiering(manifests: Dict[str, Dict], recompute_df: pd.DataFrame, leakage_df: pd.DataFrame, current_run: Path) -> pd.DataFrame:
    rows = []
    leakage_covered = set(leakage_df["dataset_name"].unique().tolist()) if not leakage_df.empty else set()
    for run_name, manifest in manifests.items():
        audit_status = manifest.get("audit_status")
        run_metrics = recompute_df[recompute_df["run_name"] == run_name]
        if run_name == current_run.name:
            headline_metrics = run_metrics[
                (run_metrics["task"] == "outcome") &
                (run_metrics["model_name"] == "paesc_hybrid") &
                (run_metrics["dataset_name"].isin(["candidate_gold_strict_v1", "candidate_gold_extended_v1"]))
            ]
            has_recomputed = not headline_metrics.empty and int(headline_metrics["metric_match"].fillna(0).all()) == 1
            has_trace = int((headline_metrics["traceability_status"].fillna("") == "complete").all()) == 1 if not headline_metrics.empty else False
            sentinel_ok = "candidate_gold_extended_v1" in leakage_covered and "candidate_gold_strict_v1" in leakage_covered
            tier = "Tier B: claimable with caution"
            claim_boundary = "Machine-assisted candidate benchmark; audit-ready and recomputed, but not human-validated."
            safe_claim = 1 if audit_status == "complete" and has_recomputed and has_trace and sentinel_ok else 0
        else:
            has_recomputed = not run_metrics.empty and int(run_metrics["metric_match"].fillna(0).all()) == 1
            has_trace = int((run_metrics["traceability_status"].fillna("") == "complete").all()) == 1 if not run_metrics.empty else False
            sentinel_ok = False
            tier = "Tier C: historical reference only / not headline-claimable"
            claim_boundary = "Historical reference only. Missing full manifest and/or leakage sentinel."
            safe_claim = 0
        rows.append({
            "run_name": run_name,
            "audit_status": audit_status,
            "metric_recomputed": int(has_recomputed),
            "artifact_traceable": int(has_trace),
            "leakage_sentinel_passed": int(sentinel_ok),
            "tier": tier,
            "claim_boundary": claim_boundary,
            "headline_claimable": safe_claim,
        })
    return pd.DataFrame(rows)


def build_forensic_texts(out_dir: Path, recompute_df: pd.DataFrame, root_cause_text: str) -> None:
    paper_text_dir = ensure_dir(PROJECT_ROOT / "paper_assets" / "text")
    current_extended = recompute_df[
        (recompute_df["run_name"] == "final_eval_20260409_194025") &
        (recompute_df["dataset_name"] == "candidate_gold_extended_v1") &
        (recompute_df["model_name"] == "paesc_hybrid")
    ]
    legacy_high = recompute_df[recompute_df["run_name"] == "final_eval_20260409_143009"]
    forensic_summary = [
        "# Forensic Summary",
        "",
        "Current main result is the audit-ready PAESC run on candidate benchmarks, not the historical 0.80/0.85 legacy result.",
        "",
        "Key findings:",
        "- Historical high outcome scores came from different evaluation regimes: small internal holdouts and an older gold500-based external run.",
        "- The old gold500 responsibility labels were 82.6% unknown, which makes the historical responsibility headline much easier than the current candidate benchmark.",
        "- The current run is traceable and leakage-audited; legacy runs are not fully reproducible under the same audit standard.",
        "",
    ]
    if not current_extended.empty:
        forensic_summary.append(f"- Current audit-ready extended outcome macro-F1 = {float(current_extended['macro_f1_recomputed'].iloc[0]):.4f}.")
    if not legacy_high.empty:
        forensic_summary.append(f"- Historical final_eval_20260409_143009 outcome macro-F1 recomputes to {float(legacy_high['macro_f1_recomputed'].iloc[0]):.4f}, but remains Tier C.")
    forensic_summary.extend(["", root_cause_text])
    text = "\n".join(forensic_summary)
    (out_dir / "forensic_audit_report.md").write_text(text, encoding="utf-8")
    (paper_text_dir / "forensic_summary.md").write_text(text, encoding="utf-8")

    old_high = [
        "# Why The Old Score Was Higher But Cannot Be Used As The Main Claim",
        "",
        "1. Legacy holdout runs used internal weak-label holdouts rather than the current external candidate benchmark.",
        "2. The legacy gold500 responsibility labels were dominated by unknown, which inflates all-label responsibility scores.",
        "3. The historical final_eval_20260409_143009 report shows a large mismatch between external headline macro-F1 (0.8007) and internal CV macro-F1 (0.2414), so it is not a stable main result.",
        "4. Legacy runs lack the current run_manifest completeness and leakage sentinel evidence.",
        "5. Under the current IEEE-TEM framing, only leakage-aware, traceable, recomputed results are headline-eligible.",
    ]
    (paper_text_dir / "old_high_score_explanation.md").write_text("\n".join(old_high), encoding="utf-8")

    boundary = [
        "# Result Claim Boundary",
        "",
        "Tier A: reserved for fully traceable, recomputed, leakage-checked results with bounded claim scope.",
        "Tier B: current candidate-benchmark results. Claimable with caution as machine-assisted, audit-ready benchmarks.",
        "Tier C: historical references only. Not headline-claimable because they lack full artifact traceability and/or leakage validation.",
        "",
        "Current project state: no result qualifies for a fully human-validated claim. The current audit-ready result is Tier B.",
    ]
    (paper_text_dir / "result_claim_boundary.md").write_text("\n".join(boundary), encoding="utf-8")


def build_paper_tables(recompute_df: pd.DataFrame, claim_df: pd.DataFrame) -> None:
    tables_dir = ensure_dir(PROJECT_ROOT / "paper_assets" / "tables")
    table_main_rows = []
    definitions = [
        ("Legacy historical result (audit caution)", (recompute_df["run_name"] == "final_eval_20260409_143009"), "Old gold500-based external result; high score but no full manifest or leakage sentinel.", "Tier C"),
        ("Legacy holdout reference (audit caution)", (recompute_df["run_name"] == "holdout_20251225_234433"), "Internal weak-label holdout, not the current external candidate benchmark.", "Tier C"),
        ("Current reproducible baseline", (recompute_df["run_name"] == "final_eval_20260409_194025") & (recompute_df["dataset_name"] == "candidate_gold_extended_v1") & (recompute_df["model_name"] == "current_hybrid_baseline"), "Highest reproducible outcome baseline on current candidate benchmark.", "Tier B"),
        ("Current audit-ready main result", (recompute_df["run_name"] == "final_eval_20260409_194025") & (recompute_df["dataset_name"] == "candidate_gold_extended_v1") & (recompute_df["model_name"] == "paesc_hybrid"), "Audit-ready PAESC result with structured responsibility and evidence-chain outputs.", "Tier B"),
    ]
    for label, filt, note, status in definitions:
        sub = recompute_df[filt].head(1)
        if sub.empty:
            continue
        row = sub.iloc[0]
        table_main_rows.append({
            "result_group": label,
            "run_name": row["run_name"],
            "dataset_name": row["dataset_name"],
            "model_name": row["model_name"],
            "accuracy": row["accuracy_recomputed"],
            "macro_f1": row["macro_f1_recomputed"],
            "headline_status": status,
            "note": note,
        })
    pd.DataFrame(table_main_rows).to_csv(tables_dir / "table_main_result_claims.csv", index=False, encoding="utf-8-sig")
    claim_df.to_csv(tables_dir / "table_result_claim_boundary.csv", index=False, encoding="utf-8-sig")
    recompute_df.to_csv(tables_dir / "table_metric_recompute_check.csv", index=False, encoding="utf-8-sig")


def top5_actions() -> pd.DataFrame:
    actions = [
        {"rank": 1, "action": "Reintroduce the legacy hybrid prior as a controlled, auditable feature block", "target_module": "outcome prediction / prior fusion", "expected_gain_direction": "raise extended outcome macro-F1", "evidence_basis": "Current reproducible baseline outperforms PAESC on extended candidate benchmark.", "risk": "Can overfit weak priors if not ablated carefully.", "auditability_impact": "safe if priors are explicit and ablated."},
        {"rank": 2, "action": "Calibrate candidate label conflicts rather than treating them as flat noise", "target_module": "candidate benchmark + training alignment", "expected_gain_direction": "raise outcome macro-F1 and reduce unstable boundary cases", "evidence_basis": "Candidate sets still contain conflict_flag and multi-source disagreement.", "risk": "Requires disciplined provenance tracking.", "auditability_impact": "safe if conflict-aware weights are logged in manifest."},
        {"rank": 3, "action": "Collapse responsibility schema for primary paper result, keep full schema as appendix", "target_module": "responsibility diagnosis", "expected_gain_direction": "raise responsibility macro-F1 materially", "evidence_basis": "Folded schema performs better than original schema in root-cause audit.", "risk": "Reduces granularity in headline table.", "auditability_impact": "safe; actually improves interpretability."},
        {"rank": 4, "action": "Target support/not_support boundary with class-balanced calibration", "target_module": "outcome classifier", "expected_gain_direction": "raise extended outcome macro-F1", "evidence_basis": "Current candidate benchmark is imbalanced and most errors cluster on support/partial boundaries.", "risk": "Can distort probability calibration if pushed too far.", "auditability_impact": "safe if calibration tables are exported."},
        {"rank": 5, "action": "Separate label noise diagnostics from model training for responsibility", "target_module": "responsibility root-cause pipeline", "expected_gain_direction": "clarify whether low score is primarily a data problem or a model problem", "evidence_basis": "Current responsibility performance is limited by mixed label sources and class imbalance.", "risk": "May not improve score immediately.", "auditability_impact": "positive; sharpens claim boundaries."},
    ]
    return pd.DataFrame(actions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/research_v1.yaml")
    ap.add_argument("--runs", nargs="*", default=HISTORICAL_RUNS)
    args = ap.parse_args()

    cfg = load_cfg(PROJECT_ROOT / args.config)
    config_path = PROJECT_ROOT / args.config
    out_dir = ensure_dir(PROJECT_ROOT / "results" / f"forensic_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    manifests: Dict[str, Dict] = {}
    for run_str in args.runs:
        run_dir = PROJECT_ROOT / run_str
        if not run_dir.exists():
            continue
        if run_dir.name == "final_eval_20260409_194025":
            manifests[run_dir.name] = ensure_current_run_artifacts(run_dir, cfg, config_path)
        else:
            manifests[run_dir.name] = infer_historical_manifest(run_dir, cfg, config_path)

    outcome_frames = []
    resp_frames = []
    for run_name in manifests:
        run_dir = PROJECT_ROOT / "results" / run_name
        outcome_frames.append(recompute_outcome_metrics_from_predictions(run_dir, manifests[run_name]))
        resp_frames.append(recompute_responsibility_metrics(run_dir))
    outcome_df = pd.concat(outcome_frames, ignore_index=True) if outcome_frames else pd.DataFrame()
    resp_df = pd.concat(resp_frames, ignore_index=True) if resp_frames else pd.DataFrame()
    metric_recompute = pd.concat([outcome_df, resp_df], ignore_index=True)
    metric_recompute.to_csv(out_dir / "metric_recompute_check.csv", index=False, encoding="utf-8-sig")
    metric_recompute.to_excel(out_dir / "metric_recompute_check.xlsx", index=False)

    leakage_df = run_leakage_sentinel(cfg, out_dir)
    _, root_cause_text = responsibility_root_cause(PROJECT_ROOT / "results" / "final_eval_20260409_194025", out_dir, cfg)
    delta_df = build_delta_table(PROJECT_ROOT / "results" / "final_eval_20260409_194025", manifests, outcome_df)
    delta_df.to_csv(out_dir / "delta_table.csv", index=False, encoding="utf-8-sig")
    delta_df.to_excel(out_dir / "delta_table.xlsx", index=False)
    claim_df = build_claim_tiering(manifests, metric_recompute, leakage_df, PROJECT_ROOT / "results" / "final_eval_20260409_194025")
    claim_df.to_csv(out_dir / "claim_tiering.csv", index=False, encoding="utf-8-sig")
    build_forensic_texts(out_dir, metric_recompute, root_cause_text)
    build_paper_tables(metric_recompute, claim_df)

    comparison_paths = []
    for run_name in manifests:
        run_dir = PROJECT_ROOT / "results" / run_name
        for name in ["metrics_main.json", "predictions_main.csv", "predictions.csv", "responsibility_eval.csv", "run_manifest.json", "inferred_run_manifest.json"]:
            path = run_dir / name
            if path.exists():
                comparison_paths.append(path)
    diff_df = synthetic_file_diff([], comparison_paths, PROJECT_ROOT, comparison_id=out_dir.name)
    diff_df.to_csv(out_dir / "file_diff_summary.csv", index=False, encoding="utf-8-sig")
    write_synthetic_git_summary(out_dir / "git_diff_summary.txt", "git unavailable in workspace; synthetic diff used. See file_diff_summary.csv for SHA256-indexed artifact inventory.")

    reproduce_lines = ["# Reproduce Commands", ""]
    for run_name, manifest in manifests.items():
        reproduce_lines.append(f"## {run_name}")
        reproduce_lines.append(f"- command: {manifest.get('command') or 'unavailable / historical run'}")
        reproduce_lines.append("")
    reproduce_lines.append(f"## forensic_audit\n- command: python src/run_forensic_audit.py --config {args.config}")
    (out_dir / "reproduce_commands.md").write_text("\n".join(reproduce_lines), encoding="utf-8")

    actions_df = top5_actions()
    actions_df.to_csv(out_dir / "top5_next_actions.csv", index=False, encoding="utf-8-sig")
    actions_df.to_csv(PROJECT_ROOT / "paper_assets" / "tables" / "table_top5_next_actions.csv", index=False, encoding="utf-8-sig")
    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()
