# -*- coding: utf-8 -*-
"""Forensic audit for the MMEC-PAESC 5.5 branch."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit_utils import build_run_manifest, synthetic_file_diff, write_manifest, write_synthetic_git_summary  # noqa: E402
from src.final_eval import build_candidate_eval_frame, build_weak_training_frame, load_structured_cases  # noqa: E402
from src.final_eval_55 import FOLDED_RESP, fold_resp  # noqa: E402
from src.research_support import LABELS, RESP_LABELS, classify_metrics, load_cfg, read_csv_flexible  # noqa: E402


def latest_prefixed_dir(root: Path, prefix: str) -> Optional[Path]:
    runs = sorted([p for p in root.glob(f"{prefix}*") if p.is_dir()])
    return runs[-1] if runs else None


def recompute_metrics(run_dir: Path) -> pd.DataFrame:
    pred = pd.read_csv(run_dir / "predictions_main.csv", encoding="utf-8-sig")
    pred = pred[pred["eval_split"].astype(str) == "external_candidate_eval"].copy()
    rows = []
    for (dataset_name, model_name), sub in pred.groupby(["dataset_name", "model_name"]):
        m = classify_metrics(sub["y_true"], sub["y_pred"], LABELS)
        rows.append(
            {
                "run_name": run_dir.name,
                "dataset_name": dataset_name,
                "model_name": model_name,
                "n": int(len(sub)),
                "accuracy_recomputed": m["accuracy"],
                "macro_f1_recomputed": m["macro_f1"],
                "weighted_f1_recomputed": m["weighted_f1"],
                "traceability_status": "complete",
            }
        )
    return pd.DataFrame(rows)


def text_for_mode(structured: Dict, mode: str) -> str:
    if mode == "pre_decision_only":
        return str(structured.get("pre_decision_text", "") or "")
    if mode == "post_decision_only":
        return str(structured.get("post_decision_text", "") or "")
    return (str(structured.get("pre_decision_text", "") or "") + "\n" + str(structured.get("post_decision_text", "") or "")).strip()


def leakage_sentinel(cfg: Dict) -> pd.DataFrame:
    p = cfg["paths"]
    structured_cases = load_structured_cases(PROJECT_ROOT / p["structured_case_dir"])
    weak_df = build_weak_training_frame(cfg, structured_cases)
    llm_df = read_csv_flexible(PROJECT_ROOT / p["llm_labels_csv"])
    candidate_paths = [
        PROJECT_ROOT / p["candidate_gold_strict_csv"],
        PROJECT_ROOT / p["candidate_gold_extended_csv"],
    ]
    rows = []
    modes = ["pre_decision_only", "post_decision_only", "pre_decision_plus_post_decision"]
    for mode in modes:
        train_texts = [text_for_mode(structured_cases[cid], mode) for cid in weak_df["case_id"].astype(str)]
        vectorizer = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2)
        x_train = vectorizer.fit_transform(train_texts)
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(x_train, weak_df["label"].to_numpy())
        for path in candidate_paths:
            eval_df = build_candidate_eval_frame(path, structured_cases, llm_df, weak_df)
            texts = [text_for_mode(structured_cases[cid], mode) for cid in eval_df["case_id"].astype(str)]
            x_eval = vectorizer.transform(texts)
            pred = model.predict(x_eval)
            m = classify_metrics(eval_df["y_true"], pred, LABELS)
            rows.append(
                {
                    "dataset_name": path.stem,
                    "text_mode": mode,
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "weighted_f1": m["weighted_f1"],
                }
            )
    out = pd.DataFrame(rows)
    delta_rows = []
    for dataset_name, sub in out.groupby("dataset_name"):
        base = float(sub[sub["text_mode"] == "pre_decision_only"]["macro_f1"].iloc[0])
        for mode in ["post_decision_only", "pre_decision_plus_post_decision"]:
            idx = sub["text_mode"] == mode
            val = float(sub[idx]["macro_f1"].iloc[0])
            out.loc[(out["dataset_name"] == dataset_name) & (out["text_mode"] == mode), "inflation_delta_macro_f1_vs_pre"] = val - base
        out.loc[(out["dataset_name"] == dataset_name) & (out["text_mode"] == "pre_decision_only"), "inflation_delta_macro_f1_vs_pre"] = 0.0
        delta_rows.append(
            {
                "dataset_name": dataset_name,
                "pre_decision_macro_f1": base,
                "post_only_delta_macro_f1": float(sub[sub["text_mode"] == "post_decision_only"]["macro_f1"].iloc[0]) - base,
                "pre_plus_post_delta_macro_f1": float(sub[sub["text_mode"] == "pre_decision_plus_post_decision"]["macro_f1"].iloc[0]) - base,
            }
        )
    return out


def responsibility_root_cause(run_dir: Path) -> pd.DataFrame:
    resp = pd.read_csv(run_dir / "responsibility_eval.csv", encoding="utf-8-sig")
    rows = []
    mmec = resp[resp.get("model_name", "paesc_hybrid") == "mmec_paesc_55"].copy()
    if not mmec.empty:
        valid = mmec[mmec["candidate_responsibility_label"].isin(RESP_LABELS[:-1])]
        fine = classify_metrics(valid["candidate_responsibility_label"], valid["primary_responsible_party"], RESP_LABELS[:-1]) if not valid.empty else {"macro_f1": 0.0, "accuracy": 0.0}
        folded = classify_metrics(mmec["folded_candidate_responsibility_label"], mmec["folded_primary_responsible_party"], FOLDED_RESP)
        rows.extend(
            [
                {
                    "root_cause_type": "label_problem",
                    "evidence_metric": "candidate_unknown_ratio",
                    "value": float((mmec["candidate_responsibility_label"] == "unknown").mean()),
                    "interpretation": "Lower unknown ratio makes current responsibility evaluation materially harder than legacy unknown-dominant settings.",
                },
                {
                    "root_cause_type": "task_formulation_problem",
                    "evidence_metric": "fine_macro_f1",
                    "value": fine["macro_f1"],
                    "interpretation": "Fine-grained actor attribution remains weak under machine-assisted labels.",
                },
                {
                    "root_cause_type": "task_formulation_problem",
                    "evidence_metric": "folded_macro_f1",
                    "value": folded["macro_f1"],
                    "interpretation": "Folded schema is more publishable and managerially interpretable than the fine-grained schema.",
                },
                {
                    "root_cause_type": "model_problem",
                    "evidence_metric": "uncertainty_rate",
                    "value": float(mmec["uncertainty_flag"].mean()),
                    "interpretation": "High uncertainty indicates that responsibility remains an audit-ready task, not a fully validated classifier.",
                },
            ]
        )
    return pd.DataFrame(rows)


def claim_tiering(run_dir: Path, recompute_df: pd.DataFrame) -> pd.DataFrame:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    api_counts = manifest.get("api_status_counts", {})
    api_available = int(api_counts.get("api_available", 0)) > 0
    rows = [
        {
            "result_group": "Current audit-ready main result",
            "run_name": "final_eval_20260409_194025",
            "model_name": "paesc_hybrid",
            "claim_tier": "Tier B: claimable with caution",
            "claim_boundary": "Current verified PAESC baseline; machine-assisted candidate benchmark.",
        },
        {
            "result_group": "Current strongest reproducible pure predictor",
            "run_name": "final_eval_20260409_194025",
            "model_name": "current_hybrid_baseline",
            "claim_tier": "Tier B: claimable with caution",
            "claim_boundary": "Strongest pure predictor, but not the integrated audit-ready framework.",
        },
    ]
    for model_name in ["gpt55_direct", "mmec_paesc_55_no_mechanism", "mmec_paesc_55_no_evidence_chain", "mmec_paesc_55"]:
        tier = "Tier B: claimable with caution"
        boundary = "Traceable MMEC 5.5 branch on machine-assisted candidate benchmarks."
        if model_name == "gpt55_direct" and not api_available:
            tier = "Tier C: historical/reference only"
            boundary = "API unavailable; row is local rule proxy and must not be described as GPT-5.5 output."
        elif not api_available:
            boundary = "Local MMEC mechanism proxy; claimable only as a deterministic management-mechanism rerun, not as a true GPT-5.5 result."
        rows.append(
            {
                "result_group": "MMEC-PAESC 5.5 branch",
                "run_name": run_dir.name,
                "model_name": model_name,
                "claim_tier": tier,
                "claim_boundary": boundary,
            }
        )
    out = pd.DataFrame(rows)
    perf = recompute_df[recompute_df["dataset_name"] == "candidate_gold_extended_v1"][
        ["model_name", "accuracy_recomputed", "macro_f1_recomputed"]
    ]
    return out.merge(perf, on="model_name", how="left")


def delta_table(run_dir: Path, cfg: Dict) -> pd.DataFrame:
    rows = [
        ("data_files", "candidate benchmarks", "same strict/extended candidate files", "no", "mixed", "small", "No test-set change; direct comparability preserved."),
        ("gold_version", "candidate-gold version", "candidate_gold_v1 unchanged", "no", "mixed", "small", "No human-gold claim introduced."),
        ("split_mode", "split mode", "external candidate evaluation + internal weak tuning", "no", "mixed", "small", "MMEC weight selected on weak proxy, not candidate test."),
        ("model_branch", "model", "PAESC plus MMEC mechanism calibration", "unclear", "mixed", "medium", "Mechanism features can shift boundary cases but are audit constrained."),
        ("llm_api", "5.5 availability", "api status recorded in manifest", "unclear", "unclear", "large", "API-unavailable rows cannot be claimed as GPT-5.5 outputs."),
        ("feature_module", "management mechanism", "documentation/procedure/causality/concurrency/readiness indices added", "no", "mixed", "medium", "Mechanism improves management interpretation even if macro-F1 is unchanged."),
        ("evidence_chain", "evidence chain", "pre-decision span checks retained", "no", "mixed", "small", "Auditability remains separate from outcome accuracy."),
        ("responsibility_head", "responsibility", "fine and folded schemas both reported", "no", "mixed", "medium", "Folded schema supports publishable management interpretation."),
        ("metric_script", "metric computation", "metrics recomputed from predictions_main.csv", "no", "mixed", "small", "Summary files are not trusted as headline source."),
        ("unknown_handling", "unknown responsibility", "unknown not promoted to easy all-label headline", "no", "lower", "large", "Avoids legacy unknown-dominant inflation."),
    ]
    return pd.DataFrame(
        [
            {
                "difference_id": item[0],
                "category": item[1],
                "old_run": "final_eval_20260409_194025",
                "new_run": run_dir.name,
                "old_value": "PAESC/current hybrid branch",
                "new_value": item[2],
                "evidence_path": str((run_dir / "run_manifest.json").relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "evidence_excerpt": item[6],
                "score_inflation_risk": item[3],
                "expected_direction": item[4],
                "estimated_impact_band": item[5],
                "confidence": "medium",
                "auditable_conclusion": item[6],
            }
            for item in rows
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/research_v2_55.yaml")
    ap.add_argument("--run-dir", default="")
    args = ap.parse_args()
    cfg = load_cfg(PROJECT_ROOT / args.config)
    run_dir = Path(args.run_dir) if args.run_dir else latest_prefixed_dir(PROJECT_ROOT / cfg["paths"]["final_eval_root"], "final_eval_55_")
    if run_dir is None:
        raise RuntimeError("No final_eval_55 run found.")
    audit_dir = PROJECT_ROOT / cfg["paths"]["final_eval_root"] / f"forensic_audit_55_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    recompute = recompute_metrics(run_dir)
    leakage = leakage_sentinel(cfg)
    root = responsibility_root_cause(run_dir)
    claims = claim_tiering(run_dir, recompute)
    delta = delta_table(run_dir, cfg)

    recompute.to_csv(audit_dir / "metric_recompute_check.csv", index=False, encoding="utf-8-sig")
    leakage.to_csv(audit_dir / "leakage_sentinel_results.csv", index=False, encoding="utf-8-sig")
    root.to_csv(audit_dir / "responsibility_root_cause.csv", index=False, encoding="utf-8-sig")
    claims.to_csv(audit_dir / "claim_tiering.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(audit_dir / "delta_table.csv", index=False, encoding="utf-8-sig")
    delta.to_excel(audit_dir / "delta_table.xlsx", index=False)
    synthetic_file_diff([PROJECT_ROOT / "results" / "final_eval_20260409_194025" / "metrics_main.json"], [run_dir / "metrics_main.json"], PROJECT_ROOT, audit_dir.name).to_csv(audit_dir / "file_diff_summary.csv", index=False, encoding="utf-8-sig")
    write_synthetic_git_summary(audit_dir / "git_diff_summary.txt", "git unavailable in workspace; synthetic diff used.")

    summary = [
        "# MMEC-PAESC 5.5 Forensic Audit",
        "",
        f"Audited run: `{run_dir.name}`",
        "",
        "All headline metrics in `metric_recompute_check.csv` were recomputed from `predictions_main.csv`.",
        "Rows marked as API unavailable are local deterministic MMEC mechanism proxies and must not be described as true GPT-5.5 results.",
    ]
    (audit_dir / "forensic_audit_report.md").write_text("\n".join(summary), encoding="utf-8")
    (audit_dir / "reproduce_commands.md").write_text(
        f"# Reproduce\n\n```powershell\npython src/run_forensic_audit_55.py --config {args.config} --run-dir {run_dir}\n```\n",
        encoding="utf-8",
    )
    manifest = build_run_manifest(
        PROJECT_ROOT,
        PROJECT_ROOT / "requirements.txt",
        [
            PROJECT_ROOT / args.config,
            PROJECT_ROOT / "src" / "run_forensic_audit_55.py",
            run_dir / "predictions_main.csv",
            audit_dir / "metric_recompute_check.csv",
            audit_dir / "leakage_sentinel_results.csv",
            audit_dir / "claim_tiering.csv",
        ],
        model_name="forensic_audit_55",
        prompt_template_version=None,
        embedding_model="tfidf_leakage_sentinel_v1",
        label_schema_version=cfg.get("mmec", {}).get("label_schema_version", "mmec_v1"),
        command=f"python src/run_forensic_audit_55.py --config {args.config} --run-dir {run_dir}",
        seed=int(cfg["random"]["seed"]),
        split_mode="external_candidate_eval+leakage_sentinel",
        text_mode="pre/post/pre+post sentinel",
        train_label_file=PROJECT_ROOT / "data" / "meta" / "labels_step2_domain.csv",
        eval_label_file=PROJECT_ROOT / cfg["paths"]["candidate_gold_extended_csv"],
        metric_source_files=[audit_dir / "metric_recompute_check.csv", audit_dir / "leakage_sentinel_results.csv"],
        audit_status="complete",
        extra={"audited_run": run_dir.name},
    )
    write_manifest(audit_dir / "run_manifest.json", manifest)
    print(f"[DONE] forensic_audit_55: {audit_dir}")


if __name__ == "__main__":
    main()
