# -*- coding: utf-8 -*-
"""Publication-style figures for the true LLM Copilot branch.

The figures are built from real artifacts in
``results/true_llm_copilot_20260519_112238``. No synthetic metrics are used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = PROJECT_ROOT / "results" / "true_llm_copilot_20260519_112238"
FIG_DIR = PROJECT_ROOT / "paper_assets" / "figures_true_llm"
DATA_DIR = PROJECT_ROOT / "paper_assets" / "figure_data_true_llm"
TABLE_DIR = PROJECT_ROOT / "paper_assets" / "tables_true_llm"

LABELS = ["support", "partial", "not_support"]
DATASET_LABELS = {
    "candidate_gold_strict_v1": "Strict candidate",
    "candidate_gold_extended_v1": "Extended candidate",
}
MODEL_LABELS = {
    "current_hybrid_baseline": "Hybrid baseline",
    "paesc_hybrid": "PAESC",
    "true_qwen_direct": "True Qwen direct",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def ensure_dirs() -> None:
    for path in [FIG_DIR, DATA_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(DATA_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    df.to_excel(DATA_DIR / f"{name}.xlsx", index=False)


def save_fig(fig: plt.Figure, name: str) -> None:
    for ext in ["png", "pdf", "svg", "tiff"]:
        dpi = 600 if ext in {"png", "tiff"} else None
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def model_comparison() -> pd.DataFrame:
    df = pd.read_csv(RUN_DIR / "model_comparison.csv", encoding="utf-8-sig")
    order = ["current_hybrid_baseline", "paesc_hybrid", "true_qwen_direct"]
    df = df[df["model_name"].isin(order)].copy()
    df["model_label"] = df["model_name"].map(MODEL_LABELS)
    df["dataset_label"] = df["dataset_name"].map(DATASET_LABELS)
    df["model_order"] = df["model_name"].map({m: i for i, m in enumerate(order)})
    df["dataset_order"] = df["dataset_name"].map({"candidate_gold_strict_v1": 0, "candidate_gold_extended_v1": 1})
    df = df.sort_values(["dataset_order", "model_order"])
    save_table(df, "fig_true1_model_comparison_macro_f1")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), sharex=True)
    colors = {
        "current_hybrid_baseline": "#4C6A92",
        "paesc_hybrid": "#5D8A72",
        "true_qwen_direct": "#A65F57",
    }
    for ax, (dataset, sub) in zip(axes, df.groupby("dataset_name", sort=False)):
        y = np.arange(len(sub))[::-1]
        for yi, (_, row) in zip(y, sub.iterrows()):
            ax.hlines(yi, 0, row["macro_f1"], color="#D7D7D7", lw=1.2, zorder=1)
            ax.plot(row["macro_f1"], yi, "o", color=colors[row["model_name"]], ms=5, zorder=2)
            label_x = min(row["macro_f1"] + 0.018, 0.735)
            ax.text(
                label_x,
                yi,
                f"{row['macro_f1']:.3f}",
                va="center",
                fontsize=6.8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.85},
            )
        ax.set_yticks(y)
        ax.set_yticklabels(sub["model_label"])
        ax.set_xlim(0, 0.75)
        ax.set_xlabel("Macro-F1")
        ax.set_title(DATASET_LABELS.get(dataset, dataset), loc="left", fontweight="bold")
        ax.grid(axis="x", color="#ECECEC", lw=0.6)
    fig.suptitle("Outcome prediction: verified true LLM branch versus reproducible baselines", x=0.02, y=1.05, ha="left", fontsize=8.5, fontweight="bold")
    fig.text(0.02, -0.03, "Metrics are recomputed from prediction-level artifacts; candidate benchmarks are machine-assisted, not human gold.", fontsize=7, color="#555555")
    save_fig(fig, "fig_true1_model_comparison_macro_f1")
    return df


def confusion_matrices() -> pd.DataFrame:
    df = pd.read_csv(RUN_DIR / "confusion_matrix_data.csv", encoding="utf-8-sig")
    df = df[df["model_name"].eq("true_qwen_direct")].copy()
    save_table(df, "fig_true2_qwen_confusion_matrices")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0))
    for ax, dataset in zip(axes, ["candidate_gold_strict_v1", "candidate_gold_extended_v1"]):
        sub = df[df["dataset_name"].eq(dataset)]
        mat = np.zeros((len(LABELS), len(LABELS)), dtype=int)
        for i, gold in enumerate(LABELS):
            for j, pred in enumerate(LABELS):
                value = sub[(sub["gold_label"].eq(gold)) & (sub["pred_label"].eq(pred))]["count"]
                mat[i, j] = int(value.iloc[0]) if not value.empty else 0
        ax.imshow(mat, cmap="Greys", vmin=0, vmax=max(1, mat.max()))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                color = "white" if mat[i, j] > mat.max() * 0.55 else "black"
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", color=color, fontsize=8)
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(["support", "partial", "not"], rotation=35, ha="right")
        ax.set_yticklabels(["support", "partial", "not"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Candidate label")
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
    fig.suptitle("True Qwen direct: class-confusion pattern", x=0.02, y=1.03, ha="left", fontsize=9, fontweight="bold")
    fig.text(0.02, -0.03, "The direct LLM output over-predicts partial support and under-recognizes not-support cases.", fontsize=7, color="#555555")
    save_fig(fig, "fig_true2_qwen_confusion_matrices")
    return df


def evidence_auditability() -> pd.DataFrame:
    df = pd.read_csv(RUN_DIR / "evidence_chain_eval.csv", encoding="utf-8-sig")
    metric_cols = ["valid_span_rate", "pre_decision_span_rate", "role_coverage_rate", "duplicate_chain_rate"]
    summary = df.groupby("dataset_name")[metric_cols].mean().reset_index()
    long_df = summary.melt(id_vars="dataset_name", var_name="metric", value_name="value")
    long_df["dataset_label"] = long_df["dataset_name"].map(DATASET_LABELS)
    save_table(long_df, "fig_true3_evidence_auditability")

    metric_labels = {
        "valid_span_rate": "Valid span",
        "pre_decision_span_rate": "Pre-decision span",
        "role_coverage_rate": "Role coverage",
        "duplicate_chain_rate": "Duplicate rate",
    }
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    x = np.arange(len(metric_cols))
    width = 0.34
    palette = {"candidate_gold_strict_v1": "#4C6A92", "candidate_gold_extended_v1": "#A8A8A8"}
    for offset, dataset in zip([-width / 2, width / 2], ["candidate_gold_strict_v1", "candidate_gold_extended_v1"]):
        vals = [summary[summary["dataset_name"].eq(dataset)][m].iloc[0] for m in metric_cols]
        bars = ax.bar(x + offset, vals, width=width, color=palette[dataset], label=DATASET_LABELS[dataset], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate")
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m] for m in metric_cols], rotation=15, ha="right")
    ax.set_title("Evidence-chain auditability in the true LLM Copilot branch", loc="left", fontweight="bold")
    ax.legend(ncol=2, loc="upper right")
    ax.grid(axis="y", color="#ECECEC", lw=0.6)
    save_fig(fig, "fig_true3_evidence_auditability")
    return long_df


def responsibility_and_management() -> pd.DataFrame:
    resp = pd.read_csv(RUN_DIR / "responsibility_summary.csv", encoding="utf-8-sig")
    mech = pd.read_csv(RUN_DIR / "managerial_mechanisms.csv", encoding="utf-8-sig")
    mech_cols = [
        "documentation_gap_index",
        "procedural_compliance_risk",
        "causality_ambiguity",
        "concurrency_risk",
        "critical_path_support",
        "negotiation_readiness_score",
    ]
    mech_summary = mech.groupby("dataset_name")[mech_cols].mean().reset_index()
    resp_long = resp.melt(
        id_vars="dataset_name",
        value_vars=["fine_macro_f1", "folded_macro_f1", "uncertainty_rate", "evidence_consistency_rate"],
        var_name="metric",
        value_name="value",
    )
    mech_long = mech_summary.melt(id_vars="dataset_name", var_name="metric", value_name="value")
    out = pd.concat([resp_long.assign(panel="responsibility"), mech_long.assign(panel="management")], ignore_index=True)
    out["dataset_label"] = out["dataset_name"].map(DATASET_LABELS)
    save_table(out, "fig_true4_responsibility_management")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    left_metrics = ["fine_macro_f1", "folded_macro_f1", "uncertainty_rate", "evidence_consistency_rate"]
    left_labels = ["Fine F1", "Folded F1", "Uncertainty", "Evidence\nconsistency"]
    x = np.arange(len(left_metrics))
    width = 0.34
    colors = ["#4C6A92", "#A8A8A8"]
    for idx, dataset in enumerate(["candidate_gold_strict_v1", "candidate_gold_extended_v1"]):
        vals = [resp[resp["dataset_name"].eq(dataset)][m].iloc[0] for m in left_metrics]
        axes[0].bar(x + (idx - 0.5) * width, vals, width=width, color=colors[idx], label=DATASET_LABELS[dataset], edgecolor="white", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(left_labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Metric value")
    axes[0].set_title("(a) Responsibility diagnosis", loc="left", fontweight="bold")
    axes[0].legend(loc="upper left", fontsize=6.5)
    axes[0].grid(axis="y", color="#ECECEC", lw=0.6)

    mech_vals = mech_summary[mech_summary["dataset_name"].eq("candidate_gold_extended_v1")][mech_cols].iloc[0].values
    y = np.arange(len(mech_cols))[::-1]
    axes[1].barh(y, mech_vals, color="#5D8A72", edgecolor="white", linewidth=0.5)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(["Doc. gap", "Procedure risk", "Causality ambiguity", "Concurrency risk", "Critical-path support", "Negotiation readiness"])
    axes[1].set_xlim(0, 0.75)
    axes[1].set_xlabel("Mean score, extended candidate")
    axes[1].set_title("(b) Management mechanism indicators", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color="#ECECEC", lw=0.6)
    for yi, val in zip(y, mech_vals):
        axes[1].text(val + 0.015, yi, f"{val:.2f}", va="center", fontsize=7)

    fig.suptitle("Responsibility remains difficult, while management mechanism outputs are structured", x=0.02, y=1.04, ha="left", fontsize=9, fontweight="bold")
    save_fig(fig, "fig_true4_responsibility_management")
    return out


def summary_tables() -> None:
    metrics = json.loads((RUN_DIR / "metrics_main.json").read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = []
    for dataset, obj in metrics["candidate_gold_evaluation"].items():
        m = obj["true_qwen_direct"]
        rows.append(
            {
                "dataset_name": dataset,
                "model_name": "true_qwen_direct",
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "api_success_rate": m["api_success_rate"],
                "audit_status": m["audit_status"],
            }
        )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table_true_llm_main_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_excel(TABLE_DIR / "table_true_llm_main_metrics.xlsx", index=False)
    pd.read_csv(RUN_DIR / "model_comparison.csv", encoding="utf-8-sig").to_excel(TABLE_DIR / "table_true_llm_model_comparison.xlsx", index=False)


def main() -> None:
    setup_style()
    ensure_dirs()
    model_comparison()
    confusion_matrices()
    evidence_auditability()
    responsibility_and_management()
    summary_tables()
    print(f"Saved true LLM figures to {FIG_DIR}")
    print(f"Saved figure source data to {DATA_DIR}")


if __name__ == "__main__":
    main()
