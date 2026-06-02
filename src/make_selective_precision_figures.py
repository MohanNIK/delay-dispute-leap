# -*- coding: utf-8 -*-
"""Nature-style figures for selective high-confidence prediction results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = ["support", "partial", "not_support"]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "legend.frameon": False,
            "figure.dpi": 120,
        }
    )


def save_pub(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def read_inputs(result_dir: Path) -> Dict[str, pd.DataFrame]:
    data = {
        "coverage": pd.read_csv(result_dir / "selective_coverage_curve.csv", encoding="utf-8-sig"),
        "subsets": pd.read_csv(result_dir / "target_accuracy_subsets.csv", encoding="utf-8-sig"),
        "comparison": pd.read_csv(result_dir / "model_comparison.csv", encoding="utf-8-sig"),
    }
    high90 = result_dir / "high_conf_subset_90.csv"
    data["high90"] = pd.read_csv(high90, encoding="utf-8-sig") if high90.exists() else pd.DataFrame()
    return data


def write_source_data(data: Dict[str, pd.DataFrame], data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_csv(data_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(data_dir / "figure_source_data.xlsx", engine="openpyxl") as writer:
        for name, df in data.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


def subset_summary(subsets: pd.DataFrame) -> pd.DataFrame:
    cols = ["subset_name", "target_accuracy", "n", "coverage", "accuracy", "macro_f1", "weighted_f1", "threshold"]
    if subsets.empty:
        return pd.DataFrame(columns=cols)
    return subsets[cols].drop_duplicates().sort_values("target_accuracy", ascending=False).reset_index(drop=True)


def plot_coverage_accuracy(data: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    curve = data["coverage"].copy()
    summary = subset_summary(data["subsets"])
    fig, ax = plt.subplots(figsize=(3.55, 2.55))
    ax.plot(curve["actual_coverage"], curve["accuracy"], color="#2B6C8A", lw=1.5, marker="o", ms=3.2)
    ax.axhline(0.90, color="#8A3B2B", lw=0.8, ls="--")
    ax.axhline(0.85, color="#777777", lw=0.7, ls=":")
    for _, row in summary.iterrows():
        if row["n"] <= 0:
            continue
        ax.scatter(row["coverage"], row["accuracy"], s=28, color="#B46A3C", zorder=4)
        label = f"{int(row['n'])} cases\nacc={row['accuracy']:.3f}"
        ax.annotate(label, (row["coverage"], row["accuracy"]), xytext=(4, 6), textcoords="offset points", fontsize=6)
    ax.set_xlabel("Selective coverage")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.05, 1.02)
    ax.set_ylim(0.62, 1.01)
    ax.set_title("Risk-controlled selective prediction", loc="left", fontsize=7.5, fontweight="bold")
    ax.text(0.07, 0.635, "Full 500 accuracy remains reported separately", fontsize=5.8, color="#555555")
    save_pub(fig, out_dir / "fig_selective_coverage_accuracy")


def plot_high90_confusion(data: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    high90 = data["high90"].copy()
    if high90.empty:
        return
    mat = np.zeros((3, 3), dtype=int)
    for i, gold in enumerate(LABELS):
        for j, pred in enumerate(LABELS):
            mat[i, j] = int(((high90["y_true"] == gold) & (high90["y_pred"] == pred)).sum())
    fig, ax = plt.subplots(figsize=(2.85, 2.65))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(3), LABELS, rotation=35, ha="right")
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Candidate label")
    for i in range(3):
        for j in range(3):
            color = "white" if mat[i, j] > mat.max() * 0.55 else "#222222"
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=7, color=color)
    acc = float(high90["correct"].mean()) if "correct" in high90 else float((high90["y_true"] == high90["y_pred"]).mean())
    ax.set_title(f"0.90+ subset confusion (n={len(high90)}, acc={acc:.3f})", loc="left", fontsize=7.5, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    save_pub(fig, out_dir / "fig_high_conf_90_confusion")


def draw_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, fc: str, ec: str = "#333333") -> None:
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec=ec, lw=0.7)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=7)


def plot_human_ai_triage(data: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    summary = subset_summary(data["subsets"])
    row90 = summary[summary["target_accuracy"].round(3).eq(0.9)].head(1)
    n90 = int(row90["n"].iloc[0]) if not row90.empty else 0
    acc90 = float(row90["accuracy"].iloc[0]) if not row90.empty else 0.0
    hard = 500 - n90
    fig, ax = plt.subplots(figsize=(4.9, 2.45))
    ax.set_axis_off()
    draw_box(ax, (0.02, 0.58), 0.18, 0.2, "500 fixed\ncases", "#F4F4F4")
    draw_box(ax, (0.30, 0.72), 0.23, 0.18, f"High-confidence\nAI triage\nn={n90}", "#DDEBF2", "#2B6C8A")
    draw_box(ax, (0.30, 0.28), 0.23, 0.18, f"Hard cases\nhuman review\nn={hard}", "#F5E8DD", "#B46A3C")
    draw_box(ax, (0.65, 0.72), 0.28, 0.18, f"Selective prediction\naccuracy={acc90:.3f}", "#EAF3EA", "#4F7D5A")
    draw_box(ax, (0.65, 0.28), 0.28, 0.18, "Evidence completion\nResponsibility review\nNegotiation prep", "#FAF7E8", "#8A7A2B")
    for y in [0.78, 0.38]:
        ax.annotate("", xy=(0.30, y), xytext=(0.20, 0.68), arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"))
        ax.annotate("", xy=(0.65, y), xytext=(0.53, y), arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"))
    ax.text(0.02, 0.08, "Decision rule: high selective score -> automated decision support; low score -> expert review.", fontsize=6.2, color="#444444")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_pub(fig, out_dir / "fig_human_ai_triage")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="results/selective_precision_90_20260521")
    parser.add_argument("--figure-dir", default="paper_assets/figures/selective_precision")
    parser.add_argument("--data-dir", default="paper_assets/figure_data/selective_precision")
    args = parser.parse_args(argv)

    result_dir = PROJECT_ROOT / args.result_dir
    fig_dir = PROJECT_ROOT / args.figure_dir
    data_dir = PROJECT_ROOT / args.data_dir
    setup_style()
    data = read_inputs(result_dir)
    write_source_data(data, data_dir)
    plot_coverage_accuracy(data, fig_dir)
    plot_high90_confusion(data, fig_dir)
    plot_human_ai_triage(data, fig_dir)
    qa = {
        "backend": "Python/matplotlib",
        "archetype": "quantitative grid + schematic-led triage panel",
        "core_conclusion": "Risk-controlled selective prediction reaches 0.90+ accuracy on a smaller high-confidence subset while full-500 performance remains separately reported.",
        "exports": sorted(str(p.relative_to(PROJECT_ROOT)) for p in fig_dir.glob("fig_selective_coverage_accuracy.*"))
        + sorted(str(p.relative_to(PROJECT_ROOT)) for p in fig_dir.glob("fig_high_conf_90_confusion.*"))
        + sorted(str(p.relative_to(PROJECT_ROOT)) for p in fig_dir.glob("fig_human_ai_triage.*")),
    }
    (data_dir / "figure_qa_notes.json").write_text(__import__("json").dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FIGURE_DIR={fig_dir}")
    print(f"FIGURE_DATA_DIR={data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
