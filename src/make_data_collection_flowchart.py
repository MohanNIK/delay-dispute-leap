# -*- coding: utf-8 -*-
"""Generate a publication-style data collection and screening flowchart."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "paper_assets/figures/dataset_flow"
DATA_DIR = PROJECT_ROOT / "paper_assets/figure_data/dataset_flow"

INITIAL_CASES = 4592
FINAL_CASES = 2892
EXCLUDED_CASES = INITIAL_CASES - FINAL_CASES


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 8.2,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def rounded_box(ax, xy, width, height, text, facecolor, edgecolor="#2F2F2F", fontsize=7.6, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.016,rounding_size=0.018",
        linewidth=0.75,
        edgecolor=edgecolor,
        facecolor=facecolor,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#222222",
        linespacing=1.18,
    )
    return patch


def arrow(ax, start, end, color="#404040", lw=0.85):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=lw,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)
    return arr


def export_source_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "step": 1,
            "stage": "Data source",
            "description": "Public adjudicated case documents from China Judgments Online",
            "count": "",
        },
        {
            "step": 2,
            "stage": "Keyword retrieval",
            "description": "Construction project; construction contract; schedule delay; extension of time; liquidated damages",
            "count": "",
        },
        {
            "step": 3,
            "stage": "Initial retrieval",
            "description": "Structured adjudicated cases retrieved before screening",
            "count": INITIAL_CASES,
        },
        {
            "step": 4,
            "stage": "Screening and de-identification",
            "description": "Retain substantive delay disputes; remove duplicates, procedural-only records, enforcement/withdrawal cases, severely missing facts, unclear labels, and insufficient pre-decision facts; de-identify sensitive information",
            "count": f"excluded {EXCLUDED_CASES}",
        },
        {
            "step": 5,
            "stage": "Final modeling dataset",
            "description": "Screened construction schedule-delay dispute cases for descriptive analysis and supervised modeling",
            "count": FINAL_CASES,
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "fig2_data_collection_flowchart_source.csv", index=False, encoding="utf-8-sig")
    df.to_excel(DATA_DIR / "fig2_data_collection_flowchart_source.xlsx", index=False)


def draw_flowchart() -> None:
    set_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    export_source_data()

    fig, ax = plt.subplots(figsize=(7.3, 2.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue = "#DCEAF7"
    blue_dark = "#B7D4EC"
    gray = "#F3F3F3"
    green = "#DDEEDC"
    amber = "#F7E8C7"

    y = 0.55
    w = 0.145
    h = 0.25
    xs = [0.035, 0.225, 0.415, 0.615, 0.815]

    rounded_box(
        ax,
        (xs[0], y),
        w,
        h,
        "China Judgments\nOnline\npublic court\ndocuments",
        facecolor=gray,
        fontsize=7.1,
        weight="bold",
    )
    rounded_box(
        ax,
        (xs[1], y),
        w,
        h,
        "Keyword\nretrieval\nconstruction +\nschedule delay",
        facecolor=blue,
        fontsize=7.1,
        weight="bold",
    )
    rounded_box(
        ax,
        (xs[2], y),
        w,
        h,
        f"Initial\nretrieval\nn = {INITIAL_CASES:,}\nstructured cases",
        facecolor=blue_dark,
        fontsize=7.25,
        weight="bold",
    )
    rounded_box(
        ax,
        (xs[3], y),
        w,
        h,
        "Multi-stage\nscreening\nand\nde-identification",
        facecolor=amber,
        fontsize=7.1,
        weight="bold",
    )
    rounded_box(
        ax,
        (xs[4], y),
        w,
        h,
        f"Final screened\ndataset\nn = {FINAL_CASES:,}\nschedule-delay\ncases",
        facecolor=green,
        fontsize=7.25,
        weight="bold",
    )

    for i in range(4):
        arrow(ax, (xs[i] + w + 0.006, y + h / 2), (xs[i + 1] - 0.006, y + h / 2))

    # Screening detail lane.
    lane_y = 0.165
    lane_x = 0.50
    lane_w = 0.305
    lane_h = 0.205
    rounded_box(
        ax,
        (lane_x, lane_y),
        lane_w,
        lane_h,
        "Screening criteria\nsubstantive delay issue | sufficient factual record | evidence mention\nexclude duplicates, procedural-only, enforcement, withdrawal,\nseverely missing facts, unclear outcome coding",
        facecolor="#FBF8EF",
        edgecolor="#A9872B",
        fontsize=6.35,
    )
    arrow(ax, (xs[3] + w / 2, y), (lane_x + lane_w / 2, lane_y + lane_h + 0.01), color="#A9872B", lw=0.75)

    rounded_box(
        ax,
        (0.825, lane_y),
        0.13,
        lane_h,
        f"Removed /\nexcluded\nn = {EXCLUDED_CASES:,}",
        facecolor="#F4F4F4",
        edgecolor="#666666",
        fontsize=7.0,
    )
    arrow(ax, (lane_x + lane_w, lane_y + lane_h / 2), (0.825, lane_y + lane_h / 2), color="#666666", lw=0.7)

    ax.text(
        0.035,
        0.075,
        "Note: post-decision text is used for label derivation only; model inputs are restricted to pre-decision information.",
        fontsize=6.6,
        ha="left",
        va="center",
        color="#4A4A4A",
    )

    fig.savefig(FIG_DIR / "fig2_data_collection_flowchart.png", bbox_inches="tight", dpi=600)
    fig.savefig(FIG_DIR / "fig2_data_collection_flowchart.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_data_collection_flowchart.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    draw_flowchart()
    print(f"Generated flowchart at {FIG_DIR}")
    print(f"Final dataset n={FINAL_CASES}; excluded n={EXCLUDED_CASES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
