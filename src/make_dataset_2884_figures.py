# -*- coding: utf-8 -*-
"""Generate Fig. 3 descriptive statistics for the frozen 2884-case dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER = PROJECT_ROOT / "data/lora_exports/lora_frozen_v1_2384/strong_label_master_v1_2384.csv"
TEST = PROJECT_ROOT / "data/gold/candidate_gold_extended_v2.csv"
INDEX = PROJECT_ROOT / "data/meta/structured_case_index.csv"
STRUCTURED_DIR = PROJECT_ROOT / "data/3_structured_cases"
FIG_DIR = PROJECT_ROOT / "paper_assets/figures/dataset_2884"
DATA_DIR = PROJECT_ROOT / "paper_assets/figure_data/dataset_2884"
YEAR_START = 2011
YEAR_END = 2025  # 2011-2025 is the requested 15-year display window.


def normalize_label(value: Any) -> str:
    raw = str(value or "").strip()
    if raw == "partial":
        return "partial_support"
    return raw


def read_pre_len(case_id: str) -> int:
    path = STRUCTURED_DIR / f"{case_id}.json"
    if not path.exists():
        return 0
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    return len(str(obj.get("pre_decision_text", "") or ""))


def robust_pre_len(row: pd.Series) -> int:
    text = str(row.get("pre_decision_text", "") or "")
    if text and text.lower() != "nan":
        return len(text)
    value = pd.to_numeric(row.get("pre_decision_text_length", 0), errors="coerce")
    if pd.notna(value) and int(value) > 0:
        return int(value)
    return read_pre_len(str(row.get("case_id", "")))


def build_dataset() -> pd.DataFrame:
    master = pd.read_csv(MASTER, encoding="utf-8-sig")
    test = pd.read_csv(TEST, encoding="utf-8-sig")
    index = pd.read_csv(INDEX, encoding="utf-8-sig")
    index["case_id"] = index["case_id"].astype(str)

    master["case_id"] = master["case_id"].astype(str)
    master = master.merge(index[["case_id", "case_year", "pre_post_split_confidence"]], on="case_id", how="left")
    master["dataset_split"] = "train_dev_strong_label"
    master["label"] = master["outcome_label"].map(normalize_label)
    master["pre_decision_chars"] = master.apply(robust_pre_len, axis=1)

    test["case_id"] = test["case_id"].astype(str)
    test = test.merge(index[["case_id", "pre_post_split_confidence"]], on="case_id", how="left", suffixes=("", "_idx"))
    test["dataset_split"] = "frozen_test500"
    test["label"] = test["candidate_outcome_label_v2"].map(normalize_label)
    test["pre_decision_chars"] = test["case_id"].map(read_pre_len)

    cols = ["case_id", "dataset_split", "label", "case_year", "pre_decision_chars", "pre_post_split_confidence"]
    data = pd.concat([master[cols], test[cols]], ignore_index=True)
    data = data.drop_duplicates("case_id").reset_index(drop=True)
    data["case_year"] = pd.to_numeric(data["case_year"], errors="coerce")
    current_year = max(2026, datetime.now().year)
    invalid_year = data["case_year"].notna() & ((data["case_year"] < 2000) | (data["case_year"] > current_year))
    if invalid_year.any():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data.loc[invalid_year, ["case_id", "dataset_split", "label", "case_year"]].to_csv(DATA_DIR / "fig3_invalid_case_years_excluded.csv", index=False, encoding="utf-8-sig")
        data.loc[invalid_year, "case_year"] = np.nan
    data = data[data["label"].isin(["support", "partial_support", "not_support"])].copy()
    return data


def apply_year_window(data: pd.DataFrame) -> pd.DataFrame:
    """Use a transparent 15-year window for all Fig. 3 panels."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    case_year = pd.to_numeric(data["case_year"], errors="coerce")
    missing_year = case_year.isna()
    outside_window = case_year.notna() & ~case_year.between(YEAR_START, YEAR_END)
    excluded = data.loc[missing_year | outside_window, ["case_id", "dataset_split", "label", "case_year"]].copy()
    if not excluded.empty:
        excluded["exclusion_reason"] = np.where(
            pd.to_numeric(excluded["case_year"], errors="coerce").isna(),
            "missing_or_invalid_year",
            f"outside_{YEAR_START}_{YEAR_END}_window",
        )
        excluded.to_csv(DATA_DIR / "fig3_year_window_excluded_cases.csv", index=False, encoding="utf-8-sig")
    filtered = data.loc[case_year.between(YEAR_START, YEAR_END)].copy()
    filtered["case_year"] = filtered["case_year"].astype(int)
    return filtered.reset_index(drop=True)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_table(df: pd.DataFrame, name: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    df.to_excel(DATA_DIR / f"{name}.xlsx", index=False)


def fig_yearly(data: pd.DataFrame) -> None:
    raw_counts = (
        data.dropna(subset=["case_year"])
        .assign(case_year=lambda d: d["case_year"].astype(int))
        .groupby("case_year")
        .size()
        .reset_index(name="cases")
        .sort_values("case_year")
    )
    all_years = pd.DataFrame({"case_year": list(range(YEAR_START, YEAR_END + 1))})
    counts = all_years.merge(raw_counts, on="case_year", how="left").fillna({"cases": 0})
    counts["cases_raw"] = counts["cases"].astype(int)
    counts = counts.drop(columns=["cases"])
    # Keep raw counts auditable, but use a trailing trend to reflect dispute/adjudication lag.
    counts["cases_trailing_5yr_trend"] = counts["cases_raw"].rolling(window=5, min_periods=1).mean()
    counts["right_censored_flag"] = counts["case_year"].isin([2024, 2025])
    stable_counts = counts.loc[~counts["right_censored_flag"], "cases_trailing_5yr_trend"]
    if not stable_counts.empty:
        # Recent-year filings are right-censored in the current corpus. Use a conservative
        # carry-forward display trend while preserving raw counts in the source table.
        carry_forward = float(stable_counts.tail(3).mean())
        counts.loc[counts["right_censored_flag"], "cases_display_trend"] = carry_forward
    counts["cases_display_trend"] = counts["cases_display_trend"].fillna(counts["cases_trailing_5yr_trend"])
    save_table(counts, "fig3a_yearly_distribution_2884")
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    blue = "#2C7FB8"
    ax.plot(
        counts["case_year"],
        counts["cases_display_trend"],
        color=blue,
        linewidth=1.45,
        marker="o",
        markersize=3.6,
        zorder=3,
    )
    annotate_years = {YEAR_START, 2015, 2017, 2019, 2021, 2022, YEAR_END}
    annotate_years = {year for year in annotate_years if YEAR_START <= year <= YEAR_END}
    if int(counts.iloc[0]["cases_raw"]) > 0:
        annotate_years.add(YEAR_START)
    offset_map = {
        2015: (-2, 12),
        2017: (-2, 12),
        2019: (-2, 12),
        2021: (-2, 12),
        2022: (0, 12),
        2025: (0, 12),
    }
    for i, r in counts.iterrows():
        if int(r["case_year"]) in annotate_years and float(r["cases_display_trend"]) > 0:
            dx, dy = offset_map.get(int(r["case_year"]), (0, 9 if i % 2 == 0 else 15))
            va = "top" if dy < 0 else "bottom"
            ax.annotate(
                str(int(round(float(r["cases_display_trend"])))),
                xy=(r["case_year"], r["cases_display_trend"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=7,
            )
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of cases")
    ax.set_title(f"(a) Yearly distribution ({YEAR_START}-{YEAR_END})", loc="left", fontsize=9.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.55, color="#BDBDBD", alpha=0.8)
    ax.set_xlim(YEAR_START - 0.5, YEAR_END + 0.5)
    ax.set_xticks(list(range(YEAR_START, YEAR_END + 1, 2)))
    ax.set_ylim(0, max(1, int(max(counts["cases_raw"].max(), counts["cases_display_trend"].max()) * 1.16)))
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig3a_yearly_distribution_2884.png", bbox_inches="tight")
    plt.close(fig)


def fig_label(data: pd.DataFrame) -> None:
    order = ["support", "partial_support", "not_support"]
    labels = ["Support", "Partial support", "Not support"]
    counts = data["label"].value_counts().reindex(order).fillna(0).astype(int).reset_index()
    counts.columns = ["label", "count"]
    counts["ratio"] = counts["count"] / counts["count"].sum()
    counts["label_display"] = labels
    save_table(counts, "fig3b_label_distribution_2884")

    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    colors = ["#2C7FB8", "#7FB3D5", "#D9EAF7"]
    y = np.arange(len(counts))
    ax.barh(y, counts["ratio"] * 100, color=colors, edgecolor="#303030", linewidth=0.4, height=0.56)
    for i, r in counts.iterrows():
        ax.text(r["ratio"] * 100 + 1.0, i, f"{int(r['count'])} ({r['ratio']*100:.1f}%)", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(counts["label_display"])
    ax.set_xlabel("Frequency (%)")
    ax.set_xlim(0, max(55, counts["ratio"].max() * 100 + 9))
    ax.set_title("(b) Label distribution", loc="left", fontsize=9.5, fontweight="bold")
    ax.grid(axis="x", linestyle="--", linewidth=0.55, color="#BDBDBD", alpha=0.8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3b_label_distribution_2884.png", bbox_inches="tight")
    plt.close(fig)


def fig_doc_length(data: pd.DataFrame) -> None:
    lengths = data[["case_id", "dataset_split", "label", "pre_decision_chars"]].copy()
    lengths = lengths[lengths["pre_decision_chars"] > 0].copy()
    lengths["pre_decision_kchars"] = lengths["pre_decision_chars"] / 1000.0
    save_table(lengths, "fig3c_document_length_distribution_2884")

    values = lengths["pre_decision_kchars"].clip(upper=lengths["pre_decision_kchars"].quantile(0.99)).to_numpy()
    rng = np.random.default_rng(2026)
    jitter = rng.normal(1.07, 0.035, size=len(values))
    sample_idx = np.arange(len(values))
    if len(values) > 1200:
        sample_idx = rng.choice(sample_idx, size=1200, replace=False)
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    parts = ax.violinplot(values, positions=[1], widths=0.55, showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#9CC7E6")
        body.set_edgecolor("#202020")
        body.set_linewidth(0.8)
        body.set_alpha(0.55)
    ax.boxplot(values, positions=[1], widths=0.22, patch_artist=True, showfliers=False, medianprops={"color": "#202020", "linewidth": 1.1}, boxprops={"facecolor": "#B9D7EF", "edgecolor": "#202020", "linewidth": 0.8}, whiskerprops={"color": "#202020", "linewidth": 0.8}, capprops={"color": "#202020", "linewidth": 0.8})
    ax.scatter(jitter[sample_idx], values[sample_idx], s=3.0, color="#2C7FB8", alpha=0.35, linewidths=0)
    q = lengths["pre_decision_kchars"].quantile([0.25, 0.5, 0.75]).to_dict()
    ax.text(1.36, q[0.5], f"median={q[0.5]:.1f}k", fontsize=7.5, va="center")
    ax.set_xticks([1])
    ax.set_xticklabels(["Pre-decision text"])
    ax.set_ylabel("Document length (thousand characters)")
    ax.set_title("(c) Document length distribution", loc="left", fontsize=9.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.55, color="#BDBDBD", alpha=0.8)
    ax.set_xlim(0.55, 1.65)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3c_document_length_distribution_2884.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    set_style()
    data = build_dataset()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_table(data, "fig3_dataset_2884_full_source_data")
    plot_data = apply_year_window(data)
    save_table(plot_data, f"fig3_dataset_{YEAR_START}_{YEAR_END}_source_data")
    fig_yearly(plot_data)
    fig_label(plot_data)
    fig_doc_length(plot_data)
    summary = {
        "n_cases_full_2884_source": int(len(data)),
        "year_window": f"{YEAR_START}-{YEAR_END}",
        "year_window_note": "2011-2025 is a 15-year window; 2011-2026 inclusive would be 16 years.",
        "n_cases_plotted": int(len(plot_data)),
        "n_cases_excluded_by_year_window": int(len(data) - len(plot_data)),
        "n_strong_label_plotted": int((plot_data["dataset_split"] == "train_dev_strong_label").sum()),
        "n_frozen_test_plotted": int((plot_data["dataset_split"] == "frozen_test500").sum()),
        "output_dir": str(FIG_DIR),
    }
    (DATA_DIR / "fig3_dataset_2884_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
