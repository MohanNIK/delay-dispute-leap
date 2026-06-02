from __future__ import annotations

import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = ROOT / "data" / "lora_exports" / "lora_strict49k_mimo_v2_final_20260528_164927" / "strong_label_master_v2.csv"
RAW_MANIFEST_PATH = ROOT / "data" / "1_raw_text" / "combined_delay_dispute_corpus_20260527" / "combined_raw_text_manifest_dedup.csv"
STRICT_POOL_PATH = ROOT / "data" / "1_raw_text" / "combined_delay_dispute_corpus_20260527" / "strict_delay_usable_manifest.csv"

FIG_DIR = ROOT / "paper_assets" / "figures" / "dataset_profile_20k"
DATA_DIR = ROOT / "paper_assets" / "figure_data" / "dataset_profile_20k"
DESKTOP = Path.home() / "Desktop"

LABEL_ORDER = ["support", "partial_support", "not_support"]
LABEL_DISPLAY = {
    "support": "Support",
    "partial_support": "Partial support",
    "not_support": "Not support",
}

BLUE = "#2F7FBD"
LIGHT_BLUE = "#88BDEB"
AMBER = "#C7A35D"
GREEN = "#72A98B"
DARK = "#1F1F1F"
GRID = "#D7DDE3"
RED = "#B36B6B"


def normalize_year(value) -> float:
    if pd.isna(value):
        return math.nan
    m = re.search(r"(19\d{2}|20\d{2})", str(value))
    if not m:
        return math.nan
    year = int(m.group(1))
    if 1900 <= year <= 2026:
        return year
    return math.nan


def regex_year(text: str) -> float:
    if not isinstance(text, str):
        return math.nan
    years = [int(x) for x in re.findall(r"(?:\(|（)?(19\d{2}|20\d{2})(?:\)|）|年)?", text[:2500])]
    years = [y for y in years if 1900 <= y <= 2026]
    if not years:
        return math.nan
    # Court case numbers and trial dates usually appear before older contract dates.
    return years[0]


def contains(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


ENGINEERING_RULES = [
    ("Residential", ["住宅", "商品房", "小区", "公寓", "安置房", "房地产", "交房", "房屋预售", "购房", "楼盘"]),
    ("Commercial", ["商业", "商铺", "酒店", "宾馆", "商场", "写字楼", "综合体", "市场", "办公"]),
    ("Public building", ["学校", "医院", "幼儿园", "政府", "教学楼", "图书馆", "体育馆", "公共建筑", "办公楼", "养老院"]),
    ("Industrial/factory", ["厂房", "工业", "工业园", "车间", "生产线", "钢结构", "矿", "煤矿", "电厂", "炭素"]),
    ("Municipal/infrastructure", ["市政", "道路", "公路", "桥梁", "桥", "隧道", "地铁", "轨道", "管网", "水利", "河道", "堤防", "污水", "供水", "燃气", "高速"]),
    ("Decoration/renovation", ["装饰装修", "装饰", "装修", "精装修", "幕墙", "门窗", "改造", "修缮", "维修工程"]),
    ("Landscape/greening", ["园林", "景观", "绿化", "园区绿化"]),
    ("Installation/MEP", ["水电安装", "安装工程", "机电", "消防", "暖通", "电气", "给排水", "电缆", "管线", "永久用电", "空调"]),
    ("General/unspecified construction", ["建设工程", "施工合同", "建筑工程", "工程款", "承包工程", "工程施工"]),
]

DELIVERY_RULES = [
    ("EPC / engineering procurement construction", ["EPC", "工程总承包", "设计采购施工"]),
    ("General construction contracting", ["施工总承包", "总承包"]),
    ("Design-build", ["设计施工", "设计-施工", "设计、施工"]),
    ("Professional subcontract", ["专业分包", "分包合同", "分包工程", "专业工程"]),
    ("Labor subcontract", ["劳务分包", "劳务合同", "劳务施工"]),
    ("Procurement / supply / processing", ["采购", "供货", "买卖合同", "材料供应", "设备租赁", "加工承揽"]),
    ("Traditional construction contract", ["建设工程施工合同", "施工合同", "承包合同"]),
    ("Other/unspecified", []),
]

CONTRACT_RULES = [
    ("Lump-sum / fixed-price contract", ["固定总价", "总价合同", "包干", "一口价", "固定价"]),
    ("Unit-price / BoQ contract", ["单价合同", "综合单价", "工程量清单", "清单计价", "按工程量"]),
    ("Adjustable / measured contract", ["据实结算", "按实结算", "按实际", "可调价", "签证结算", "变更价款"]),
    ("Cost-plus-fee contract", ["成本加酬金", "成本加"]),
    ("Procurement / processing contract", ["采购合同", "供货合同", "买卖合同", "加工承揽"]),
    ("Other/unspecified", []),
]

PHASE_RULES = [
    ("Design/change coordination", ["设计变更", "图纸", "施工图", "设计单位", "变更签证", "设计"]),
    ("Start-up / mobilization", ["开工令", "开工日期", "开工", "进场", "开工报告"]),
    ("Construction execution", ["施工期间", "施工过程", "工期", "延误", "逾期", "停工", "窝工", "进度", "关键线路"]),
    ("Completion / handover", ["竣工", "验收", "交付", "交房", "备案", "移交"]),
    ("Payment / settlement", ["工程款", "结算", "欠付", "支付", "价款", "优先受偿", "审计", "鉴定"]),
    ("Warranty / repair", ["质保", "保修", "维修", "返修", "质量维修"]),
    ("Other/unspecified", []),
]


def classify(text: str, rules: list[tuple[str, list[str]]]) -> str:
    text = "" if not isinstance(text, str) else text
    for label, keywords in rules:
        if keywords and contains(text, keywords):
            return label
    return rules[-1][0]


def build_dataset() -> pd.DataFrame:
    usecols = [
        "case_id",
        "title",
        "text_sha256",
        "outcome_label",
        "label_confidence",
        "label_model",
        "label_source",
        "pre_decision_text",
        "pre_decision_chars",
        "raw_text_path",
    ]
    master = pd.read_csv(MASTER_PATH, usecols=usecols, low_memory=False)
    raw = pd.read_csv(
        RAW_MANIFEST_PATH,
        usecols=["text_sha256", "case_year", "text_chars", "usable_tier"],
        low_memory=False,
    ).drop_duplicates("text_sha256")
    df = master.merge(raw, on="text_sha256", how="left")
    df["year_from_manifest"] = df["case_year"].map(normalize_year)
    df["year_from_text"] = (df["title"].fillna("") + " " + df["pre_decision_text"].fillna("").str[:2500]).map(regex_year)
    df["case_year_final"] = df["year_from_manifest"].fillna(df["year_from_text"])

    class_text = (df["title"].fillna("") + " " + df["pre_decision_text"].fillna("")).str[:16000]
    df["engineering_type"] = class_text.map(lambda x: classify(x, ENGINEERING_RULES))
    df["project_delivery_method"] = class_text.map(lambda x: classify(x, DELIVERY_RULES))
    df["contractual_agreement_type"] = class_text.map(lambda x: classify(x, CONTRACT_RULES))
    df["project_dispute_phase"] = class_text.map(lambda x: classify(x, PHASE_RULES))
    df["pre_decision_kchars"] = df["pre_decision_chars"].astype(float) / 1000.0
    return df


def distribution_table(series: pd.Series, order: list[str] | None = None, display_map: dict[str, str] | None = None) -> pd.DataFrame:
    counts = series.value_counts(dropna=False)
    if order is not None:
        counts = counts.reindex(order).fillna(0).astype(int)
    out = counts.rename_axis("category").reset_index(name="count")
    out["ratio"] = out["count"] / out["count"].sum()
    out["percent"] = out["ratio"] * 100
    if display_map:
        out["display"] = out["category"].map(display_map).fillna(out["category"].astype(str))
    else:
        out["display"] = out["category"].astype(str)
    return out


def attribute_profile(df: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("Engineering type", "engineering_type", [x[0] for x in ENGINEERING_RULES] + ["Other/unspecified"]),
        ("Project delivery method", "project_delivery_method", [x[0] for x in DELIVERY_RULES]),
        ("Contractual agreement type", "contractual_agreement_type", [x[0] for x in CONTRACT_RULES]),
        ("Project/dispute phase", "project_dispute_phase", [x[0] for x in PHASE_RULES]),
    ]
    rows = []
    n = len(df)
    for group, col, order in groups:
        vc = df[col].value_counts()
        for cat in order:
            if cat in vc.index:
                count = int(vc[cat])
                rows.append(
                    {
                        "dimension": group,
                        "category": cat,
                        "count": count,
                        "ratio": count / n,
                        "percent": 100 * count / n,
                    }
                )
    return pd.DataFrame(rows)


def make_excel(df: pd.DataFrame, length_df: pd.DataFrame, label_df: pd.DataFrame, year_df: pd.DataFrame, attr_df: pd.DataFrame, out_xlsx: Path) -> None:
    strict_rows = sum(1 for _ in STRICT_POOL_PATH.open("rb")) - 1 if STRICT_POOL_PATH.exists() else None
    readme = pd.DataFrame(
        [
            {"item": "figure_dataset", "value": str(MASTER_PATH)},
            {"item": "figure_dataset_rows", "value": len(df)},
            {"item": "strict_delay_usable_reference_rows", "value": strict_rows},
            {"item": "note", "value": "Figure uses final outcome-labeled strong_label_master_v2; strict_delay_usable_manifest is recorded as the strict delay-pool reference."},
            {"item": "year_panel_note", "value": "Panel c shows annual case counts for 2011-2025. The linear regression is fitted on complete observed years from 2011 to 2024; 2025 is retained as a partial-release year."},
            {"item": "leakage_note", "value": "Only metadata, outcome labels, and pre-decision text lengths/categories are summarized; no post-decision text is exported."},
        ]
    )
    rule_rows = []
    for dim, rules in [
        ("Engineering type", ENGINEERING_RULES),
        ("Project delivery method", DELIVERY_RULES),
        ("Contractual agreement type", CONTRACT_RULES),
        ("Project/dispute phase", PHASE_RULES),
    ]:
        for label, kws in rules:
            rule_rows.append({"dimension": dim, "category": label, "keywords_cn": "; ".join(kws)})
    rulebook = pd.DataFrame(rule_rows)
    year_summary = pd.DataFrame(
        [
            {"metric": "rows_total", "value": len(df)},
            {"metric": "year_non_missing", "value": int(df["case_year_final"].notna().sum())},
            {"metric": "year_panel_min", "value": int(year_df["year"].min()) if len(year_df) else ""},
            {"metric": "year_panel_max", "value": int(year_df["year"].max()) if len(year_df) else ""},
            {"metric": "linear_regression_slope_per_year", "value": float(year_df["regression_slope"].dropna().iloc[0]) if "regression_slope" in year_df and year_df["regression_slope"].notna().any() else ""},
            {"metric": "linear_regression_intercept_at_2011", "value": float(year_df["regression_intercept_at_2011"].dropna().iloc[0]) if "regression_intercept_at_2011" in year_df and year_df["regression_intercept_at_2011"].notna().any() else ""},
            {"metric": "linear_regression_r2", "value": float(year_df["regression_r2"].dropna().iloc[0]) if "regression_r2" in year_df and year_df["regression_r2"].notna().any() else ""},
        ]
    )
    case_profile = df[
        [
            "case_id",
            "title",
            "outcome_label",
            "label_confidence",
            "pre_decision_chars",
            "case_year_final",
            "engineering_type",
            "project_delivery_method",
            "contractual_agreement_type",
            "project_dispute_phase",
            "label_model",
            "label_source",
            "text_sha256",
        ]
    ].copy()
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        length_df.to_excel(writer, sheet_name="document_length", index=False)
        label_df.to_excel(writer, sheet_name="outcome_label_distribution", index=False)
        year_df.to_excel(writer, sheet_name="yearly_case_distribution", index=False)
        attr_df.to_excel(writer, sheet_name="attribute_profile", index=False)
        rulebook.to_excel(writer, sheet_name="attribute_rulebook", index=False)
        year_summary.to_excel(writer, sheet_name="year_extraction_summary", index=False)
        case_profile.to_excel(writer, sheet_name="case_profile_25032", index=False)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.edgecolor": DARK,
            "axes.linewidth": 0.9,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "figure.dpi": 160,
            "savefig.dpi": 600,
        }
    )


def plot_figure(df: pd.DataFrame, label_df: pd.DataFrame, year_df: pd.DataFrame, attr_df: pd.DataFrame, out_base: Path) -> None:
    setup_style()
    rng = np.random.default_rng(42)
    fig = plt.figure(figsize=(11.2, 10.2))
    gs = GridSpec(2, 3, height_ratios=[1.0, 2.35], hspace=0.46, wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, :])

    # (a) Document length distribution
    lengths = df["pre_decision_kchars"].dropna().clip(upper=35)
    parts = ax_a.violinplot([lengths], positions=[1], widths=0.7, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#A9CDE8")
        pc.set_edgecolor(BLUE)
        pc.set_alpha(0.45)
        pc.set_linewidth(0.8)
    ax_a.boxplot(
        [lengths],
        positions=[0.78],
        widths=0.28,
        patch_artist=True,
        boxprops=dict(facecolor="#C7DCEF", color=DARK, linewidth=0.8),
        medianprops=dict(color=DARK, linewidth=1.0),
        whiskerprops=dict(color=DARK, linewidth=0.8),
        capprops=dict(color=DARK, linewidth=0.8),
        flierprops=dict(marker="", markersize=0),
    )
    sample = lengths.sample(min(4500, len(lengths)), random_state=42)
    jitter = rng.normal(1.08, 0.04, size=len(sample))
    ax_a.scatter(jitter, sample, s=2.0, c=BLUE, alpha=0.26, linewidths=0)
    median = df["pre_decision_kchars"].median()
    ax_a.axhline(median, color=BLUE, linestyle="--", linewidth=0.8)
    ax_a.text(0.54, 32.5, f"n={len(df):,}", fontsize=7)
    ax_a.text(1.18, min(34, median + 0.8), f"median={median:.1f}", fontsize=7, color=DARK)
    ax_a.set_xlim(0.45, 1.45)
    ax_a.set_ylim(0, 35.5)
    ax_a.set_xticks([1])
    ax_a.set_xticklabels(["Cases"])
    ax_a.set_ylabel("Pre-decision length (k characters)")
    ax_a.set_title("Document-length distribution")
    ax_a.grid(axis="y", color=GRID, linestyle="--", linewidth=0.6, alpha=0.8)

    # (b) Label distribution
    label_plot = label_df.copy()
    colors = [BLUE, AMBER, GREEN]
    ypos = np.arange(len(label_plot))
    ax_b.barh(ypos, label_plot["percent"], color=colors, edgecolor=DARK, linewidth=0.5, height=0.62)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels(label_plot["display"])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Frequency (%)")
    ax_b.set_title("Outcome-label distribution")
    ax_b.set_xlim(0, max(60, label_plot["percent"].max() * 1.18))
    ax_b.grid(axis="x", color=GRID, linestyle="--", linewidth=0.6, alpha=0.8)
    for y, row in zip(ypos, label_plot.itertuples(index=False)):
        ax_b.text(row.percent + 1.0, y, f"{int(row.count):,} ({row.percent:.1f}%)", va="center", fontsize=7)

    # (c) Yearly distribution
    obs_year = year_df[~year_df["partial_release"]]
    partial_year = year_df[year_df["partial_release"]]
    ax_c.axvspan(2021 - 0.5, 2023 + 0.5, color="#F4D6A0", alpha=0.22, linewidth=0)
    ax_c.plot(obs_year["year"], obs_year["case_count"], "-o", color=BLUE, linewidth=1.5, markersize=4.5, label="Number of cases")
    if len(partial_year):
        ax_c.plot(partial_year["year"], partial_year["case_count"], "o", markerfacecolor="white", markeredgecolor=RED, color=RED, markersize=5.2)
    ax_c.plot(obs_year["year"], obs_year["regression_fit"], "--", color=LIGHT_BLUE, linewidth=1.3, label="Linear regression")
    if len(year_df):
        peak = obs_year.loc[obs_year["case_count"].idxmax()]
        first = year_df.iloc[0]
        last = partial_year.iloc[-1] if len(partial_year) else year_df.iloc[-1]
        for row in [first, peak, last]:
            ax_c.text(row["year"], row["case_count"] + max(year_df["case_count"].max() * 0.035, 10), f"{int(row['case_count']):,}", ha="center", fontsize=7)
        slope = year_df["regression_slope"].dropna().iloc[0]
        intercept = year_df["regression_intercept_at_2011"].dropna().iloc[0]
        r2 = year_df["regression_r2"].dropna().iloc[0]
        ax_c.text(
            0.04,
            0.70,
            f"Cases = {slope:.1f}(Year-2011) + {intercept:.1f}\n$R^2$ = {r2:.2f}",
            transform=ax_c.transAxes,
            fontsize=6.7,
            va="top",
        )
        ax_c.text(2021, year_df["case_count"].max() * 0.13, "2021-2023\npandemic window", fontsize=6.4, color="#8A5A00", ha="center")
        if len(partial_year):
            ax_c.text(float(partial_year.iloc[-1]["year"]), float(partial_year.iloc[-1]["case_count"]) + max(year_df["case_count"].max() * 0.07, 12), "partial", ha="center", fontsize=6.2, color=RED)
    ax_c.set_title(f"Yearly case distribution ({int(year_df['year'].min())}-{int(year_df['year'].max())})")
    ax_c.set_xlabel("Year")
    ax_c.set_ylabel("Cases")
    tick_step = 2 if (year_df["year"].max() - year_df["year"].min()) > 9 else 1
    ax_c.set_xticks(list(range(int(year_df["year"].min()), int(year_df["year"].max()) + 1, tick_step)))
    ax_c.grid(axis="y", color=GRID, linestyle="--", linewidth=0.6, alpha=0.8)
    ax_c.legend(frameon=False, loc="upper left")

    # (d) Attribute profile
    attr_plot = attr_df.copy()
    attr_plot["bar_label"] = attr_plot["category"]
    y_positions = np.arange(len(attr_plot))
    ax_d.barh(y_positions, attr_plot["percent"], color=BLUE, edgecolor=BLUE, height=0.58)
    ax_d.set_yticks(y_positions)
    ax_d.set_yticklabels(attr_plot["bar_label"])
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Frequency (%)")
    ax_d.set_title("Dataset attribute profile")
    xmax = max(10, attr_plot["percent"].max() * 1.35)
    ax_d.set_xlim(0, xmax)
    ax_d.grid(axis="x", color=GRID, linestyle="--", linewidth=0.6, alpha=0.75)
    for y, row in zip(y_positions, attr_plot.itertuples(index=False)):
        if row.percent >= 0.25:
            ax_d.text(row.percent + xmax * 0.01, y, f"{row.percent:.1f}", va="center", fontsize=6.7)
    # Group separators
    group_ranges = []
    start = 0
    for dim, sub in attr_plot.groupby("dimension", sort=False):
        end = start + len(sub) - 1
        group_ranges.append((dim, start, end))
        start = end + 1
    for dim, start, end in group_ranges[:-1]:
        ax_d.axhline(end + 0.5, color=DARK, linestyle=(0, (4, 4)), linewidth=0.9)

    for ax, label in [(ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)"), (ax_d, "(d)")]:
        ax.text(0.5, -0.26 if ax is not ax_d else -0.18, label, transform=ax.transAxes, ha="center", va="top", fontsize=12, fontweight="bold")

    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataset()
    label_df = distribution_table(df["outcome_label"], LABEL_ORDER, LABEL_DISPLAY)
    length_df = df[["case_id", "pre_decision_chars", "pre_decision_kchars"]].copy()
    years = df["case_year_final"].dropna().astype(int)
    years = years[(years >= 2011) & (years <= 2025)]
    min_year = int(years.min()) if len(years) else 2011
    max_year = int(years.max()) if len(years) else 2024
    year_index = pd.Index(range(min_year, max_year + 1), name="year")
    year_df = years.value_counts().reindex(year_index, fill_value=0).rename("case_count").reset_index()
    year_df["partial_release"] = year_df["year"] >= 2025
    fit_df = year_df[~year_df["partial_release"]].copy()
    x = (fit_df["year"] - 2011).to_numpy(dtype=float)
    y = fit_df["case_count"].to_numpy(dtype=float)
    if len(fit_df) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    else:
        slope, intercept, r2 = np.nan, np.nan, np.nan
    year_df["regression_fit"] = slope * (year_df["year"] - 2011) + intercept
    year_df.loc[year_df["partial_release"], "regression_fit"] = np.nan
    year_df["regression_slope"] = slope
    year_df["regression_intercept_at_2011"] = intercept
    year_df["regression_r2"] = r2
    attr_df = attribute_profile(df)

    out_base = FIG_DIR / "fig_dataset_profile_20k_nature_style"
    xlsx = DATA_DIR / "dataset_profile_20k_source_data.xlsx"
    make_excel(df, length_df, label_df, year_df, attr_df, xlsx)
    plot_figure(df, label_df, year_df, attr_df, out_base)

    desktop_png = DESKTOP / "fig_dataset_profile_20k_nature_style.png"
    desktop_xlsx = DESKTOP / "dataset_profile_20k_source_data.xlsx"
    shutil.copy2(out_base.with_suffix(".png"), desktop_png)
    shutil.copy2(xlsx, desktop_xlsx)

    print("rows", len(df))
    print("figure_png", out_base.with_suffix(".png"))
    print("figure_pdf", out_base.with_suffix(".pdf"))
    print("figure_svg", out_base.with_suffix(".svg"))
    print("figure_tiff", out_base.with_suffix(".tiff"))
    print("source_excel", xlsx)
    print("desktop_png", desktop_png)
    print("desktop_excel", desktop_xlsx)
    print("label_distribution")
    print(label_df[["display", "count", "percent"]].to_string(index=False))
    print("year_range", min_year, max_year, "year_rows", len(year_df))


if __name__ == "__main__":
    main()
