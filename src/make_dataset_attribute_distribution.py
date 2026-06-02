# -*- coding: utf-8 -*-
"""Generate Fig. 3(d): project/contract attribute distributions.

The extraction is rule-based and auditable. It does not alter the raw case data;
all case-level assignments and keyword evidence are exported for review.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER = PROJECT_ROOT / "data/lora_exports/lora_frozen_v1_2384/strong_label_master_v1_2384.csv"
TEST = PROJECT_ROOT / "data/gold/candidate_gold_extended_v2.csv"
INDEX = PROJECT_ROOT / "data/meta/structured_case_index.csv"
STRUCTURED_DIR = PROJECT_ROOT / "data/3_structured_cases"
FIG_DIR = PROJECT_ROOT / "paper_assets/figures/dataset_2884"
DATA_DIR = PROJECT_ROOT / "paper_assets/figure_data/dataset_2884"

YEAR_START = 2011
YEAR_END = 2025
VALID_LABELS = {"support", "partial_support", "not_support"}


def normalize_label(value: Any) -> str:
    raw = str(value or "").strip()
    if raw == "partial":
        return "partial_support"
    return raw


def read_structured_text(case_id: str) -> str:
    path = STRUCTURED_DIR / f"{case_id}.json"
    if not path.exists():
        return ""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(obj.get("pre_decision_text", "") or "")


def build_dataset() -> pd.DataFrame:
    master = pd.read_csv(MASTER, encoding="utf-8-sig")
    test = pd.read_csv(TEST, encoding="utf-8-sig")
    index = pd.read_csv(INDEX, encoding="utf-8-sig")
    index["case_id"] = index["case_id"].astype(str)

    master["case_id"] = master["case_id"].astype(str)
    master = master.merge(index[["case_id", "case_year"]], on="case_id", how="left")
    master["dataset_split"] = "train_dev_strong_label"
    master["label"] = master["outcome_label"].map(normalize_label)
    master["text_for_attr"] = master["source_file"].fillna("").astype(str) + "\n" + master["pre_decision_text"].fillna("").astype(str)

    test["case_id"] = test["case_id"].astype(str)
    test = test.merge(index[["case_id", "case_year"]], on="case_id", how="left", suffixes=("", "_idx"))
    test["dataset_split"] = "frozen_test500"
    test["label"] = test["candidate_outcome_label_v2"].map(normalize_label)
    test["text_for_attr"] = test["source_file"].fillna("").astype(str) + "\n" + test["case_id"].map(read_structured_text)

    cols = ["case_id", "dataset_split", "label", "case_year", "source_file", "text_for_attr"]
    data = pd.concat([master[cols], test[cols]], ignore_index=True)
    data = data.drop_duplicates("case_id").reset_index(drop=True)
    data["case_year"] = pd.to_numeric(data["case_year"], errors="coerce")
    data = data[data["label"].isin(VALID_LABELS)].copy()
    data = data[data["case_year"].between(YEAR_START, YEAR_END)].copy()
    data["case_year"] = data["case_year"].astype(int)
    return data.reset_index(drop=True)


def contains_any(text: str, patterns: list[str]) -> list[str]:
    hits = []
    for pat in patterns:
        if re.search(pat, text, flags=re.I):
            hits.append(pat)
    return hits


PROJECT_TYPE = {
    "Residential": [r"住宅", r"商品房", r"安置房", r"公寓", r"小区", r"楼盘", r"房地产", r"别墅"],
    "Commercial": [r"商业", r"商场", r"酒店", r"宾馆", r"写字楼", r"办公楼", r"综合体", r"店铺"],
    "Public building": [r"学校", r"医院", r"政府", r"公共", r"体育馆", r"文化", r"教育", r"教学楼", r"办公用房"],
    "Industrial/factory": [r"厂房", r"车间", r"工业园", r"仓库", r"生产线", r"产业园", r"基地"],
    "Municipal/infrastructure": [r"市政", r"道路", r"公路", r"桥梁", r"隧道", r"管网", r"排水", r"污水", r"供水", r"轨道", r"交通"],
    "Decoration/renovation": [r"装饰", r"装修", r"幕墙", r"门窗", r"精装", r"改造"],
    "Landscape/greening": [r"园林", r"绿化", r"景观", r"苗木", r"绿地"],
    "Installation/MEP": [r"安装", r"机电", r"消防", r"电气", r"暖通", r"给排水", r"钢结构"],
}

DELIVERY_METHOD = {
    "EPC / engineering procurement construction": [r"\bEPC\b", r"工程总承包", r"设计采购施工"],
    "Design-build": [r"设计施工", r"设计-施工", r"设计与施工", r"设计、施工"],
    "General construction contracting": [r"施工总承包", r"总承包"],
    "Traditional construction contract": [r"建设工程施工合同", r"施工合同"],
    "Professional subcontract": [r"专业分包", r"分包合同", r"分包工程"],
    "Labor subcontract": [r"劳务分包", r"劳务合同", r"劳务作业"],
    "Procurement / supply / processing": [r"采购合同", r"买卖合同", r"供货合同", r"加工合同", r"承揽合同", r"材料供应"],
}

CONTRACT_TYPE = {
    "Lump-sum / fixed-price contract": [r"固定总价", r"总价包干", r"闭口包干", r"包干价", r"固定价", r"总价合同"],
    "Unit-price / BoQ contract": [r"固定单价", r"综合单价", r"单价合同", r"工程量清单", r"清单计价", r"清单报价"],
    "Adjustable / measured contract": [r"可调价", r"价款调整", r"据实结算", r"按实结算", r"审计结算", r"审价结算", r"最终结算"],
    "Cost-plus-fee contract": [r"成本加酬金", r"成本加", r"酬金"],
    "Procurement / processing contract": [r"采购合同", r"买卖合同", r"供货合同", r"加工合同", r"承揽合同"],
}

PROJECT_PHASE = {
    "Design/change coordination": [r"设计变更", r"图纸", r"设计交底", r"技术交底", r"深化设计", r"变更"],
    "Start-up / mobilization": [r"开工令", r"开工报告", r"进场", r"场地移交", r"施工许可证", r"开工"],
    "Construction execution": [r"施工过程中", r"停工", r"窝工", r"赶工", r"复工", r"工程进度", r"进度计划", r"关键线路"],
    "Completion / handover": [r"竣工", r"验收", r"交付", r"移交", r"整改", r"交工"],
    "Payment / settlement": [r"工程款", r"进度款", r"付款", r"结算", r"审价", r"审计", r"鉴定", r"造价"],
    "Warranty / repair": [r"保修", r"维修", r"返修", r"质量缺陷", r"质保"],
}


def classify(text: str, spec: dict[str, list[str]], default: str = "Other/unspecified") -> tuple[str, str, int]:
    scores: list[tuple[int, int, str, list[str]]] = []
    for idx, (label, patterns) in enumerate(spec.items()):
        hits = contains_any(text, patterns)
        scores.append((len(hits), -idx, label, hits))
    scores.sort(reverse=True)
    best_score, _, best_label, best_hits = scores[0]
    if best_score <= 0:
        return default, "", 0
    return best_label, "; ".join(best_hits[:5]), int(best_score)


def extract_attributes(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in data.itertuples(index=False):
        text = str(r.text_for_attr or "")
        # Use first part plus keyword-bearing full text to reduce repeated judgment boilerplate.
        text = text[:12000]
        project_type, project_hits, project_score = classify(text, PROJECT_TYPE)
        delivery, delivery_hits, delivery_score = classify(text, DELIVERY_METHOD)
        contract, contract_hits, contract_score = classify(text, CONTRACT_TYPE)
        phase, phase_hits, phase_score = classify(text, PROJECT_PHASE)
        rows.append(
            {
                "case_id": r.case_id,
                "dataset_split": r.dataset_split,
                "label": r.label,
                "case_year": r.case_year,
                "project_type": project_type,
                "project_type_hits": project_hits,
                "project_type_score": project_score,
                "delivery_method": delivery,
                "delivery_method_hits": delivery_hits,
                "delivery_method_score": delivery_score,
                "contract_type": contract,
                "contract_type_hits": contract_hits,
                "contract_type_score": contract_score,
                "project_phase": phase,
                "project_phase_hits": phase_hits,
                "project_phase_score": phase_score,
            }
        )
    out = pd.DataFrame(rows)
    return out


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 7.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


DIMENSIONS = [
    ("D1 Engineering type", "project_type", [
        "Residential", "Commercial", "Public building", "Industrial/factory",
        "Municipal/infrastructure", "Decoration/renovation", "Landscape/greening",
        "Installation/MEP", "Other/unspecified",
    ]),
    ("D2 Project delivery method", "delivery_method", [
        "Traditional construction contract", "General construction contracting",
        "EPC / engineering procurement construction", "Design-build",
        "Professional subcontract", "Labor subcontract",
        "Procurement / supply / processing", "Other/unspecified",
    ]),
    ("D3 Contractual agreement type", "contract_type", [
        "Lump-sum / fixed-price contract", "Unit-price / BoQ contract",
        "Adjustable / measured contract", "Cost-plus-fee contract",
        "Procurement / processing contract", "Other/unspecified",
    ]),
    ("D4 Project/dispute phase", "project_phase", [
        "Design/change coordination", "Start-up / mobilization",
        "Construction execution", "Completion / handover",
        "Payment / settlement", "Warranty / repair", "Other/unspecified",
    ]),
]


def summarize(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(assignments)
    for dim_name, col, order in DIMENSIONS:
        counts = assignments[col].value_counts()
        for cat in order:
            count = int(counts.get(cat, 0))
            rows.append(
                {
                    "dimension": dim_name,
                    "attribute": cat,
                    "count": count,
                    "frequency_pct": 100 * count / n if n else 0.0,
                    "n_cases": n,
                }
            )
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame) -> None:
    set_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    y = 0
    separator_after = []
    group_centers = []
    for dim_name, _, order in DIMENSIONS:
        start = y
        for cat in order:
            rec = summary[(summary["dimension"] == dim_name) & (summary["attribute"] == cat)].iloc[0].to_dict()
            rec["y"] = y
            rows.append(rec)
            y += 1
        group_centers.append((dim_name, (start + y - 1) / 2))
        separator_after.append(y - 0.5)
        y += 0.55
    plot_df = pd.DataFrame(rows)

    height = max(6.0, 0.205 * len(plot_df) + 1.2)
    fig, ax = plt.subplots(figsize=(6.4, height))
    blue = "#2C7FB8"
    ax.barh(plot_df["y"], plot_df["frequency_pct"], color=blue, height=0.38, edgecolor=blue, linewidth=0.25)
    for r in plot_df.itertuples(index=False):
        if r.frequency_pct > 0:
            ax.text(r.frequency_pct + 0.7, r.y, f"{r.frequency_pct:.1f}", va="center", ha="left", fontsize=6.7, fontweight="bold")
    for sep in separator_after[:-1]:
        ax.hlines(sep, xmin=0, xmax=100, colors="#222222", linestyles=(0, (4, 4)), linewidth=0.75)
    for dim_name, center in group_centers:
        ax.text(70.0, center, dim_name, color="#0B559F", va="center", ha="left", fontsize=7.0)

    ax.set_yticks(plot_df["y"])
    ax.set_yticklabels(plot_df["attribute"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Frequency (%)")
    ax.set_title("(d) Distribution of project and contract attributes", loc="left", fontsize=9.2, fontweight="bold")
    ax.grid(axis="x", linestyle="--", linewidth=0.45, color="#BDBDBD", alpha=0.65)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=100, decimals=0))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3d_project_contract_attribute_distribution_2884.png", bbox_inches="tight", dpi=600)
    fig.savefig(FIG_DIR / "fig3d_project_contract_attribute_distribution_2884.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3d_project_contract_attribute_distribution_2884.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = build_dataset()
    assignments = extract_attributes(data)
    summary = summarize(assignments)
    assignments.to_csv(DATA_DIR / "fig3d_project_contract_attribute_case_assignments.csv", index=False, encoding="utf-8-sig")
    assignments.to_excel(DATA_DIR / "fig3d_project_contract_attribute_case_assignments.xlsx", index=False)
    summary.to_csv(DATA_DIR / "fig3d_project_contract_attribute_distribution.csv", index=False, encoding="utf-8-sig")
    summary.to_excel(DATA_DIR / "fig3d_project_contract_attribute_distribution.xlsx", index=False)
    plot(summary)
    report = {
        "n_cases": int(len(assignments)),
        "year_window": f"{YEAR_START}-{YEAR_END}",
        "figure": str(FIG_DIR / "fig3d_project_contract_attribute_distribution_2884.png"),
        "note": "Rule-based extraction from source_file and pre_decision_text; Other/unspecified indicates no reliable keyword evidence.",
    }
    (DATA_DIR / "fig3d_project_contract_attribute_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
