# -*- coding: utf-8 -*-
"""Generate IEEE-TEM-oriented figures, tables, figure data, and manuscript text assets."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_support import latest_run_dir, load_cfg  # noqa: E402


MODEL_LABELS = {
    "majority_class": "Majority",
    "rule_baseline": "Rule",
    "tfidf_logreg": "TF-IDF+LogReg",
    "tfidf_linearsvc": "TF-IDF+LinearSVC",
    "tfidf_multinomialnb": "TF-IDF+MNB",
    "current_hybrid_baseline": "Legacy Hybrid",
    "paesc_hybrid": "PAESC Hybrid",
    "gpt55_direct": "5.5 direct/proxy",
    "mmec_paesc_55_no_mechanism": "MMEC w/o mechanism",
    "mmec_paesc_55_no_evidence_chain": "MMEC w/o evidence",
    "mmec_paesc_55": "MMEC-PAESC 5.5",
}

SETTING_LABELS = {
    "full_model": "Full model",
    "remove_pre_decision_constraint": "Leakage stress test",
    "remove_structured_events": "No event structure",
    "remove_procedural_signals": "No procedure cues",
    "remove_evidence_chain": "No evidence chain",
    "remove_responsibility_head": "No responsibility head",
    "remove_retrieval": "No retrieval",
    "remove_irac_verifier": "No verifier",
}

DATASET_LABELS = {
    "candidate_gold_strict_v1": "Strict candidate benchmark",
    "candidate_gold_extended_v1": "Extended candidate benchmark",
}

RESP_LABELS = {
    "owner": "Owner",
    "contractor": "Contractor",
    "subcontractor": "Subcontractor",
    "designer_supervisor": "Designer/Supervisor",
    "both": "Shared",
    "force_majeure_policy": "Force majeure/policy",
    "unknown": "Unknown",
}

OUTCOME_LABELS = {
    "support": "Support",
    "partial": "Partial support",
    "not_support": "Not support",
}


def load_yaml(path: Path, default: Dict) -> Dict:
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_style(style_cfg: Dict) -> None:
    style = style_cfg["style"]
    layout = style_cfg["layout"]
    plt.rcParams["font.family"] = [style["font_family"], "SimSun", style.get("fallback_font_family", "DejaVu Serif")]
    plt.rcParams["font.size"] = style["font_size"]
    plt.rcParams["axes.titlesize"] = style["title_size"]
    plt.rcParams["axes.labelsize"] = style["axis_label_size"]
    plt.rcParams["xtick.labelsize"] = style["tick_size"]
    plt.rcParams["ytick.labelsize"] = style["tick_size"]
    plt.rcParams["legend.fontsize"] = style["legend_size"]
    plt.rcParams["axes.linewidth"] = style["axis_width"]
    plt.rcParams["grid.alpha"] = style["grid_alpha"]
    plt.rcParams["lines.linewidth"] = style["line_width"]
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams["savefig.dpi"] = 600
    if layout.get("tight_layout", True):
        plt.rcParams["figure.autolayout"] = True


def save_figure(fig, out_base: Path, formats: List[str]) -> None:
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), bbox_inches="tight")


def save_data_bundle(name: str, df: pd.DataFrame, figure_data_dir: Path, workbook_sheets: Dict[str, pd.DataFrame]) -> None:
    out_csv = figure_data_dir / f"{name}.csv"
    out_xlsx = figure_data_dir / f"{name}.xlsx"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df.to_excel(out_xlsx, index=False)
    workbook_sheets[name[:31]] = df


def save_table_bundle(name: str, df: pd.DataFrame, tables_dir: Path, workbook_sheets: Dict[str, pd.DataFrame]) -> None:
    out_csv = tables_dir / f"{name}.csv"
    out_xlsx = tables_dir / f"{name}.xlsx"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df.to_excel(out_xlsx, index=False)
    workbook_sheets[name[:31]] = df


def flat_resp_table(metrics_main: Dict) -> pd.DataFrame:
    rows = []
    for dataset_name in ["candidate_gold_strict_v1", "candidate_gold_extended_v1"]:
        resp = metrics_main["candidate_gold_evaluation"][dataset_name]["responsibility_task"]
        per_class = resp.get("per_class_performance", {})
        for label, vals in per_class.items():
            if label.endswith("avg") or label == "micro avg":
                continue
            rows.append({
                "dataset_name": dataset_name,
                "responsibility_label": label,
                "precision": vals["precision"],
                "recall": vals["recall"],
                "f1_score": vals["f1-score"],
                "support": vals["support"],
            })
    return pd.DataFrame(rows)


def auditability_table(metrics_main: Dict) -> pd.DataFrame:
    rows = []
    for dataset_name in ["candidate_gold_strict_v1", "candidate_gold_extended_v1"]:
        aud = metrics_main["candidate_gold_evaluation"][dataset_name]["evidence_chain_auditability"]
        row = {"dataset_name": dataset_name}
        row.update(aud)
        rows.append(row)
    return pd.DataFrame(rows)


def dataset_profile(strict_df: pd.DataFrame, extended_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in [("candidate_gold_strict_v1", strict_df), ("candidate_gold_extended_v1", extended_df)]:
        for label, count in df["candidate_outcome_label"].value_counts().items():
            rows.append({
                "dataset_name": name,
                "task": "outcome",
                "label": label,
                "count": int(count),
                "ratio": float(count / len(df)),
            })
        for label, count in df["candidate_responsibility_label"].value_counts().items():
            rows.append({
                "dataset_name": name,
                "task": "responsibility",
                "label": label,
                "count": int(count),
                "ratio": float(count / len(df)),
            })
    return pd.DataFrame(rows)


def confusion_data(pred_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    sub = pred_df[(pred_df["dataset_name"] == dataset_name) & (pred_df["model_name"] == "paesc_hybrid")]
    conf = pd.crosstab(sub["y_true"], sub["y_pred"], dropna=False).reindex(index=["support", "partial", "not_support"], columns=["support", "partial", "not_support"], fill_value=0)
    conf = conf.reset_index().rename(columns={"y_true": "true_label"})
    return conf


def split_summary(metrics_main: Dict) -> pd.DataFrame:
    rows = []
    random_m = metrics_main["internal_proxy_validation"]["random_split"]["paesc_hybrid"]
    rows.append({"evaluation_slice": "weak_random_split", "macro_f1": random_m["macro_f1"], "accuracy": random_m["accuracy"]})
    time_m = metrics_main["internal_proxy_validation"].get("time_split", {})
    if "paesc_hybrid" in time_m:
        rows.append({"evaluation_slice": f"weak_time_split_after_{time_m.get('cutoff_year', 'na')}", "macro_f1": time_m["paesc_hybrid"]["macro_f1"], "accuracy": time_m["paesc_hybrid"]["accuracy"]})
    for dataset_name in ["candidate_gold_strict_v1", "candidate_gold_extended_v1"]:
        m = metrics_main["candidate_gold_evaluation"][dataset_name]["paesc_hybrid"]
        rows.append({"evaluation_slice": dataset_name, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})
    return pd.DataFrame(rows)


def write_markdown(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def latest_prefixed_dir(root: Path, prefix: str) -> Optional[Path]:
    matches = sorted([d for d in root.glob(f"{prefix}*") if d.is_dir()])
    return matches[-1] if matches else None


def safe_sheet_name(name: str) -> str:
    clean = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    return clean[:31]


def wrap(text: object, width: int) -> str:
    raw = " ".join(str(text or "").replace("\n", " ").split())
    return textwrap.fill(raw, width=width) if raw else ""


def shorten(text: object, max_len: int = 180) -> str:
    raw = " ".join(str(text or "").replace("\n", " ").split())
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 3].rstrip() + "..."


def literal_list(value: object) -> List[dict]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.0,
    linestyle: str = "-",
    fontsize: float = 8.2,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def add_arrow(ax, xy_from: Tuple[float, float], xy_to: Tuple[float, float], color: str, linewidth: float = 1.0) -> None:
    ax.annotate(
        "",
        xy=xy_to,
        xytext=xy_from,
        arrowprops=dict(arrowstyle="->", lw=linewidth, color=color, shrinkA=2, shrinkB=2),
    )


def build_main_profile_table(strict_df: pd.DataFrame, extended_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_name, df in [
        ("candidate_gold_strict_v1", strict_df),
        ("candidate_gold_extended_v1", extended_df),
    ]:
        outcome_counts = df["candidate_outcome_label"].value_counts()
        resp_counts = df["candidate_responsibility_label"].value_counts()
        rows.append(
            {
                "benchmark": DATASET_LABELS[dataset_name],
                "dataset_name": dataset_name,
                "n_cases": int(len(df)),
                "support_n": int(outcome_counts.get("support", 0)),
                "support_ratio": round(float(outcome_counts.get("support", 0) / len(df)), 3),
                "partial_n": int(outcome_counts.get("partial", 0)),
                "partial_ratio": round(float(outcome_counts.get("partial", 0) / len(df)), 3),
                "not_support_n": int(outcome_counts.get("not_support", 0)),
                "not_support_ratio": round(float(outcome_counts.get("not_support", 0) / len(df)), 3),
                "owner_n": int(resp_counts.get("owner", 0)),
                "owner_ratio": round(float(resp_counts.get("owner", 0) / len(df)), 3),
                "contractor_n": int(resp_counts.get("contractor", 0)),
                "contractor_ratio": round(float(resp_counts.get("contractor", 0) / len(df)), 3),
                "unknown_n": int(resp_counts.get("unknown", 0)),
                "unknown_ratio": round(float(resp_counts.get("unknown", 0) / len(df)), 3),
            }
        )
    return pd.DataFrame(rows)


def build_main_resp_summary(
    metrics_main: Dict,
    resp_table: pd.DataFrame,
    audit_table: pd.DataFrame,
    root_cause_df: pd.DataFrame,
) -> pd.DataFrame:
    if not root_cause_df.empty and {"evidence_metric", "dataset_name", "value"}.issubset(root_cause_df.columns):
        unknown_map = (
            root_cause_df[root_cause_df["evidence_metric"] == "candidate_unknown_ratio"]
            .set_index("dataset_name")["value"]
            .to_dict()
        )
    else:
        unknown_map = {}
    rows = []
    for dataset_name in ["candidate_gold_strict_v1", "candidate_gold_extended_v1"]:
        task = metrics_main["candidate_gold_evaluation"][dataset_name]["responsibility_task"]
        sub = resp_table[resp_table["dataset_name"] == dataset_name].copy().sort_values("f1_score", ascending=False)
        audit_row = audit_table[audit_table["dataset_name"] == dataset_name].iloc[0]
        strongest = sub.iloc[0] if not sub.empty else None
        weakest = sub.iloc[-1] if not sub.empty else None
        rows.append(
            {
                "benchmark": DATASET_LABELS[dataset_name],
                "dataset_name": dataset_name,
                "responsibility_accuracy": task.get("accuracy", np.nan),
                "responsibility_macro_f1": task.get("macro_f1", np.nan),
                "candidate_unknown_ratio": round(float(unknown_map.get(dataset_name, np.nan)), 3),
                "strongest_class": strongest["responsibility_label"] if strongest is not None else "",
                "strongest_class_f1": strongest["f1_score"] if strongest is not None else np.nan,
                "weakest_class": weakest["responsibility_label"] if weakest is not None else "",
                "weakest_class_f1": weakest["f1_score"] if weakest is not None else np.nan,
                "valid_span_rate": audit_row["valid_span_rate"],
                "pre_decision_span_rate": audit_row["pre_decision_span_rate"],
                "duplicate_chain_rate": audit_row["duplicate_chain_rate"],
                "role_coverage_rate": audit_row["role_coverage_rate"],
                "missing_role_rate": audit_row["missing_role_rate"],
            }
        )
    return pd.DataFrame(rows)


def choose_main_cases(rep_df: pd.DataFrame, n_cases: int = 2) -> pd.DataFrame:
    if rep_df.empty:
        return rep_df
    data = rep_df.copy()
    # The 5.5 evaluation branch exports a leaner representative_cases.csv than
    # the legacy plotting pipeline. Fill optional text fields so the case-panel
    # selector remains backward-compatible without changing evaluated cases.
    for col, default in [
        ("source_file", ""),
        ("case_snippet", ""),
        ("candidate_responsibility_label", "unknown"),
        ("primary_responsible_party", "unknown"),
        ("procedural_compliance_status", ""),
        ("delay_events_preview", ""),
        ("claims_preview", ""),
        ("defenses_preview", ""),
        ("evidence_spans", "[]"),
        ("responsibility_type", ""),
        ("documentation_integrity_flag", ""),
        ("explanation_text", ""),
        ("uncertainty_flag", 0),
        ("confidence", 0.0),
        ("candidate_confidence", 0.0),
        ("high_dispute_flag", False),
        ("error_category", "unspecified"),
    ]:
        if col not in data.columns:
            data[col] = default
    data["construction_like"] = (
        data["source_file"].fillna("").str.contains("建设|施工|工程", regex=True)
        | data["case_snippet"].fillna("").str.contains("工期|建设工程|施工合同", regex=True)
    )
    data["non_labor"] = ~data["source_file"].fillna("").str.contains("劳动", regex=True)
    data = data[data["construction_like"] & data["non_labor"]].copy()
    if data.empty:
        data = rep_df.copy()
        for col, default in [
            ("source_file", ""),
            ("case_snippet", ""),
            ("candidate_responsibility_label", "unknown"),
            ("primary_responsible_party", "unknown"),
            ("procedural_compliance_status", ""),
            ("delay_events_preview", ""),
            ("claims_preview", ""),
            ("defenses_preview", ""),
            ("evidence_spans", "[]"),
            ("responsibility_type", ""),
            ("documentation_integrity_flag", ""),
            ("explanation_text", ""),
            ("uncertainty_flag", 0),
            ("confidence", 0.0),
            ("candidate_confidence", 0.0),
            ("high_dispute_flag", False),
            ("error_category", "unspecified"),
        ]:
            if col not in data.columns:
                data[col] = default

    preferred_errors = [
        "ambiguous_causality",
        "insufficient_evidence",
        "procedural_noncompliance_confusion",
        "concurrent_delay_conflict",
        "partial_support_boundary_confusion",
        "responsibility_evidence_inconsistency",
    ]
    chosen = []
    for err in preferred_errors:
        sub = data[data["error_category"] == err].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["high_dispute_flag", "candidate_confidence", "confidence"], ascending=[False, False, False])
        row = sub.iloc[0]
        if row["case_id"] not in {picked["case_id"] for picked in chosen}:
            chosen.append(row)
        if len(chosen) >= n_cases:
            break
    if len(chosen) < n_cases:
        fallback = data.sort_values(["high_dispute_flag", "candidate_confidence", "confidence"], ascending=[False, False, False])
        for _, row in fallback.iterrows():
            if row["case_id"] not in {picked["case_id"] for picked in chosen}:
                chosen.append(row)
            if len(chosen) >= n_cases:
                break
    return pd.DataFrame(chosen).reset_index(drop=True)


def draw_case_columns(ax, case_row: pd.Series, palette: List[str], panel_label: str) -> None:
    ax.axis("off")
    gray = ["#111111", "#666666", "#BBBBBB"]
    ax.text(0.0, 1.02, panel_label, fontsize=10, fontweight="bold", ha="left", va="bottom")
    columns = [
        ("Case context", 0.00, 0.22),
        ("Evidence-chain excerpts", 0.24, 0.32),
        ("Outcome assessment", 0.58, 0.18),
        ("Responsibility + governance cue", 0.78, 0.21),
    ]
    for title, x0, width in columns:
        add_box(ax, x0, 0.05, width, 0.82, "", "white", gray[2], linewidth=0.8)
        ax.text(x0 + 0.01, 0.84, title, ha="left", va="top", fontsize=8.5, fontweight="bold")

    context_text = (
        f"Case ID: {case_row['case_id']}\n"
        f"Dataset: {DATASET_LABELS.get(case_row['dataset_name'], case_row['dataset_name'])}\n"
        f"Error type: {case_row.get('error_category', '')}\n"
        f"Snippet: {wrap(shorten(case_row.get('case_snippet', ''), 220), 30)}"
    )
    ax.text(0.015, 0.79, context_text, ha="left", va="top", fontsize=7.9)

    evidence_units = literal_list(case_row.get("evidence_spans"))
    evidence_lines = []
    for unit in evidence_units[:4]:
        evidence_lines.append(f"{str(unit.get('role_label', '')).replace('_', ' ')}: {shorten(unit.get('text', ''), 95)}")
    if not evidence_lines:
        evidence_lines.append("No evidence spans exported.")
    ax.text(0.255, 0.79, wrap("\n".join(evidence_lines), 42), ha="left", va="top", fontsize=7.7)

    outcome_text = (
        f"True label: {case_row.get('y_true', '')}\n"
        f"Predicted label: {case_row.get('y_pred', '')}\n"
        f"Confidence: {float(case_row.get('confidence', 0.0)):.3f}\n"
        f"High-dispute flag: {int(case_row.get('high_dispute_flag', 0))}\n"
        f"Uncertainty flag: {int(case_row.get('uncertainty_flag', 0))}"
    )
    ax.text(0.595, 0.79, outcome_text, ha="left", va="top", fontsize=7.9)

    gov_text = (
        f"Primary responsibility: {RESP_LABELS.get(str(case_row.get('primary_responsible_party', '')), str(case_row.get('primary_responsible_party', '')))}\n"
        f"Type: {case_row.get('responsibility_type', '')}\n"
        f"Procedure status: {case_row.get('procedural_compliance_status', '')}\n"
        f"Documentation: {case_row.get('documentation_integrity_flag', '')}\n"
        f"Explanation: {wrap(shorten(case_row.get('explanation_text', ''), 165), 26)}"
    )
    ax.text(0.795, 0.79, gov_text, ha="left", va="top", fontsize=7.7)

    ax.add_patch(Rectangle((0.24, 0.71), 0.32, 0.04, facecolor=palette[2], alpha=0.08, edgecolor="none"))
    ax.add_patch(Rectangle((0.58, 0.71), 0.18, 0.04, facecolor=palette[1], alpha=0.08, edgecolor="none"))
    ax.add_patch(Rectangle((0.78, 0.71), 0.21, 0.04, facecolor=palette[3], alpha=0.08, edgecolor="none"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/research_v1.yaml")
    ap.add_argument("--style", type=str, default="config/figure_style_sci.yaml")
    ap.add_argument("--run_dir", type=str, default="")
    args = ap.parse_args()

    cfg = load_cfg(PROJECT_ROOT / args.config)
    style_cfg = load_yaml(PROJECT_ROOT / args.style, {})
    set_style(style_cfg)

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(PROJECT_ROOT / cfg["paths"]["final_eval_root"])
    if run_dir is None:
        raise FileNotFoundError("No final_eval_* run directory found.")
    run_dir = run_dir.resolve()

    paper_root = ensure_dir(PROJECT_ROOT / cfg["paths"]["paper_assets_dir"])
    figures_dir = ensure_dir(paper_root / cfg["figures"]["out_dir_name"])
    figure_data_dir = ensure_dir(paper_root / cfg["figures"]["figure_data_dir_name"])
    tables_dir = ensure_dir(paper_root / cfg["figures"]["tables_dir_name"])
    captions_dir = ensure_dir(paper_root / cfg["figures"]["captions_dir_name"])
    text_dir = ensure_dir(paper_root / cfg["figures"]["text_dir_name"])

    formats = cfg["figures"]["formats"]
    palette = style_cfg["style"]["color_palette"]
    gray = style_cfg["style"].get("grayscale_palette", ["#202020", "#5E5E5E", "#919191", "#C8C8C8"])

    strict_df = pd.read_csv(PROJECT_ROOT / cfg["paths"]["candidate_gold_strict_csv"])
    extended_df = pd.read_csv(PROJECT_ROOT / cfg["paths"]["candidate_gold_extended_csv"])
    baseline_df = pd.read_csv(run_dir / "baseline_comparison.csv")
    pred_df = pd.read_csv(run_dir / "predictions_main.csv")
    per_class_df = pd.read_csv(run_dir / "per_class_results.csv")
    resp_df = pd.read_csv(run_dir / "responsibility_eval.csv")
    chain_df = pd.read_csv(run_dir / "evidence_chain_eval.csv")
    ablation_df = pd.read_csv(run_dir / "ablation_results.csv")
    error_df = pd.read_csv(run_dir / "error_analysis.csv")
    rep_df = pd.read_csv(run_dir / "representative_cases.csv")
    metrics_main = json.loads((run_dir / "metrics_main.json").read_text(encoding="utf-8"))

    forensic_dir = latest_prefixed_dir(PROJECT_ROOT / cfg["paths"]["final_eval_root"], "forensic_audit_")
    claim_tiering_df = pd.read_csv(forensic_dir / "claim_tiering.csv") if forensic_dir else pd.DataFrame()
    leakage_df = pd.read_csv(forensic_dir / "leakage_sentinel_results.csv") if forensic_dir else pd.DataFrame()
    delta_df = pd.read_csv(forensic_dir / "delta_table.csv") if forensic_dir else pd.DataFrame()
    root_cause_df = pd.read_csv(forensic_dir / "responsibility_root_cause.csv") if forensic_dir else pd.DataFrame()

    workbook_sheets: Dict[str, pd.DataFrame] = {}

    profile_df = dataset_profile(strict_df, extended_df)
    save_table_bundle("table1_dataset_profile", profile_df, tables_dir, workbook_sheets)
    save_table_bundle("table2_baseline_performance", baseline_df, tables_dir, workbook_sheets)
    save_table_bundle("table3_ablation_results", ablation_df, tables_dir, workbook_sheets)
    resp_table = flat_resp_table(metrics_main)
    save_table_bundle("table4_responsibility_metrics", resp_table, tables_dir, workbook_sheets)
    audit_table = auditability_table(metrics_main)
    save_table_bundle("table5_auditability_metrics", audit_table, tables_dir, workbook_sheets)
    save_table_bundle("table6_representative_cases", rep_df, tables_dir, workbook_sheets)

    # Fig1: methodology flow
    fig1_data = pd.DataFrame([
        {"step": 1, "module": "Leakage-aware parsing", "input": "DOCX / parsed JSON", "output": "pre_decision_text + post_decision_text"},
        {"step": 2, "module": "Candidate-gold governance", "input": "weak labels + seed reference + post anchors", "output": "candidate_gold_strict / extended"},
        {"step": 3, "module": "Outcome prediction", "input": "pre_decision_text", "output": "support / partial / not_support"},
        {"step": 4, "module": "Responsibility diagnosis", "input": "pre-decision evidence", "output": "primary responsibility + procedure status"},
        {"step": 5, "module": "Evidence-chain audit", "input": "span pointers", "output": "auditability metrics + review subset"},
        {"step": 6, "module": "Managerial reporting", "input": "model outputs", "output": "figures / tables / review package"},
    ])
    save_data_bundle("fig1_method_flow", fig1_data, figure_data_dir, workbook_sheets)
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.axis("off")
    for i, row in fig1_data.iterrows():
        x = 0.08 + i * 0.15
        ax.text(x, 0.5, f"{row['module']}\n{row['output']}", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=palette[0], lw=1.2))
        if i < len(fig1_data) - 1:
            ax.annotate("", xy=(x + 0.07, 0.5), xytext=(x + 0.12, 0.5), arrowprops=dict(arrowstyle="->", color=palette[3], lw=1.2))
    ax.set_title("DelayDispute Copilot workflow")
    save_figure(fig, figures_dir / "fig1_method_flow", formats)
    plt.close(fig)

    # Fig2: class distribution
    fig2_data = profile_df[profile_df["task"] == "outcome"].copy()
    save_data_bundle("fig2_class_distribution", fig2_data, figure_data_dir, workbook_sheets)
    pivot2 = fig2_data.pivot(index="label", columns="dataset_name", values="count").reindex(["support", "partial", "not_support"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    pivot2.plot(kind="bar", ax=ax, color=[palette[0], palette[2]])
    ax.set_title("Candidate benchmark label distribution")
    ax.set_xlabel("Outcome label")
    ax.set_ylabel("Count")
    ax.legend(title="Dataset", frameon=False)
    save_figure(fig, figures_dir / "fig2_class_distribution", formats)
    plt.close(fig)

    # Fig3: traditional vs hybrid comparison
    fig3_data = baseline_df.copy()
    fig3_data["model_label"] = fig3_data["model_name"].map(MODEL_LABELS)
    save_data_bundle("fig3_outcome_performance", fig3_data, figure_data_dir, workbook_sheets)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, dataset_name in zip(axes, ["candidate_gold_strict_v1", "candidate_gold_extended_v1"]):
        sub = fig3_data[fig3_data["dataset_name"] == dataset_name].copy()
        ax.bar(sub["model_label"], sub["macro_f1"], color=palette[:len(sub)])
        ax.set_title(dataset_name.replace("_v1", ""))
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Macro-F1")
    fig.suptitle("Traditional models versus hybrid systems")
    save_figure(fig, figures_dir / "fig3_outcome_performance", formats)
    plt.close(fig)

    # Fig4: confusion matrix
    fig4_data = confusion_data(pred_df, "candidate_gold_extended_v1")
    save_data_bundle("fig4_confusion_matrix", fig4_data, figure_data_dir, workbook_sheets)
    conf_mat = fig4_data.set_index("true_label")[["support", "partial", "not_support"]]
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(conf_mat.values, cmap="Blues")
    ax.set_xticks(range(3), conf_mat.columns)
    ax.set_yticks(range(3), conf_mat.index)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(j, i, int(conf_mat.iloc[i, j]), ha="center", va="center")
    ax.set_title("PAESC confusion matrix on extended candidate benchmark")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save_figure(fig, figures_dir / "fig4_confusion_matrix", formats)
    plt.close(fig)

    # Fig5: responsibility performance
    fig5_data = resp_table.copy()
    save_data_bundle("fig5_responsibility_performance", fig5_data, figure_data_dir, workbook_sheets)
    pivot5 = fig5_data.pivot(index="responsibility_label", columns="dataset_name", values="f1_score").fillna(0)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    pivot5.plot(kind="bar", ax=ax, color=[palette[1], palette[4]])
    ax.set_title("Responsibility diagnosis per-class F1")
    ax.set_xlabel("Responsibility label")
    ax.set_ylabel("F1-score")
    ax.legend(title="Dataset", frameon=False)
    save_figure(fig, figures_dir / "fig5_responsibility_performance", formats)
    plt.close(fig)

    # Fig6: evidence-chain auditability
    fig6_data = audit_table.copy().melt(id_vars="dataset_name", var_name="metric", value_name="value")
    save_data_bundle("fig6_evidence_auditability", fig6_data, figure_data_dir, workbook_sheets)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for idx, dataset_name in enumerate(fig6_data["dataset_name"].unique()):
        sub = fig6_data[fig6_data["dataset_name"] == dataset_name]
        ax.plot(sub["metric"], sub["value"], marker="o", label=dataset_name, color=palette[idx])
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Evidence-chain auditability metrics")
    ax.set_ylabel("Rate")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir / "fig6_evidence_auditability", formats)
    plt.close(fig)

    # Fig7: ablation
    fig7_data = ablation_df.copy()
    fig7_data["setting_label"] = fig7_data["ablation_setting"].map(SETTING_LABELS)
    save_data_bundle("fig7_ablation", fig7_data, figure_data_dir, workbook_sheets)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ext_ab = fig7_data[fig7_data["dataset_name"] == "candidate_gold_extended_v1"]
    ax.barh(ext_ab["setting_label"], ext_ab["macro_f1"], color=palette[0])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Ablation results on the extended candidate benchmark")
    save_figure(fig, figures_dir / "fig7_ablation", formats)
    plt.close(fig)

    # Fig8: error category distribution
    fig8_data = error_df.groupby(["dataset_name", "error_category"]).size().reset_index(name="count")
    save_data_bundle("fig8_error_distribution", fig8_data, figure_data_dir, workbook_sheets)
    pivot8 = fig8_data.pivot(index="error_category", columns="dataset_name", values="count").fillna(0)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    pivot8.plot(kind="bar", ax=ax, color=[palette[3], palette[5]])
    ax.set_title("Error category distribution")
    ax.set_xlabel("Error category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Dataset", frameon=False)
    save_figure(fig, figures_dir / "fig8_error_distribution", formats)
    plt.close(fig)

    # Fig9: split comparison
    fig9_data = split_summary(metrics_main)
    save_data_bundle("fig9_split_comparison", fig9_data, figure_data_dir, workbook_sheets)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(fig9_data["evaluation_slice"], fig9_data["macro_f1"], marker="o", color=palette[0], label="Macro-F1")
    ax.plot(fig9_data["evaluation_slice"], fig9_data["accuracy"], marker="s", color=palette[3], label="Accuracy")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("Validation slices and benchmark performance")
    ax.legend(frameon=False)
    save_figure(fig, figures_dir / "fig9_split_comparison", formats)
    plt.close(fig)

    # Fig10: representative case panel
    fig10_data = rep_df.head(4).copy()
    save_data_bundle("fig10_representative_cases", fig10_data, figure_data_dir, workbook_sheets)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= len(fig10_data):
            continue
        row = fig10_data.iloc[idx]
        txt = (
            f"Case: {row['case_id']}\n"
            f"Dataset: {row['dataset_name']}\n"
            f"True/Pred: {row['y_true']} / {row['y_pred']}\n"
            f"Error: {row['error_category']}\n"
            f"Resp: {row.get('primary_responsible_party', 'unknown')}\n"
            f"Snippet: {str(row.get('case_snippet', ''))[:170]}"
        )
        ax.text(0.01, 0.98, txt, va="top", ha="left", wrap=True)
    fig.suptitle("Representative error cases")
    save_figure(fig, figures_dir / "fig10_representative_cases", formats)
    plt.close(fig)

    # Final main-text package: 6 figures and 5 tables
    claim_table_path = tables_dir / "table_main_result_claims.csv"
    claim_df = pd.read_csv(claim_table_path) if claim_table_path.exists() else pd.DataFrame()
    main_profile_df = build_main_profile_table(strict_df, extended_df)
    main_resp_summary_df = build_main_resp_summary(metrics_main, resp_table, audit_table, root_cause_df)
    ablation_required = [
        "dataset_name",
        "ablation_setting",
        "removed_component",
        "accuracy",
        "macro_f1",
        "responsibility_macro_f1",
        "evidence_consistency_rate",
        "valid_span_rate",
        "pre_decision_span_rate",
        "high_dispute_rate",
    ]
    main_ablation_df = ablation_df.copy()
    if "removed_component" not in main_ablation_df.columns:
        main_ablation_df["removed_component"] = main_ablation_df.get("model_name", main_ablation_df.get("ablation_setting", ""))
    for col in ablation_required:
        if col not in main_ablation_df.columns:
            main_ablation_df[col] = np.nan
    main_ablation_df = main_ablation_df[ablation_required].copy()
    if not claim_tiering_df.empty and {"result_group", "run_name", "model_name", "claim_tier", "claim_boundary"}.issubset(claim_tiering_df.columns):
        main_claim_df = claim_tiering_df.copy()
        if "dataset_name" not in main_claim_df.columns:
            main_claim_df["dataset_name"] = ""
        if "accuracy" not in main_claim_df.columns:
            main_claim_df["accuracy"] = main_claim_df.get("accuracy_recomputed", np.nan)
        if "macro_f1" not in main_claim_df.columns:
            main_claim_df["macro_f1"] = main_claim_df.get("macro_f1_recomputed", np.nan)
        if "audit_status" not in main_claim_df.columns:
            main_claim_df["audit_status"] = np.where(main_claim_df["claim_tier"].astype(str).str.contains("Tier C"), "audit_incomplete", "complete")
        main_claim_df = main_claim_df[
            [
                "result_group",
                "run_name",
                "dataset_name",
                "model_name",
                "accuracy",
                "macro_f1",
                "audit_status",
                "claim_tier",
                "claim_boundary",
            ]
        ]
    elif not claim_df.empty and not claim_tiering_df.empty:
        main_claim_df = claim_df.merge(claim_tiering_df, on="run_name", how="left")[
            [
                "result_group",
                "run_name",
                "dataset_name",
                "model_name",
                "accuracy",
                "macro_f1",
                "audit_status",
                "tier",
                "claim_boundary",
            ]
        ].rename(columns={"tier": "claim_tier"})
    else:
        main_claim_df = pd.DataFrame()

    main_figure_names = [
        "fig1_delaydispute_workflow",
        "fig2_leakage_benchmark_governance",
        "fig3_outcome_prediction_comparison",
        "fig4_responsibility_auditability",
        "fig5_forensic_claim_boundary",
        "fig6_representative_case_panel",
    ]
    appendix_figure_names = [
        "figS1_confusion_matrices",
        "figS2_additional_representative_cases",
        "figS3_error_distribution",
        "figS4_split_comparison",
    ]
    main_table_names = [
        "table1_candidate_benchmark_profile",
        "table2_main_outcome_results",
        "table3_ablation_leakage_stress",
        "table4_responsibility_auditability_summary",
        "table5_forensic_claim_boundary",
    ]
    appendix_table_names = [
        "tableS1_per_class_outcome_metrics",
        "tableS2_per_class_responsibility_metrics",
        "tableS3_leakage_sentinel_results",
        "tableS4_forensic_delta_audit",
        "tableS5_responsibility_root_cause",
        "tableS6_expanded_ablation",
        "tableS7_candidate_profile_long",
    ]

    save_table_bundle(main_table_names[0], main_profile_df, tables_dir, workbook_sheets)
    save_table_bundle(main_table_names[1], baseline_df, tables_dir, workbook_sheets)
    save_table_bundle(main_table_names[2], main_ablation_df, tables_dir, workbook_sheets)
    save_table_bundle(main_table_names[3], main_resp_summary_df, tables_dir, workbook_sheets)
    if not main_claim_df.empty:
        save_table_bundle(main_table_names[4], main_claim_df, tables_dir, workbook_sheets)
    save_table_bundle(appendix_table_names[0], per_class_df, tables_dir, workbook_sheets)
    save_table_bundle(appendix_table_names[1], resp_table, tables_dir, workbook_sheets)
    if not leakage_df.empty:
        save_table_bundle(appendix_table_names[2], leakage_df, tables_dir, workbook_sheets)
    if not delta_df.empty:
        save_table_bundle(appendix_table_names[3], delta_df, tables_dir, workbook_sheets)
    if not root_cause_df.empty:
        save_table_bundle(appendix_table_names[4], root_cause_df, tables_dir, workbook_sheets)
    save_table_bundle(appendix_table_names[5], ablation_df, tables_dir, workbook_sheets)
    save_table_bundle(appendix_table_names[6], profile_df, tables_dir, workbook_sheets)

    # Main Fig. 1
    final_fig1_df = pd.DataFrame(
        [
            {"lane": "core", "block": "Document parsing", "output": "Structured segments"},
            {"lane": "core", "block": "Pre/post split", "output": "Prospective input boundary"},
            {"lane": "core", "block": "Candidate benchmark governance", "output": "Strict + extended candidate sets"},
            {"lane": "core", "block": "Outcome prediction", "output": "Support / partial / not support"},
            {"lane": "core", "block": "Responsibility + evidence chain", "output": "Structured outputs + spans"},
            {"lane": "audit", "block": "Leakage sentinel", "output": "Pre / post / pre+post"},
            {"lane": "audit", "block": "Claim boundary", "output": "Tier B vs Tier C"},
            {"lane": "audit", "block": "Review-ready package", "output": "Audit-ready exports"},
        ]
    )
    save_data_bundle(main_figure_names[0], final_fig1_df, figure_data_dir, workbook_sheets)
    fig = plt.figure(figsize=(7.6, 3.8))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    core_boxes = [
        (0.04, 0.58, 0.14, 0.16, "Document\nparsing"),
        (0.22, 0.58, 0.14, 0.16, "Pre/post\nsplit"),
        (0.40, 0.58, 0.18, 0.16, "Candidate benchmark\ngovernance"),
        (0.62, 0.58, 0.14, 0.16, "Outcome\nprediction"),
        (0.80, 0.58, 0.16, 0.16, "Responsibility +\nevidence chain"),
    ]
    for idx, (x, y, w, h, label) in enumerate(core_boxes):
        add_box(ax, x, y, w, h, label, "white", "#777777", linewidth=0.9)
        if idx < len(core_boxes) - 1:
            next_box = core_boxes[idx + 1]
            add_arrow(ax, (x + w, y + h / 2), (next_box[0], next_box[1] + next_box[3] / 2), "#444444", 0.9)
    add_box(ax, 0.40, 0.80, 0.26, 0.10, "Post-decision anchors\n(labels + forensic only)", "white", palette[3], linewidth=0.9, linestyle="--", fontsize=8.0)
    add_arrow(ax, (0.29, 0.74), (0.53, 0.80), palette[3], 0.9)
    ax.text(0.69, 0.84, "Not used for inference", fontsize=7.6, color=palette[3], ha="left")
    audit_boxes = [
        (0.12, 0.16, 0.18, 0.13, "Leakage\nsentinel"),
        (0.36, 0.16, 0.18, 0.13, "Claim\nboundary"),
        (0.60, 0.16, 0.22, 0.13, "Review-ready\npackage"),
    ]
    for idx, (x, y, w, h, label) in enumerate(audit_boxes):
        add_box(ax, x, y, w, h, label, palette[1], "none", linewidth=0.0, fontsize=8.0)
        if idx < len(audit_boxes) - 1:
            nxt = audit_boxes[idx + 1]
            add_arrow(ax, (x + w, y + h / 2), (nxt[0], nxt[1] + nxt[3] / 2), "#AAAAAA", 0.8)
    add_arrow(ax, (0.49, 0.58), (0.21, 0.29), "#AAAAAA", 0.8)
    add_arrow(ax, (0.49, 0.58), (0.45, 0.29), "#AAAAAA", 0.8)
    add_arrow(ax, (0.88, 0.58), (0.71, 0.29), "#AAAAAA", 0.8)
    ax.text(0.04, 0.94, "DelayDispute Copilot: leakage-aware decision-support workflow", fontsize=10.0, fontweight="bold", ha="left")
    save_figure(fig, figures_dir / main_figure_names[0], formats)
    plt.close(fig)

    # Main Fig. 2
    legacy_unknown_ratio = np.nan
    if not root_cause_df.empty:
        legacy_rows = root_cause_df[root_cause_df["evidence_metric"] == "old_gold500_unknown_ratio"]
        if not legacy_rows.empty:
            legacy_unknown_ratio = float(legacy_rows.iloc[0]["value"])
    final_fig2_rows = []
    for dataset_name, df in [("candidate_gold_strict_v1", strict_df), ("candidate_gold_extended_v1", extended_df)]:
        n_cases = len(df)
        for label in ["support", "partial", "not_support"]:
            count = int((df["candidate_outcome_label"] == label).sum())
            final_fig2_rows.append(
                {
                    "dataset_name": dataset_name,
                    "series": "outcome_distribution",
                    "label": label,
                    "count": count,
                    "ratio": round(float(count / n_cases), 3),
                }
            )
        final_fig2_rows.append(
            {
                "dataset_name": dataset_name,
                "series": "responsibility_unknown_ratio",
                "label": "unknown",
                "count": int((df["candidate_responsibility_label"] == "unknown").sum()),
                "ratio": round(float((df["candidate_responsibility_label"] == "unknown").mean()), 3),
            }
        )
    if not math.isnan(legacy_unknown_ratio):
        final_fig2_rows.append(
            {
                "dataset_name": "gold500_v1_historical_reference",
                "series": "responsibility_unknown_ratio",
                "label": "unknown",
                "count": np.nan,
                "ratio": legacy_unknown_ratio,
            }
        )
    final_fig2_df = pd.DataFrame(final_fig2_rows)
    save_data_bundle(main_figure_names[1], final_fig2_df, figure_data_dir, workbook_sheets)
    fig = plt.figure(figsize=(7.5, 4.1))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1.35], wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_a.axis("off")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    add_panel_label(ax_a, "(a)")
    add_box(ax_a, 0.04, 0.70, 0.24, 0.15, "Raw case\ndocument", "white", "#777777", 0.9)
    add_box(ax_a, 0.38, 0.70, 0.28, 0.15, "Pre-decision\nevidence view", palette[1], "none", 0.0)
    add_box(ax_a, 0.72, 0.70, 0.24, 0.15, "Post-decision\nanchors", "white", palette[3], 0.9, "--")
    add_arrow(ax_a, (0.28, 0.775), (0.38, 0.775), "#444444", 0.9)
    add_arrow(ax_a, (0.28, 0.775), (0.72, 0.775), palette[3], 0.9)
    ax_a.plot([0.69, 0.69], [0.20, 0.92], linestyle="--", color=palette[3], linewidth=0.9)
    ax_a.text(0.695, 0.92, "Inference boundary", fontsize=7.6, color=palette[3], ha="left")
    add_box(ax_a, 0.08, 0.42, 0.26, 0.14, "Outcome\nprediction", "white", "#AAAAAA", 0.8)
    add_box(ax_a, 0.40, 0.42, 0.26, 0.14, "Responsibility\ndiagnosis", "white", "#AAAAAA", 0.8)
    add_box(ax_a, 0.08, 0.18, 0.26, 0.14, "Evidence-chain\naudit", "white", "#AAAAAA", 0.8)
    add_box(ax_a, 0.40, 0.18, 0.26, 0.14, "Labels + leakage\nsentinel only", "white", palette[3], 0.8, "--")
    add_arrow(ax_a, (0.52, 0.70), (0.21, 0.56), "#AAAAAA", 0.8)
    add_arrow(ax_a, (0.52, 0.70), (0.53, 0.56), "#AAAAAA", 0.8)
    add_arrow(ax_a, (0.84, 0.70), (0.53, 0.32), palette[3], 0.8)
    add_arrow(ax_a, (0.21, 0.42), (0.21, 0.32), "#AAAAAA", 0.8)
    add_panel_label(ax_b, "(b)")
    ax_b.set_title("Candidate benchmark composition", pad=10)
    bar_data = final_fig2_df[final_fig2_df["series"] == "outcome_distribution"].copy()
    bench_order = ["candidate_gold_strict_v1", "candidate_gold_extended_v1"]
    y_pos = np.arange(len(bench_order))
    left = np.zeros(len(bench_order))
    for color, label in zip([palette[2], "#BBBBBB", "#333333"], ["support", "partial", "not_support"]):
        values = [
            float(bar_data[(bar_data["dataset_name"] == dataset_name) & (bar_data["label"] == label)]["ratio"].iloc[0])
            for dataset_name in bench_order
        ]
        ax_b.barh(y_pos, values, left=left, color=color, edgecolor="white", height=0.34, label=OUTCOME_LABELS[label])
        left += np.array(values)
    ax_b.set_xlim(0, 1)
    ax_b.set_yticks(y_pos, ["Strict (n=250)", "Extended (n=500)"])
    ax_b.set_xlabel("Outcome-label ratio")
    ax_b.invert_yaxis()
    ax_b.legend(frameon=False, loc="lower right")
    inset = ax_b.inset_axes([0.57, 0.12, 0.39, 0.34])
    unknown_rows = [
        ("Strict", float(main_profile_df.loc[main_profile_df["dataset_name"] == "candidate_gold_strict_v1", "unknown_ratio"].iloc[0])),
        ("Extended", float(main_profile_df.loc[main_profile_df["dataset_name"] == "candidate_gold_extended_v1", "unknown_ratio"].iloc[0])),
    ]
    if not math.isnan(legacy_unknown_ratio):
        unknown_rows.append(("Legacy gold500", legacy_unknown_ratio))
    inset_df = pd.DataFrame(unknown_rows, columns=["name", "ratio"])
    inset_y = np.arange(len(inset_df))
    inset.barh(inset_y, inset_df["ratio"], color=[palette[1], palette[1], palette[3]][: len(inset_df)])
    inset.set_xlim(0, 1)
    inset.set_yticks(inset_y, inset_df["name"])
    inset.set_title("Unknown-label ratio", fontsize=8.0)
    inset.tick_params(axis="both", labelsize=7.0)
    for idx, ratio in enumerate(inset_df["ratio"]):
        inset.text(min(ratio + 0.02, 0.98), idx, f"{ratio:.3f}", va="center", ha="left", fontsize=6.8)
    save_figure(fig, figures_dir / main_figure_names[1], formats)
    plt.close(fig)

    # Main Fig. 3
    model_order = {
        "majority_class": 0,
        "rule_baseline": 1,
        "tfidf_logreg": 2,
        "tfidf_linearsvc": 3,
        "tfidf_multinomialnb": 4,
        "current_hybrid_baseline": 5,
        "paesc_hybrid": 6,
        "gpt55_direct": 7,
        "mmec_paesc_55_no_mechanism": 8,
        "mmec_paesc_55_no_evidence_chain": 9,
        "mmec_paesc_55": 10,
    }
    model_labels = {
        "majority_class": "Majority",
        "rule_baseline": "Rule",
        "tfidf_logreg": "TF-IDF + LogReg",
        "tfidf_linearsvc": "TF-IDF + LinearSVC",
        "tfidf_multinomialnb": "TF-IDF + MNB",
        "current_hybrid_baseline": "Reproducible hybrid",
        "paesc_hybrid": "PAESC hybrid",
        "gpt55_direct": "5.5 direct/proxy",
        "mmec_paesc_55_no_mechanism": "MMEC w/o mechanism",
        "mmec_paesc_55_no_evidence_chain": "MMEC w/o evidence",
        "mmec_paesc_55": "MMEC-PAESC 5.5",
    }
    final_fig3_df = baseline_df.copy()
    final_fig3_df["model_order"] = final_fig3_df["model_name"].map(model_order)
    final_fig3_df["model_label"] = final_fig3_df["model_name"].map(model_labels).fillna(final_fig3_df["model_name"])
    final_fig3_df = final_fig3_df.sort_values(["dataset_name", "model_order"])
    save_data_bundle(main_figure_names[2], final_fig3_df, figure_data_dir, workbook_sheets)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), sharex=True, sharey=True)
    for ax, dataset_name, panel_label in zip(
        axes,
        ["candidate_gold_strict_v1", "candidate_gold_extended_v1"],
        ["a", "b"],
    ):
        sub = final_fig3_df[final_fig3_df["dataset_name"] == dataset_name].sort_values("model_order").reset_index(drop=True)
        y_pos = np.arange(len(sub))[::-1]
        colors = []
        for model_name in sub["model_name"]:
            if model_name == "paesc_hybrid":
                colors.append(palette[2])
            elif model_name == "current_hybrid_baseline":
                colors.append(palette[1])
            else:
                colors.append(gray[2])
        ax.hlines(y_pos, sub["macro_f1_ci_low"], sub["macro_f1_ci_high"], color=gray[1], lw=0.9, zorder=1)
        ax.scatter(sub["macro_f1"], y_pos, color=colors, s=36, zorder=2)
        ax.set_yticks(y_pos, sub["model_label"])
        ax.set_xlim(0, 0.9)
        ax.grid(axis="x", alpha=0.15)
        ax.set_xlabel("Macro-F1")
        ax.set_title(DATASET_LABELS.get(dataset_name, dataset_name), fontsize=10.0)
        add_panel_label(ax, panel_label)
    axes[0].set_ylabel("Model")
    axes[1].tick_params(axis="y", labelleft=True)
    save_figure(fig, figures_dir / main_figure_names[2], formats)
    plt.close(fig)

    # Main Fig. 4
    resp_ext = resp_table[resp_table["dataset_name"] == "candidate_gold_extended_v1"].copy()
    resp_ext["responsibility_display"] = resp_ext["responsibility_label"].map(RESP_LABELS).fillna(resp_ext["responsibility_label"])
    resp_ext = resp_ext.sort_values("f1_score", ascending=True)
    audit_metric_labels = {
        "valid_span_rate": "Valid span",
        "pre_decision_span_rate": "Pre-decision span",
        "duplicate_chain_rate": "Duplicate chain",
        "role_coverage_rate": "Role coverage",
        "missing_role_rate": "Missing role",
    }
    audit_plot_df = audit_table.melt(
        id_vars=["dataset_name"],
        value_vars=list(audit_metric_labels.keys()),
        var_name="metric",
        value_name="value",
    )
    audit_plot_df["metric_label"] = audit_plot_df["metric"].map(audit_metric_labels)
    final_fig4_df = pd.concat(
        [
            resp_ext.assign(panel="responsibility"),
            audit_plot_df.assign(panel="auditability"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_data_bundle(main_figure_names[3], final_fig4_df, figure_data_dir, workbook_sheets)
    fig = plt.figure(figsize=(7.6, 4.1))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.02, 1.18], wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    y_pos = np.arange(len(resp_ext))
    ax_a.hlines(y_pos, 0, resp_ext["f1_score"], color=gray[1], lw=0.8)
    ax_a.scatter(resp_ext["f1_score"], y_pos, color=palette[1], s=34)
    ax_a.set_yticks(y_pos, resp_ext["responsibility_display"])
    ax_a.set_xlim(0, max(resp_ext["f1_score"].max() + 0.08, 0.4))
    ax_a.set_xlabel("Extended-benchmark F1")
    ax_a.set_ylabel("Responsibility class")
    ax_a.grid(axis="x", alpha=0.12)
    add_panel_label(ax_a, "a")
    width = 0.36
    metric_order = list(audit_metric_labels.keys())
    metric_x = np.arange(len(metric_order))
    for idx, dataset_name in enumerate(["candidate_gold_strict_v1", "candidate_gold_extended_v1"]):
        sub = audit_plot_df[audit_plot_df["dataset_name"] == dataset_name].set_index("metric").reindex(metric_order)
        ax_b.bar(
            metric_x + (idx - 0.5) * width,
            sub["value"],
            width=width,
            color=palette[idx + 1],
            label=DATASET_LABELS.get(dataset_name, dataset_name),
        )
    ax_b.set_xticks(metric_x, [audit_metric_labels[m] for m in metric_order], rotation=18, ha="right")
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel("Rate")
    ax_b.grid(axis="y", alpha=0.12)
    ax_b.legend(frameon=False, loc="lower left")
    add_panel_label(ax_b, "b")
    save_figure(fig, figures_dir / main_figure_names[3], formats)
    plt.close(fig)

    # Main Fig. 5
    leak_plot_df = pd.DataFrame()
    if not leakage_df.empty:
        leak_plot_df = leakage_df[leakage_df["text_mode"] != "pre_decision_only"].copy()
        if "inflation_delta_macro_f1" not in leak_plot_df.columns and "inflation_delta_macro_f1_vs_pre" in leak_plot_df.columns:
            leak_plot_df["inflation_delta_macro_f1"] = leak_plot_df["inflation_delta_macro_f1_vs_pre"]
        leak_plot_df["dataset_label"] = leak_plot_df["dataset_name"].map(DATASET_LABELS).fillna(leak_plot_df["dataset_name"])
        leak_plot_df["mode_label"] = leak_plot_df["text_mode"].map(
            {
                "post_decision_only": "Post only",
                "pre_decision_plus_post_decision": "Pre + post",
            }
        ).fillna(leak_plot_df["text_mode"])
        leak_plot_df["comparison"] = leak_plot_df["dataset_label"] + " | " + leak_plot_df["mode_label"]
    claim_plot_df = main_claim_df.copy()
    if not claim_plot_df.empty:
        claim_plot_df["tier_short"] = (
            claim_plot_df["claim_tier"].fillna("unclassified").astype(str).str.split(":").str[0]
        )
        claim_plot_df["result_label"] = claim_plot_df["result_group"].replace(
            {
                "Legacy historical result (audit caution)": "Legacy external high score",
                "Legacy holdout reference (audit caution)": "Legacy internal holdout",
                "Current reproducible baseline": "Current reproducible baseline",
                "Current audit-ready main result": "Current audit-ready main result",
            }
        )
    final_fig5_df = pd.concat(
        [
            leak_plot_df.assign(panel="leakage"),
            claim_plot_df.assign(panel="claim_boundary"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_data_bundle(main_figure_names[4], final_fig5_df, figure_data_dir, workbook_sheets)
    fig = plt.figure(figsize=(7.6, 4.0))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.95, 1.15], wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    if not leak_plot_df.empty:
        leak_plot_df = leak_plot_df.sort_values("inflation_delta_macro_f1", ascending=True)
        bar_colors = [
            palette[3] if "Pre + post" in label else palette[4]
            for label in leak_plot_df["comparison"]
        ]
        ax_a.barh(leak_plot_df["comparison"], leak_plot_df["inflation_delta_macro_f1"], color=bar_colors)
        ax_a.set_xlabel("Macro-F1 inflation vs pre-only")
        ax_a.axvline(0, color=gray[1], lw=0.8)
        for idx, value in enumerate(leak_plot_df["inflation_delta_macro_f1"]):
            ax_a.text(value + 0.001, idx, f"{value:.3f}", va="center", fontsize=7.2)
    ax_a.set_title("Leakage sentinel deltas", fontsize=10.0)
    ax_a.grid(axis="x", alpha=0.12)
    add_panel_label(ax_a, "a")
    if not claim_plot_df.empty:
        tier_color = {"Tier B": palette[1], "Tier C": palette[3]}
        claim_order = [
            "Legacy external high score",
            "Legacy internal holdout",
            "Current reproducible baseline",
            "Current audit-ready main result",
        ]
        claim_plot_df["claim_order"] = claim_plot_df["result_label"].map({name: idx for idx, name in enumerate(claim_order)})
        claim_plot_df = claim_plot_df.sort_values("claim_order", ascending=True)
        y_pos = np.arange(len(claim_plot_df))
        ax_b.barh(
            y_pos,
            claim_plot_df["macro_f1"],
            color=[tier_color.get(t, gray[2]) for t in claim_plot_df["tier_short"]],
            height=0.55,
        )
        ax_b.set_yticks(y_pos, claim_plot_df["result_label"])
        ax_b.set_xlim(0, 0.9)
        ax_b.set_xlabel("Macro-F1")
        ax_b.grid(axis="x", alpha=0.12)
        for idx, row in claim_plot_df.reset_index(drop=True).iterrows():
            ax_b.text(
                min(row["macro_f1"] + 0.015, 0.88),
                idx,
                row["tier_short"],
                va="center",
                fontsize=7.3,
                color=gray[0],
            )
    ax_b.set_title("Claim boundary", fontsize=10.0)
    add_panel_label(ax_b, "b")
    save_figure(fig, figures_dir / main_figure_names[4], formats)
    plt.close(fig)

    # Main Fig. 6
    main_cases = choose_main_cases(rep_df, n_cases=2)
    fig6_cols = [
        "case_id",
        "dataset_name",
        "y_true",
        "y_pred",
        "confidence",
        "candidate_responsibility_label",
        "primary_responsible_party",
        "procedural_compliance_status",
        "high_dispute_flag",
        "error_category",
        "delay_events_preview",
        "claims_preview",
        "defenses_preview",
        "evidence_spans",
    ]
    if not main_cases.empty:
        for col, default in [
            ("candidate_responsibility_label", "unknown"),
            ("primary_responsible_party", "unknown"),
            ("procedural_compliance_status", ""),
            ("delay_events_preview", ""),
            ("claims_preview", ""),
            ("defenses_preview", ""),
            ("evidence_spans", "[]"),
            ("responsibility_type", ""),
            ("documentation_integrity_flag", ""),
            ("explanation_text", ""),
            ("uncertainty_flag", 0),
            ("case_snippet", ""),
        ]:
            if col not in main_cases.columns:
                main_cases[col] = default
    final_fig6_df = main_cases[fig6_cols].copy() if not main_cases.empty else pd.DataFrame(columns=fig6_cols)
    save_data_bundle(main_figure_names[5], final_fig6_df, figure_data_dir, workbook_sheets)
    n_case_panels = max(len(main_cases), 1)
    fig = plt.figure(figsize=(7.6, 2.55 * n_case_panels))
    gs = GridSpec(n_case_panels, 1, figure=fig, hspace=0.18)
    if main_cases.empty:
        ax = fig.add_subplot(gs[0, 0])
        ax.axis("off")
        ax.text(0.5, 0.5, "No representative cases available", ha="center", va="center")
    else:
        for idx, (_, row) in enumerate(main_cases.iterrows()):
            ax = fig.add_subplot(gs[idx, 0])
            draw_case_columns(ax, row.to_dict(), palette, chr(97 + idx) if len(main_cases) > 1 else "")
    save_figure(fig, figures_dir / main_figure_names[5], formats)
    plt.close(fig)

    # Appendix Fig. S1
    conf_strict = confusion_data(pred_df, "candidate_gold_strict_v1").assign(dataset_name="candidate_gold_strict_v1")
    conf_extended = confusion_data(pred_df, "candidate_gold_extended_v1").assign(dataset_name="candidate_gold_extended_v1")
    figS1_df = pd.concat([conf_strict, conf_extended], ignore_index=True)
    save_data_bundle(appendix_figure_names[0], figS1_df, figure_data_dir, workbook_sheets)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    outcome_order = list(OUTCOME_LABELS.keys())
    for ax, dataset_name, panel_label in zip(
        axes,
        ["candidate_gold_strict_v1", "candidate_gold_extended_v1"],
        ["a", "b"],
    ):
        mat = (
            figS1_df[figS1_df["dataset_name"] == dataset_name]
            .set_index("true_label")[outcome_order]
            .reindex(outcome_order)
            .fillna(0)
        )
        im = ax.imshow(mat.values, cmap="Greys", vmin=0, vmax=max(mat.values.max(), 1))
        ax.set_xticks(range(len(outcome_order)), [OUTCOME_LABELS[label] for label in outcome_order], rotation=18, ha="right")
        ax.set_yticks(range(len(outcome_order)), [OUTCOME_LABELS[label] for label in outcome_order])
        ax.set_xlabel("Predicted label")
        ax.set_title(DATASET_LABELS.get(dataset_name, dataset_name), fontsize=10.0)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, int(mat.values[i, j]), ha="center", va="center", fontsize=7.4)
        add_panel_label(ax, panel_label)
    axes[0].set_ylabel("True label")
    fig.colorbar(im, ax=axes, fraction=0.028, pad=0.03)
    save_figure(fig, figures_dir / appendix_figure_names[0], formats)
    plt.close(fig)

    # Appendix Fig. S2
    extra_cases = rep_df[~rep_df["case_id"].isin(main_cases["case_id"])].copy() if not main_cases.empty else rep_df.copy()
    if not extra_cases.empty:
        for col, default in [
            ("candidate_responsibility_label", "unknown"),
            ("primary_responsible_party", "unknown"),
            ("procedural_compliance_status", ""),
            ("delay_events_preview", ""),
            ("claims_preview", ""),
            ("defenses_preview", ""),
            ("evidence_spans", "[]"),
            ("responsibility_type", ""),
            ("documentation_integrity_flag", ""),
            ("explanation_text", ""),
            ("uncertainty_flag", 0),
            ("confidence", 0.0),
            ("high_dispute_flag", False),
            ("case_snippet", ""),
        ]:
            if col not in extra_cases.columns:
                extra_cases[col] = default
    extra_cases = extra_cases.sort_values(
        ["dataset_name", "high_dispute_flag", "confidence"],
        ascending=[True, False, True],
    ).head(3)
    figS2_df = extra_cases[fig6_cols].copy() if not extra_cases.empty else pd.DataFrame(columns=fig6_cols)
    save_data_bundle(appendix_figure_names[1], figS2_df, figure_data_dir, workbook_sheets)
    n_extra = max(len(extra_cases), 1)
    fig = plt.figure(figsize=(7.6, 2.3 * n_extra))
    gs = GridSpec(n_extra, 1, figure=fig, hspace=0.16)
    if extra_cases.empty:
        ax = fig.add_subplot(gs[0, 0])
        ax.axis("off")
        ax.text(0.5, 0.5, "No additional representative cases", ha="center", va="center")
    else:
        for idx, (_, row) in enumerate(extra_cases.iterrows()):
            ax = fig.add_subplot(gs[idx, 0])
            draw_case_columns(ax, row.to_dict(), palette, chr(97 + idx))
    save_figure(fig, figures_dir / appendix_figure_names[1], formats)
    plt.close(fig)

    # Appendix Fig. S3
    figS3_df = (
        error_df.groupby(["dataset_name", "error_category"])
        .size()
        .reset_index(name="count")
        .sort_values(["dataset_name", "count"], ascending=[True, False])
    )
    save_data_bundle(appendix_figure_names[2], figS3_df, figure_data_dir, workbook_sheets)
    fig, ax = plt.subplots(figsize=(7.3, 3.7))
    pivot_s3 = figS3_df.pivot(index="error_category", columns="dataset_name", values="count").fillna(0)
    pivot_s3.plot(kind="bar", ax=ax, color=[palette[1], palette[3]])
    ax.set_xlabel("Error category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.12)
    save_figure(fig, figures_dir / appendix_figure_names[2], formats)
    plt.close(fig)

    # Appendix Fig. S4
    figS4_df = split_summary(metrics_main)
    save_data_bundle(appendix_figure_names[3], figS4_df, figure_data_dir, workbook_sheets)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(figS4_df["evaluation_slice"], figS4_df["macro_f1"], marker="o", color=palette[1], label="Macro-F1")
    ax.plot(figS4_df["evaluation_slice"], figS4_df["accuracy"], marker="s", color=palette[3], label="Accuracy")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=16)
    ax.grid(axis="y", alpha=0.12)
    ax.legend(frameon=False)
    save_figure(fig, figures_dir / appendix_figure_names[3], formats)
    plt.close(fig)

    # Workbook pack
    pack_path = paper_root / "paper_data_pack.xlsx"
    with pd.ExcelWriter(pack_path, engine="openpyxl") as writer:
        for sheet_name, df in workbook_sheets.items():
            df.to_excel(writer, sheet_name=safe_sheet_name(sheet_name), index=False)

    write_markdown(
        captions_dir / "figure_captions.md",
        f"""
        # Figure Captions
        Fig. 1. Leakage-aware workflow of DelayDispute Copilot for construction delay disputes. The workflow keeps post-decision material outside the inference path and routes it only to candidate-label derivation, forensic audit, and review-ready packaging.
        Fig. 2. Leakage-control design and candidate-benchmark governance used in the present study. Panel (a) shows the prospective pre-decision boundary, whereas panel (b) contrasts strict and extended candidate benchmarks and reports the unknown-label burden that differentiated the legacy reference setting.
        Fig. 3. Outcome-prediction performance on the strict and extended candidate benchmarks. The reproducible hybrid baseline remains the strongest pure predictor, while the PAESC hybrid is retained as the main audit-ready framework because it jointly exports responsibility and evidence-chain artifacts.
        Fig. 4. Responsibility-diagnosis performance and evidence-chain auditability on the candidate benchmarks. Panel (a) shows that responsibility prediction is materially harder than the main outcome task; panel (b) shows that traceability-oriented evidence metrics remain substantially stronger than fine-grained responsibility classification.
        Fig. 5. Forensic audit of leakage-related score inflation risk and result claim boundary. Panel (a) quantifies macro-F1 inflation under leakage stress settings; panel (b) distinguishes Tier B results that are claimable with caution from Tier C historical references that cannot serve as headline manuscript results.
        Fig. 6. Representative audit-ready case panel showing pre-decision evidence, predicted outcome, structured responsibility diagnosis, and uncertainty cues. The panels illustrate how the system supports managerial review rather than autonomous adjudication.
        Fig. S1. Confusion matrices of the PAESC hybrid on the strict and extended candidate benchmarks.
        Fig. S2. Additional representative audit-ready cases retained for appendix-level illustration.
        Fig. S3. Distribution of mechanism-oriented error categories across the candidate benchmarks.
        Fig. S4. Comparison of internal proxy validation and external candidate-benchmark performance slices.
        """,
    )
    write_markdown(
        captions_dir / "figure_captions_sci.md",
        """
        # SCI Figure Captions
        Fig. 1. Leakage-aware workflow of DelayDispute Copilot for construction delay disputes.
        Fig. 2. Leakage-control design and candidate-benchmark governance used in the present study.
        Fig. 3. Outcome-prediction performance on the strict and extended candidate benchmarks.
        Fig. 4. Responsibility-diagnosis performance and evidence-chain auditability on the candidate benchmarks.
        Fig. 5. Forensic audit of leakage-related score inflation risk and result claim boundary.
        Fig. 6. Representative audit-ready case panel showing pre-decision evidence, predicted outcome, structured responsibility diagnosis, and uncertainty cues.
        Fig. S1. Confusion matrices of the PAESC hybrid on the strict and extended candidate benchmarks.
        Fig. S2. Additional representative audit-ready cases retained for appendix-level illustration.
        Fig. S3. Distribution of mechanism-oriented error categories across the candidate benchmarks.
        Fig. S4. Comparison of internal proxy validation and external candidate-benchmark performance slices.
        """,
    )

    write_markdown(
        captions_dir / "table_notes.md",
        """
        # Table Notes
        1. `candidate_gold_strict_v1` and `candidate_gold_extended_v1` are machine-assisted candidate benchmarks, not human gold labels.
        2. `remove_pre_decision_constraint` is reported only as a leakage stress test. It is not a valid deployment setting and must not be interpreted as a claimable operating mode.
        3. Responsibility-diagnosis and evidence-chain outputs are machine-evaluated and audit-ready; they should not be described as fully human-validated findings.
        4. Historical results without complete manifest and leakage sentinel coverage remain Tier C reference material only.
        """,
    )
    write_markdown(
        captions_dir / "table_notes_sci.md",
        """
        # SCI Table Notes
        Table 1. Candidate benchmarks are machine-assisted and should not be described as true human gold.
        Table 2. The reproducible hybrid baseline is the strongest predictive comparator, whereas PAESC is retained as the integrated audit-ready framework.
        Table 3. Leakage-inclusive rows are stress tests only and are not deployment-valid settings.
        Table 4. Responsibility and evidence-chain metrics are automatic proxy metrics pending formal human review.
        Table 5. Tier B results are claimable with caution; Tier C results are historical references only.
        """,
    )

    strict_macro = metrics_main["candidate_gold_evaluation"]["candidate_gold_strict_v1"]["paesc_hybrid"]["macro_f1"]
    ext_macro = metrics_main["candidate_gold_evaluation"]["candidate_gold_extended_v1"]["paesc_hybrid"]["macro_f1"]
    ext_resp = metrics_main["candidate_gold_evaluation"]["candidate_gold_extended_v1"]["responsibility_task"]["macro_f1"]
    leak_delta_col = (
        "inflation_delta_macro_f1"
        if "inflation_delta_macro_f1" in leakage_df.columns
        else "inflation_delta_macro_f1_vs_pre"
    )
    leak_strict = float(
        leakage_df[
            (leakage_df["dataset_name"] == "candidate_gold_strict_v1")
            & (leakage_df["text_mode"] == "pre_decision_plus_post_decision")
        ][leak_delta_col].iloc[0]
    ) if not leakage_df.empty else float("nan")
    leak_extended = float(
        leakage_df[
            (leakage_df["dataset_name"] == "candidate_gold_extended_v1")
            & (leakage_df["text_mode"] == "pre_decision_plus_post_decision")
        ][leak_delta_col].iloc[0]
    ) if not leakage_df.empty else float("nan")

    write_markdown(
        text_dir / "managerial_relevance_statement.md",
        f"""
        DelayDispute Copilot is positioned as a managerial decision-support framework rather than an autonomous adjudicator. By enforcing a pre-decision input boundary, reconstructing auditable evidence chains, and packaging structured responsibility signals with uncertainty cues, the system helps project teams identify documentation gaps, procedural vulnerabilities, and high-dispute cases before formal adjudication. On the extended candidate benchmark, the PAESC hybrid reached an outcome macro-F1 of {ext_macro:.3f} while preserving evidence-auditability rates above 0.98, which supports governance-oriented triage, claim preparation, and review workflows under human oversight.
        """,
    )

    write_markdown(
        text_dir / "contribution_statement.md",
        f"""
        This study contributes a leakage-aware, audit-ready decision-support framework for construction schedule delay disputes. First, it operationalizes a prospective evaluation setting by separating pre-decision information from post-decision anchors. Second, it upgrades responsibility diagnosis from informal hints to structured outputs tied to procedural compliance, documentation integrity, and evidence-citation fields. Third, it introduces a forensic claim-boundary layer that distinguishes claimable current results from historical references that remain audit-incomplete. Under this framing, the PAESC hybrid achieved macro-F1 values of {strict_macro:.3f} on the strict candidate benchmark and {ext_macro:.3f} on the extended candidate benchmark, substantially above traditional TF-IDF baselines while retaining responsibility and evidence outputs.
        """,
    )

    write_markdown(
        text_dir / "validation_statement.md",
        f"""
        Validation currently proceeds at three levels. Level A reports automatic outcome-prediction metrics on internal proxy validation and candidate-benchmark evaluation. Level B reports responsibility-diagnosis metrics and uncertainty indicators; on the extended candidate benchmark, responsibility macro-F1 remained {ext_resp:.3f}, indicating that this task is materially harder and should be interpreted cautiously. Level C reports evidence-chain auditability metrics such as valid-span rate and pre-decision-span rate, both of which remained above 0.98 on the current candidate benchmarks. These outputs are machine-evaluated and audit-ready; they are not equivalent to completed human validation.
        """,
    )

    write_markdown(
        text_dir / "limitations_statement.md",
        """
        The main limitation is that the available evaluation resources remain machine-assisted rather than fully human-validated. The strict and extended candidate benchmarks are appropriate for controlled comparison and reproducibility recovery, but they should not be described as true human gold. Responsibility diagnosis also remains sensitive to label-source heterogeneity, class imbalance, and schema granularity. Accordingly, the current system should be framed as an auditable decision-support tool that supports managerial review rather than a substitute for expert adjudication.
        """,
    )

    write_markdown(
        text_dir / "methodology_summary.md",
        f"""
        The implemented workflow integrates leakage-aware parsing, candidate-benchmark governance, multi-model outcome evaluation, structured responsibility diagnosis, evidence-chain reconstruction, forensic audit, and paper-asset generation. The resulting package reports both predictive performance and claim-boundary logic. The leakage sentinel shows macro-F1 inflation of {leak_strict:.3f} on the strict benchmark and {leak_extended:.3f} on the extended benchmark when pre- and post-decision text are combined, which justifies the prospective pre-decision constraint adopted throughout the manuscript.
        """,
    )

    write_markdown(
        text_dir / "figure_table_placement_outline.md",
        """
        # Figure and Table Placement Outline
        ## Main-text figures
        1. Fig. 1. DelayDispute Copilot workflow
        2. Fig. 2. Leakage control and candidate-benchmark governance
        3. Fig. 3. Outcome-prediction performance comparison
        4. Fig. 4. Responsibility difficulty and evidence-chain auditability
        5. Fig. 5. Forensic audit of score inflation risk and claim boundary
        6. Fig. 6. Representative audit-ready case panel

        ## Main-text tables
        1. Table 1. Candidate benchmark profile
        2. Table 2. Main outcome-prediction results
        3. Table 3. Ablation and leakage-stress results
        4. Table 4. Responsibility and auditability summary
        5. Table 5. Forensic claim boundary

        ## Appendix / supplementary package
        - Fig. S1. Extended confusion matrices
        - Fig. S2. Additional representative cases
        - Fig. S3. Error-category distribution
        - Fig. S4. Split-comparison figure
        - Table S1. Full per-class outcome metrics
        - Table S2. Full per-class responsibility metrics
        - Table S3. Full leakage sentinel results
        - Table S4. Full forensic delta audit
        - Table S5. Responsibility root-cause evidence
        - Table S6. Expanded ablation sheet
        - Table S7. Candidate-benchmark long profile
        """,
    )

    write_markdown(
        text_dir / "final_visual_package.md",
        """
        # Final Visual Blueprint
        The manuscript package is constrained to six main-text figures and five main-text tables in order to match the restrained visual grammar of AIC, AEI, and IEEE-TEM papers. Figures carry workflow, benchmark governance, performance patterns, auditability contrasts, forensic claim boundaries, and one representative case panel. Tables carry exact counts, benchmark definitions, main numerical results, ablations, auditability summaries, and claim boundary logic.

        ## Main-text figures
        - `fig1_delaydispute_workflow`: horizontal workflow figure with an audit lane.
        - `fig2_leakage_benchmark_governance`: two-panel governance figure showing leakage boundary and candidate-benchmark composition.
        - `fig3_outcome_prediction_comparison`: two-panel dot-whisker comparison across traditional baselines, the reproducible hybrid baseline, and the PAESC hybrid.
        - `fig4_responsibility_auditability`: two-panel figure showing responsibility-diagnosis difficulty and evidence-chain auditability.
        - `fig5_forensic_claim_boundary`: leakage-sentinel delta plot plus claim-tier comparison.
        - `fig6_representative_case_panel`: structured case panel for audit-ready review.

        ## Main-text tables
        - `table1_candidate_benchmark_profile`
        - `table2_main_outcome_results`
        - `table3_ablation_leakage_stress`
        - `table4_responsibility_auditability_summary`
        - `table5_forensic_claim_boundary`

        ## Appendix allocation
        Detailed confusion matrices, per-class sheets, full forensic deltas, leakage-sentinel rows, responsibility root-cause diagnostics, expanded ablations, and additional case panels remain in appendix or online supplementary material because they are important for audit traceability but too granular for the manuscript core.
        """,
    )

    write_markdown(
        text_dir / "second_stage_implementation_plan.md",
        """
        # Second-Stage Implementation Plan
        1. Freeze the current six-figure/five-table package as the manuscript default.
        2. Use Tables 1, 2, and 5 plus Figures 1, 3, and 4 as the safest first submission batch.
        3. Keep Figure 5 and Table 5 aligned with the forensic narrative; update them only when the audit logic changes.
        4. Treat Figure 6 as a compact analytical exhibit, not a screenshot replacement.
        5. Move confusion matrices, full root-cause diagnostics, and all full-width audit sheets to appendix.
        """,
    )

    write_markdown(
        text_dir / "visual_design_checklist.md",
        """
        # How to Make These Figures and Tables Look Like Published AIC/AEI Papers Rather Than Homemade Graphics
        - Keep exact metrics in tables, not sprayed across figures.
        - Use no more than three to four muted colors in any figure.
        - Prefer horizontal workflows and compact multi-panel layouts.
        - Use dot-whisker or lollipop plots rather than heavy grouped bars for benchmark comparison.
        - Put caveats, benchmark definitions, and claim boundaries in captions and table notes.
        - Avoid icons, gradients, shadows, and presentation-style callouts.
        - Use light borders, consistent line weights, and generous whitespace.
        - Keep labels short and noun-based.
        - Show one representative case well rather than many cases poorly.
        - Never visually imply that candidate benchmarks are human gold or that responsibility outputs are fully human-validated.
        """,
    )

    write_markdown(
        paper_root / "CODE_FILE_GUIDE.md",
        """
        # Code File Guide
        - `src/pipeline_all_in_one.py`: Parses dispute documents, separates pre-decision and post-decision segments, and exports structured case records.
        - `src/research_support.py`: Shared utilities for leakage-aware parsing, metrics, evidence extraction, and configuration management.
        - `src/build_candidate_gold.py`: Builds candidate-benchmark assets, manifests, guidelines, and review-ready subsets.
        - `src/llm_step2_fast_qc.py`: Runs the existing LLM labeling stage and preserves local key-loading behavior while exporting QC artifacts.
        - `src/final_eval.py`: Executes traditional baselines, the reproducible hybrid baseline, and the PAESC hybrid; exports outcome, responsibility, and evidence-chain outputs.
        - `src/run_ablation.py`: Evaluates removal-based ablations under the current candidate-benchmark protocol.
        - `src/error_analysis.py`: Produces mechanism-oriented error categories and representative-case exports.
        - `src/run_forensic_audit.py`: Recomputes historical metrics, quantifies leakage inflation risk, and assigns claim tiers.
        - `src/make_paper_figures.py`: Generates the final six-figure/five-table manuscript package, appendix assets, and figure/table source data.
        - `src/run_full_research.py`: Orchestrates the end-to-end reproducible workflow from structuring to paper assets.
        """,
    )

    print(f"[DONE] paper assets written to {paper_root}")


if __name__ == "__main__":
    main()
