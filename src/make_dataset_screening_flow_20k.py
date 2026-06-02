from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = ROOT / "data" / "lora_exports" / "lora_strict49k_mimo_v2_final_20260528_164927" / "strong_label_master_v2.csv"
STRICT_PATH = ROOT / "data" / "1_raw_text" / "combined_delay_dispute_corpus_20260527" / "strict_delay_usable_manifest.csv"
FIG_DIR = ROOT / "paper_assets" / "figures" / "dataset_profile_20k"
DATA_DIR = ROOT / "paper_assets" / "figure_data" / "dataset_profile_20k"
DESKTOP = Path.home() / "Desktop"


def load_counts() -> dict:
    master = pd.read_csv(MASTER_PATH, usecols=["case_id", "text_sha256", "outcome_label"], low_memory=False)
    strict = pd.read_csv(STRICT_PATH, usecols=["case_id", "text_sha256", "usable_tier"], low_memory=False)
    master_sha = set(master["text_sha256"].dropna())
    strict_sha = set(strict["text_sha256"].dropna())
    overlap_sha = master_sha & strict_sha
    return {
        "deduplicated_corpus": 112212,
        "chinese_delay_research_candidate": 67913,
        "excluded_after_keyword_screening": 112212 - 67913,
        "chinese_delay_usable": 49506,
        "excluded_after_usability_screening": 67913 - 49506,
        "strict_delay_usable": len(strict_sha),
        "excluded_after_strict_screening": 49506 - len(strict_sha),
        "strong_label_master": len(master_sha),
        "lora_train": 22028,
        "lora_dev": 3004,
        "strict_labeled_overlap": len(overlap_sha),
        "labeled_not_strict": len(master_sha - strict_sha),
        "strict_not_labeled": len(strict_sha - master_sha),
        "master_label_distribution": master["outcome_label"].value_counts().rename_axis("label").reset_index(name="count"),
        "strict_labeled_distribution": master[master["text_sha256"].isin(overlap_sha)]["outcome_label"].value_counts().rename_axis("label").reset_index(name="count"),
        "strict_tier_distribution": strict["usable_tier"].value_counts().rename_axis("usable_tier").reset_index(name="count"),
    }


def add_box(ax, xy, wh, text, fontsize=9, dashed=True, lw=1.0, fc="white", ec="#222222"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.004",
        linewidth=lw,
        linestyle=(0, (4, 3)) if dashed else "solid",
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, linespacing=1.25)
    return patch


def arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#111111", shrinkA=2, shrinkB=2))


def stage_label(ax, y, text):
    ax.add_patch(Rectangle((0.025, y - 0.075), 0.055, 0.15, facecolor="#E8F2FB", edgecolor="#1D6AA8", linewidth=0.9))
    ax.text(0.052, y, text, rotation=90, ha="center", va="center", fontsize=11, fontstyle="italic", fontweight="bold")


def red_n(n: int) -> str:
    return f"n = {n:,}"


def draw_flow(counts: dict, out_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "savefig.dpi": 600,
        }
    )
    fig, ax = plt.subplots(figsize=(8.6, 10.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title = "Identification and screening of schedule-delay dispute cases"
    ax.add_patch(
        FancyBboxPatch(
            (0.10, 0.935),
            0.84,
            0.045,
            boxstyle="round,pad=0.01,rounding_size=0.008",
            facecolor="#FFF5EC",
            edgecolor="#F26C23",
            linewidth=1.2,
        )
    )
    ax.text(0.52, 0.957, title, ha="center", va="center", fontsize=11, fontweight="bold")

    stage_label(ax, 0.80, "Identification")
    stage_label(ax, 0.55, "Screening")
    stage_label(ax, 0.32, "Eligibility")
    stage_label(ax, 0.16, "Included")

    add_box(
        ax,
        (0.14, 0.76),
        (0.34, 0.13),
        "Records identified from:\n• Public adjudicated documents\n• Construction-related disputes\n• Schedule-delay search terms",
        fontsize=8.6,
    )
    add_box(
        ax,
        (0.56, 0.76),
        (0.34, 0.13),
        "Keyword combinations:\n• construction project / contract\n• schedule delay / extension of time\n• liquidated damages / responsibility",
        fontsize=8.6,
    )

    add_box(
        ax,
        (0.32, 0.665),
        (0.36, 0.055),
        f"Deduplicated candidate corpus: $\\bf{{{red_n(counts['deduplicated_corpus'])}}}$",
        fontsize=9.4,
    )
    arrow(ax, (0.31, 0.76), (0.43, 0.72))
    arrow(ax, (0.73, 0.76), (0.57, 0.72))

    add_box(
        ax,
        (0.27, 0.565),
        (0.45, 0.065),
        f"Schedule-delay research candidates: $\\bf{{{red_n(counts['chinese_delay_research_candidate'])}}}$",
        fontsize=9.2,
    )
    arrow(ax, (0.50, 0.665), (0.50, 0.63))

    add_box(
        ax,
        (0.735, 0.545),
        (0.22, 0.105),
        f"Excluded: $\\bf{{{red_n(counts['excluded_after_keyword_screening'])}}}$\n• weak delay relevance\n• weak construction context\n• duplicate / noisy records",
        fontsize=7.7,
    )
    arrow(ax, (0.72, 0.597), (0.735, 0.597))

    add_box(
        ax,
        (0.29, 0.43),
        (0.42, 0.068),
        f"Usable Chinese delay pool: $\\bf{{{red_n(counts['chinese_delay_usable'])}}}$",
        fontsize=9.2,
    )
    arrow(ax, (0.50, 0.565), (0.50, 0.498))
    add_box(
        ax,
        (0.735, 0.405),
        (0.22, 0.13),
        f"Excluded: $\\bf{{{red_n(counts['excluded_after_usability_screening'])}}}$\n• insufficient facts\n• procedural-only records\n• unclear delay issue\n• low evidence completeness",
        fontsize=7.5,
    )
    arrow(ax, (0.71, 0.464), (0.735, 0.464))

    add_box(
        ax,
        (0.275, 0.285),
        (0.45, 0.072),
        f"Strict schedule-delay usable pool: $\\bf{{{red_n(counts['strict_delay_usable'])}}}$",
        fontsize=9.2,
    )
    arrow(ax, (0.50, 0.43), (0.50, 0.357))
    add_box(
        ax,
        (0.735, 0.265),
        (0.22, 0.13),
        f"Excluded: $\\bf{{{red_n(counts['excluded_after_strict_screening'])}}}$\n• weaker management relevance\n• weaker pre-decision facts\n• lower evidence/procedure signal",
        fontsize=7.5,
    )
    arrow(ax, (0.725, 0.321), (0.735, 0.321))

    add_box(
        ax,
        (0.12, 0.105),
        (0.36, 0.12),
        f"Outcome-labeled LoRA master:\n$\\bf{{{red_n(counts['strong_label_master'])}}}$\ntrain/dev = {counts['lora_train']:,}/{counts['lora_dev']:,}",
        fontsize=9.0,
        fc="#F7FAFD",
        ec="#2F7FBD",
        dashed=False,
        lw=1.1,
    )
    add_box(
        ax,
        (0.54, 0.105),
        (0.34, 0.12),
        f"Strict ∩ labeled subset:\n$\\bf{{{red_n(counts['strict_labeled_overlap'])}}}$\nstrict-only supervised setting",
        fontsize=9.0,
        fc="#F7FAFD",
        ec="#2F7FBD",
        dashed=False,
        lw=1.1,
    )
    arrow(ax, (0.50, 0.285), (0.69, 0.225))
    arrow(ax, (0.48, 0.165), (0.54, 0.165))

    ax.text(
        0.50,
        0.065,
        f"Important boundary: {counts['strong_label_master']:,} is not reduced to {counts['strict_delay_usable']:,}. "
        f"Labeled but outside strict pool = {counts['labeled_not_strict']:,}; strict pool not yet outcome-labeled = {counts['strict_not_labeled']:,}.",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#333333",
    )

    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_excel(counts: dict, out_xlsx: Path) -> None:
    rows = [
        ("deduplicated_candidate_corpus", counts["deduplicated_corpus"], "Initial deduplicated corpus from combined raw-text manifest."),
        ("schedule_delay_research_candidates", counts["chinese_delay_research_candidate"], "Broad Chinese delay research candidate pool."),
        ("excluded_after_keyword_screening", counts["excluded_after_keyword_screening"], "Deduplicated corpus minus broad delay research candidates."),
        ("usable_chinese_delay_pool", counts["chinese_delay_usable"], "Stricter usable Chinese delay pool."),
        ("excluded_after_usability_screening", counts["excluded_after_usability_screening"], "Research candidates minus usable Chinese delay pool."),
        ("strict_schedule_delay_usable_pool", counts["strict_delay_usable"], "Current strict delay-dispute research pool."),
        ("excluded_after_strict_screening", counts["excluded_after_strict_screening"], "Usable Chinese delay pool minus strict delay usable pool."),
        ("outcome_labeled_lora_master", counts["strong_label_master"], "Final outcome-labeled supervised LoRA master."),
        ("lora_train", counts["lora_train"], "Train split from outcome-labeled LoRA master."),
        ("lora_dev", counts["lora_dev"], "Dev split from outcome-labeled LoRA master."),
        ("strict_labeled_overlap", counts["strict_labeled_overlap"], "Intersection between strict pool and outcome-labeled master by text_sha256."),
        ("labeled_not_strict", counts["labeled_not_strict"], "Outcome-labeled master records not in strict delay pool."),
        ("strict_not_labeled", counts["strict_not_labeled"], "Strict delay pool records without outcome label in current master."),
    ]
    flow_counts = pd.DataFrame(rows, columns=["item", "count", "definition"])
    interpretation = pd.DataFrame(
        [
            {
                "question": "Can 25,032 be reduced to 21,233?",
                "answer": "No. They are two different pools. 25,032 is an outcome-labeled LoRA master; 21,233 is a strict delay-dispute research pool.",
            },
            {
                "question": "Which pool is for supervised LoRA training?",
                "answer": "Use 25,032 if broad labeled negatives are allowed. Use 7,636 if the experiment requires both strict-delay membership and existing outcome labels.",
            },
            {
                "question": "Which pool is for corpus statistics/RAG?",
                "answer": "Use the strict delay usable pool of 21,233, because it has stronger delay-dispute relevance but not all rows are outcome-labeled.",
            },
        ]
    )
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        flow_counts.to_excel(writer, sheet_name="screening_flow_counts", index=False)
        interpretation.to_excel(writer, sheet_name="interpretation", index=False)
        counts["master_label_distribution"].to_excel(writer, sheet_name="master_label_distribution", index=False)
        counts["strict_labeled_distribution"].to_excel(writer, sheet_name="strict_labeled_distribution", index=False)
        counts["strict_tier_distribution"].to_excel(writer, sheet_name="strict_tier_distribution", index=False)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    counts = load_counts()
    out_base = FIG_DIR / "fig_dataset_screening_flow_20k"
    out_xlsx = DATA_DIR / "dataset_screening_flow_20k_source_data.xlsx"
    draw_flow(counts, out_base)
    write_excel(counts, out_xlsx)
    desktop_png = DESKTOP / "fig_dataset_screening_flow_20k.png"
    desktop_xlsx = DESKTOP / "dataset_screening_flow_20k_source_data.xlsx"
    shutil.copy2(out_base.with_suffix(".png"), desktop_png)
    shutil.copy2(out_xlsx, desktop_xlsx)
    print("screening_flow_png", out_base.with_suffix(".png"))
    print("screening_flow_pdf", out_base.with_suffix(".pdf"))
    print("screening_flow_svg", out_base.with_suffix(".svg"))
    print("screening_flow_excel", out_xlsx)
    print("desktop_png", desktop_png)
    print("desktop_excel", desktop_xlsx)
    print({k: v for k, v in counts.items() if isinstance(v, int)})


if __name__ == "__main__":
    main()
