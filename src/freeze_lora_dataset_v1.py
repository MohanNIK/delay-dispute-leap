# -*- coding: utf-8 -*-
"""Freeze LoRA v1 dataset artifacts from the finalized 2384 strong-label set."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


SOURCE_DIR = PROJECT_ROOT / "data/lora_exports/high_conf_lora_qwen_flash_full_20260522"
DEFAULT_OUT = PROJECT_ROOT / "data/lora_exports/lora_frozen_v1_2384"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def label_export(label: str) -> str:
    return "partial_support" if label == "partial" else str(label)


def write_split_report(out_dir: Path, master: pd.DataFrame, train: pd.DataFrame, dev: pd.DataFrame, test_labels: pd.DataFrame) -> None:
    lines = [
        "# LoRA Dataset Split Report v1",
        "",
        f"Created at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Frozen datasets",
        f"- strong_label_master_v1_2384: {len(master)} cases",
        f"- train_v1_2098: {len(train)} cases",
        f"- dev_v1_286: {len(dev)} cases",
        f"- frozen_test500_v1: {len(test_labels)} cases",
        "",
        "## Label distributions",
        "",
        "| split | support | partial_support | not_support | total |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, df, col in [
        ("master", master, "outcome_label"),
        ("train", train, "outcome_label"),
        ("dev", dev, "outcome_label"),
    ]:
        counts = df[col].map(label_export).value_counts().to_dict()
        lines.append(f"| {name} | {counts.get('support',0)} | {counts.get('partial_support',0)} | {counts.get('not_support',0)} | {len(df)} |")
    counts = test_labels["private_label"].value_counts().to_dict()
    lines.append(f"| frozen_test500_private | {counts.get('support',0)} | {counts.get('partial_support',0)} | {counts.get('not_support',0)} | {len(test_labels)} |")
    lines += [
        "",
        "## Leakage discipline",
        "- LoRA train/dev inputs contain pre-decision information only.",
        "- Post-decision text was used only to construct machine-assisted labels and is not included in train/dev JSONL or raw prompt files.",
        "- Frozen test labels are kept private and are not included in the external LoRA training zip.",
        "",
        "## Recommended training files",
        "- Primary: `lora_train_alpaca.jsonl`, `lora_dev_alpaca.jsonl`.",
        "- Alternative raw prompt format: `lora_train_raw.txt`, `lora_dev_raw.txt`.",
        "- Evidence-conditioned optional files may be used in a separate experiment.",
    ]
    (out_dir / "split_report_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_manifest(out_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in sorted(out_dir.glob("*")):
        if path.is_file() and path.name != "dataset_manifest_v1.csv":
            rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "dataset_manifest_v1.csv", index=False, encoding="utf-8-sig")
    return manifest


def make_external_zip(out_dir: Path) -> Path:
    include = [
        "lora_train_alpaca.jsonl",
        "lora_dev_alpaca.jsonl",
        "lora_train_raw.txt",
        "lora_dev_raw.txt",
        "lora_train_evidence_conditioned_alpaca.jsonl",
        "lora_dev_evidence_conditioned_alpaca.jsonl",
        "lora_data_readme.md",
        "split_report_v1.md",
        "label_distribution_v1.csv",
        "class_weight_metadata.csv",
        "dataset_manifest_v1.csv",
    ]
    zip_path = out_dir / "data_package_for_external_lora_finetuning.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in include:
            path = out_dir / name
            if path.exists():
                zf.write(path, arcname=name)
    return zip_path


def freeze_dataset(source_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    copies = {
        "strong_label_master.csv": "strong_label_master_v1_2384.csv",
        "lora_train_manifest.csv": "train_manifest_v1.csv",
        "lora_dev_manifest.csv": "dev_manifest_v1.csv",
        "frozen_test_input_only.jsonl": "frozen_test500_input_only_v1.jsonl",
        "frozen_test_labels_private.csv": "frozen_test500_labels_private_v1.csv",
        "lora_train_alpaca.jsonl": "lora_train_alpaca.jsonl",
        "lora_dev_alpaca.jsonl": "lora_dev_alpaca.jsonl",
        "lora_train_raw.txt": "lora_train_raw.txt",
        "lora_dev_raw.txt": "lora_dev_raw.txt",
        "lora_train_evidence_conditioned_alpaca.jsonl": "lora_train_evidence_conditioned_alpaca.jsonl",
        "lora_dev_evidence_conditioned_alpaca.jsonl": "lora_dev_evidence_conditioned_alpaca.jsonl",
        "lora_data_readme.md": "lora_data_readme.md",
        "class_weight_metadata.csv": "class_weight_metadata.csv",
    }
    for src_name, dst_name in copies.items():
        copy_file(source_dir / src_name, out_dir / dst_name)

    master = pd.read_csv(out_dir / "strong_label_master_v1_2384.csv", encoding="utf-8-sig")
    train = pd.read_csv(out_dir / "train_manifest_v1.csv", encoding="utf-8-sig")
    dev = pd.read_csv(out_dir / "dev_manifest_v1.csv", encoding="utf-8-sig")
    test_labels = pd.read_csv(out_dir / "frozen_test500_labels_private_v1.csv", encoding="utf-8-sig")

    label_rows = []
    for name, df, col in [
        ("strong_label_master_v1_2384", master, "outcome_label"),
        ("train_v1_2098", train, "outcome_label"),
        ("dev_v1_286", dev, "outcome_label"),
    ]:
        labels = df[col].map(label_export)
        total = len(labels)
        for label, count in labels.value_counts().to_dict().items():
            label_rows.append({"split": name, "label": label, "count": count, "ratio": count / total if total else 0.0})
    total = len(test_labels)
    for label, count in test_labels["private_label"].value_counts().to_dict().items():
        label_rows.append({"split": "frozen_test500_v1_private", "label": label, "count": count, "ratio": count / total if total else 0.0})
    pd.DataFrame(label_rows).to_csv(out_dir / "label_distribution_v1.csv", index=False, encoding="utf-8-sig")

    write_split_report(out_dir, master, train, dev, test_labels)
    make_manifest(out_dir)
    make_external_zip(out_dir)
    make_manifest(out_dir)
    print(f"FROZEN_LORA_V1_DIR={out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(SOURCE_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    freeze_dataset(PROJECT_ROOT / args.source_dir if not Path(args.source_dir).is_absolute() else Path(args.source_dir), PROJECT_ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
