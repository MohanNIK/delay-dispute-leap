# -*- coding: utf-8 -*-
"""Evaluate roommate LoRA prediction CSV against frozen test500 labels.

Expected prediction CSV columns:
    case_id, prediction
Optional:
    confidence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = ["support", "partial_support", "not_support"]


def normalize_label(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw == "partial":
        return "partial_support"
    if raw in LABELS:
        return raw
    if "partial" in raw or "部分" in raw:
        return "partial_support"
    if "not" in raw or "驳回" in raw or "不予" in raw:
        return "not_support"
    if "support" in raw or "支持" in raw:
        return "support"
    return "unknown"


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def compute_metrics(rows: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    df = pd.DataFrame(list(rows))
    df = df[df["y_true"].isin(LABELS) & df["y_pred"].isin(LABELS)].copy()
    n = len(df)
    correct = int((df["y_true"] == df["y_pred"]).sum()) if n else 0
    per, cm, f1s = [], [], []
    weighted = 0.0
    for lab in LABELS:
        tp = int(((df["y_true"] == lab) & (df["y_pred"] == lab)).sum())
        fp = int(((df["y_true"] != lab) & (df["y_pred"] == lab)).sum())
        fn = int(((df["y_true"] == lab) & (df["y_pred"] != lab)).sum())
        support = int((df["y_true"] == lab).sum())
        p, r, f = prf(tp, fp, fn)
        f1s.append(f)
        weighted += f * support
        per.append({"label": lab, "precision": p, "recall": r, "f1": f, "support": support})
        for pred in LABELS:
            cm.append({"y_true": lab, "y_pred": pred, "count": int(((df["y_true"] == lab) & (df["y_pred"] == pred)).sum())})
    return {"n": n, "accuracy": correct / n if n else 0.0, "macro_f1": sum(f1s) / len(f1s), "weighted_f1": weighted / n if n else 0.0}, per, cm


def evaluate(pred_path: Path, label_path: Path, out_dir: Path, pred_col: str, conf_col: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(pred_path, encoding="utf-8-sig")
    labels = pd.read_csv(label_path, encoding="utf-8-sig")
    if pred_col not in pred.columns:
        raise ValueError(f"Prediction column {pred_col!r} not found in {pred_path}")
    pred["case_id"] = pred["case_id"].astype(str)
    labels["case_id"] = labels["case_id"].astype(str)
    pred["y_pred"] = pred[pred_col].map(normalize_label)
    labels["y_true"] = labels["private_label"].map(normalize_label)
    merged = labels.merge(pred, on="case_id", how="left")
    merged["y_pred"] = merged["y_pred"].fillna("unknown")
    if conf_col in merged.columns:
        merged["confidence"] = pd.to_numeric(merged[conf_col], errors="coerce")
    else:
        merged["confidence"] = pd.NA
    merged.to_csv(out_dir / "lora_predictions_joined.csv", index=False, encoding="utf-8-sig")

    valid = merged[merged["y_pred"].isin(LABELS)].copy()
    rows = valid[["case_id", "y_true", "y_pred", "confidence"]].to_dict("records")
    metrics, per, cm = compute_metrics(rows)
    metrics.update({"prediction_file": str(pred_path), "label_file": str(label_path), "coverage": len(valid) / len(labels) if len(labels) else 0.0})
    (out_dir / "metrics_main.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(per).to_csv(out_dir / "per_class_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cm).to_csv(out_dir / "confusion_matrix_data.csv", index=False, encoding="utf-8-sig")

    if "confidence" in valid.columns and valid["confidence"].notna().any():
        curves = []
        for threshold in [i / 100 for i in range(0, 101, 5)]:
            sub = valid[pd.to_numeric(valid["confidence"], errors="coerce").fillna(-1) >= threshold]
            m, _, _ = compute_metrics(sub[["case_id", "y_true", "y_pred"]].to_dict("records"))
            curves.append({"threshold": threshold, "n": len(sub), "coverage": len(sub) / len(labels) if len(labels) else 0.0, **m})
        pd.DataFrame(curves).to_csv(out_dir / "selective_threshold_results.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["threshold", "n", "coverage", "accuracy", "macro_f1", "weighted_f1"]).to_csv(out_dir / "selective_threshold_results.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", default="data/lora_exports/lora_frozen_v1_2384/frozen_test500_labels_private_v1.csv")
    parser.add_argument("--out-dir", default="results/lora_prediction_eval")
    parser.add_argument("--pred-col", default="prediction")
    parser.add_argument("--conf-col", default="confidence")
    args = parser.parse_args()
    evaluate(PROJECT_ROOT / args.predictions, PROJECT_ROOT / args.labels, PROJECT_ROOT / args.out_dir, args.pred_col, args.conf_col)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
