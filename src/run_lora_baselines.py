# -*- coding: utf-8 -*-
"""Run traditional baselines on frozen LoRA v1 train/dev/test splits."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = ["support", "partial_support", "not_support"]


def normalize_label(value: Any) -> str:
    raw = str(value or "").strip()
    return "partial_support" if raw == "partial" else raw


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
    per_class: List[Dict[str, Any]] = []
    cm: List[Dict[str, Any]] = []
    f1s: List[float] = []
    weighted = 0.0
    for lab in LABELS:
        tp = int(((df["y_true"] == lab) & (df["y_pred"] == lab)).sum())
        fp = int(((df["y_true"] != lab) & (df["y_pred"] == lab)).sum())
        fn = int(((df["y_true"] == lab) & (df["y_pred"] != lab)).sum())
        support = int((df["y_true"] == lab).sum())
        p, r, f = prf(tp, fp, fn)
        f1s.append(f)
        weighted += f * support
        per_class.append({"label": lab, "precision": p, "recall": r, "f1": f, "support": support})
        for pred in LABELS:
            cm.append({"y_true": lab, "y_pred": pred, "count": int(((df["y_true"] == lab) & (df["y_pred"] == pred)).sum())})
    return {"n": n, "accuracy": correct / n if n else 0.0, "macro_f1": sum(f1s) / len(f1s), "weighted_f1": weighted / n if n else 0.0}, per_class, cm


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train_manifest_v1.csv", encoding="utf-8-sig")
    dev = pd.read_csv(data_dir / "dev_manifest_v1.csv", encoding="utf-8-sig")
    test_x = pd.read_json(data_dir / "frozen_test500_input_only_v1.jsonl", lines=True)
    test_y = pd.read_csv(data_dir / "frozen_test500_labels_private_v1.csv", encoding="utf-8-sig")
    test = test_x.merge(test_y, on="case_id", how="inner")
    train["label"] = train["outcome_label"].map(normalize_label)
    dev["label"] = dev["outcome_label"].map(normalize_label)
    test["label"] = test["private_label"].map(normalize_label)
    train["text"] = train["pre_decision_text"].fillna("").astype(str)
    dev["text"] = dev["pre_decision_text"].fillna("").astype(str)
    test["text"] = test["input"].fillna("").astype(str)
    return train, dev, test


def model_specs() -> Dict[str, Any]:
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=80000)
    char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=80000)
    union = FeatureUnion([("word", word), ("char", char)])
    return {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "tfidf_logreg_balanced": Pipeline([("tfidf", union), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1))]),
        "tfidf_linearsvc_balanced": Pipeline([("tfidf", union), ("clf", LinearSVC(class_weight="balanced"))]),
        "tfidf_multinomialnb": Pipeline([("tfidf", union), ("clf", MultinomialNB(alpha=0.2))]),
    }


def predict_model(model: Any, x: pd.Series) -> List[str]:
    return [normalize_label(v) for v in model.predict(x)]


def run_baselines(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train, dev, test = load_data(data_dir)
    train_all = pd.concat([train, dev], ignore_index=True)
    rows: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []
    per_rows: List[Dict[str, Any]] = []
    cm_rows: List[Dict[str, Any]] = []

    for name, model in model_specs().items():
        model.fit(train["text"], train["label"])
        for split_name, split_df in [("dev", dev)]:
            preds = predict_model(model, split_df["text"])
            pred_rows = [{"case_id": cid, "split": split_name, "model_name": name, "y_true": y, "y_pred": p} for cid, y, p in zip(split_df["case_id"], split_df["label"], preds)]
            rows.extend(pred_rows)
            m, per, cm = compute_metrics(pred_rows)
            metrics_rows.append({"split": split_name, "model_name": name, **m})
            per_rows.extend([{**r, "split": split_name, "model_name": name} for r in per])
            cm_rows.extend([{**r, "split": split_name, "model_name": name} for r in cm])

        model.fit(train_all["text"], train_all["label"])
        preds = predict_model(model, test["text"])
        pred_rows = [{"case_id": cid, "split": "frozen_test500", "model_name": name, "y_true": y, "y_pred": p} for cid, y, p in zip(test["case_id"], test["label"], preds)]
        rows.extend(pred_rows)
        m, per, cm = compute_metrics(pred_rows)
        metrics_rows.append({"split": "frozen_test500", "model_name": name, **m})
        per_rows.extend([{**r, "split": "frozen_test500", "model_name": name} for r in per])
        cm_rows.extend([{**r, "split": "frozen_test500", "model_name": name} for r in cm])

    pd.DataFrame(rows).to_csv(out_dir / "baseline_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics_rows).to_csv(out_dir / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_rows).to_csv(out_dir / "baseline_per_class_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cm_rows).to_csv(out_dir / "baseline_confusion_matrix_data.csv", index=False, encoding="utf-8-sig")
    (out_dir / "baseline_run_manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "python_version": sys.version,
                "platform": platform.platform(),
                "data_dir": str(data_dir),
                "train_n": len(train),
                "dev_n": len(dev),
                "test_n": len(test),
                "models": list(model_specs().keys()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(metrics_rows).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/lora_exports/lora_frozen_v1_2384")
    parser.add_argument("--out-dir", default="results/lora_v1_baselines")
    args = parser.parse_args()
    run_baselines(PROJECT_ROOT / args.data_dir, PROJECT_ROOT / args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
