from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = ["support", "partial", "not_support"]


def normalize_probs(frame: pd.DataFrame) -> np.ndarray:
    arr = frame[[f"{label}_prob" for label in LABELS]].to_numpy(dtype=float)
    arr = np.clip(arr, 1e-12, 1.0)
    arr = arr / arr.sum(axis=1, keepdims=True)
    return arr


def label_indices(frame: pd.DataFrame) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(LABELS)}
    return frame["y_true"].map(mapping).to_numpy(dtype=int)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / float(temperature)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def negative_log_likelihood(probs: np.ndarray, y_true: np.ndarray) -> float:
    return float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true])))


def multiclass_brier(probs: np.ndarray, y_true: np.ndarray) -> float:
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        if idx < n_bins - 1:
            mask = (confidence > edges[idx]) & (confidence <= edges[idx + 1])
        else:
            mask = (confidence > edges[idx]) & (confidence <= edges[idx + 1] + 1e-12)
        if mask.any():
            ece += abs(correct[mask].mean() - confidence[mask].mean()) * mask.mean()
    return float(ece)


def summarize_thresholds(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    error = predicted != y_true
    return {
        "mean_confidence": float(confidence.mean()),
        "low_confidence_rate_055": float((confidence < 0.55).mean()),
        "low_confidence_rate_065": float((confidence < 0.65).mean()),
        "overconfident_error_rate_080": float(((confidence >= 0.80) & error).mean()),
    }


def summarize_split(name: str, probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    pred = probs.argmax(axis=1)
    summary = {
        "split_name": name,
        "n_cases": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "nll": negative_log_likelihood(probs, y_true),
        "brier": multiclass_brier(probs, y_true),
        "ece": expected_calibration_error(probs, y_true),
    }
    summary.update(summarize_thresholds(probs, y_true))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Appendix-only post-hoc outcome-boundary calibration")
    parser.add_argument(
        "--run_dir",
        default=str(PROJECT_ROOT / "results" / "final_eval_20260409_194025"),
        help="Path to the audit-ready final_eval run",
    )
    parser.add_argument(
        "--grid_min",
        type=float,
        default=0.50,
        help="Minimum temperature to search",
    )
    parser.add_argument(
        "--grid_max",
        type=float,
        default=3.00,
        help="Maximum temperature to search",
    )
    parser.add_argument(
        "--grid_points",
        type=int,
        default=251,
        help="Number of temperatures in the search grid",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    pred_path = run_dir / "predictions_main.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")

    df = pd.read_csv(pred_path)
    paesc = df[df["model_name"] == "paesc_hybrid"].copy()

    validation_df = paesc[paesc["dataset_name"] == "weak_internal_proxy"].copy()
    strict_df = paesc[paesc["dataset_name"] == "candidate_gold_strict_v1"].copy()
    extended_df = paesc[paesc["dataset_name"] == "candidate_gold_extended_v1"].copy()
    if validation_df.empty or strict_df.empty or extended_df.empty:
        raise ValueError("Calibration requires weak_internal_proxy, strict, and extended PAESC predictions.")

    validation_probs = normalize_probs(validation_df)
    validation_y = label_indices(validation_df)

    best_temperature = 1.0
    best_nll = negative_log_likelihood(validation_probs, validation_y)
    search_grid = np.linspace(args.grid_min, args.grid_max, args.grid_points)
    for temperature in search_grid:
        candidate_nll = negative_log_likelihood(apply_temperature(validation_probs, float(temperature)), validation_y)
        if candidate_nll < best_nll:
            best_nll = candidate_nll
            best_temperature = float(temperature)

    rows: List[Dict[str, float]] = []
    threshold_rows: List[Dict[str, float]] = []
    appendix_note: List[str] = []
    appendix_note.append("# Outcome-Boundary Calibration Note")
    appendix_note.append("")
    appendix_note.append(
        "This appendix-only analysis applies post-hoc temperature scaling to the current PAESC outcome probabilities."
    )
    appendix_note.append(
        "The temperature is fitted on the internal weak-proxy validation slices only and then transferred unchanged to the external candidate benchmarks."
    )
    appendix_note.append("")
    appendix_note.append(f"- Selected temperature: `{best_temperature:.3f}`")
    appendix_note.append("- Main-text class predictions remain unchanged because temperature scaling preserves argmax labels.")
    appendix_note.append(
        "- This analysis is retained as an appendix-only reliability check because external candidate-benchmark NLL improvements are not stable enough for a headline claim."
    )

    split_map = {
        "validation_internal_proxy": validation_df,
        "strict_candidate_benchmark": strict_df,
        "extended_candidate_benchmark": extended_df,
    }
    for split_name, frame in split_map.items():
        probs = normalize_probs(frame)
        y_true = label_indices(frame)
        before = summarize_split(split_name, probs, y_true)
        before["mode"] = "before_calibration"
        after_probs = apply_temperature(probs, best_temperature)
        after = summarize_split(split_name, after_probs, y_true)
        after["mode"] = "after_calibration"
        rows.extend([before, after])

        before_thr = {"split_name": split_name, "mode": "before_calibration"}
        before_thr.update(summarize_thresholds(probs, y_true))
        after_thr = {"split_name": split_name, "mode": "after_calibration"}
        after_thr.update(summarize_thresholds(after_probs, y_true))
        threshold_rows.extend([before_thr, after_thr])

    metrics_df = pd.DataFrame(rows)
    threshold_df = pd.DataFrame(threshold_rows)

    out_dir = PROJECT_ROOT / "results" / f"outcome_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(out_dir / "calibration_metrics.csv", index=False, encoding="utf-8-sig")
    threshold_df.to_csv(out_dir / "calibration_threshold_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "calibration_appendix_note.md").write_text("\n".join(appendix_note) + "\n", encoding="utf-8")
    (out_dir / "reproduce_commands.md").write_text(
        "# Reproduce Command\n\n"
        f"```powershell\npython src/run_outcome_boundary_calibration.py --run_dir {run_dir}\n```\n",
        encoding="utf-8",
    )

    payload = {
        "source_run_dir": str(run_dir),
        "selected_temperature": best_temperature,
        "validation_strategy": "combined weak_internal_proxy random_split and time_split_after_2020",
        "grid_min": args.grid_min,
        "grid_max": args.grid_max,
        "grid_points": args.grid_points,
        "headline_use": "appendix_only",
        "rationale": "Calibration leaves headline class metrics unchanged and yields mixed external NLL effects; retained only as an uncertainty-boundary check.",
    }
    (out_dir / "calibration_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] appendix-only calibration bundle written to {out_dir}")


if __name__ == "__main__":
    main()
