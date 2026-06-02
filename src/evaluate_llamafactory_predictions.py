import argparse
import csv
import json
import re
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


LABELS = ["support", "partial_support", "not_support"]


def load_records(path: Path):
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "predictions", "records"):
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported file type: {path}")


def normalize_label(text):
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-zA-Z_一-龥]", " ", s)

    # Check longer labels first, because "support" is contained in "partial_support".
    if "partial_support" in s or "partially_support" in s or "partial" in s:
        return "partial_support"
    if "not_support" in s or "unsupported" in s or "reject" in s or "dismiss" in s:
        return "not_support"
    if re.search(r"\bsupport\b", s):
        return "support"

    if "部分支持" in s or "部分" in s:
        return "partial_support"
    if "不支持" in s or "驳回" in s or "未支持" in s:
        return "not_support"
    if "支持" in s:
        return "support"
    return ""


def pick_field(row, candidates):
    for key in candidates:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="Gold Alpaca JSON/JSONL/CSV with output labels.")
    parser.add_argument("--pred", required=True, help="LLaMA-Factory prediction JSON/JSONL/CSV.")
    parser.add_argument("--out-dir", required=True, help="Directory for metric outputs.")
    args = parser.parse_args()

    gold_path = Path(args.gold)
    pred_path = Path(args.pred)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_rows = load_records(gold_path)
    pred_rows = load_records(pred_path)

    if len(pred_rows) != len(gold_rows):
        print(f"WARNING: pred rows {len(pred_rows)} != gold rows {len(gold_rows)}; scoring aligned prefix only.")
    n = min(len(gold_rows), len(pred_rows))

    scored = []
    y_true = []
    y_pred = []

    for i in range(n):
        gold = gold_rows[i]
        pred = pred_rows[i]
        true_label = normalize_label(
            pick_field(gold, ["output", "label", "private_label", "gold", "gold_label", "target", "answer"])
        )
        pred_text = pick_field(
            pred,
            [
                "predict",
                "prediction",
                "pred",
                "generated_text",
                "generated_response",
                "response",
                "output",
                "answer",
            ],
        )
        pred_label = normalize_label(pred_text)
        y_true.append(true_label)
        y_pred.append(pred_label)
        scored.append(
            {
                "row_id": i,
                "true_label": true_label,
                "pred_label": pred_label,
                "raw_prediction": "" if pred_text is None else str(pred_text).replace("\n", " ")[:500],
                "correct": int(true_label == pred_label),
            }
        )

    valid_true_count = sum(1 for t in y_true if t in LABELS)
    valid_pred_count = sum(1 for p in y_pred if p in LABELS)
    if valid_true_count == 0:
        raise SystemExit("No valid gold labels found. Check gold field names.")
    # Invalid model generations are counted as wrong predictions for full-set accuracy.
    yt = [t for t in y_true if t in LABELS]
    yp = [p if p in LABELS else "__invalid__" for t, p in zip(y_true, y_pred) if t in LABELS]

    metrics = {
        "n_gold": len(gold_rows),
        "n_pred": len(pred_rows),
        "n_scored": len(yt),
        "valid_prediction_count": valid_pred_count,
        "invalid_prediction_count": sum(1 for t, p in zip(y_true, y_pred) if t in LABELS and p not in LABELS),
        "valid_prediction_rate": valid_pred_count / max(1, valid_true_count),
        "accuracy": accuracy_score(yt, yp),
        "macro_f1": f1_score(yt, yp, labels=LABELS, average="macro"),
        "weighted_f1": f1_score(yt, yp, labels=LABELS, average="weighted"),
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "scored_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored)

    report = classification_report(yt, yp, labels=LABELS, output_dict=True, zero_division=0)
    with (out_dir / "per_class_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(yt, yp, labels=LABELS)
    with (out_dir / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *LABELS])
        for label, row in zip(LABELS, cm):
            writer.writerow([label, *row.tolist()])

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
