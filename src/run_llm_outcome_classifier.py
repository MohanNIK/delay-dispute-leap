# -*- coding: utf-8 -*-
"""Direct LLM outcome classification for DelayDispute datasets.

This script calls a real OpenAI-compatible DashScope/Qwen endpoint. It uses
pre_decision_text only as model input and recomputes metrics from prediction
level artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABELS = ["support", "partial_support", "not_support"]
INSTRUCTION = (
    "Based only on the pre-decision information, predict the outcome label of "
    "the delay-related construction claim. Output exactly one label from: "
    "support, partial_support, not_support."
)

DEFAULT_CFG: Dict[str, Any] = {
    "dataset_csv": "data/lora_exports/high_conf_lora_qwen_flash_full_20260522/strong_label_master.csv",
    "dataset_name": "strong_label_master_2384_machine_assisted",
    "case_id_col": "case_id",
    "gold_col": "outcome_label",
    "text_col": "pre_decision_text",
    "provider": "dashscope_qwen",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key_env": "DASHSCOPE_API_KEY",
    "model_name": "qwen-flash",
    "temperature": 0.0,
    "max_tokens": 160,
    "timeout": 90,
    "retries": 1,
    "workers": 8,
    "max_input_chars": 5000,
    "seed": 2026,
}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CFG)
    if path and path.exists():
        raw = path.read_text(encoding="utf-8")
        extra = yaml.safe_load(raw) if yaml else json.loads(raw)
        cfg.update(extra or {})
    return cfg


def normalize_label(value: Any) -> str:
    raw = str(value or "").strip().lower()
    mapping = {
        "support": "support",
        "supported": "support",
        "支持": "support",
        "予以支持": "support",
        "partial": "partial_support",
        "partial_support": "partial_support",
        "partially_support": "partial_support",
        "partially supported": "partial_support",
        "部分支持": "partial_support",
        "酌情支持": "partial_support",
        "not_support": "not_support",
        "not-support": "not_support",
        "not support": "not_support",
        "not_supported": "not_support",
        "rejected": "not_support",
        "reject": "not_support",
        "不支持": "not_support",
        "不予支持": "not_support",
        "驳回": "not_support",
    }
    if raw in mapping:
        return mapping[raw]
    if "partial" in raw or "部分" in raw or "酌情" in raw:
        return "partial_support"
    if "not" in raw or "reject" in raw or "驳回" in raw or "不予" in raw or "不支持" in raw:
        return "not_support"
    if "support" in raw or "支持" in raw:
        return "support"
    return "unknown"


def compact_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    keywords = ["工期", "延误", "延期", "停工", "签证", "通知", "索赔", "关键线路", "进度", "证据", "违约", "鉴定"]
    head = text[: int(max_chars * 0.40)]
    hits: List[str] = []
    for sent in re.split(r"(?<=[。！？；])", text):
        if any(k in sent for k in keywords):
            hits.append(sent.strip())
        if sum(len(x) for x in hits) >= int(max_chars * 0.45):
            break
    tail = text[-int(max_chars * 0.12) :]
    return (head + "\n[Delay/evidence-focused snippets]\n" + "\n".join(hits) + "\n[tail]\n" + tail)[:max_chars]


def parse_llm_label(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(cleaned[start : end + 1])
            label = normalize_label(obj.get("outcome_label", obj.get("label", "")))
            return {
                "outcome_label": label,
                "confidence": float(obj.get("confidence", obj.get("outcome_confidence", 0.0)) or 0.0),
                "reason_short": str(obj.get("reason_short", obj.get("reason", "")))[:500],
            }
    except Exception:
        pass
    return {"outcome_label": normalize_label(text), "confidence": 0.0, "reason_short": ""}


def build_prompt(row: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    case_id = str(row.get(cfg["case_id_col"], ""))
    text = compact_text(str(row.get(cfg["text_col"], "") or ""), int(cfg["max_input_chars"]))
    payload = {
        "case_id": case_id,
        "task": "one_label_outcome_prediction",
        "hard_constraints": [
            "Use only pre_decision_text.",
            "Do not infer from judgment/ruling/post-decision language.",
            "Return exactly one JSON object.",
        ],
        "label_schema": LABELS,
        "required_json": {
            "outcome_label": "support|partial_support|not_support",
            "confidence": "float 0-1",
            "reason_short": "short reason based only on pre-decision information",
        },
        "instruction": INSTRUCTION,
        "pre_decision_text": text,
    }
    return [
        {"role": "system", "content": "You are a construction delay-dispute outcome classifier. Return valid JSON only."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def call_llm(row: Dict[str, Any], cfg: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    started = time.time()
    case_id = str(row.get(cfg["case_id_col"], ""))
    y_true = normalize_label(row.get(cfg["gold_col"], ""))
    messages = build_prompt(row, cfg)
    payload = {
        "model": cfg["model_name"],
        "messages": messages,
        "temperature": float(cfg["temperature"]),
        "max_tokens": int(cfg["max_tokens"]),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = ""
    for attempt in range(int(cfg["retries"]) + 1):
        try:
            resp = requests.post(f"{str(cfg['base_url']).rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=int(cfg["timeout"]))
            resp.raise_for_status()
            obj = resp.json()
            raw = obj["choices"][0]["message"]["content"]
            parsed = parse_llm_label(raw)
            return {
                "case_id": case_id,
                "dataset_name": cfg["dataset_name"],
                "model_name": cfg["model_name"],
                "api_status": "api_available",
                "attempts": attempt + 1,
                "y_true": y_true,
                "y_pred": parsed["outcome_label"],
                "confidence": parsed["confidence"],
                "correct": int(y_true == parsed["outcome_label"]),
                "reason_short": parsed["reason_short"],
                "latency_sec": round(time.time() - started, 4),
                "raw_response": raw,
            }
        except Exception as exc:
            last_error = str(exc)[:1000]
            if attempt < int(cfg["retries"]):
                time.sleep(1 + attempt)
    return {
        "case_id": case_id,
        "dataset_name": cfg["dataset_name"],
        "model_name": cfg["model_name"],
        "api_status": "api_error",
        "attempts": int(cfg["retries"]) + 1,
        "y_true": y_true,
        "y_pred": "unknown",
        "confidence": 0.0,
        "correct": 0,
        "error": last_error,
        "latency_sec": round(time.time() - started, 4),
    }


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def compute_metrics(rows: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return {"n": 0, "accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}, [], []
    df = df[df["y_pred"].isin(LABELS) & df["y_true"].isin(LABELS)].copy()
    n = len(df)
    correct = int((df["y_true"] == df["y_pred"]).sum()) if n else 0
    per_class: List[Dict[str, Any]] = []
    cm: List[Dict[str, Any]] = []
    weighted_sum = 0.0
    f1s = []
    for lab in LABELS:
        tp = int(((df["y_true"] == lab) & (df["y_pred"] == lab)).sum())
        fp = int(((df["y_true"] != lab) & (df["y_pred"] == lab)).sum())
        fn = int(((df["y_true"] == lab) & (df["y_pred"] != lab)).sum())
        support = int((df["y_true"] == lab).sum())
        p, r, f = prf(tp, fp, fn)
        f1s.append(f)
        weighted_sum += f * support
        per_class.append({"label": lab, "precision": p, "recall": r, "f1": f, "support": support})
        for pred in LABELS:
            cm.append({"y_true": lab, "y_pred": pred, "count": int(((df["y_true"] == lab) & (df["y_pred"] == pred)).sum())})
    metrics = {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "weighted_f1": weighted_sum / n if n else 0.0,
    }
    return metrics, per_class, cm


def run_predictions(df: pd.DataFrame, cfg: Dict[str, Any], out_dir: Path, resume: bool) -> pd.DataFrame:
    api_key = os.getenv(str(cfg["api_key_env"]), "").strip()
    if not api_key:
        raise RuntimeError(f"Missing {cfg['api_key_env']}")
    existing = pd.DataFrame()
    pred_path = out_dir / "predictions_main.csv"
    if resume and pred_path.exists():
        existing = pd.read_csv(pred_path, encoding="utf-8-sig")
    done = set(existing["case_id"].astype(str)) if not existing.empty and "case_id" in existing else set()
    todo = df[~df[cfg["case_id_col"]].astype(str).isin(done)].copy()
    rows = existing.to_dict("records") if not existing.empty else []
    workers = max(1, int(cfg["workers"]))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(call_llm, row, cfg, api_key) for row in todo.to_dict("records")]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 25 == 0:
                pd.DataFrame(rows).to_csv(pred_path, index=False, encoding="utf-8-sig")
                print(f"progress predictions: {len(rows)}/{len(df)}")
    out = pd.DataFrame(rows).drop_duplicates("case_id", keep="last")
    out.to_csv(pred_path, index=False, encoding="utf-8-sig")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--dataset-csv", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(PROJECT_ROOT / args.config if args.config else None)
    if args.dataset_csv:
        cfg["dataset_csv"] = args.dataset_csv
    if args.model_name:
        cfg["model_name"] = args.model_name
    if args.workers:
        cfg["workers"] = args.workers
    out_dir = PROJECT_ROOT / (args.out_dir or f"results/llm_outcome_classifier_{now_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROJECT_ROOT / cfg["dataset_csv"], encoding="utf-8-sig")
    df[cfg["case_id_col"]] = df[cfg["case_id_col"]].astype(str)
    df[cfg["gold_col"]] = df[cfg["gold_col"]].map(normalize_label)
    df = df[df[cfg["gold_col"]].isin(LABELS)].drop_duplicates(cfg["case_id_col"]).reset_index(drop=True)
    if args.max_cases > 0:
        df = df.head(args.max_cases).copy()

    predictions = run_predictions(df, cfg, out_dir, resume=args.resume)
    valid = predictions[predictions["api_status"].eq("api_available")].copy()
    metrics, per_class, cm = compute_metrics(valid.to_dict("records"))
    metrics.update(
        {
            "dataset_csv": cfg["dataset_csv"],
            "dataset_name": cfg["dataset_name"],
            "model_name": cfg["model_name"],
            "api_success_n": int(predictions["api_status"].eq("api_available").sum()) if not predictions.empty else 0,
            "api_total_n": int(len(predictions)),
            "api_success_rate": float(predictions["api_status"].eq("api_available").mean()) if not predictions.empty else 0.0,
            "interpretation": "machine-assisted strong-label agreement; not human-gold accuracy",
        }
    )
    (out_dir / "metrics_main.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(per_class).to_csv(out_dir / "per_class_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cm).to_csv(out_dir / "confusion_matrix_data.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "model_name": cfg["model_name"],
        "provider": cfg["provider"],
        "text_mode": "pre_decision_only",
        "prompt_template_version": "llm_direct_outcome_classifier_v1",
        "dataset_csv": cfg["dataset_csv"],
        "n_input": len(df),
        "n_predictions": len(predictions),
        "command": " ".join(sys.argv),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"OUT_DIR={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
