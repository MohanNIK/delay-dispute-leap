# -*- coding: utf-8 -*-
"""Precision-lift experiments for candidate_gold_extended_v2.

This script is deliberately algorithm-only: it does not rewrite manuscripts or
change benchmark labels. Prediction inputs are restricted to pre-decision text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_candidate_gold_v2_qwen import (  # noqa: E402
    QwenClient,
    build_case_text,
    compact_text,
    extract_json_object,
    normalize_label_v2,
    probe_model,
    resolve_api_key,
    sha256_text,
    validate_evidence_chain,
)


LABELS = ["support", "partial", "not_support"]
VIEW_SPECS = [
    ("legal_result", "从裁判结果预测角度判断请求被支持、部分支持或不支持。"),
    ("evidence_sufficiency", "从证据充分性角度判断索赔是否足以成立。"),
    ("management_responsibility", "从工程管理责任、程序履约、因果链和关键线路角度判断。"),
]
DEFAULT_CFG = {
    "paths": {
        "raw_docx_dir": "data/0_raw_docx",
        "structured_case_dir": "data/3_structured_cases",
        "label_file": "data/gold/candidate_gold_extended_v2.csv",
        "strict_label_file": "data/gold/candidate_gold_strict_v2.csv",
        "candidate_v2_run_dir": "results/candidate_gold_v2_qwen_20260519_131703",
        "previous_final_eval_predictions": "results/final_eval_20260409_194025/predictions_main.csv",
        "legacy_key_source_py": "src/llm_step2_fast_qc.py",
        "output_root": "results",
    },
    "llm": {
        "provider": "dashscope_qwen",
        "api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "debug_api_key": "",
        "reuse_legacy_debug_key": False,
        "model_candidates": ["qwen-max", "qwen-plus"],
        "temperature": 0.0,
        "timeout": 120,
        "retries": 2,
        "workers": 8,
        "sleep_seconds": 0.02,
        "max_chars_pre": 9000,
        "max_tokens_vote": 1400,
        "prompt_template_version": "precision_lift_85_self_consistency_v1",
        "allow_rule_fallback": False,
    },
    "eval": {
        "seed": 2026,
        "n_splits": 5,
        "top_k": 5,
        "clean_filter": {"needs_review": 0, "conflict_flag": 0},
    },
    "run": {"run_name_prefix": "precision_lift_85"},
}


def deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(base, ensure_ascii=False))
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_default_config(path: Path) -> None:
    if path.exists():
        return
    ensure_parent(path)
    path.write_text(json.dumps(DEFAULT_CFG, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(path: Path) -> Dict[str, Any]:
    write_default_config(path)
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text) or {}
        except Exception as exc:
            raise RuntimeError(f"Cannot parse config {path}: {exc}") from exc
    return deep_update(DEFAULT_CFG, data)


def normalize_label(value: Any) -> str:
    return normalize_label_v2(value)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def build_stratified_folds(rows: Sequence[Dict[str, Any]], label_key: str, n_splits: int, seed: int) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    by_label: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        by_label[str(row[label_key])].append(str(row["case_id"]))
    folds: List[List[str]] = [[] for _ in range(n_splits)]
    for label in sorted(by_label):
        ids = by_label[label]
        rng.shuffle(ids)
        for idx, cid in enumerate(ids):
            folds[idx % n_splits].append(cid)
    return folds


def tokenize(text: str) -> set[str]:
    text = re.sub(r"\s+", " ", str(text or "").lower())
    zh = set(re.findall(r"[\u4e00-\u9fff]{2,4}", text))
    latin = set(re.findall(r"[a-z0-9_]{2,}", text))
    return zh | latin


def retrieve_topk_no_self(test_case_id: str, train_case_ids: Sequence[str], corpus: Dict[str, str], top_k: int) -> List[Dict[str, Any]]:
    test_tokens = tokenize(corpus.get(test_case_id, ""))
    rows: List[Dict[str, Any]] = []
    for cid in train_case_ids:
        if str(cid) == str(test_case_id):
            continue
        cand_tokens = tokenize(corpus.get(str(cid), ""))
        if not test_tokens or not cand_tokens:
            score = 0.0
        else:
            score = len(test_tokens & cand_tokens) / max(1, len(test_tokens | cand_tokens))
        rows.append({"case_id": str(cid), "similarity": float(score)})
    rows.sort(key=lambda x: x["similarity"], reverse=True)
    return rows[:top_k]


def majority_vote(votes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [v for v in votes if normalize_label(v.get("label")) in LABELS]
    if not valid:
        return {"label": "unknown", "confidence": 0.0, "agreement": 0.0}
    grouped: Dict[str, List[float]] = defaultdict(list)
    for vote in valid:
        grouped[normalize_label(vote.get("label"))].append(safe_float(vote.get("confidence"), 0.0))
    best_label = sorted(
        grouped,
        key=lambda lab: (len(grouped[lab]), sum(grouped[lab]) / max(1, len(grouped[lab]))),
        reverse=True,
    )[0]
    return {
        "label": best_label,
        "confidence": float(sum(grouped[best_label]) / max(1, len(grouped[best_label]))),
        "agreement": float(len(grouped[best_label]) / max(1, len(valid))),
    }


def recompute_metrics(rows: Sequence[Dict[str, Any]], labels: Sequence[str] = LABELS) -> Dict[str, Any]:
    pairs = [(normalize_label(r.get("y_true")), normalize_label(r.get("y_pred"))) for r in rows]
    pairs = [(a, b) for a, b in pairs if a in labels and b in labels]
    n = len(pairs)
    accuracy = sum(1 for a, b in pairs if a == b) / n if n else 0.0
    per_class: Dict[str, Dict[str, float]] = {}
    weighted_num = 0.0
    cm = [[0 for _ in labels] for _ in labels]
    idx = {lab: i for i, lab in enumerate(labels)}
    for a, b in pairs:
        cm[idx[a]][idx[b]] += 1
    for lab in labels:
        tp = sum(1 for a, b in pairs if a == lab and b == lab)
        fp = sum(1 for a, b in pairs if a != lab and b == lab)
        fn = sum(1 for a, b in pairs if a == lab and b != lab)
        support = sum(1 for a, _ in pairs if a == lab)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[lab] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        weighted_num += f1 * support
    macro_f1 = sum(per_class[lab]["f1"] for lab in labels) / len(labels)
    weighted_f1 = weighted_num / n if n else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "n_eval_rows": n,
    }


def label_to_probs(label: Any, confidence: Any = 1.0) -> Dict[str, float]:
    lab = normalize_label(label)
    conf = max(0.0, min(1.0, safe_float(confidence, 1.0)))
    off = (1.0 - conf) / (len(LABELS) - 1)
    probs = {x: off for x in LABELS}
    if lab in LABELS:
        probs[lab] = conf
    else:
        probs = {x: 1.0 / len(LABELS) for x in LABELS}
    s = sum(probs.values())
    return {k: v / s for k, v in probs.items()}


def vote_records_by_case(vote_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not vote_path.exists():
        return out
    for line in vote_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row.get("api_status") == "api_available":
                out[str(row["case_id"])].append(row)
        except Exception:
            continue
    return out


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in ["pandas", "numpy", "sklearn", "scipy", "requests"]:
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            versions[name] = "not_installed"
    return versions


def load_extended_dataset(cfg: Dict[str, Any]) -> pd.DataFrame:
    path = PROJECT_ROOT / cfg["paths"]["label_file"]
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["dataset_name"] = "candidate_gold_extended_v2"
    df["candidate_outcome_label"] = df["candidate_outcome_label"].map(normalize_label)
    df = df[df["candidate_outcome_label"].isin(LABELS)].copy()
    df["case_id"] = df["case_id"].astype(str)
    return df.drop_duplicates("case_id", keep="first").reset_index(drop=True)


def load_case_texts(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    cases: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        case = build_case_text(row, cfg)
        cases[str(row["case_id"])] = case
    return cases


def load_existing_prediction_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    base_run = PROJECT_ROOT / cfg["paths"]["candidate_v2_run_dir"]
    pred_path = base_run / "predictions_main.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    qwen = pd.read_csv(pred_path, encoding="utf-8-sig")
    qwen = qwen[qwen["dataset_name"].eq("candidate_gold_extended_v2")].copy()
    qwen = qwen.rename(columns={"y_pred": "qwen_direct_pred"})
    keep = [
        "case_id",
        "qwen_direct_pred",
        "outcome_confidence",
        "valid_span_rate",
        "pre_decision_span_rate",
        "role_coverage_rate",
        "documentation_gap_index",
        "procedural_compliance_risk",
        "causality_ambiguity",
        "concurrency_risk",
        "critical_path_support",
        "negotiation_readiness_score",
        "uncertainty_flag",
    ]
    qwen = qwen[[c for c in keep if c in qwen.columns]].copy()
    ref_path = PROJECT_ROOT / cfg["paths"]["previous_final_eval_predictions"]
    old = pd.read_csv(ref_path, encoding="utf-8-sig")
    old = old[old["model_name"].isin(["current_hybrid_baseline", "paesc_hybrid"])].copy()
    old["dataset_name"] = old["dataset_name"].replace(
        {"candidate_gold_extended_v1": "candidate_gold_extended_v2", "candidate_gold_strict_v1": "candidate_gold_strict_v2"}
    )
    old = old[old["dataset_name"].eq("candidate_gold_extended_v2")].copy()
    old_piv = old.pivot_table(index="case_id", columns="model_name", values="y_pred", aggfunc="first").reset_index()
    merged = df[["case_id", "candidate_outcome_label", "needs_review", "conflict_flag"]].merge(qwen, on="case_id", how="left").merge(old_piv, on="case_id", how="left")
    for col in ["qwen_direct_pred", "current_hybrid_baseline", "paesc_hybrid"]:
        if col in merged:
            merged[col] = merged[col].map(normalize_label)
    for col in [
        "outcome_confidence",
        "valid_span_rate",
        "pre_decision_span_rate",
        "role_coverage_rate",
        "documentation_gap_index",
        "procedural_compliance_risk",
        "causality_ambiguity",
        "concurrency_risk",
        "critical_path_support",
        "negotiation_readiness_score",
        "uncertainty_flag",
    ]:
        if col not in merged:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return merged


def build_vote_prompt(case: Dict[str, Any], view_name: str, view_instruction: str, exemplars: Sequence[Dict[str, Any]], cfg: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    payload = {
        "task": "pre_decision_delay_dispute_precision_lift_vote",
        "view_name": view_name,
        "view_instruction": view_instruction,
        "hard_constraints": [
            "Only use pre_decision_text and the given pre-decision exemplar summaries.",
            "Do not infer from court judgment, dispositive result, or post-decision reasoning.",
            "Return JSON only.",
            "Evidence spans must be exact excerpts from pre_decision_text when possible.",
        ],
        "label_schema": LABELS,
        "few_shot_exemplars": exemplars,
        "required_json": {
            "outcome_label": "support|partial|not_support",
            "outcome_confidence": "float 0-1",
            "rationale_short": "one short Chinese sentence",
            "evidence_chain": [{"role_label": "ENT|NOT|CAU|IMP|DOC", "span_text": "exact excerpt", "reason": "short reason"}],
        },
        "pre_decision_text": compact_text(case["pre_decision_text"], int(cfg["llm"].get("max_chars_pre", 9000))),
    }
    prompt = json.dumps(payload, ensure_ascii=False)
    return [
        {"role": "system", "content": "你是建设工程工期延误纠纷预测助手。只输出合法 JSON，不要 Markdown。"},
        {"role": "user", "content": prompt},
    ], prompt


def call_with_retries(client: QwenClient, messages: List[Dict[str, str]], max_tokens: int, retries: int) -> Tuple[str, Dict[str, Any], int]:
    last: Optional[Exception] = None
    for attempt in range(1, retries + 2):
        try:
            content, usage = client.chat(messages, max_tokens=max_tokens)
            return content, usage, attempt
        except Exception as exc:
            last = exc
            time.sleep(min(8.0, 1.5 * attempt))
    raise RuntimeError(str(last))


def run_vote_task(task: Dict[str, Any], client: QwenClient, cfg: Dict[str, Any]) -> Dict[str, Any]:
    cid = str(task["case_id"])
    view_name = str(task["view_name"])
    messages, prompt = build_vote_prompt(task["case"], view_name, str(task["view_instruction"]), task["exemplars"], cfg)
    row: Dict[str, Any] = {
        "case_id": cid,
        "fold_id": task["fold_id"],
        "view_name": view_name,
        "model_name": client.model_name,
        "prompt_sha256": sha256_text(prompt),
        "prompt_chars": len(prompt),
    }
    try:
        content, usage, attempts = call_with_retries(client, messages, int(cfg["llm"].get("max_tokens_vote", 1400)), int(cfg["llm"].get("retries", 2)))
        parsed = extract_json_object(content)
        chain, chain_metrics = validate_evidence_chain(parsed.get("evidence_chain", []), task["case"].get("pre_decision_text", ""))
        row.update(
            {
                "api_status": "api_available",
                "attempts": attempts,
                "outcome_label": normalize_label(parsed.get("outcome_label", "unknown")),
                "outcome_confidence": safe_float(parsed.get("outcome_confidence"), 0.0),
                "rationale_short": str(parsed.get("rationale_short", ""))[:500],
                "valid_span_rate": chain_metrics["valid_span_rate"],
                "pre_decision_span_rate": chain_metrics["pre_decision_span_rate"],
                "role_coverage_rate": chain_metrics["role_coverage_rate"],
                "usage_json": json.dumps(usage, ensure_ascii=False),
                "evidence_chain_json": json.dumps(chain, ensure_ascii=False),
                "raw_response": content,
            }
        )
    except Exception as exc:
        row.update({"api_status": "api_error", "attempts": 0, "outcome_label": "unknown", "outcome_confidence": 0.0, "error": str(exc)[:1000]})
    return row


def append_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_folds_and_retrieval(df: pd.DataFrame, cases: Dict[str, Dict[str, Any]], cfg: Dict[str, Any], out_dir: Path) -> Tuple[List[List[str]], Dict[Tuple[str, str], List[Dict[str, Any]]]]:
    rows = [{"case_id": r["case_id"], "label": r["candidate_outcome_label"]} for _, r in df.iterrows()]
    folds = build_stratified_folds(rows, "label", int(cfg["eval"].get("n_splits", 5)), int(cfg["eval"].get("seed", 2026)))
    corpus = {cid: cases[cid].get("pre_decision_text", "") for cid in cases}
    labels_by_id = dict(zip(df["case_id"], df["candidate_outcome_label"]))
    retrieval: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    manifest_rows: List[Dict[str, Any]] = []
    top_k = int(cfg["eval"].get("top_k", 5))
    all_ids = set(df["case_id"])
    for fold_id, test_ids in enumerate(folds):
        train_ids = sorted(all_ids - set(test_ids))
        for cid in test_ids:
            retrieved = retrieve_topk_no_self(cid, train_ids, corpus, top_k)
            exemplars = []
            for item in retrieved:
                rid = item["case_id"]
                ex = {
                    "case_id": rid,
                    "similarity": item["similarity"],
                    "pre_decision_summary": compact_text(corpus.get(rid, ""), 700),
                    "outcome_label": labels_by_id.get(rid, ""),
                }
                exemplars.append(ex)
                manifest_rows.append({"fold_id": fold_id, "case_id": cid, "retrieved_case_id": rid, "similarity": item["similarity"], "retrieved_label": labels_by_id.get(rid, "")})
            retrieval[(str(cid), str(fold_id))] = exemplars
    pd.DataFrame(manifest_rows).to_csv(out_dir / "rag_retrieval_manifest.csv", index=False, encoding="utf-8-sig")
    return folds, retrieval


def run_llm_votes(df: pd.DataFrame, cases: Dict[str, Dict[str, Any]], folds: List[List[str]], retrieval: Dict[Tuple[str, str], List[Dict[str, Any]]], cfg: Dict[str, Any], out_dir: Path, max_cases: Optional[int] = None) -> pd.DataFrame:
    vote_path = out_dir / "llm_vote_records.jsonl"
    existing = vote_records_by_case(vote_path)
    done_keys = set()
    for cid, rows in existing.items():
        for row in rows:
            done_keys.add((cid, str(row.get("fold_id")), str(row.get("view_name"))))
    tasks: List[Dict[str, Any]] = []
    selected_ids = set(df["case_id"].astype(str).tolist()[:max_cases]) if max_cases else None
    for fold_id, test_ids in enumerate(folds):
        for cid in test_ids:
            if selected_ids is not None and str(cid) not in selected_ids:
                continue
            for view_name, instruction in VIEW_SPECS:
                key = (str(cid), str(fold_id), view_name)
                if key in done_keys:
                    continue
                tasks.append(
                    {
                        "case_id": str(cid),
                        "fold_id": fold_id,
                        "view_name": view_name,
                        "view_instruction": instruction,
                        "case": cases[str(cid)],
                        "exemplars": retrieval.get((str(cid), str(fold_id)), []),
                    }
                )
    if tasks:
        api_key, api_key_source = resolve_api_key(cfg)
        if not api_key:
            raise RuntimeError("No API key available for Qwen self-consistency votes")
        model = probe_model(cfg, api_key, out_dir)
        client = QwenClient(cfg, api_key, model)
        batch: List[Dict[str, Any]] = []
        workers = int(cfg["llm"].get("workers", 8))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_vote_task, task, client, cfg) for task in tasks]
            for idx, fut in enumerate(as_completed(futures), start=1):
                row = fut.result()
                row["api_key_source"] = api_key_source
                batch.append(row)
                if len(batch) >= 20:
                    append_jsonl(vote_path, batch)
                    print(f"saved vote records: {idx}/{len(tasks)}")
                    batch.clear()
        if batch:
            append_jsonl(vote_path, batch)
    rows = []
    for line in vote_path.read_text(encoding="utf-8").splitlines() if vote_path.exists() else []:
        if line.strip():
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def ml_available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


def simple_fallback_predict(train: pd.DataFrame, test: pd.DataFrame, corpus: Dict[str, str]) -> Dict[str, str]:
    labels_by_id = dict(zip(train["case_id"], train["candidate_outcome_label"]))
    preds: Dict[str, str] = {}
    for cid in test["case_id"]:
        retrieved = retrieve_topk_no_self(cid, train["case_id"].astype(str).tolist(), corpus, 7)
        counts = Counter(labels_by_id.get(item["case_id"], "partial") for item in retrieved)
        preds[str(cid)] = counts.most_common(1)[0][0] if counts else "partial"
    return preds


def train_predict_tfidf_models(train: pd.DataFrame, test: pd.DataFrame, corpus: Dict[str, str], seed: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    if not ml_available():
        preds = simple_fallback_predict(train, test, corpus)
        return {"tfidf_fallback_knn": {cid: label_to_probs(pred, 0.65) for cid, pred in preds.items()}}
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    x_train = [corpus[str(cid)] for cid in train["case_id"]]
    y_train = train["candidate_outcome_label"].tolist()
    x_test = [corpus[str(cid)] for cid in test["case_id"]]
    ids = test["case_id"].astype(str).tolist()
    models = {
        "tfidf_word_logreg": make_pipeline(
            TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), max_features=40000, min_df=2),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        ),
        "tfidf_char_logreg": make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=60000, min_df=2),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        ),
        "tfidf_nb": make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=60000, min_df=2),
            MultinomialNB(alpha=0.2),
        ),
        "tfidf_linearsvc": make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=60000, min_df=2),
            CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=seed), cv=3),
        ),
    }
    outputs: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, model in models.items():
        try:
            model.fit(x_train, y_train)
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(x_test)
                classes = list(model.classes_)
                outputs[name] = {cid: {lab: float(prob[i, classes.index(lab)]) if lab in classes else 0.0 for lab in LABELS} for i, cid in enumerate(ids)}
            else:
                pred = model.predict(x_test)
                outputs[name] = {cid: label_to_probs(pred[i], 0.75) for i, cid in enumerate(ids)}
        except Exception as exc:
            print(f"model {name} failed: {exc}")
    return outputs


def estimate_model_weights(train_df: pd.DataFrame, model_cols: Sequence[str]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for col in model_cols:
        if col not in train_df:
            continue
        valid = train_df[col].isin(LABELS)
        if valid.any():
            acc = (train_df.loc[valid, col] == train_df.loc[valid, "candidate_outcome_label"]).mean()
            weights[col] = max(0.05, float(acc))
    return weights


def combine_probabilities(prob_maps: Sequence[Tuple[Dict[str, float], float]]) -> Tuple[str, Dict[str, float]]:
    total = {lab: 0.0 for lab in LABELS}
    denom = 0.0
    for probs, weight in prob_maps:
        denom += weight
        for lab in LABELS:
            total[lab] += weight * safe_float(probs.get(lab), 0.0)
    if denom <= 0:
        return "partial", {lab: 1.0 / len(LABELS) for lab in LABELS}
    total = {lab: total[lab] / denom for lab in LABELS}
    return max(LABELS, key=lambda lab: total[lab]), total


def apply_two_stage_boundary(probs: Dict[str, float], mechanism: pd.Series) -> str:
    not_score = probs.get("not_support", 0.0)
    partial_score = probs.get("partial", 0.0)
    support_score = probs.get("support", 0.0)
    evidence_gap = safe_float(mechanism.get("documentation_gap_index"), 0.0)
    procedure_risk = safe_float(mechanism.get("procedural_compliance_risk"), 0.0)
    causality = safe_float(mechanism.get("causality_ambiguity"), 0.0)
    if not_score + 0.08 * evidence_gap + 0.05 * procedure_risk + 0.05 * causality >= max(support_score, partial_score) + 0.05:
        return "not_support"
    if partial_score + 0.08 * causality + 0.04 * procedure_risk >= support_score:
        return "partial"
    return "support"


def one_hot_label_features(prefix: str, label: Any) -> Dict[str, float]:
    lab = normalize_label(label)
    return {f"{prefix}_{x}": 1.0 if lab == x else 0.0 for x in LABELS}


def build_meta_feature_row(row: pd.Series, sc: Dict[str, Any], rag_label: str) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    feats.update(one_hot_label_features("qwen", row.get("qwen_direct_pred")))
    feats.update(one_hot_label_features("sc", sc.get("label")))
    feats.update(one_hot_label_features("hybrid", row.get("current_hybrid_baseline")))
    feats.update(one_hot_label_features("paesc", row.get("paesc_hybrid")))
    feats.update(one_hot_label_features("rag", rag_label))
    feats.update(
        {
            "qwen_conf": safe_float(row.get("outcome_confidence"), 0.0),
            "sc_conf": safe_float(sc.get("confidence"), 0.0),
            "sc_agreement": safe_float(sc.get("agreement"), 0.0),
            "valid_span_rate": safe_float(row.get("valid_span_rate"), 0.0),
            "pre_decision_span_rate": safe_float(row.get("pre_decision_span_rate"), 0.0),
            "role_coverage_rate": safe_float(row.get("role_coverage_rate"), 0.0),
            "documentation_gap_index": safe_float(row.get("documentation_gap_index"), 0.0),
            "procedural_compliance_risk": safe_float(row.get("procedural_compliance_risk"), 0.0),
            "causality_ambiguity": safe_float(row.get("causality_ambiguity"), 0.0),
            "concurrency_risk": safe_float(row.get("concurrency_risk"), 0.0),
            "critical_path_support": safe_float(row.get("critical_path_support"), 0.0),
            "negotiation_readiness_score": safe_float(row.get("negotiation_readiness_score"), 0.0),
            "uncertainty_flag": safe_float(row.get("uncertainty_flag"), 0.0),
        }
    )
    return feats


def rag_label_vote(exemplars: Sequence[Dict[str, Any]]) -> Tuple[str, float]:
    scores = {lab: 0.0 for lab in LABELS}
    for item in exemplars:
        lab = normalize_label(item.get("outcome_label"))
        if lab in LABELS:
            scores[lab] += max(0.0001, safe_float(item.get("similarity"), 0.0))
    if not any(scores.values()):
        return "partial", 0.0
    total = sum(scores.values())
    lab = max(LABELS, key=lambda x: scores[x])
    return lab, scores[lab] / total if total else 0.0


def train_meta_logreg(train_feat: pd.DataFrame, train_y: Sequence[str], test_feat: pd.DataFrame) -> Tuple[List[str], List[Dict[str, float]]]:
    if not ml_available() or train_feat.empty or test_feat.empty:
        return [], []
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cols = sorted(set(train_feat.columns) | set(test_feat.columns))
    x_train = train_feat.reindex(columns=cols, fill_value=0.0)
    x_test = test_feat.reindex(columns=cols, fill_value=0.0)
    model = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0.0),
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=2026, C=0.7),
    )
    model.fit(x_train, list(train_y))
    prob = model.predict_proba(x_test)
    classes = list(model.classes_)
    preds = [classes[int(np.argmax(prob[i]))] for i in range(prob.shape[0])]
    prob_rows = [{lab: float(prob[i, classes.index(lab)]) if lab in classes else 0.0 for lab in LABELS} for i in range(prob.shape[0])]
    return preds, prob_rows


def compute_oof_predictions(df: pd.DataFrame, feature_df: pd.DataFrame, cases: Dict[str, Dict[str, Any]], folds: List[List[str]], retrieval: Dict[Tuple[str, str], List[Dict[str, Any]]], votes: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    corpus = {cid: cases[cid].get("pre_decision_text", "") for cid in cases}
    base = df.merge(feature_df, on=["case_id", "candidate_outcome_label"], how="left", suffixes=("", "_feat"))
    vote_major: Dict[str, Dict[str, Any]] = {}
    if not votes.empty:
        for cid, sub in votes[votes["api_status"].eq("api_available")].groupby("case_id"):
            vote_major[str(cid)] = majority_vote([{"label": r["outcome_label"], "confidence": r.get("outcome_confidence", 0.0)} for _, r in sub.iterrows()])
    all_rows: List[Dict[str, Any]] = []
    seed = int(cfg["eval"].get("seed", 2026))
    all_ids = set(base["case_id"].astype(str))
    for fold_id, test_ids in enumerate(folds):
        test_ids_set = set(map(str, test_ids))
        train = base[~base["case_id"].astype(str).isin(test_ids_set)].copy()
        test = base[base["case_id"].astype(str).isin(test_ids_set)].copy()
        tfidf_outputs = train_predict_tfidf_models(train, test, corpus, seed + fold_id)
        weights = estimate_model_weights(train, ["qwen_direct_pred", "current_hybrid_baseline", "paesc_hybrid"])
        # Tfidf models receive conservative fold-local weights; exact training accuracy is not used.
        for model_name in tfidf_outputs:
            weights[model_name] = 0.48
        weights["qwen_self_consistency_3view"] = max(0.05, weights.get("qwen_direct_pred", 0.45) + 0.04)
        # Fold-local meta learner: trained only on training cases and then
        # applied to the held-out fold.
        train_meta_rows: List[Dict[str, float]] = []
        train_y: List[str] = []
        train_ids_for_meta = train["case_id"].astype(str).tolist()
        train_label_by_id = dict(zip(train["case_id"].astype(str), train["candidate_outcome_label"]))
        for _, tr in train.iterrows():
            tr_cid = str(tr["case_id"])
            tr_sc = vote_major.get(tr_cid, {"label": normalize_label(tr.get("qwen_direct_pred")), "confidence": tr.get("outcome_confidence", 0.0), "agreement": 0.0})
            tr_neighbors = retrieve_topk_no_self(tr_cid, train_ids_for_meta, corpus, int(cfg["eval"].get("top_k", 5)))
            tr_exemplars = [{"outcome_label": train_label_by_id.get(x["case_id"], ""), "similarity": x["similarity"]} for x in tr_neighbors]
            tr_rag, _ = rag_label_vote(tr_exemplars)
            train_meta_rows.append(build_meta_feature_row(tr, tr_sc, tr_rag))
            train_y.append(normalize_label(tr["candidate_outcome_label"]))
        test_meta_rows: List[Dict[str, float]] = []
        test_meta_case_ids: List[str] = []
        for _, te in test.iterrows():
            te_cid = str(te["case_id"])
            te_sc = vote_major.get(te_cid, {"label": normalize_label(te.get("qwen_direct_pred")), "confidence": te.get("outcome_confidence", 0.0), "agreement": 0.0})
            te_rag, _ = rag_label_vote(retrieval.get((te_cid, str(fold_id)), []))
            test_meta_rows.append(build_meta_feature_row(te, te_sc, te_rag))
            test_meta_case_ids.append(te_cid)
        meta_preds, meta_probs = train_meta_logreg(pd.DataFrame(train_meta_rows), train_y, pd.DataFrame(test_meta_rows))
        meta_pred_by_id = {cid: meta_preds[i] for i, cid in enumerate(test_meta_case_ids)} if meta_preds else {}
        meta_prob_by_id = {cid: meta_probs[i] for i, cid in enumerate(test_meta_case_ids)} if meta_probs else {}

        for _, row in test.iterrows():
            cid = str(row["case_id"])
            y_true = normalize_label(row["candidate_outcome_label"])
            direct = normalize_label(row.get("qwen_direct_pred"))
            sc = vote_major.get(cid, {"label": direct, "confidence": row.get("outcome_confidence", 0.0), "agreement": 0.0})
            rag_pred, rag_conf = rag_label_vote(retrieval.get((cid, str(fold_id)), []))
            all_rows.append({"case_id": cid, "dataset_name": "candidate_gold_extended_v2", "fold_id": fold_id, "model_name": "qwen_direct_v2", "y_true": y_true, "y_pred": direct, "confidence": safe_float(row.get("outcome_confidence"), 0.0)})
            all_rows.append({"case_id": cid, "dataset_name": "candidate_gold_extended_v2", "fold_id": fold_id, "model_name": "qwen_self_consistency_3view", "y_true": y_true, "y_pred": sc["label"], "confidence": sc["confidence"], "vote_agreement": sc.get("agreement", 0.0)})
            all_rows.append({"case_id": cid, "dataset_name": "candidate_gold_extended_v2", "fold_id": fold_id, "model_name": "rag_fewshot_label_vote", "y_true": y_true, "y_pred": rag_pred, "confidence": rag_conf})
            for col in ["current_hybrid_baseline", "paesc_hybrid"]:
                pred = normalize_label(row.get(col))
                if pred in LABELS:
                    all_rows.append({"case_id": cid, "dataset_name": "candidate_gold_extended_v2", "fold_id": fold_id, "model_name": col, "y_true": y_true, "y_pred": pred, "confidence": 1.0})
            for model_name, prob_by_id in tfidf_outputs.items():
                probs = prob_by_id.get(cid, {lab: 1.0 / len(LABELS) for lab in LABELS})
                pred = max(LABELS, key=lambda lab: probs.get(lab, 0.0))
                all_rows.append({"case_id": cid, "dataset_name": "candidate_gold_extended_v2", "fold_id": fold_id, "model_name": model_name, "y_true": y_true, "y_pred": pred, "confidence": max(probs.values())})
            prob_maps: List[Tuple[Dict[str, float], float]] = []
            if direct in LABELS:
                prob_maps.append((label_to_probs(direct, row.get("outcome_confidence", 0.65)), weights.get("qwen_direct_pred", 0.45)))
            if sc["label"] in LABELS:
                prob_maps.append((label_to_probs(sc["label"], sc.get("confidence", 0.65)), weights.get("qwen_self_consistency_3view", 0.49)))
            for col in ["current_hybrid_baseline", "paesc_hybrid"]:
                pred = normalize_label(row.get(col))
                if pred in LABELS:
                    prob_maps.append((label_to_probs(pred, 0.70), weights.get(col, 0.35)))
            for model_name, prob_by_id in tfidf_outputs.items():
                prob_maps.append((prob_by_id.get(cid, {lab: 1.0 / len(LABELS) for lab in LABELS}), weights.get(model_name, 0.45)))
            fusion_pred, fusion_probs = combine_probabilities(prob_maps)
            all_rows.append({"case_id": cid, "dataset_name": "candidate_gold_extended_v2", "fold_id": fold_id, "model_name": "paesc_llm_fusion_85", "y_true": y_true, "y_pred": fusion_pred, "confidence": max(fusion_probs.values())})
            two_stage = apply_two_stage_boundary(fusion_probs, row)
            all_rows.append({"case_id": cid, "dataset_name": "candidate_gold_extended_v2", "fold_id": fold_id, "model_name": "two_stage_boundary_model", "y_true": y_true, "y_pred": two_stage, "confidence": max(fusion_probs.values())})
            meta_pred = normalize_label(meta_pred_by_id.get(cid, fusion_pred))
            meta_probs_row = meta_prob_by_id.get(cid, fusion_probs)
            all_rows.append({"case_id": cid, "dataset_name": "candidate_gold_extended_v2", "fold_id": fold_id, "model_name": "stacked_meta_logreg", "y_true": y_true, "y_pred": meta_pred, "confidence": max(meta_probs_row.values()) if meta_probs_row else 0.0})
    return pd.DataFrame(all_rows)


def write_metric_artifacts(pred: pd.DataFrame, out_dir: Path, clean_ids: set[str]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"candidate_gold_extended_v2": {}, "clean397": {}}
    rows: List[Dict[str, Any]] = []
    per_rows: List[Dict[str, Any]] = []
    cm_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    for scope, subdf in [("candidate_gold_extended_v2", pred), ("clean397", pred[pred["case_id"].isin(clean_ids)].copy())]:
        for model_name, sub in subdf.groupby("model_name"):
            metric = recompute_metrics(sub[["y_true", "y_pred"]].to_dict("records"), LABELS)
            metrics[scope][model_name] = metric
            rows.append({"scope": scope, "model_name": model_name, "n": metric["n_eval_rows"], "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"], "weighted_f1": metric["weighted_f1"]})
            for lab, item in metric["per_class"].items():
                per_rows.append({"scope": scope, "model_name": model_name, "class_label": lab, **item})
            for i, gold in enumerate(LABELS):
                for j, pred_lab in enumerate(LABELS):
                    cm_rows.append({"scope": scope, "model_name": model_name, "gold_label": gold, "pred_label": pred_lab, "count": metric["confusion_matrix"][i][j]})
            for fold_id, fold_sub in sub.groupby("fold_id"):
                fm = recompute_metrics(fold_sub[["y_true", "y_pred"]].to_dict("records"), LABELS)
                fold_rows.append({"scope": scope, "model_name": model_name, "fold_id": fold_id, "n": fm["n_eval_rows"], "accuracy": fm["accuracy"], "macro_f1": fm["macro_f1"], "weighted_f1": fm["weighted_f1"]})
    pred.to_csv(out_dir / "predictions_main.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).sort_values(["scope", "accuracy"], ascending=[True, False]).to_csv(out_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_rows).to_csv(out_dir / "per_class_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cm_rows).to_csv(out_dir / "confusion_matrix_data.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    (out_dir / "metrics_main.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    errors = pred[(pred["model_name"].eq("paesc_llm_fusion_85")) & (~pred["y_true"].eq(pred["y_pred"]))].copy()
    errors["error_category"] = np.where(errors["y_true"].eq("partial") | errors["y_pred"].eq("partial"), "partial_boundary_confusion", "other_class_confusion")
    errors.to_csv(out_dir / "error_analysis.csv", index=False, encoding="utf-8-sig")
    return metrics


def macro_f1_from_lists(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(recompute_metrics([{"y_true": a, "y_pred": b} for a, b in zip(y_true, y_pred)], LABELS)["macro_f1"])


def optimized_vote_predict(case_rows: pd.DataFrame, model_weights: Dict[str, float], class_bias: Dict[str, float]) -> str:
    scores = {lab: class_bias.get(lab, 0.0) for lab in LABELS}
    for _, row in case_rows.iterrows():
        model = str(row["model_name"])
        if model not in model_weights:
            continue
        pred = normalize_label(row.get("y_pred"))
        if pred in LABELS:
            conf = safe_float(row.get("confidence"), 1.0)
            scores[pred] += model_weights[model] * max(0.25, min(1.0, conf))
    return max(LABELS, key=lambda lab: scores[lab])


def add_optimized_weighted_ensemble(pred: pd.DataFrame, seed: int = 2026) -> pd.DataFrame:
    """Add fold-local optimized weighted-vote ensembles.

    Base predictions are already out-of-fold. For each held-out fold, weights
    are selected only on the other folds, then applied to the held-out fold.
    """
    base_models = [
        "qwen_direct_v2",
        "qwen_self_consistency_3view",
        "tfidf_word_logreg",
        "tfidf_char_logreg",
        "tfidf_nb",
        "tfidf_linearsvc",
        "rag_fewshot_label_vote",
        "current_hybrid_baseline",
        "paesc_hybrid",
        "paesc_llm_fusion_85",
        "two_stage_boundary_model",
        "stacked_meta_logreg",
    ]
    available = [m for m in base_models if m in set(pred["model_name"])]
    rng = np.random.default_rng(seed)
    added: List[Dict[str, Any]] = []
    case_truth = pred.groupby("case_id")["y_true"].first().to_dict()
    for fold_id in sorted(pred["fold_id"].dropna().unique()):
        fold_id = int(fold_id)
        train_cases = sorted(set(pred.loc[pred["fold_id"].ne(fold_id), "case_id"]))
        test_cases = sorted(set(pred.loc[pred["fold_id"].eq(fold_id), "case_id"]))
        train_sub = pred[pred["case_id"].isin(train_cases) & pred["model_name"].isin(available)].copy()
        grouped_train = {cid: g for cid, g in train_sub.groupby("case_id")}
        y_train = [case_truth[cid] for cid in train_cases if cid in grouped_train]
        # Seed candidates: uniform, model-accuracy weights, and random searches.
        candidates: List[Tuple[Dict[str, float], Dict[str, float]]] = []
        candidates.append(({m: 1.0 for m in available}, {lab: 0.0 for lab in LABELS}))
        acc_weights = {}
        for m in available:
            msub = train_sub[train_sub["model_name"].eq(m)]
            if len(msub):
                acc_weights[m] = max(0.01, float((msub["y_true"].map(normalize_label) == msub["y_pred"].map(normalize_label)).mean()))
        candidates.append((acc_weights, {lab: 0.0 for lab in LABELS}))
        class_counts = Counter(y_train)
        candidates.append((acc_weights, {lab: -math.log(max(1, class_counts.get(lab, 1))) * 0.05 for lab in LABELS}))
        for _ in range(600):
            weights = {m: float(rng.uniform(0.0, 2.0)) for m in available}
            # Keep very weak historical models from dominating unless random
            # search finds a clear fold-local benefit.
            bias = {lab: float(rng.uniform(-0.25, 0.25)) for lab in LABELS}
            candidates.append((weights, bias))
        best_score = -1.0
        best_weights: Dict[str, float] = {}
        best_bias: Dict[str, float] = {}
        for weights, bias in candidates:
            preds_train = [optimized_vote_predict(grouped_train[cid], weights, bias) for cid in train_cases if cid in grouped_train]
            score = 0.55 * (sum(1 for a, b in zip(y_train, preds_train) if a == b) / max(1, len(y_train))) + 0.45 * macro_f1_from_lists(y_train, preds_train)
            if score > best_score:
                best_score = score
                best_weights, best_bias = weights, bias
        test_sub = pred[pred["case_id"].isin(test_cases) & pred["model_name"].isin(available)].copy()
        for cid, g in test_sub.groupby("case_id"):
            y_pred = optimized_vote_predict(g, best_weights, best_bias)
            added.append(
                {
                    "case_id": cid,
                    "dataset_name": "candidate_gold_extended_v2",
                    "fold_id": fold_id,
                    "model_name": "optimized_weighted_ensemble",
                    "y_true": case_truth[cid],
                    "y_pred": y_pred,
                    "confidence": 0.0,
                    "train_objective_score": best_score,
                }
            )
    if not added:
        return pred
    return pd.concat([pred, pd.DataFrame(added)], ignore_index=True)


def write_run_manifest(out_dir: Path, cfg_path: Path, cfg: Dict[str, Any], extra_files: Sequence[Path]) -> None:
    critical = [
        cfg_path,
        PROJECT_ROOT / cfg["paths"]["label_file"],
        PROJECT_ROOT / cfg["paths"]["candidate_v2_run_dir"] / "prediction_v2_records.csv",
        PROJECT_ROOT / cfg["paths"]["previous_final_eval_predictions"],
        PROJECT_ROOT / "src" / "run_precision_lift_85.py",
        out_dir / "predictions_main.csv",
        out_dir / "metrics_main.json",
        out_dir / "llm_vote_records.jsonl",
    ]
    critical.extend(extra_files)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": get_package_versions(),
        "model_name": "paesc_llm_fusion_85",
        "prompt_template_version": cfg["llm"].get("prompt_template_version"),
        "embedding_model": "tfidf_fold_local_rag_v1",
        "label_schema_version": "outcome_v1__candidate_gold_v2_qwen_temporary",
        "seed": cfg["eval"].get("seed"),
        "split_mode": "5_fold_out_of_fold",
        "text_mode": "pre_decision_only",
        "no_post_decision_prediction": True,
        "metric_source_files": ["predictions_main.csv", "model_comparison.csv"],
        "artifact_hashes": {str(p.relative_to(PROJECT_ROOT) if p.is_absolute() and PROJECT_ROOT in p.parents else p): sha256_file(p) for p in critical},
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/research_precision_lift_85.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--max-cases", type=int, default=0, help="Smoke-test limit. 0 means full dataset.")
    parser.add_argument("--skip-api", action="store_true", help="Do not call Qwen; reuse existing llm_vote_records.jsonl or direct predictions.")
    parser.add_argument("--recompute-only", action="store_true", help="Recompute metrics from existing outputs.")
    args = parser.parse_args(argv)

    cfg_path = PROJECT_ROOT / args.config
    cfg = load_config(cfg_path)
    if args.out_dir:
        out_dir = PROJECT_ROOT / args.out_dir
    else:
        out_dir = PROJECT_ROOT / cfg["paths"].get("output_root", "results") / f"{cfg['run'].get('run_name_prefix', 'precision_lift_85')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.recompute_only and (out_dir / "predictions_main.csv").exists():
        pred = pd.read_csv(out_dir / "predictions_main.csv", encoding="utf-8-sig")
        df = load_extended_dataset(cfg)
        clean = df[(pd.to_numeric(df.get("needs_review", 0), errors="coerce").fillna(0).astype(int).eq(0)) & (pd.to_numeric(df.get("conflict_flag", 0), errors="coerce").fillna(0).astype(int).eq(0))]
        write_metric_artifacts(pred, out_dir, set(clean["case_id"].astype(str)))
        write_run_manifest(out_dir, cfg_path, cfg, [])
        return 0

    df = load_extended_dataset(cfg)
    if args.max_cases and args.max_cases > 0:
        df = df.head(args.max_cases).copy()
    cases = load_case_texts(df, cfg)
    feature_df = load_existing_prediction_features(df, cfg)
    folds, retrieval = prepare_folds_and_retrieval(df, cases, cfg, out_dir)
    votes = pd.DataFrame()
    if not args.skip_api:
        votes = run_llm_votes(df, cases, folds, retrieval, cfg, out_dir, max_cases=args.max_cases or None)
    else:
        vote_path = out_dir / "llm_vote_records.jsonl"
        if vote_path.exists():
            votes = pd.DataFrame([json.loads(line) for line in vote_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    pred = compute_oof_predictions(df, feature_df, cases, folds, retrieval, votes, cfg)
    pred = add_optimized_weighted_ensemble(pred, int(cfg["eval"].get("seed", 2026)))
    clean = df[(pd.to_numeric(df.get("needs_review", 0), errors="coerce").fillna(0).astype(int).eq(0)) & (pd.to_numeric(df.get("conflict_flag", 0), errors="coerce").fillna(0).astype(int).eq(0))]
    metrics = write_metric_artifacts(pred, out_dir, set(clean["case_id"].astype(str)))
    write_run_manifest(out_dir, cfg_path, cfg, [out_dir / "rag_retrieval_manifest.csv"])
    comp = pd.read_csv(out_dir / "model_comparison.csv", encoding="utf-8-sig")
    print(comp.to_string(index=False))
    best = comp[comp["scope"].eq("candidate_gold_extended_v2")].sort_values("accuracy", ascending=False).head(1)
    if not best.empty:
        print(f"BEST_FULL_500 {best.iloc[0]['model_name']} accuracy={best.iloc[0]['accuracy']:.4f} macro_f1={best.iloc[0]['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
