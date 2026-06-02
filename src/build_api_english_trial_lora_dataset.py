# -*- coding: utf-8 -*-
"""Translate a 100+100 English trial LoRA dataset through an API model.

This script intentionally stops at a small trial set. It first prechecks the
full summary train/val files, then translates only train100 and val100. It does
not use frozen test labels or post-decision text.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "lora_exports"
    / "summary_compressed_train5000_val1000_20260529_210034"
)
DEFAULT_TRAIN = DEFAULT_SOURCE_DIR / "FT_data_summary_train5000.json"
DEFAULT_VAL = DEFAULT_SOURCE_DIR / "FT_data_summary_val1000.json"
DEFAULT_OUT_DIR = DEFAULT_SOURCE_DIR / "api_english_trial_100"

LABELS = {"support", "partial_support", "not_support"}
HEADERS = [
    "Project context",
    "Claims and defenses",
    "Delay and responsibility facts",
    "Evidence and procedure",
    "Dispute focus",
]
INSTRUCTION = (
    "Based only on the pre-decision information, predict the outcome label of "
    "the delay-related construction claim. Output only one label from: support, "
    "partial_support, not_support."
)
POST_DECISION_RISK_TERMS = [
    "judgment result",
    "the court held",
    "the court finds",
    "the claim is supported",
    "the claim is rejected",
    "uphold the original judgment",
    "revoke the original judgment",
    "support / partial_support / not_support",
    "support, partial_support, not_support",
    "判决如下",
    "本院认为",
    "不予支持",
    "予以支持",
    "驳回",
    "维持原判",
]


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    return data


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def chinese_char_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text or ""))


def normalize_label(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw == "partial":
        return "partial_support"
    return raw if raw in LABELS else "invalid"


def extract_case_id(question: str) -> str:
    match = re.search(r"Case ID:\s*([^\s]+)", question or "")
    return match.group(1).strip() if match else ""


def split_instruction(question: str) -> Tuple[str, str]:
    pos = (question or "").find("Case ID:")
    if pos == -1:
        return INSTRUCTION, question or ""
    return (question[:pos].strip() or INSTRUCTION), question[pos:].strip()


def parse_sections(question_body: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    current = ""
    buf: List[str] = []
    for line in (question_body or "").splitlines():
        stripped = line.strip()
        match = re.match(r"^([A-Za-z and]+):\s*(.*)$", stripped)
        if match and match.group(1) in HEADERS:
            if current:
                sections[current] = "\n".join(buf).strip()
            current = match.group(1)
            buf = [match.group(2).strip()] if match.group(2).strip() else []
        elif current:
            buf.append(stripped)
    if current:
        sections[current] = "\n".join(buf).strip()
    return sections


def post_decision_hits(text: str) -> List[str]:
    lowered = (text or "").lower()
    return [term for term in POST_DECISION_RISK_TERMS if term.lower() in lowered]


def source_record_precheck(row: Dict[str, Any], split: str, row_id: int) -> Dict[str, Any]:
    question = str(row.get("Question", ""))
    label = normalize_label(row.get("Response"))
    _, body = split_instruction(question)
    case_id = extract_case_id(body)
    sections = parse_sections(body)
    missing = [h for h in HEADERS if not sections.get(h, "").strip()]
    summary_text = "\n".join(sections.get(h, "") for h in HEADERS)
    return {
        "split": split,
        "row_id": row_id,
        "case_id": case_id,
        "case_id_missing": not bool(case_id),
        "response": label,
        "response_valid": label in LABELS,
        "missing_fields": "|".join(missing),
        "missing_field_count": len(missing),
        "post_decision_risk": bool(post_decision_hits(summary_text)),
        "post_decision_terms": "|".join(post_decision_hits(summary_text)),
        "summary_chars": len(summary_text),
        "summary_chinese_chars": chinese_char_count(summary_text),
        "field_parse_ok": bool(case_id) and label in LABELS and len(missing) == 0,
    }


def write_precheck(train: List[Dict[str, Any]], val: List[Dict[str, Any]], out_dir: Path) -> None:
    rows = []
    for split, data in [("train", train), ("val", val)]:
        for idx, row in enumerate(data, 1):
            rows.append(source_record_precheck(row, split, idx))
    train_questions = {str(r.get("Question", "")) for r in train}
    val_questions = {str(r.get("Question", "")) for r in val}
    overlap = train_questions & val_questions
    for row in rows:
        row["train_val_question_overlap_count"] = len(overlap)
    write_csv(out_dir / "dataset_translation_precheck_report.csv", rows)


def build_translation_prompt(case_id: str, sections: Dict[str, str]) -> List[Dict[str, str]]:
    payload = {
        "task": "translate_pre_decision_factual_summary_only",
        "case_id": case_id,
        "rules": [
            "Translate only the Chinese factual content into concise legal-engineering English.",
            "Do not add facts, do not infer outcome, do not explain the judgment.",
            "Do not include support, partial_support, or not_support in the translation.",
            "Keep the five section headers exactly.",
            "If a source section is empty, return an empty string for that section.",
            "Use consistent terms: 工期顺延=extension of time; 逾期交房=delayed delivery; 违约金=liquidated damages/contractual penalty; 监理确认=supervision confirmation; 签证=site instruction/variation confirmation.",
        ],
        "required_json": {h: "English translation string" for h in HEADERS},
        "source_sections": {h: sections.get(h, "") for h in HEADERS},
    }
    return [
        {"role": "system", "content": "Return exactly one valid JSON object. No Markdown. No extra text."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in API response")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("API response JSON is not an object")
    return obj


def call_chat(base_url: str, api_key: str, model: str, messages: List[Dict[str, str]], timeout: int, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    obj = resp.json()
    return obj["choices"][0]["message"]["content"], obj.get("usage", {})


def translate_one(
    row: Dict[str, Any],
    split: str,
    row_id: int,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    max_tokens: int,
    retries: int,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    question = str(row.get("Question", ""))
    instruction, body = split_instruction(question)
    case_id = extract_case_id(body)
    label_before = normalize_label(row.get("Response"))
    sections = parse_sections(body)
    messages = build_translation_prompt(case_id, sections)
    last_error = ""
    raw_response = ""
    usage: Dict[str, Any] = {}
    translations: Dict[str, str] = {h: "" for h in HEADERS}
    status = "api_error"

    for attempt in range(retries + 1):
        try:
            raw_response, usage = call_chat(base_url, api_key, model, messages, timeout, max_tokens)
            parsed = extract_json_object(raw_response)
            translations = {h: str(parsed.get(h, "") or "").strip() for h in HEADERS}
            status = "ok"
            break
        except Exception as exc:  # noqa: BLE001 - record API/JSON failures for report
            last_error = str(exc)[:1000]
            if attempt < retries:
                time.sleep(1 + attempt)

    translated_lines = [instruction.strip(), "", f"Case ID: {case_id}", "Pre-decision factual summary:"]
    for header in HEADERS:
        translated_lines.append(f"{header}: {translations.get(header, '')}")
    translated_question = "\n".join(translated_lines)
    label_after = label_before
    translated_text = "\n".join(translations.values())
    missing = [h for h in HEADERS if h not in translations]
    report = {
        "split": split,
        "row_id": row_id,
        "case_id": case_id,
        "api_status": status,
        "error": last_error,
        "label_before": label_before,
        "label_after": label_after,
        "label_changed": label_before != label_after,
        "response_valid": label_after in LABELS,
        "large_chinese_residue": chinese_char_count(translated_text) > 20,
        "chinese_residue_chars": chinese_char_count(translated_text),
        "english_summary_chars": len(translated_text),
        "missing_fields": "|".join(missing),
        "missing_field_count": len(missing),
        "field_parse_ok": bool(case_id) and status == "ok" and label_after in LABELS and not missing,
        "post_decision_risk": bool(post_decision_hits(translated_text)),
        "post_decision_terms": "|".join(post_decision_hits(translated_text)),
        "translation_failed": status != "ok",
        "usage_json": json.dumps(usage, ensure_ascii=False),
        "raw_response_head": raw_response[:500].replace("\n", " "),
    }
    return {"Question": translated_question, "Complex_CoT": "", "Response": label_after}, report


def to_alpaca(row: Dict[str, str]) -> Dict[str, str]:
    parts = row["Question"].split("\n\n", 1)
    if len(parts) == 2:
        instruction, model_input = parts
    else:
        instruction, model_input = INSTRUCTION, row["Question"]
    return {"instruction": instruction.strip(), "input": model_input.strip(), "output": row["Response"]}


def convert_split(
    rows: List[Dict[str, Any]],
    split: str,
    base_url: str,
    api_key: str,
    model: str,
    workers: int,
    timeout: int,
    max_tokens: int,
    retries: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    outputs: List[Dict[str, str] | None] = [None] * len(rows)
    reports: List[Dict[str, Any] | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {
            ex.submit(translate_one, row, split, i + 1, base_url, api_key, model, timeout, max_tokens, retries): i
            for i, row in enumerate(rows)
        }
        done = 0
        for fut in as_completed(futs):
            idx = futs[fut]
            outputs[idx], reports[idx] = fut.result()
            done += 1
            if done % 10 == 0:
                print(f"{split} translated {done}/{len(rows)}", flush=True)
    return [r for r in outputs if r is not None], [r for r in reports if r is not None]


def run_trial(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_all = load_json_array(Path(args.train))
    val_all = load_json_array(Path(args.val))
    write_precheck(train_all, val_all, out_dir)

    train = train_all[: args.limit]
    val = val_all[: args.limit]
    api_key = os.getenv(args.api_key_env, "").strip()
    base_url = os.getenv(args.base_url_env, "").strip().rstrip("/") or args.base_url.rstrip("/")
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")
    if not base_url:
        raise RuntimeError("Missing API base URL")

    train_en, train_report = convert_split(train, "train", base_url, api_key, args.model, args.workers, args.timeout, args.max_tokens, args.retries)
    val_en, val_report = convert_split(val, "val", base_url, api_key, args.model, args.workers, args.timeout, args.max_tokens, args.retries)

    files = {
        "FT_data_summary_train100_en.json": train_en,
        "FT_data_summary_val100_en.json": val_en,
        "FT_data_summary_train100_en_alpaca.jsonl": [to_alpaca(r) for r in train_en],
        "FT_data_summary_val100_en_alpaca.jsonl": [to_alpaca(r) for r in val_en],
    }
    for name, rows in files.items():
        path = out_dir / name
        if name.endswith(".jsonl"):
            write_jsonl(path, rows)
        else:
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    quality = train_report + val_report
    write_csv(out_dir / "translation_quality_report_100.csv", quality)
    failed = [r for r in quality if r["translation_failed"]]
    write_csv(out_dir / "translation_failed_samples_100.csv", failed)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "api_translation_trial_only",
        "provider_base_url_env": args.base_url_env,
        "api_key_env": args.api_key_env,
        "model": args.model,
        "limit_per_split": args.limit,
        "train_rows": len(train_en),
        "val_rows": len(val_en),
        "translation_failed": len(failed),
        "large_chinese_residue": sum(1 for r in quality if r["large_chinese_residue"]),
        "label_changed": sum(1 for r in quality if r["label_changed"]),
        "post_decision_risk": sum(1 for r in quality if r["post_decision_risk"]),
        "outputs": sorted(files),
        "precheck_report": "dataset_translation_precheck_report.csv",
        "quality_report": "translation_quality_report_100.csv",
    }
    (out_dir / "manifest_api_trial_100.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = out_dir / "DelayDispute_api_english_trial_train100_val100.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in [
            *[out_dir / name for name in sorted(files)],
            out_dir / "translation_quality_report_100.csv",
            out_dir / "translation_failed_samples_100.csv",
            out_dir / "dataset_translation_precheck_report.csv",
            out_dir / "manifest_api_trial_100.json",
        ]:
            zf.write(p, arcname=p.name)
    desktop_path = Path.home() / "Desktop" / zip_path.name
    desktop_path.write_bytes(zip_path.read_bytes())
    manifest["zip_path"] = str(zip_path)
    manifest["desktop_zip"] = str(desktop_path)
    (out_dir / "manifest_api_trial_100.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--val", default=str(DEFAULT_VAL))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--base-url-env", default="MIMO_OPENAI_BASE_URL")
    parser.add_argument("--api-key-env", default="MIMO_API_KEY")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="mimo-v2.5-pro")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    manifest = run_trial(parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
