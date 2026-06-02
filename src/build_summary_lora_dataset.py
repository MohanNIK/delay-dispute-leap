# -*- coding: utf-8 -*-
"""Build summary-compressed LoRA datasets for DelayDispute classification.

This script is deliberately local and deterministic: it does not call any LLM
API. It compresses long pre-decision inputs into structured factual summaries
so Qwen2.5-7B LoRA training stays closer to a 2048-token window.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = {"support", "partial_support", "not_support"}
INSTRUCTION = (
    "Based only on the pre-decision information, predict the outcome label of "
    "the delay-related construction claim. Output only one label from: support, "
    "partial_support, not_support."
)

DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "lora_exports"
    / "llamafactory_qwen25_7b_balanced_train5000_val1000_20260529_003410"
)
DEFAULT_TRAIN = DEFAULT_SOURCE_DIR / "delay_dispute_train5000_balanced_alpaca.json"
DEFAULT_VAL = DEFAULT_SOURCE_DIR / "delay_dispute_val1000_balanced_alpaca.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "lora_exports"

CLAIM_TERMS = [
    "诉讼请求",
    "请求",
    "主张",
    "辩称",
    "反诉",
    "上诉",
    "抗辩",
    "答辩",
    "原告",
    "被告",
]
DELAY_TERMS = [
    "工期",
    "延误",
    "逾期",
    "顺延",
    "延期",
    "停工",
    "窝工",
    "竣工",
    "开工",
    "进度",
    "关键线路",
    "违约金",
]
EVIDENCE_TERMS = [
    "证据",
    "签证",
    "通知",
    "索赔",
    "施工日志",
    "监理",
    "会议纪要",
    "进度计划",
    "鉴定",
    "验收",
    "合同",
    "付款",
]
ISSUE_TERMS = ["争议焦点", "焦点", "本案争议", "是否", "如何确定", "责任", "原因", "举证"]
CONSTRUCTION_TERMS = ["建设工程", "施工", "工程", "承包", "发包", "分包", "竣工", "质量"]


def normalize_label(value: Any) -> str:
    raw = str(value or "").strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if raw == "partial":
        return "partial_support"
    if raw in LABELS:
        return raw
    if "partial_support" in raw or "partial" in raw:
        return "partial_support"
    if "not_support" in raw or "unsupported" in raw or "reject" in raw or "dismiss" in raw:
        return "not_support"
    if raw == "support" or re.search(r"\bsupport\b", raw):
        return "support"
    return "invalid"


def clean_text(text: str) -> str:
    text = str(text or "").replace("\u3000", " ")
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_case_id(text: str) -> str:
    match = re.search(r"Case ID:\s*([^\s]+)", text)
    return match.group(1).strip() if match else ""


def extract_title(text: str) -> str:
    match = re.search(r"Title:\s*(.+?)(?:\n|$)", text)
    return match.group(1).strip() if match else ""


def strip_input_prefix(text: str) -> str:
    body = re.sub(r"^Case ID:\s*[^\n]+\n?", "", text, flags=re.I).strip()
    body = re.sub(r"^Title:\s*.+?\n", "", body, flags=re.I).strip()
    body = re.sub(r"^Pre-decision (?:factual )?information:\s*", "", body, flags=re.I).strip()
    return body


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = re.split(r"(?<=[。！？；;])\s*|\n+", text)
    sentences = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > 260:
            chunks = [part[i : i + 240] for i in range(0, len(part), 240)]
            sentences.extend(chunks)
        else:
            sentences.append(part)
    return sentences


def count_terms(text: str, terms: Sequence[str]) -> int:
    return sum(1 for term in terms if term and term in text)


def score_sentence(sentence: str, index: int) -> int:
    score = 0
    score += 4 * count_terms(sentence, DELAY_TERMS)
    score += 3 * count_terms(sentence, EVIDENCE_TERMS)
    score += 3 * count_terms(sentence, CLAIM_TERMS)
    score += 2 * count_terms(sentence, ISSUE_TERMS)
    score += count_terms(sentence, CONSTRUCTION_TERMS)
    if index < 6:
        score += 2
    if 35 <= len(sentence) <= 220:
        score += 1
    return score


def pick_section(sentences: Sequence[str], terms: Sequence[str], limit_chars: int, fallback_start: int = 0) -> str:
    scored: List[Tuple[int, int, str]] = []
    for idx, sentence in enumerate(sentences):
        hits = count_terms(sentence, terms)
        if hits:
            scored.append((hits * 10 + score_sentence(sentence, idx), idx, sentence))
    if not scored and fallback_start < len(sentences):
        scored = [(score_sentence(s, i), i, s) for i, s in enumerate(sentences[fallback_start : fallback_start + 8], fallback_start)]
    chosen = sorted(sorted(scored, reverse=True)[:6], key=lambda item: item[1])
    text = " ".join(item[2] for item in chosen)
    return text[:limit_chars].strip()


def truncate_at_sentence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last = max(cut.rfind("。"), cut.rfind("；"), cut.rfind("."), cut.rfind("\n"))
    if last > max_chars * 0.72:
        return cut[: last + 1].strip()
    return cut.rstrip() + "..."


def summarize_input(text: str, target_min_chars: int = 1200, target_max_chars: int = 1600) -> str:
    text = clean_text(text)
    case_id = extract_case_id(text)
    title = extract_title(text)
    body = strip_input_prefix(text)
    sentences = split_sentences(body)

    context = " ".join(sentences[:3])[:360].strip()
    claims = pick_section(sentences, CLAIM_TERMS, 360, fallback_start=0)
    delay = pick_section(sentences, DELAY_TERMS, 420, fallback_start=3)
    evidence = pick_section(sentences, EVIDENCE_TERMS, 420, fallback_start=6)
    issue = pick_section(sentences, ISSUE_TERMS, 300, fallback_start=9)

    lines = []
    if case_id:
        lines.append(f"Case ID: {case_id}")
    if title:
        lines.append(f"Title: {title[:120]}")
    lines.append("Pre-decision factual summary:")
    if context:
        lines.append(f"Project context: {context}")
    if claims:
        lines.append(f"Claims and defenses: {claims}")
    if delay:
        lines.append(f"Delay and responsibility facts: {delay}")
    if evidence:
        lines.append(f"Evidence and procedure: {evidence}")
    if issue:
        lines.append(f"Dispute focus: {issue}")

    summary = "\n".join(lines)
    if len(summary) < target_min_chars and len(body) > len(summary):
        used = set()
        for chunk in [context, claims, delay, evidence, issue]:
            for s in split_sentences(chunk):
                used.add(s)
        extras = [
            s
            for i, s in sorted(((score_sentence(s, i), s) for i, s in enumerate(sentences)), reverse=True)
            if s not in used
        ]
        appendix = []
        for s in extras:
            if len(summary) + len(" ".join(appendix)) + len(s) + 32 > target_max_chars:
                break
            appendix.append(s)
            if len(summary) + len(" ".join(appendix)) >= target_min_chars:
                break
        if appendix:
            summary += "\nAdditional relevant facts: " + " ".join(reversed(appendix))

    return truncate_at_sentence(summary, target_max_chars)


def build_question(instruction: str, summarized_input: str) -> str:
    instruction = (instruction or INSTRUCTION).strip()
    return f"{instruction}\n\n{summarized_input.strip()}"


def build_detection_record(source: Dict[str, Any], split: str, row_id: int) -> Tuple[Dict[str, str], Dict[str, Any]]:
    label = normalize_label(source.get("output") or source.get("Response"))
    raw_input = str(source.get("input") or source.get("Question") or "")
    summary = summarize_input(raw_input)
    question = build_question(str(source.get("instruction") or INSTRUCTION), summary)
    case_id = extract_case_id(raw_input) or f"{split}_{row_id:05d}"
    valid_label = label in LABELS
    has_delay_signal = bool(count_terms(summary, DELAY_TERMS))
    has_construction_signal = bool(count_terms(summary, CONSTRUCTION_TERMS))
    needs_review = (not valid_label) or (not has_construction_signal) or len(summary) < 250
    record = {
        "Question": question,
        "Complex_CoT": "",
        "Response": label if valid_label else "invalid",
    }
    meta = {
        "split": split,
        "row_id": row_id,
        "case_id": case_id,
        "valid_label": valid_label,
        "label": label,
        "original_chars": len(raw_input),
        "summary_chars": len(summary),
        "question_chars": len(question),
        "compression_ratio": round(len(summary) / max(1, len(raw_input)), 4),
        "has_delay_signal": has_delay_signal,
        "has_construction_signal": has_construction_signal,
        "needs_review": needs_review,
        "review_reason": ";".join(
            reason
            for reason, flag in [
                ("invalid_label", not valid_label),
                ("missing_construction_signal", not has_construction_signal),
                ("summary_too_short", len(summary) < 250),
            ]
            if flag
        ),
    }
    return record, meta


def to_alpaca(record: Dict[str, str]) -> Dict[str, str]:
    question = record["Question"]
    if "\n\n" in question:
        instruction, model_input = question.split("\n\n", 1)
    else:
        instruction, model_input = INSTRUCTION, question
    return {"instruction": instruction.strip(), "input": model_input.strip(), "output": record["Response"]}


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
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


def build_split(rows: List[Dict[str, Any]], split: str) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    records, metas = [], []
    seen = set()
    for idx, row in enumerate(rows, 1):
        rec, meta = build_detection_record(row, split, idx)
        if rec["Question"] in seen:
            meta["needs_review"] = True
            meta["review_reason"] = (meta["review_reason"] + ";duplicate_question").strip(";")
        seen.add(rec["Question"])
        records.append(rec)
        metas.append(meta)
    return records, metas


def aggregate_report(split_name: str, records: List[Dict[str, str]], metas: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = Counter(r["Response"] for r in records)
    lengths = [int(m["summary_chars"]) for m in metas]
    q_lengths = [int(m["question_chars"]) for m in metas]
    return {
        "split": split_name,
        "rows": len(records),
        "valid_labels": sum(1 for m in metas if m["valid_label"]),
        "needs_review": sum(1 for m in metas if m["needs_review"]),
        "duplicate_questions": len(records) - len({r["Question"] for r in records}),
        "label_support": labels.get("support", 0),
        "label_partial_support": labels.get("partial_support", 0),
        "label_not_support": labels.get("not_support", 0),
        "summary_chars_min": min(lengths) if lengths else 0,
        "summary_chars_p50": statistics.median(lengths) if lengths else 0,
        "summary_chars_p90": sorted(lengths)[int(len(lengths) * 0.9)] if lengths else 0,
        "summary_chars_max": max(lengths) if lengths else 0,
        "question_chars_p90": sorted(q_lengths)[int(len(q_lengths) * 0.9)] if q_lengths else 0,
        "question_chars_max": max(q_lengths) if q_lengths else 0,
    }


def build_dataset(train_path: Path, val_path: Path, out_dir: Path, desktop_zip: bool) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_json_array(train_path)
    val_rows = load_json_array(val_path)

    train_records, train_meta = build_split(train_rows, "train")
    val_records, val_meta = build_split(val_rows, "val")

    train_questions = {r["Question"] for r in train_records}
    val_questions = {r["Question"] for r in val_records}
    train_val_overlap = train_questions & val_questions
    if train_val_overlap:
        raise RuntimeError(f"train/val duplicate Question count: {len(train_val_overlap)}")

    outputs = {
        "FT_data_summary_train5000.json": train_records,
        "FT_data_summary_val1000.json": val_records,
        "FT_data_summary_train5000_alpaca.jsonl": [to_alpaca(r) for r in train_records],
        "FT_data_summary_val1000_alpaca.jsonl": [to_alpaca(r) for r in val_records],
    }
    for name, rows in outputs.items():
        path = out_dir / name
        if name.endswith(".jsonl"):
            write_jsonl(path, rows)
        else:
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    write_csv(out_dir / "summary_quality_report.csv", train_meta + val_meta)
    needs = [m for m in train_meta + val_meta if m["needs_review"]]
    write_csv(out_dir / "needs_relabel_or_review.csv", needs)

    report_rows = [
        aggregate_report("train", train_records, train_meta),
        aggregate_report("val", val_records, val_meta),
    ]
    write_csv(out_dir / "dataset_validation_report.csv", report_rows)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "local_deterministic_keyword_summary_no_api",
        "train_source": str(train_path),
        "val_source": str(val_path),
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "train_val_question_overlap": len(train_val_overlap),
        "labels": sorted(LABELS),
        "outputs": sorted(outputs),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = (
        "# DelayDispute summary-compressed LoRA data\n\n"
        "This package is generated locally without LLM/API calls. It keeps the English classification instruction "
        "and replaces long pre-decision text with a structured Chinese factual summary.\n\n"
        "Primary files for training:\n"
        "- FT_data_summary_train5000.json\n"
        "- FT_data_summary_val1000.json\n\n"
        "Alternative Alpaca JSONL files are included for LLaMA-Factory style loaders.\n"
    )
    (out_dir / "README_summary_data.md").write_text(readme, encoding="utf-8")

    zip_path = out_dir / "DelayDispute_summary_train5000_val1000_package.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in [
            out_dir / "FT_data_summary_train5000.json",
            out_dir / "FT_data_summary_val1000.json",
            out_dir / "FT_data_summary_train5000_alpaca.jsonl",
            out_dir / "FT_data_summary_val1000_alpaca.jsonl",
            out_dir / "dataset_validation_report.csv",
            out_dir / "summary_quality_report.csv",
            out_dir / "needs_relabel_or_review.csv",
            out_dir / "manifest.json",
            out_dir / "README_summary_data.md",
        ]:
            zf.write(p, arcname=p.name)

    if desktop_zip:
        desktop_path = Path.home() / "Desktop" / zip_path.name
        desktop_path.write_bytes(zip_path.read_bytes())
        manifest["desktop_zip"] = str(desktop_path)
    manifest["zip_path"] = str(zip_path)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--val", default=str(DEFAULT_VAL))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--no-desktop-zip", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUTPUT_ROOT / f"summary_compressed_train5000_val1000_{timestamp}"
    manifest = build_dataset(Path(args.train), Path(args.val), out_dir, desktop_zip=not args.no_desktop_zip)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
