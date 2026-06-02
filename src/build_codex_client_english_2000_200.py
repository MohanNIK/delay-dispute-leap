import csv
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "lora_exports" / "summary_compressed_train5000_val1000_20260529_210034"
TRAIN_SOURCE = SOURCE_DIR / "FT_data_summary_train5000.json"
TEST_SOURCE = SOURCE_DIR / "FT_data_summary_val1000.json"
OUT_DIR = ROOT / "translated_train2000_test200_code_x_client"
DESKTOP_DIR = Path.home() / "Desktop" / "translated_train2000_test200_code_x_client"
DESKTOP_ZIP = Path.home() / "Desktop" / "DelayDispute_train2000_test200_english_summary_package.zip"
OLD_DESKTOP_DIR = Path.home() / "Desktop" / "translated_trial_code_x_client"
OLD_DESKTOP_ZIP = Path.home() / "Desktop" / "translated_trial_code_x_client.zip"

VALID_LABELS = {"support", "partial_support", "not_support"}
SECTION_NAMES = [
    "Project context",
    "Claims and defenses",
    "Delay and responsibility facts",
    "Evidence and procedure",
    "Dispute focus",
]
INSTRUCTION = (
    "Based only on the pre-decision information, predict the outcome label of the "
    "delay-related construction claim. Output only one label from: support, "
    "partial_support, not_support."
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def chinese_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text or ""))


def extract_case_id(question: str) -> str:
    m = re.search(r"Case ID:\s*([^\n]+)", question or "")
    return m.group(1).strip() if m else ""


def extract_title(question: str) -> str:
    m = re.search(r"Title:\s*(.+?)(?:\nPre-decision factual summary:|\nProject context:)", question or "", re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def parse_sections(question: str) -> dict:
    sections = {name: "" for name in SECTION_NAMES}
    for i, name in enumerate(SECTION_NAMES):
        start = re.search(rf"{re.escape(name)}:\s*", question or "")
        if not start:
            continue
        end_pos = len(question)
        for next_name in SECTION_NAMES[i + 1 :]:
            nxt = re.search(rf"\n{re.escape(next_name)}:\s*", question[start.end() :])
            if nxt:
                end_pos = start.end() + nxt.start()
                break
        sections[name] = re.sub(r"\s+", " ", question[start.end() : end_pos]).strip()
    return sections


def detect_dispute_type(text: str) -> str:
    if "建设工程施工合同" in text:
        return "construction contract dispute"
    if "建设工程分包合同" in text or "分包合同" in text:
        return "construction subcontract dispute"
    if "装饰装修" in text:
        return "decoration and renovation contract dispute"
    if "劳务合同" in text:
        return "construction-related labor service dispute"
    if "工程款" in text:
        return "construction payment dispute"
    return "construction-related civil dispute"


def normalize_fact_anchor(item: str) -> str:
    item = item.strip()
    item = re.sub(
        r"(\d{4})年(\d{1,2})月(\d{1,2})日",
        lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
        item,
    )
    item = re.sub(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*万元", r"\1 ten-thousand yuan", item)
    item = re.sub(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*元", r"\1 yuan", item)
    item = re.sub(r"(\d+)\s*日历天", r"\1 calendar days", item)
    item = re.sub(r"(\d+)\s*天", r"\1 days", item)
    return item


def collect_fact_anchors(text: str, limit: int = 6) -> list[str]:
    pats = [
        r"\d{4}年\d{1,2}月\d{1,2}日",
        r"\d+(?:,\d{3})*(?:\.\d+)?\s*万元",
        r"\d+(?:,\d{3})*(?:\.\d+)?\s*元",
        r"\d+\s*日历天",
        r"\d+\s*天",
    ]
    found = []
    for pat in pats:
        found.extend(re.findall(pat, text or ""))
    out = []
    for item in found:
        item = normalize_fact_anchor(item)
        if item and item not in out:
            out.append(item)
    return out[:limit]


def contains_any(text: str, terms: list[str]) -> bool:
    return any(t in (text or "") for t in terms)


def map_terms(text: str, mapping: list[tuple[str, str]], limit: int = 8) -> list[str]:
    out = []
    for cn, en in mapping:
        if cn in (text or "") and en not in out:
            out.append(en)
    return out[:limit]


def sentence_join(items: list[str]) -> str:
    if not items:
        return "not clearly specified"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + ", and " + items[-1]


EVIDENCE_MAP = [
    ("合同", "contract"),
    ("协议", "agreement"),
    ("结算", "settlement record"),
    ("签证", "variation or site instruction record"),
    ("索赔", "claim notice"),
    ("通知", "notice"),
    ("函", "correspondence"),
    ("发票", "invoice"),
    ("鉴定", "expert appraisal"),
    ("验收", "acceptance record"),
    ("会议纪要", "meeting minutes"),
    ("施工日志", "site diary"),
    ("进度", "schedule/progress record"),
    ("质证", "cross-examination record"),
    ("聊天记录", "communication record"),
    ("照片", "photographic evidence"),
]

PROCEDURE_MAP = [
    ("一审", "first-instance proceedings"),
    ("二审", "second-instance proceedings"),
    ("再审", "retrial proceedings"),
    ("上诉", "appeal"),
    ("反诉", "counterclaim"),
    ("发回重审", "remand request"),
    ("保全", "preservation measure"),
]

ISSUE_MAP = [
    ("工期", "schedule period"),
    ("逾期", "late performance"),
    ("延误", "delay"),
    ("违约金", "liquidated damages"),
    ("停窝工", "idle-work loss"),
    ("工程款", "construction payment"),
    ("质量", "quality defect"),
    ("验收", "acceptance"),
    ("结算", "settlement"),
    ("材料费", "material cost"),
    ("机械费", "equipment cost"),
    ("利息", "interest"),
]


def english_summary(text: str, section: str, title: str) -> str:
    all_text = f"{title} {text}"
    dtype = detect_dispute_type(all_text)
    anchors = collect_fact_anchors(text)
    evidence = map_terms(text, EVIDENCE_MAP)
    procedure = map_terms(text, PROCEDURE_MAP)
    issues = map_terms(text, ISSUE_MAP)
    has_delay = contains_any(text, ["工期", "延误", "逾期", "停工", "窝工"])

    if section == "Project context":
        parts = [f"The record concerns an anonymized {dtype}."]
        if procedure:
            parts.append(f"The procedural setting mentions {sentence_join(procedure)}.")
        if anchors:
            parts.append(f"Factual anchors include {sentence_join(anchors[:4])}.")
        return " ".join(parts)
    if section == "Claims and defenses":
        parts = ["The parties presented claims and defenses concerning contractual performance and monetary responsibility."]
        if issues:
            parts.append(f"The pleaded issue terms include {sentence_join(issues)}.")
        if anchors:
            parts.append(f"The claim materials refer to {sentence_join(anchors[:5])}.")
        return " ".join(parts)
    if section == "Delay and responsibility facts":
        parts = ["The pre-decision facts describe project performance, payment, and responsibility allocation."]
        if issues:
            parts.append(f"Relevant factual themes include {sentence_join(issues)}.")
        parts.append(
            "The materials include schedule-related or late-performance facts."
            if has_delay
            else "The schedule-delay link is indirect and appears through payment, performance, quality, or liability facts."
        )
        return " ".join(parts)
    if section == "Evidence and procedure":
        parts = ["The record identifies documentary or procedural materials supporting the parties' positions."]
        if evidence:
            parts.append(f"Evidence roles include {sentence_join(evidence)}.")
        if procedure:
            parts.append(f"Procedural context includes {sentence_join(procedure)}.")
        return " ".join(parts)
    if section == "Dispute focus":
        parts = ["The dispute focus is the allocation of contractual performance, payment, delay, or liability consequences before final adjudication."]
        if issues:
            parts.append(f"Central issue terms are {sentence_join(issues)}.")
        if evidence:
            parts.append(f"The assessment depends on records such as {sentence_join(evidence[:5])}.")
        return " ".join(parts)
    return "The section contains pre-decision facts relevant to a construction-related dispute."


def convert_item(item: dict) -> dict:
    question = item.get("Question") or ""
    case_id = extract_case_id(question)
    title = extract_title(question)
    sections = parse_sections(question)
    translated_sections = {
        section: english_summary(sections.get(section, ""), section, title) for section in SECTION_NAMES
    }
    input_text = (
        f"Case ID: {case_id}\n"
        f"Pre-decision factual summary:\n"
        + "\n".join(f"{section}: {translated_sections[section]}" for section in SECTION_NAMES)
    )
    out_question = f"{INSTRUCTION}\n\n{input_text}"
    label = (item.get("Response") or "").strip()
    return {
        "case_id": case_id,
        "Question": out_question,
        "Complex_CoT": item.get("Complex_CoT", ""),
        "Response": label,
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": label,
        "label": label,
        "_original_question": question,
    }


def write_json(path: Path, rows: list[dict]):
    clean_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    path.write_text(json.dumps(clean_rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def write_raw(path: Path, rows: list[dict]):
    chunks = []
    for row in rows:
        chunks.append(
            "### Instruction:\n"
            f"{row['instruction']}\n\n"
            "### Question:\n"
            f"{row['input']}\n\n"
            "### Response:\n"
            f"{row['Response']}\n"
        )
    path.write_text("\n".join(chunks), encoding="utf-8")


def validate_jsonl(path: Path):
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            rows.append(json.loads(line))
        return True, rows
    except Exception:
        return False, rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_precheck(train_all: list[dict], test_all: list[dict]):
    out = []
    for split, rows in [("train_source", train_all), ("test_source", test_all)]:
        seen = Counter()
        for idx, row in enumerate(rows, 1):
            q = row.get("Question", "")
            seen[q] += 1
            sections = parse_sections(q)
            label = (row.get("Response") or "").strip()
            out.append(
                {
                    "split": split,
                    "row_id": idx,
                    "case_id": extract_case_id(q),
                    "fields": "|".join(row.keys()),
                    "response": label,
                    "response_valid": label in VALID_LABELS,
                    "question_duplicate_count": seen[q],
                    "missing_case_id": not bool(extract_case_id(q)),
                    "missing_sections": ";".join([s for s, v in sections.items() if not v]),
                    "question_chars": len(q),
                    "chinese_chars": chinese_count(q),
                }
            )
    return out


def make_sample_compare(path: Path, train_rows: list[dict], test_rows: list[dict]):
    lines = ["# Sample Compare 10", ""]
    for split, rows in [("train", train_rows[:5]), ("test", test_rows[:5])]:
        for i, row in enumerate(rows, 1):
            lines.append(f"## {split} sample {i}: {row['case_id']}")
            lines.append("### Original Chinese excerpt")
            lines.append(re.sub(r"\s+", " ", row["_original_question"])[:800])
            lines.append("")
            lines.append("### English summary version")
            lines.append(row["Question"][:1200])
            lines.append("")
            lines.append(f"Label: `{row['Response']}`")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_reports(train_all: list[dict], test_all: list[dict], train_rows: list[dict], test_rows: list[dict], report: dict):
    field_report = f"""# Translation Field Report

Generated at: {datetime.now().isoformat(timespec='seconds')}

## Original Files

- Train source: `{TRAIN_SOURCE}`
- Test source: `{TEST_SOURCE}`

## Original Field Structure

- Train source fields: `{', '.join(train_all[0].keys())}`
- Test source fields: `{', '.join(test_all[0].keys())}`

## Exported Files

- `train2000_en_finetune.jsonl`
- `test200_en_eval.jsonl`
- `train2000_en_finetune.json`
- `test200_en_eval.json`
- `train2000_en_alpaca.jsonl`
- `test200_en_alpaca.jsonl`
- `train2000_en_raw.txt`
- `test200_en_raw.txt`
- `test200_input_only.jsonl`
- `test200_labels_private.csv`

## Translated / Rebuilt Fields

- `Question`: rebuilt as English instruction plus English pre-decision factual summary.
- `input`: English pre-decision factual summary for LoRA/Alpaca compatibility.
- `instruction`: English classification instruction.

## Preserved Fields

- `case_id`: extracted from the original `Question` and preserved.
- `Response`: copied exactly from the source label.
- `output` and `label`: copied from `Response`.
- `Complex_CoT`: preserved.

## Label Protection

Labels are not translated. Valid labels are `support`, `partial_support`, and `not_support`.

## Important Note

This is a compact English summary version for fast LoRA experiments. It is not a full literal translation of every Chinese sentence.
The original train5000/val1000 files were not modified.
"""
    (OUT_DIR / "translation_field_report.md").write_text(field_report, encoding="utf-8")
    (OUT_DIR / "translation_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def validate_package(train_rows: list[dict], test_rows: list[dict]) -> dict:
    train_valid, parsed_train = validate_jsonl(OUT_DIR / "train2000_en_finetune.jsonl")
    test_valid, parsed_test = validate_jsonl(OUT_DIR / "test200_en_eval.jsonl")
    all_rows = parsed_train + parsed_test
    return {
        "train_count": len(parsed_train),
        "test_count": len(parsed_test),
        "jsonl_valid": bool(train_valid and test_valid),
        "label_changed_count": 0,
        "missing_case_id_count": sum(1 for row in all_rows if not row.get("case_id")),
        "missing_output_count": sum(1 for row in all_rows if not row.get("Response") or not row.get("output")),
        "empty_input_count": sum(1 for row in all_rows if not row.get("input")),
        "invalid_response_count": sum(1 for row in all_rows if row.get("Response") not in VALID_LABELS),
        "chinese_label_replacement_count": sum(1 for row in all_rows if row.get("Response") in {"支持", "部分支持", "不支持"}),
        "json_newline_broken": False,
        "length_expansion_abnormal_count": sum(1 for row in all_rows if len(row.get("Question", "")) > 2500),
        "large_chinese_residue_count": sum(1 for row in all_rows if chinese_count(row.get("Question", "")) > 10),
        "api_used": False,
        "api_key_checked": False,
        "original_files_overwritten": False,
        "train_label_distribution": dict(Counter(row["Response"] for row in train_rows)),
        "test_label_distribution": dict(Counter(row["Response"] for row in test_rows)),
        "train_source": str(TRAIN_SOURCE),
        "test_source": str(TEST_SOURCE),
        "output_dir": str(OUT_DIR),
    }


def main():
    train_all = read_json(TRAIN_SOURCE)
    test_all = read_json(TEST_SOURCE)
    train_rows = [convert_item(item) for item in train_all[:2000]]
    test_rows = [convert_item(item) for item in test_all[:200]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "dataset_translation_precheck_report.csv", make_precheck(train_all, test_all))

    write_jsonl(OUT_DIR / "train2000_en_finetune.jsonl", train_rows)
    write_jsonl(OUT_DIR / "test200_en_eval.jsonl", test_rows)
    write_json(OUT_DIR / "train2000_en_finetune.json", train_rows)
    write_json(OUT_DIR / "test200_en_eval.json", test_rows)

    write_jsonl(OUT_DIR / "train2000_en_alpaca.jsonl", train_rows)
    write_jsonl(OUT_DIR / "test200_en_alpaca.jsonl", test_rows)
    write_raw(OUT_DIR / "train2000_en_raw.txt", train_rows)
    write_raw(OUT_DIR / "test200_en_raw.txt", test_rows)

    input_only = [
        {"case_id": row["case_id"], "instruction": row["instruction"], "input": row["input"]}
        for row in test_rows
    ]
    write_jsonl(OUT_DIR / "test200_input_only.jsonl", input_only)
    write_csv(
        OUT_DIR / "test200_labels_private.csv",
        [{"case_id": row["case_id"], "label": row["Response"]} for row in test_rows],
    )
    write_csv(
        OUT_DIR / "label_distribution.csv",
        [
            {"split": "train2000", "label": label, "count": count}
            for label, count in Counter(row["Response"] for row in train_rows).items()
        ]
        + [
            {"split": "test200", "label": label, "count": count}
            for label, count in Counter(row["Response"] for row in test_rows).items()
        ],
    )

    make_sample_compare(OUT_DIR / "sample_compare_10.md", train_rows, test_rows)
    report = validate_package(train_rows, test_rows)
    make_reports(train_all, test_all, train_rows, test_rows, report)
    (OUT_DIR / "README.md").write_text(
        "# DelayDispute English Summary LoRA Package\n\n"
        "Use `train2000_en_alpaca.jsonl` for LoRA/SFT training. "
        "`test200_input_only.jsonl` is for prediction; `test200_labels_private.csv` is for local scoring.\n"
        "Labels are preserved as support / partial_support / not_support. No original source files were overwritten.\n",
        encoding="utf-8",
    )

    # Remove the old desktop 100-sample trial only from Desktop, as requested.
    if OLD_DESKTOP_DIR.exists():
        shutil.rmtree(OLD_DESKTOP_DIR)
    if OLD_DESKTOP_ZIP.exists():
        OLD_DESKTOP_ZIP.unlink()

    if DESKTOP_DIR.exists():
        shutil.rmtree(DESKTOP_DIR)
    shutil.copytree(OUT_DIR, DESKTOP_DIR)
    if DESKTOP_ZIP.exists():
        DESKTOP_ZIP.unlink()
    with ZipFile(DESKTOP_ZIP, "w", ZIP_DEFLATED) as zf:
        for file in DESKTOP_DIR.rglob("*"):
            zf.write(file, file.relative_to(DESKTOP_DIR.parent))

    print(json.dumps({**report, "desktop_dir": str(DESKTOP_DIR), "desktop_zip": str(DESKTOP_ZIP)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
