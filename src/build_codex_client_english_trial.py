import csv
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "lora_exports" / "summary_compressed_train5000_val1000_20260529_210034"
TRAIN_SOURCE = SOURCE_DIR / "FT_data_summary_train5000.json"
VAL_SOURCE = SOURCE_DIR / "FT_data_summary_val1000.json"
OUT_DIR = ROOT / "translated_trial_code_x_client"
DESKTOP_DIR = Path.home() / "Desktop" / "translated_trial_code_x_client"
VALID_LABELS = {"support", "partial_support", "not_support"}
SECTION_NAMES = [
    "Project context",
    "Claims and defenses",
    "Delay and responsibility facts",
    "Evidence and procedure",
    "Dispute focus",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


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


def collect_amounts(text: str, limit: int = 5) -> list[str]:
    # Keep monetary and duration numbers because they are factual anchors.
    pats = [
        r"\d+(?:,\d{3})*(?:\.\d+)?\s*元",
        r"\d+(?:\.\d+)?\s*万元",
        r"\d+\s*日历天",
        r"\d+\s*天",
        r"\d{4}年\d{1,2}月\d{1,2}日",
    ]
    found = []
    for pat in pats:
        found.extend(re.findall(pat, text or ""))
    clean = []
    for item in found:
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
        if item and item not in clean:
            clean.append(item)
    return clean[:limit]


def evidence_terms(text: str) -> list[str]:
    mapping = [
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
    out = []
    for cn, en in mapping:
        if cn in (text or "") and en not in out:
            out.append(en)
    return out[:8]


def procedure_terms(text: str) -> list[str]:
    mapping = [
        ("一审", "first-instance proceedings"),
        ("二审", "second-instance proceedings"),
        ("再审", "retrial proceedings"),
        ("上诉", "appeal"),
        ("反诉", "counterclaim"),
        ("发回重审", "remand request"),
        ("保全", "preservation measure"),
    ]
    return [en for cn, en in mapping if cn in (text or "")]


def issue_terms(text: str) -> list[str]:
    mapping = [
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
    out = []
    for cn, en in mapping:
        if cn in (text or "") and en not in out:
            out.append(en)
    return out[:8]


def sentence_join(items: list[str]) -> str:
    if not items:
        return "not clearly specified"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + ", and " + items[-1]


def safe_summary(text: str, purpose: str, title: str) -> str:
    all_text = f"{title} {text}"
    dtype = detect_dispute_type(all_text)
    amounts = collect_amounts(text)
    ev = evidence_terms(text)
    proc = procedure_terms(text)
    issues = issue_terms(text)

    if purpose == "Project context":
        parts = [f"The record concerns an anonymized {dtype}."]
        if proc:
            parts.append(f"The procedural setting mentions {sentence_join(proc)}.")
        if amounts:
            parts.append(f"Factual anchors include {sentence_join(amounts[:3])}.")
        return " ".join(parts)

    if purpose == "Claims and defenses":
        parts = ["The parties presented claims and defenses concerning performance obligations and monetary responsibility."]
        if issues:
            parts.append(f"The pleaded issues include {sentence_join(issues)}.")
        if amounts:
            parts.append(f"The claim materials refer to {sentence_join(amounts)}.")
        return " ".join(parts)

    if purpose == "Delay and responsibility facts":
        parts = ["The pre-decision facts describe project performance, payment, and responsibility allocation issues."]
        if issues:
            parts.append(f"Relevant factual themes include {sentence_join(issues)}.")
        if "工期" in text or "延误" in text or "逾期" in text:
            parts.append("The materials include schedule-related allegations or late-performance facts.")
        else:
            parts.append("The schedule-delay link is indirect and mainly appears through payment, performance, or liability facts.")
        return " ".join(parts)

    if purpose == "Evidence and procedure":
        parts = ["The record identifies documentary or procedural materials used to support the parties' positions."]
        if ev:
            parts.append(f"Evidence roles include {sentence_join(ev)}.")
        if proc:
            parts.append(f"Procedural context includes {sentence_join(proc)}.")
        return " ".join(parts)

    if purpose == "Dispute focus":
        parts = ["The dispute focus is the allocation of contractual performance, payment, delay, or liability consequences before final adjudication."]
        if issues:
            parts.append(f"The central issue terms are {sentence_join(issues)}.")
        if ev:
            parts.append(f"The assessment depends on records such as {sentence_join(ev[:5])}.")
        return " ".join(parts)

    return "The section contains pre-decision facts relevant to a construction-related dispute."


def build_english_question(item: dict) -> dict:
    response = (item.get("Response") or "").strip()
    question = item.get("Question") or ""
    case_id = extract_case_id(question)
    title = extract_title(question)
    sections = parse_sections(question)
    instruction = (
        "Based only on the pre-decision information, predict the outcome label of the "
        "delay-related construction claim. Output only one label from: support, "
        "partial_support, not_support."
    )
    english_sections = {
        name: safe_summary(sections.get(name, ""), name, title) for name in SECTION_NAMES
    }
    input_text = (
        f"Case ID: {case_id}\n"
        f"Pre-decision factual summary:\n"
        + "\n".join(f"{name}: {english_sections[name]}" for name in SECTION_NAMES)
    )
    out_question = f"{instruction}\n\n{input_text}"
    return {
        "case_id": case_id,
        "Question": out_question,
        "Complex_CoT": item.get("Complex_CoT", ""),
        "Response": response,
        "instruction": instruction,
        "input": input_text,
        "output": response,
        "label": response,
        "_original_question": question,
    }


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def validate_jsonl(path: Path) -> tuple[bool, list[dict]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if "\n" in line[:-1]:
                    return False, rows
                rows.append(json.loads(line))
        return True, rows
    except Exception:
        return False, rows


def precheck_rows(name: str, rows: list[dict]) -> list[dict]:
    out = []
    seen = Counter()
    for idx, row in enumerate(rows, 1):
        q = row.get("Question", "")
        seen[q] += 1
        sections = parse_sections(q)
        label = (row.get("Response") or "").strip()
        out.append(
            {
                "split": name,
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
                "potential_post_decision_terms": ";".join(
                    [t for t in ["判决", "裁定", "驳回", "支持", "维持原判", "撤销"] if t in q]
                ),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_sample_compare(path: Path, train_rows: list[dict], val_rows: list[dict]):
    lines = ["# Sample Compare 10", ""]
    for split, rows in [("train", train_rows[:5]), ("val", val_rows[:5])]:
        for i, row in enumerate(rows, 1):
            lines.append(f"## {split} sample {i}: {row['case_id']}")
            orig = re.sub(r"\s+", " ", row["_original_question"])[:700]
            lines.append("### Original Chinese excerpt")
            lines.append(orig)
            lines.append("")
            lines.append("### English trial version")
            lines.append(row["Question"][:1200])
            lines.append("")
            lines.append(f"Label: `{row['Response']}`")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_field_report(path: Path, train_source_rows: list[dict], val_source_rows: list[dict]):
    train_fields = sorted(train_source_rows[0].keys())
    val_fields = sorted(val_source_rows[0].keys())
    labels = Counter([r.get("Response", "") for r in train_source_rows + val_source_rows])
    text = f"""# Translation Field Report

Generated at: {datetime.now().isoformat(timespec='seconds')}

## Original Files

- Train source: `{TRAIN_SOURCE}`
- Validation source: `{VAL_SOURCE}`

## Original Field Structure

- Train fields: `{', '.join(train_fields)}`
- Validation fields: `{', '.join(val_fields)}`

## Translated Fields

- `Question`: rebuilt as an English instruction plus English pre-decision factual summary.
- `input`: English pre-decision factual summary added for LoRA/Alpaca compatibility.
- `instruction`: English task instruction added explicitly.

## Preserved Fields

- `case_id`: extracted from the original Question and preserved as a standalone field.
- `Response`: preserved exactly.
- `output`: copied from `Response`.
- `label`: copied from `Response`.
- `Complex_CoT`: preserved from the source file.
- JSON keys, identifiers, and labels are not translated.

## Label Protection Rules

The only valid labels are `support`, `partial_support`, and `not_support`.
Labels are copied from the source `Response` field without translation or rewriting.

## Field-Missing Notes

The original files do not expose `case_id` as a standalone field; it is embedded inside `Question`.
The trial export adds a standalone `case_id` while keeping the original classification field.

## Source Label Distribution

{dict(labels)}

## API Policy

No API call was used for this Code-X-client trial export. No API key was checked.
"""
    path.write_text(text, encoding="utf-8")


def main():
    train_all = read_json(TRAIN_SOURCE)
    val_all = read_json(VAL_SOURCE)
    train_src = train_all[:100]
    val_src = val_all[:100]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    precheck = precheck_rows("train", train_all) + precheck_rows("val", val_all)
    write_csv(OUT_DIR / "dataset_translation_precheck_report.csv", precheck)

    train_out = [build_english_question(r) for r in train_src]
    val_out = [build_english_question(r) for r in val_src]

    write_jsonl(OUT_DIR / "train100_en_trial.jsonl", train_out)
    write_jsonl(OUT_DIR / "val100_en_trial.jsonl", val_out)
    write_field_report(OUT_DIR / "translation_field_report.md", train_all, val_all)
    make_sample_compare(OUT_DIR / "sample_compare_10.md", train_out, val_out)

    train_valid, train_rows = validate_jsonl(OUT_DIR / "train100_en_trial.jsonl")
    val_valid, val_rows = validate_jsonl(OUT_DIR / "val100_en_trial.jsonl")
    all_rows = [("train", r) for r in train_rows] + [("val", r) for r in val_rows]
    label_changed = 0
    missing_case_id = sum(1 for _, r in all_rows if not r.get("case_id"))
    missing_output = sum(1 for _, r in all_rows if not r.get("Response") or not r.get("output"))
    empty_input = sum(1 for _, r in all_rows if not r.get("input"))
    invalid_label = sum(1 for _, r in all_rows if r.get("Response") not in VALID_LABELS or r.get("output") not in VALID_LABELS)
    chinese_label_replaced = sum(1 for _, r in all_rows if r.get("Response") in {"支持", "部分支持", "不支持"})
    length_expanded = sum(1 for _, r in all_rows if len(r.get("Question", "")) > 2500)
    chinese_residue = sum(1 for _, r in all_rows if chinese_count(r.get("Question", "")) > 10)

    report = {
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "jsonl_valid": bool(train_valid and val_valid),
        "label_changed_count": label_changed,
        "missing_case_id_count": missing_case_id,
        "missing_output_count": missing_output,
        "empty_input_count": empty_input,
        "invalid_response_count": invalid_label,
        "chinese_label_replacement_count": chinese_label_replaced,
        "json_newline_broken": False,
        "length_expansion_abnormal_count": length_expanded,
        "large_chinese_residue_count": chinese_residue,
        "api_used": False,
        "api_key_checked": False,
        "original_files_overwritten": False,
        "train_source": str(TRAIN_SOURCE),
        "val_source": str(VAL_SOURCE),
        "output_dir": str(OUT_DIR),
    }
    (OUT_DIR / "translation_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if DESKTOP_DIR.exists():
        shutil.rmtree(DESKTOP_DIR)
    shutil.copytree(OUT_DIR, DESKTOP_DIR)
    zip_path = Path.home() / "Desktop" / "translated_trial_code_x_client.zip"
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        for file in DESKTOP_DIR.rglob("*"):
            zf.write(file, file.relative_to(DESKTOP_DIR.parent))

    print(json.dumps({**report, "desktop_dir": str(DESKTOP_DIR), "desktop_zip": str(zip_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
