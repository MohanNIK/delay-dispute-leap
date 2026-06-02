# -*- coding: utf-8 -*-
"""Create English summary LoRA data from summary-compressed DelayDispute data.

The conversion is local and deterministic. It does not use frozen test labels,
post-decision text, or any external API. The goal is a compact English factual
representation for LoRA experiments while keeping case_id and labels unchanged.
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
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "lora_exports"
    / "summary_compressed_train5000_val1000_20260529_210034"
)
DEFAULT_TRAIN = DEFAULT_SOURCE_DIR / "FT_data_summary_train5000.json"
DEFAULT_VAL = DEFAULT_SOURCE_DIR / "FT_data_summary_val1000.json"
DEFAULT_OUT_DIR = DEFAULT_SOURCE_DIR

LABELS = {"support", "partial_support", "not_support"}
INSTRUCTION = (
    "Based only on the pre-decision information, predict the outcome label of "
    "the delay-related construction claim. Output only one label from: support, "
    "partial_support, not_support."
)

SECTION_HEADERS = [
    "Project context",
    "Claims and defenses",
    "Delay and responsibility facts",
    "Evidence and procedure",
    "Dispute focus",
    "Additional relevant facts",
]

POST_DECISION_RISK_TERMS = [
    "本院认为",
    "法院认为",
    "判决如下",
    "裁定如下",
    "依照《",
    "驳回诉讼请求",
    "予以支持",
    "不予支持",
    "维持原判",
    "撤销原判",
    "判决结果",
    "裁判结果",
    "审理认为",
]

TERM_MAP = {
    "工期顺延": "extension of time",
    "顺延工期": "extension of time",
    "逾期交房": "delayed delivery",
    "逾期竣工": "late completion",
    "工期延误": "schedule delay",
    "延误": "delay",
    "延期": "delay",
    "停工": "suspension of work",
    "窝工": "idle labor",
    "违约金": "liquidated damages",
    "违约责任": "contractual liability",
    "监理确认": "supervision confirmation",
    "监理": "supervision",
    "签证": "site instruction / variation confirmation",
    "工程签证": "variation confirmation",
    "索赔": "claim",
    "施工日志": "site log",
    "会议纪要": "meeting minutes",
    "进度计划": "schedule plan",
    "关键线路": "critical path",
    "鉴定意见": "expert appraisal opinion",
    "鉴定": "expert appraisal",
    "验收": "acceptance inspection",
    "竣工": "completion",
    "开工": "commencement",
    "发包人": "owner",
    "承包人": "contractor",
    "分包": "subcontracting",
    "工程款": "project payment",
    "质量问题": "quality defects",
    "付款": "payment",
    "利息": "interest",
    "证据": "evidence",
    "举证": "burden of proof",
    "争议焦点": "disputed issue",
    "诉讼请求": "claim request",
    "反诉": "counterclaim",
    "上诉": "appeal",
    "再审": "retrial",
    "一审": "first-instance",
    "二审": "second-instance",
    "合同": "contract",
    "施工合同": "construction contract",
    "劳务合同": "labor service contract",
    "建设工程": "construction project",
    "装饰装修": "decoration works",
    "施工": "construction",
}

FEATURES = [
    ("extension_of_time", ["工期顺延", "顺延工期", "延期", "延长期限"]),
    ("schedule_delay", ["工期延误", "延误", "逾期竣工", "逾期交房", "延期交付"]),
    ("liquidated_damages", ["违约金", "工期违约", "逾期违约", "罚款"]),
    ("idle_work_loss", ["停窝工", "窝工", "停工损失", "人工费", "机械费"]),
    ("payment_dispute", ["工程款", "欠付", "付款", "结算", "价款", "利息"]),
    ("quality_defect", ["质量问题", "不合格", "维修", "返修", "保修"]),
    ("owner_cause", ["发包人原因", "甲方原因", "业主原因", "未提供图纸", "未提供场地", "未付款"]),
    ("contractor_cause", ["承包人原因", "乙方原因", "施工方原因", "未按期", "擅自停工", "施工质量"]),
    ("concurrent_or_shared", ["双方", "共同", "各自", "部分责任", "均有"]),
    ("site_instruction", ["签证", "工程签证", "变更", "增加工程", "洽商"]),
    ("notice_or_procedure", ["通知", "报审", "申请", "函", "催告"]),
    ("supervision_confirmation", ["监理", "监理确认", "监理通知", "监理单位"]),
    ("site_log_or_minutes", ["施工日志", "会议纪要", "记录", "聊天记录"]),
    ("expert_appraisal", ["鉴定", "造价鉴定", "质量鉴定", "鉴定意见"]),
    ("acceptance_record", ["验收", "竣工验收", "交付使用", "备案"]),
    ("burden_of_proof", ["证据不足", "未举证", "举证不能", "无法证明"]),
]

CASE_TYPE_MAP = [
    ("建设工程施工合同纠纷", "construction contract dispute"),
    ("建设工程分包合同纠纷", "construction subcontract dispute"),
    ("建设工程合同纠纷", "construction project contract dispute"),
    ("劳务合同纠纷", "labor service contract dispute"),
    ("装饰装修合同纠纷", "decoration works contract dispute"),
    ("买卖合同纠纷", "sales contract dispute"),
    ("民事再审", "civil retrial"),
    ("民事二审", "civil second-instance proceeding"),
    ("民事一审", "civil first-instance proceeding"),
]


def chinese_char_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text or ""))


def normalize_label(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw == "partial":
        return "partial_support"
    if raw in LABELS:
        return raw
    return "invalid"


def high_risk_post_decision_hits(text: str) -> List[str]:
    return [term for term in POST_DECISION_RISK_TERMS if term in (text or "")]


def extract_case_id(question: str) -> str:
    match = re.search(r"Case ID:\s*([^\s]+)", question or "")
    return match.group(1).strip() if match else ""


def extract_title(question: str) -> str:
    match = re.search(r"^Title:\s*(.+)$", question or "", flags=re.M)
    return match.group(1).strip() if match else ""


def split_instruction(question: str) -> Tuple[str, str]:
    marker = "Case ID:"
    pos = question.find(marker)
    if pos == -1:
        return INSTRUCTION, question
    instruction = question[:pos].strip() or INSTRUCTION
    body = question[pos:].strip()
    return instruction, body


def parse_sections(body: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    current = ""
    buffer: List[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        header_match = re.match(r"^([A-Za-z and]+):\s*(.*)$", stripped)
        if header_match and header_match.group(1) in SECTION_HEADERS:
            if current:
                sections[current] = "\n".join(buffer).strip()
            current = header_match.group(1)
            buffer = [header_match.group(2).strip()] if header_match.group(2).strip() else []
        elif current:
            buffer.append(stripped)
    if current:
        sections[current] = "\n".join(buffer).strip()
    return sections


def source_record_precheck(row: Dict[str, Any], split: str, row_id: int) -> Dict[str, Any]:
    question = str(row.get("Question", ""))
    label = normalize_label(row.get("Response"))
    instruction, body = split_instruction(question)
    case_id = extract_case_id(body)
    sections = parse_sections(body)
    section_missing = [header for header in SECTION_HEADERS[:5] if not sections.get(header, "").strip()]
    summary_text = "\n".join(sections.get(header, "") for header in SECTION_HEADERS)
    hits = high_risk_post_decision_hits(summary_text)
    return {
        "split": split,
        "row_id": row_id,
        "case_id": case_id,
        "case_id_missing": not bool(case_id),
        "question_present": bool(question.strip()),
        "response": label,
        "response_valid": label in LABELS,
        "project_context_missing": "Project context" in section_missing,
        "claims_and_defenses_missing": "Claims and defenses" in section_missing,
        "delay_and_responsibility_facts_missing": "Delay and responsibility facts" in section_missing,
        "evidence_and_procedure_missing": "Evidence and procedure" in section_missing,
        "dispute_focus_missing": "Dispute focus" in section_missing,
        "missing_field_count": len(section_missing),
        "missing_fields": "|".join(section_missing),
        "post_decision_risk": bool(hits),
        "post_decision_terms": "|".join(hits),
        "summary_chars": len(summary_text),
        "summary_chinese_chars": chinese_char_count(summary_text),
        "question_chars": len(question),
        "field_parse_ok": bool(question.strip()) and bool(case_id) and label in LABELS,
    }


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？；;])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def detect_features(text: str) -> List[str]:
    found = []
    for name, terms in FEATURES:
        if any(term in text for term in terms):
            found.append(name)
    return found


def extract_amounts_dates(text: str, max_items: int = 8) -> Tuple[List[str], List[str]]:
    amounts = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?\s*(?:元|万元|亿元|%|天|日|个月|年)", text or "")
    dates = re.findall(r"\d{4}年\d{1,2}月(?:\d{1,2}日)?", text or "")
    return list(dict.fromkeys(amounts))[:max_items], list(dict.fromkeys(dates))[:max_items]


def translate_title(title: str) -> str:
    if not title:
        return ""
    parts = [en for cn, en in CASE_TYPE_MAP if cn in title]
    if not parts:
        if "工程" in title:
            parts.append("construction-related dispute")
        elif "合同" in title:
            parts.append("contract dispute")
        else:
            parts.append("civil dispute")
    if "再审" in title:
        parts.append("retrial")
    elif "二审" in title:
        parts.append("second-instance")
    elif "一审" in title:
        parts.append("first-instance")
    return "; ".join(dict.fromkeys(parts))


def glossary_phrase(text: str, max_terms: int = 10) -> List[str]:
    terms = []
    for cn, en in TERM_MAP.items():
        if cn in text and en not in terms:
            terms.append(en)
    return terms[:max_terms]


def section_sentence(header: str, chinese_text: str, title_en: str = "") -> str:
    features = detect_features(chinese_text)
    terms = glossary_phrase(chinese_text)
    amounts, dates = extract_amounts_dates(chinese_text)
    has_claimant = any(t in chinese_text for t in ["原告", "上诉人", "申请人", "反诉原告"])
    has_defendant = any(t in chinese_text for t in ["被告", "被上诉人", "被申请人", "反诉被告"])

    if header == "Project context":
        clauses = []
        if title_en:
            clauses.append(f"Case type: {title_en}")
        if has_claimant or has_defendant:
            clauses.append("The pre-decision record identifies the parties and procedural posture")
        if dates:
            clauses.append("Key dates: " + ", ".join(dates[:4]))
        if not clauses:
            clauses.append("The record provides basic project and party context")
        return ". ".join(clauses) + "."

    if header == "Claims and defenses":
        clauses = []
        if has_claimant:
            clauses.append("The claimant/appellant advanced construction-related claims")
        if has_defendant:
            clauses.append("The opposing party raised defenses or counterclaims")
        if "工程款" in chinese_text or "付款" in chinese_text:
            clauses.append("payment or settlement was disputed")
        if "违约金" in chinese_text:
            clauses.append("liquidated damages or contractual penalty was claimed")
        if amounts:
            clauses.append("Amounts mentioned: " + ", ".join(amounts[:5]))
        if not clauses:
            clauses.append("The section summarizes the parties' claims and defenses")
        return "; ".join(clauses) + "."

    if header == "Delay and responsibility facts":
        clauses = []
        if "工期" in chinese_text or "延误" in chinese_text or "逾期" in chinese_text:
            clauses.append("The facts concern schedule delay, late completion, or extension of time")
        if "发包" in chinese_text or "甲方" in chinese_text or "业主" in chinese_text:
            clauses.append("possible owner-side causes are mentioned")
        if "承包" in chinese_text or "乙方" in chinese_text or "施工方" in chinese_text:
            clauses.append("possible contractor-side causes are mentioned")
        if "质量" in chinese_text:
            clauses.append("quality defects are linked to the dispute")
        if dates:
            clauses.append("Schedule dates: " + ", ".join(dates[:4]))
        if not clauses:
            clauses.append("The section records responsibility-related project facts")
        return "; ".join(clauses) + "."

    if header == "Evidence and procedure":
        clauses = []
        if terms:
            clauses.append("Evidence/procedure terms: " + ", ".join(terms[:8]))
        if "证据不足" in chinese_text or "未举证" in chinese_text or "举证不能" in chinese_text:
            clauses.append("the record refers to insufficient proof or burden of proof")
        if "鉴定" in chinese_text:
            clauses.append("expert appraisal evidence is involved")
        if "验收" in chinese_text:
            clauses.append("acceptance or completion records are involved")
        if not clauses:
            clauses.append("The section lists documentary or procedural materials")
        return "; ".join(clauses) + "."

    if header == "Dispute focus":
        clauses = []
        if "是否" in chinese_text:
            clauses.append("The dispute is framed as whether the asserted delay/payment responsibility is established")
        if "责任" in chinese_text:
            clauses.append("responsibility allocation is at issue")
        if "工程款" in chinese_text or "价款" in chinese_text:
            clauses.append("project payment calculation is at issue")
        if "工期" in chinese_text:
            clauses.append("schedule-delay consequences are at issue")
        if not clauses:
            clauses.append("The section states the main disputed issue")
        return "; ".join(clauses) + "."

    if features or terms:
        return "Additional relevant facts involve " + ", ".join(dict.fromkeys(features + terms)[:10]) + "."
    return "Additional relevant facts are recorded in the pre-decision materials."


def translate_sections(question: str) -> Tuple[str, Dict[str, Any]]:
    instruction, body = split_instruction(question)
    case_id = extract_case_id(body)
    title = extract_title(body)
    title_en = translate_title(title)
    sections = parse_sections(body)
    all_text = "\n".join(sections.values())

    lines = [instruction.strip(), "", f"Case ID: {case_id}", "Pre-decision factual summary:"]
    if title_en:
        lines.append(f"Project context: {section_sentence('Project context', sections.get('Project context', ''), title_en)}")
    else:
        lines.append(f"Project context: {section_sentence('Project context', sections.get('Project context', ''), title_en)}")

    for header in [
        "Claims and defenses",
        "Delay and responsibility facts",
        "Evidence and procedure",
        "Dispute focus",
    ]:
        lines.append(f"{header}: {section_sentence(header, sections.get(header, ''), title_en)}")

    # Preserve observed terms and numerical anchors without keeping Chinese text.
    features = detect_features(all_text)
    amounts, dates = extract_amounts_dates(all_text)
    terms = glossary_phrase(all_text, max_terms=12)
    if features or terms or amounts or dates:
        anchor_parts = []
        if features:
            anchor_parts.append("Observed issue signals: " + ", ".join(features[:12]))
        if terms:
            anchor_parts.append("Normalized terms: " + ", ".join(terms[:12]))
        if amounts:
            anchor_parts.append("Numerical anchors: " + ", ".join(amounts[:8]))
        if dates:
            anchor_parts.append("Date anchors: " + ", ".join(dates[:8]))
        lines.append("Additional relevant facts: " + "; ".join(anchor_parts) + ".")

    out = "\n".join(lines)
    meta = {
        "case_id": case_id,
        "source_chinese_chars": chinese_char_count(question),
        "translated_chinese_chars": chinese_char_count(out),
        "sections_found": "|".join(sorted(sections)),
        "features": "|".join(features),
        "amount_count": len(amounts),
        "date_count": len(dates),
        "case_id_missing": not bool(case_id),
        "large_chinese_residue": chinese_char_count(out) > 20,
    }
    return out, meta


def to_alpaca(record: Dict[str, str]) -> Dict[str, str]:
    question = record["Question"]
    parts = question.split("\n\n", 1)
    if len(parts) == 2:
        instruction, model_input = parts
    else:
        instruction, model_input = INSTRUCTION, question
    return {"instruction": instruction.strip(), "input": model_input.strip(), "output": record["Response"]}


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


def convert_split(rows: List[Dict[str, Any]], split: str) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    out_rows = []
    report = []
    for idx, row in enumerate(rows, 1):
        label_before = normalize_label(row.get("Response"))
        q_en, meta = translate_sections(str(row.get("Question", "")))
        out = {"Question": q_en, "Complex_CoT": "", "Response": label_before}
        label_after = normalize_label(out["Response"])
        out_rows.append(out)
        report.append(
            {
                "split": split,
                "row_id": idx,
                **meta,
                "label_before": label_before,
                "label_after": label_after,
                "label_changed": label_before != label_after,
                "field_parse_ok": bool(meta["case_id"]) and label_after in LABELS and bool(q_en),
                "question_chars": len(q_en),
            }
        )
    return out_rows, report


def write_precheck_report(train: List[Dict[str, Any]], val: List[Dict[str, Any]], out_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for split, data in [("train", train), ("val", val)]:
        for idx, row in enumerate(data, 1):
            rows.append(source_record_precheck(row, split, idx))
    write_csv(out_dir / "dataset_translation_precheck_report.csv", rows)

    summary_rows = []
    for split in ["train", "val"]:
        sub = [r for r in rows if r["split"] == split]
        lengths = [int(r["summary_chinese_chars"]) for r in sub]
        labels = Counter(r["response"] for r in sub)
        summary_rows.append(
            {
                "split": split,
                "rows": len(sub),
                "response_invalid": sum(1 for r in sub if not r["response_valid"]),
                "case_id_missing": sum(1 for r in sub if r["case_id_missing"]),
                "field_parse_fail": sum(1 for r in sub if not r["field_parse_ok"]),
                "missing_any_required_section": sum(1 for r in sub if int(r["missing_field_count"]) > 0),
                "post_decision_risk": sum(1 for r in sub if r["post_decision_risk"]),
                "summary_chinese_chars_min": min(lengths) if lengths else 0,
                "summary_chinese_chars_p50": statistics.median(lengths) if lengths else 0,
                "summary_chinese_chars_p90": sorted(lengths)[int(len(lengths) * 0.9)] if lengths else 0,
                "summary_chinese_chars_max": max(lengths) if lengths else 0,
                "support": labels.get("support", 0),
                "partial_support": labels.get("partial_support", 0),
                "not_support": labels.get("not_support", 0),
            }
        )
    write_csv(out_dir / "dataset_translation_precheck_summary.csv", summary_rows)
    return rows


def build_outputs(
    train_path: Path,
    val_path: Path,
    out_dir: Path,
    desktop_zip: bool = True,
    limit: int = 0,
    precheck_only: bool = False,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = load_json_array(train_path)
    val = load_json_array(val_path)
    precheck_rows = write_precheck_report(train, val, out_dir)
    if precheck_only:
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "method": "precheck_only",
            "train_source": str(train_path),
            "val_source": str(val_path),
            "train_rows": len(train),
            "val_rows": len(val),
            "precheck_report": "dataset_translation_precheck_report.csv",
            "precheck_summary": "dataset_translation_precheck_summary.csv",
        }
        (out_dir / "manifest_en_precheck.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    if limit and limit > 0:
        train = train[:limit]
        val = val[:limit]
    train_en, train_report = convert_split(train, "train")
    val_en, val_report = convert_split(val, "val")

    train_questions = {r["Question"] for r in train_en}
    val_questions = {r["Question"] for r in val_en}
    overlap = train_questions & val_questions
    for r in train_report + val_report:
        r["train_val_duplicate"] = r["split"] == "train" and False
    if overlap:
        overlap_set = overlap
        for rows, reports in [(train_en, train_report), (val_en, val_report)]:
            for rec, rep in zip(rows, reports):
                if rec["Question"] in overlap_set:
                    rep["train_val_duplicate"] = True
    else:
        for r in train_report + val_report:
            r["train_val_duplicate"] = False

    size_suffix_train = f"train{len(train_en)}"
    size_suffix_val = f"val{len(val_en)}"
    files = {
        f"FT_data_summary_{size_suffix_train}_en.json": train_en,
        f"FT_data_summary_{size_suffix_val}_en.json": val_en,
        f"FT_data_summary_{size_suffix_train}_en_alpaca.jsonl": [to_alpaca(r) for r in train_en],
        f"FT_data_summary_{size_suffix_val}_en_alpaca.jsonl": [to_alpaca(r) for r in val_en],
    }
    for name, rows in files.items():
        path = out_dir / name
        if name.endswith(".jsonl"):
            write_jsonl(path, rows)
        else:
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    quality = train_report + val_report
    quality_report_name = f"translation_quality_report_{limit}.csv" if limit and limit > 0 else "translation_quality_report.csv"
    write_csv(out_dir / quality_report_name, quality)
    summary = []
    for split, rows, report in [("train", train_en, train_report), ("val", val_en, val_report)]:
        labels = Counter(r["Response"] for r in rows)
        residue = [r["translated_chinese_chars"] for r in report]
        summary.append(
            {
                "split": split,
                "rows": len(rows),
                "case_id_missing": sum(1 for r in report if r["case_id_missing"]),
                "label_changed": sum(1 for r in report if r["label_changed"]),
                "large_chinese_residue": sum(1 for r in report if r["large_chinese_residue"]),
                "field_parse_fail": sum(1 for r in report if not r["field_parse_ok"]),
                "duplicate_with_other_split": sum(1 for r in report if r["train_val_duplicate"]),
                "support": labels.get("support", 0),
                "partial_support": labels.get("partial_support", 0),
                "not_support": labels.get("not_support", 0),
                "translated_chinese_chars_p50": statistics.median(residue) if residue else 0,
                "translated_chinese_chars_max": max(residue) if residue else 0,
            }
        )
    write_csv(out_dir / "translation_quality_summary.csv", summary)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "local_deterministic_english_fact_abstraction_no_api",
        "train_source": str(train_path),
        "val_source": str(val_path),
        "train_rows": len(train_en),
        "val_rows": len(val_en),
        "train_val_question_overlap": len(overlap),
        "outputs": sorted(files),
        "limit_per_split": limit,
        "precheck_report": "dataset_translation_precheck_report.csv",
        "precheck_rows": len(precheck_rows),
        "quality_report": quality_report_name,
        "quality_summary": "translation_quality_summary.csv",
        "labels_unchanged": all(not r["label_changed"] for r in quality),
        "case_id_missing_count": sum(1 for r in quality if r["case_id_missing"]),
        "large_chinese_residue_count": sum(1 for r in quality if r["large_chinese_residue"]),
    }
    (out_dir / "manifest_en.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = (
        "# DelayDispute English summary LoRA data\n\n"
        "This package converts the Chinese pre-decision factual summaries into compact English fact abstractions.\n"
        "It does not use frozen test500 labels, post-decision text, or external API calls.\n\n"
        "Primary files:\n"
        "- FT_data_summary_train5000_en.json\n"
        "- FT_data_summary_val1000_en.json\n"
        "- FT_data_summary_train5000_en_alpaca.jsonl\n"
        "- FT_data_summary_val1000_en_alpaca.jsonl\n\n"
        "Labels remain exactly support / partial_support / not_support.\n"
    )
    (out_dir / "README_english_summary_data.md").write_text(readme, encoding="utf-8")

    zip_name = (
        f"DelayDispute_summary_train{len(train_en)}_val{len(val_en)}_en_trial_package.zip"
        if limit and limit > 0
        else "DelayDispute_summary_train5000_val1000_en_package.zip"
    )
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in [
            *[out_dir / name for name in sorted(files)],
            out_dir / quality_report_name,
            out_dir / "dataset_translation_precheck_report.csv",
            out_dir / "dataset_translation_precheck_summary.csv",
            out_dir / "translation_quality_summary.csv",
            out_dir / "manifest_en.json",
            out_dir / "README_english_summary_data.md",
        ]:
            zf.write(p, arcname=p.name)
    manifest["zip_path"] = str(zip_path)
    if desktop_zip:
        desktop_path = Path.home() / "Desktop" / zip_path.name
        desktop_path.write_bytes(zip_path.read_bytes())
        manifest["desktop_zip"] = str(desktop_path)
        (out_dir / "manifest_en.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--val", default=str(DEFAULT_VAL))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--limit", type=int, default=0, help="Limit rows per split for trial export; 0 means full export.")
    parser.add_argument("--precheck-only", action="store_true", help="Only write dataset_translation_precheck_report.csv and summary.")
    parser.add_argument("--no-desktop-zip", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_outputs(
        Path(args.train),
        Path(args.val),
        Path(args.out_dir),
        desktop_zip=not args.no_desktop_zip,
        limit=args.limit,
        precheck_only=args.precheck_only,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
