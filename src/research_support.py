# -*- coding: utf-8 -*-
"""Shared research utilities for the DelayDispute Copilot study.

This module keeps the paper-facing logic explicit:
- leakage-aware pre/post split
- candidate-gold derivation helpers
- auditable evidence-chain extraction
- structured responsibility diagnosis helpers
- evaluation metrics and export utilities
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, f1_score

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


LABELS = ["support", "partial", "not_support"]
RESP_LABELS = [
    "owner",
    "contractor",
    "subcontractor",
    "designer_supervisor",
    "both",
    "force_majeure_policy",
    "unknown",
]
RESP_ZH_TO_EN = {
    "业主": "owner",
    "发包人": "owner",
    "建设单位": "owner",
    "甲方": "owner",
    "承包商": "contractor",
    "承包人": "contractor",
    "施工单位": "contractor",
    "乙方": "contractor",
    "分包商": "subcontractor",
    "分包": "subcontractor",
    "设计/监理": "designer_supervisor",
    "设计": "designer_supervisor",
    "监理": "designer_supervisor",
    "双方": "both",
    "不可抗力/政策": "force_majeure_policy",
    "不可抗力": "force_majeure_policy",
    "政策": "force_majeure_policy",
    "unknown": "unknown",
    "": "unknown",
}
RESP_EN_TO_ZH = {
    "owner": "业主",
    "contractor": "承包商",
    "subcontractor": "分包商",
    "designer_supervisor": "设计/监理",
    "both": "双方",
    "force_majeure_policy": "不可抗力/政策",
    "unknown": "unknown",
}

CONSTRUCTION_KEYWORDS = [
    "建设工程", "工程", "施工", "承包", "发包", "监理", "竣工", "工期", "工程款", "签证", "索赔", "进度款",
]
DELAY_KEYWORDS = [
    "延误", "延期", "顺延", "逾期", "停工", "拖延", "误期", "工期", "竣工日期", "交付时间", "抢工",
]
OWNER_CAUSE_KEYWORDS = [
    "未按时支付工程款", "未及时支付", "设计变更", "审批滞后", "未提供场地", "未交付图纸", "指令停工", "资金不到位", "甲方原因",
]
CONTRACTOR_CAUSE_KEYWORDS = [
    "管理不善", "组织不力", "施工人员", "未按时施工", "拖延施工", "举证不足", "未履行通知义务", "材料准备不足", "承包人原因",
]
PROCEDURE_KEYWORDS = [
    "通知", "签证", "索赔", "备案", "验收", "报审", "报告", "申请", "顺延申请", "签认", "监理日志", "会议纪要",
]
EVIDENCE_KEYWORDS = [
    "证据", "纪要", "签证", "日志", "函件", "往来函", "邮件", "图纸", "监理", "报告", "清单", "支付凭证", "照片",
]
ROLE_KEYWORDS = {
    "entitlement": ["合同约定", "合同条款", "约定", "应予顺延", "享有", "工期", "索赔依据"],
    "notice_substantiation": ["通知", "签证", "报审", "申请", "备案", "报告", "监理", "纪要"],
    "causality": ["导致", "原因", "因", "造成", "影响", "阻碍", "拖延"],
    "impact_schedule_relevance": ["关键线路", "工期", "竣工", "进度", "停工", "顺延", "延期", "交付"],
    "documentation_integrity": ["证据", "资料", "日志", "函件", "纪要", "签证", "证明", "记录"],
}
OUTCOME_PATTERNS = {
    "partial": [r"部分支持", r"其余.*驳回", r"酌情支持", r"部分予以支持"],
    "support": [r"予以支持", r"应予支持", r"判令", r"确认.*请求", r"准许"],
    "not_support": [r"不予支持", r"驳回", r"不予采信", r"不予认可"],
}
LEAKAGE_PATTERNS = [r"判决如下", r"裁定如下", r"本院认为", r"综上所述", r"判令", r"驳回"]
YEAR_ASCII = re.compile(r"(20\d{2})")
YEAR_ZH = re.compile(r"([〇零一二三四五六七八九○ＯoO]{4})年")
SENT_SPLIT = re.compile(r"(?<=[。！？；\n])")


@dataclass
class EvidenceUnit:
    role_label: str
    text: str
    span_start: int
    span_end: int
    pre_decision_flag: int
    duplicate_flag: int


DEFAULT_CFG = {
    "project": {"name": "delay_dispute_madra", "root": "."},
    "paths": {
        "parsed_json_dir": "data/2_parsed_json",
        "structured_case_dir": "data/3_structured_cases",
        "meta_labels_csv": "data/meta/labels_step2.csv",
        "llm_labels_csv": "results/labels_step2_delay_outcome_llm.csv",
        "seed_reference_csv": "data/gold/gold65_v1.csv",
        "candidate_gold_strict_csv": "data/gold/candidate_gold_strict_v1.csv",
        "candidate_gold_extended_csv": "data/gold/candidate_gold_extended_v1.csv",
        "gold_guideline_md": "data/gold/annotation_guideline_v1.md",
        "gold_qc_txt": "data/gold/qc_report_v1.txt",
        "gold_sampling_manifest_csv": "data/gold/sampling_manifest_v1.csv",
        "gold_provenance_manifest_csv": "data/gold/provenance_manifest_v1.csv",
        "review_dir": "data/review",
        "paper_assets_dir": "paper_assets",
        "final_eval_root": "results",
    },
    "random": {"seed": 2026},
    "candidate_gold": {
        "strict_target_size": 250,
        "extended_target_size": 500,
        "strict_min_per_label": 50,
        "extended_min_per_label": 100,
        "strict_confidence_threshold": 0.78,
        "audit_subset_size": 60,
    },
    "eval": {
        "max_features": 50000,
        "ngram_min": 1,
        "ngram_max": 2,
        "bootstrap_rounds": 300,
        "retrieval_top_k": 5,
        "random_test_size": 0.25,
        "time_split_holdout_ratio": 0.25,
        "baselines": [
            "majority_class",
            "rule_baseline",
            "tfidf_logreg",
            "tfidf_linearsvc",
            "tfidf_multinomialnb",
            "current_hybrid_baseline",
            "paesc_hybrid",
        ],
    },
    "figures": {"dpi": 600, "formats": ["png", "pdf", "svg"]},
}


def load_cfg(path: Path) -> Dict:
    if yaml is None or not path.exists():
        return DEFAULT_CFG
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    merged = json.loads(json.dumps(DEFAULT_CFG, ensure_ascii=False))
    deep_update(merged, cfg or {})
    return merged


def deep_update(base: Dict, updates: Dict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value


def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_label(x: object) -> str:
    x = str(x or "").strip().lower()
    mapping = {
        "support": "support",
        "partial": "partial",
        "partial_support": "partial",
        "partially_support": "partial",
        "not_support": "not_support",
        "not-support": "not_support",
        "notsupport": "not_support",
        "支持": "support",
        "部分支持": "partial",
        "不支持": "not_support",
        "驳回": "not_support",
        "unknown": "unknown",
        "": "unknown",
        "nan": "unknown",
    }
    return mapping.get(x, "unknown")


def normalize_resp(x: object) -> str:
    x = str(x or "").strip()
    if x in RESP_LABELS:
        return x
    return RESP_ZH_TO_EN.get(x, "unknown")


def split_sentences(text: str) -> List[Tuple[str, int, int]]:
    text = str(text or "")
    if not text:
        return []
    spans: List[Tuple[str, int, int]] = []
    start = 0
    for m in SENT_SPLIT.finditer(text):
        end = m.end()
        sent = text[start:end].strip()
        if sent:
            sent_start = text.find(sent, start)
            spans.append((sent, sent_start, sent_start + len(sent)))
        start = end
    tail = text[start:].strip()
    if tail:
        tail_start = text.find(tail, start)
        spans.append((tail, tail_start, tail_start + len(tail)))
    return spans


def chinese_year_to_int(token: str) -> Optional[int]:
    mapping = {"零": "0", "〇": "0", "○": "0", "Ｏ": "0", "o": "0", "O": "0", "一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
    digits = "".join(mapping.get(ch, "") for ch in token)
    if len(digits) == 4 and digits.isdigit():
        return int(digits)
    return None


def extract_year(text: str) -> Optional[int]:
    text = str(text or "")
    m = YEAR_ASCII.search(text)
    if m:
        year = int(m.group(1))
        if 1990 <= year <= 2035:
            return year
    m = YEAR_ZH.search(text)
    if m:
        year = chinese_year_to_int(m.group(1))
        if year and 1990 <= year <= 2035:
            return year
    return None


def detect_domain_case(source_file: str, raw_text: str) -> bool:
    text = f"{source_file}\n{raw_text}"
    has_construction = any(k in text for k in CONSTRUCTION_KEYWORDS)
    has_delay = any(k in text for k in DELAY_KEYWORDS)
    return bool(has_construction and has_delay)


def _collect_keyword_sentences(text: str, keywords: Sequence[str], limit: int = 8) -> List[Dict[str, object]]:
    hits: List[Dict[str, object]] = []
    for sent, start, end in split_sentences(text):
        if any(k in sent for k in keywords):
            hits.append({"text": sent, "span_start": start, "span_end": end})
            if len(hits) >= limit:
                break
    return hits


def extract_claims_defenses(pre_text: str) -> Dict[str, List[Dict[str, object]]]:
    claim_keywords = ["原告", "申请人", "上诉人", "索赔", "请求", "主张"]
    defense_keywords = ["被告", "被申请人", "被上诉人", "辩称", "抗辩", "答辩"]
    claims = []
    defenses = []
    for sent, start, end in split_sentences(pre_text):
        if any(k in sent for k in claim_keywords):
            claims.append({"text": sent, "span_start": start, "span_end": end})
        if any(k in sent for k in defense_keywords):
            defenses.append({"text": sent, "span_start": start, "span_end": end})
    return {"claims": claims[:10], "defenses": defenses[:10]}


def extract_project_context(pre_text: str, source_file: str) -> Dict[str, object]:
    sents = split_sentences(pre_text)
    summary = " ".join(sent for sent, _, _ in sents[:3])[:500]
    return {
        "source_file": source_file,
        "summary": summary,
        "has_construction_terms": any(k in pre_text for k in CONSTRUCTION_KEYWORDS),
        "has_delay_terms": any(k in pre_text for k in DELAY_KEYWORDS),
    }


def extract_delay_events(pre_text: str) -> List[Dict[str, object]]:
    return _collect_keyword_sentences(pre_text, DELAY_KEYWORDS, limit=12)


def extract_procedural_compliance_cues(pre_text: str) -> List[Dict[str, object]]:
    return _collect_keyword_sentences(pre_text, PROCEDURE_KEYWORDS, limit=12)


def extract_evidence_mentions(pre_text: str) -> List[Dict[str, object]]:
    return _collect_keyword_sentences(pre_text, EVIDENCE_KEYWORDS, limit=15)


def build_evidence_chain(pre_text: str) -> List[Dict[str, object]]:
    used_texts = set()
    chain: List[Dict[str, object]] = []
    for role, keywords in ROLE_KEYWORDS.items():
        picked = None
        for sent, start, end in split_sentences(pre_text):
            if any(k in sent for k in keywords):
                duplicate = int(sent in used_texts)
                picked = {
                    "role_label": role,
                    "text": sent,
                    "span_start": start,
                    "span_end": end,
                    "pre_decision_flag": 1,
                    "duplicate_flag": duplicate,
                }
                used_texts.add(sent)
                break
        if picked is None:
            picked = {
                "role_label": role,
                "text": "",
                "span_start": -1,
                "span_end": -1,
                "pre_decision_flag": 0,
                "duplicate_flag": 0,
            }
        chain.append(picked)
    return chain


def evidence_chain_metrics(chain: List[Dict[str, object]]) -> Dict[str, float]:
    if not chain:
        return {
            "valid_span_rate": 0.0,
            "pre_decision_span_rate": 0.0,
            "duplicate_chain_rate": 0.0,
            "role_coverage_rate": 0.0,
            "missing_role_rate": 1.0,
        }
    valid = [1 for x in chain if int(x.get("span_start", -1)) >= 0 and str(x.get("text", "")).strip()]
    pre = [1 for x in chain if int(x.get("pre_decision_flag", 0)) == 1 and str(x.get("text", "")).strip()]
    dup = [1 for x in chain if int(x.get("duplicate_flag", 0)) == 1]
    covered = [1 for x in chain if str(x.get("text", "")).strip()]
    n = len(chain)
    return {
        "valid_span_rate": len(valid) / n,
        "pre_decision_span_rate": len(pre) / n,
        "duplicate_chain_rate": len(dup) / n,
        "role_coverage_rate": len(covered) / n,
        "missing_role_rate": 1.0 - len(covered) / n,
    }


def build_structured_case(case_obj: Dict) -> Dict:
    sections = case_obj.get("sections", {}) or {}
    facts = str(sections.get("facts") or "").strip()
    issues = str(sections.get("issues") or "").strip()
    reasoning = str(sections.get("reasoning") or "").strip()
    decision = str(sections.get("decision") or "").strip()
    raw_text = str(sections.get("full_text") or "").strip()
    pre_text = "\n".join([x for x in [facts, issues] if x]).strip()
    post_text = "\n".join([x for x in [reasoning, decision] if x]).strip()
    if not pre_text and raw_text:
        pre_text = raw_text[: max(0, len(raw_text) // 2)]
    if not post_text and raw_text:
        post_text = raw_text[max(0, len(raw_text) // 2):]
    claims_defenses = extract_claims_defenses(pre_text)
    delay_events = extract_delay_events(pre_text)
    procedure_cues = extract_procedural_compliance_cues(pre_text)
    evidence_mentions = extract_evidence_mentions(pre_text)
    chain = build_evidence_chain(pre_text)
    leakage_flag = int(any(re.search(p, pre_text) for p in LEAKAGE_PATTERNS))
    anchor_found = int(bool(reasoning or decision))
    split_confidence = round(0.9 if facts or issues else 0.55 if raw_text else 0.0, 2)
    structured = {
        "case_id": case_obj.get("case_id"),
        "source_file": case_obj.get("source_file", ""),
        "raw_text": raw_text,
        "structured_segments": sections,
        "pre_decision_text": pre_text,
        "post_decision_text": post_text,
        "project_context": extract_project_context(pre_text, case_obj.get("source_file", "")),
        "delay_events": delay_events,
        "claims_defenses": claims_defenses,
        "procedural_compliance_cues": procedure_cues,
        "evidence_mentions": evidence_mentions,
        "source_span_pointers": chain,
        "pre_post_split_confidence": split_confidence,
        "anchor_found_flag": anchor_found,
        "potential_leakage_flag": leakage_flag,
        "case_year": extract_year(post_text or raw_text),
    }
    return structured


def derive_outcome_from_post(post_text: str) -> Tuple[str, str]:
    text = str(post_text or "")
    for label in ["partial", "not_support", "support"]:
        for pattern in OUTCOME_PATTERNS[label]:
            m = re.search(pattern, text)
            if m:
                snippet = text[max(0, m.start() - 80): m.end() + 120].replace("\n", " ").strip()
                return label, snippet
    return "unknown", ""


def derive_responsibility_candidate(post_text: str, llm_hint: str = "") -> Tuple[str, float, str]:
    text = str(post_text or "")
    scores = Counter()
    evidence_bits: List[str] = []

    causal_cues = ["原因", "导致", "造成", "责任", "过错", "未", "拖延", "延误", "停工", "不予支持"]

    def sent_score(label: str, role_keywords: Sequence[str], trigger_keywords: Sequence[str], weight: float) -> None:
        for sent, _, _ in split_sentences(text):
            if any(rk in sent for rk in role_keywords) and any(tk in sent for tk in trigger_keywords):
                scores[label] += weight
                evidence_bits.append(sent[:40])

    for explicit in ["双方均有责任", "双方对此均存在", "双方均存在", "各方均有责任", "双方均有过错"]:
        if explicit in text:
            scores["both"] += 3.0
            evidence_bits.append(explicit)

    for explicit in ["不可抗力", "政府行为", "疫情", "环保管控", "大气管控", "政策调整"]:
        if explicit in text:
            scores["force_majeure_policy"] += 2.4
            evidence_bits.append(explicit)

    sent_score("owner", ["发包人", "业主", "建设单位", "甲方", "被告"], OWNER_CAUSE_KEYWORDS + causal_cues, 1.6)
    sent_score("contractor", ["承包人", "承包商", "施工单位", "乙方", "原告"], CONTRACTOR_CAUSE_KEYWORDS + causal_cues, 1.6)
    sent_score("subcontractor", ["分包", "劳务分包"], ["原因", "责任", "过错", "管理", "协调", "拖延"], 1.3)
    sent_score("designer_supervisor", ["监理", "设计单位", "设计变更", "监理工程师"], ["原因", "责任", "过错", "迟延", "变更", "审批"], 1.2)

    hint = normalize_resp(llm_hint)
    if hint != "unknown":
        scores[hint] += 1.2
        evidence_bits.append(f"llm_hint:{hint}")

    if not scores:
        return hint if hint != "unknown" else "unknown", 0.35 if hint != "unknown" else 0.2, "; ".join(evidence_bits[:4])

    top_two = scores.most_common(2)
    label = top_two[0][0]
    margin = top_two[0][1] - (top_two[1][1] if len(top_two) > 1 else 0.0)
    confidence = min(0.95, 0.55 + 0.1 * top_two[0][1] + 0.05 * margin)
    return label, float(round(confidence, 4)), "; ".join(evidence_bits[:6])


def diagnose_responsibility_from_pre(pre_text: str, llm_hint: str, chain: Optional[List[Dict[str, object]]] = None) -> Dict[str, object]:
    text = str(pre_text or "")
    scores = Counter()
    reasons: List[str] = []

    causal_cues = ["原因", "导致", "造成", "影响", "未", "延误", "停工", "拖延", "责任", "过错"]

    def sent_scan(label: str, role_keywords: Sequence[str], trigger_keywords: Sequence[str], weight: float) -> None:
        for sent, _, _ in split_sentences(text):
            if any(rk in sent for rk in role_keywords) and any(tk in sent for tk in trigger_keywords):
                scores[label] += weight
                reasons.append(sent[:40])

    sent_scan("owner", ["业主", "发包人", "甲方", "建设单位"], OWNER_CAUSE_KEYWORDS + causal_cues, 0.9)
    sent_scan("contractor", ["承包人", "承包商", "施工单位", "乙方"], CONTRACTOR_CAUSE_KEYWORDS + causal_cues, 0.9)
    sent_scan("subcontractor", ["分包", "劳务分包"], ["原因", "责任", "过错", "协调", "管理"], 0.8)
    sent_scan("designer_supervisor", ["监理", "设计单位", "设计变更", "监理工程师"], ["原因", "责任", "迟延", "审批"], 0.75)
    sent_scan("force_majeure_policy", ["不可抗力", "政策", "政府行为", "环保", "疫情"], ["停工", "延误", "影响", "管控"], 1.0)
    for explicit in ["双方", "各方", "共同责任", "均有责任"]:
        if explicit in text:
            scores["both"] += 1.1
            reasons.append(explicit)

    hint = normalize_resp(llm_hint)
    if hint != "unknown":
        scores[hint] += 0.9
        reasons.append(f"llm_hint:{hint}")

    primary = scores.most_common(1)[0][0] if scores else (hint if hint != "unknown" else "unknown")
    secondary = "unknown"
    if len(scores) > 1:
        top_two = scores.most_common(2)
        if top_two[1][1] >= max(0.8, top_two[0][1] * 0.7):
            secondary = top_two[1][0]
    procedure_hits = sum(1 for kw in PROCEDURE_KEYWORDS if kw in text)
    evidence_hits = sum(1 for kw in EVIDENCE_KEYWORDS if kw in text)
    doc_flag = "incomplete" if any(x in text for x in ["缺乏证据", "未提交", "未提供", "资料缺失", "举证不足"]) else "complete" if evidence_hits >= 2 else "uncertain"
    proc_status = "noncompliant" if any(x in text for x in ["未通知", "未报审", "未备案", "未签证", "未验收", "未履行"]) else "partially_compliant" if procedure_hits >= 1 else "uncertain"
    causality_summary = "；".join(x["text"] for x in (chain or []) if x.get("role_label") == "causality" and x.get("text"))[:180]
    if not causality_summary:
        causality_summary = "；".join(s for s, _, _ in split_sentences(text) if any(k in s for k in ["导致", "造成", "影响", "拖延"]))[:180]
    evidence_spans = []
    for item in (chain or []):
        if item.get("text"):
            evidence_spans.append({
                "role_label": item.get("role_label"),
                "text": str(item.get("text"))[:120],
                "span_start": item.get("span_start"),
                "span_end": item.get("span_end"),
            })
        if len(evidence_spans) >= 4:
            break
    confidence = min(0.95, round(0.35 + 0.08 * sum(scores.values()) + 0.05 * procedure_hits + 0.04 * evidence_hits, 4))
    uncertainty_flag = int(primary == "unknown" or confidence < 0.62)
    return {
        "primary_responsible_party": primary,
        "secondary_responsible_party": secondary,
        "responsibility_type": "shared" if primary == "both" or secondary != "unknown" else "single_party" if primary != "unknown" else "uncertain",
        "evidence_spans": evidence_spans,
        "procedural_compliance_status": proc_status,
        "causality_chain_summary": causality_summary,
        "documentation_integrity_flag": doc_flag,
        "confidence": confidence,
        "uncertainty_flag": uncertainty_flag,
        "explanation_text": "；".join(reasons[:8]),
    }


def evidence_sufficiency_score(structured_case: Dict) -> float:
    chain_metrics = evidence_chain_metrics(structured_case.get("source_span_pointers", []))
    procedure_hits = len(structured_case.get("procedural_compliance_cues", []))
    evidence_hits = len(structured_case.get("evidence_mentions", []))
    claim_hits = len((structured_case.get("claims_defenses", {}) or {}).get("claims", []))
    defense_hits = len((structured_case.get("claims_defenses", {}) or {}).get("defenses", []))
    score = (
        chain_metrics["role_coverage_rate"] * 0.45
        + min(procedure_hits / 5.0, 1.0) * 0.15
        + min(evidence_hits / 5.0, 1.0) * 0.20
        + min((claim_hits + defense_hits) / 6.0, 1.0) * 0.20
    )
    return float(round(max(0.0, min(score, 1.0)), 4))


def structured_numeric_features(structured_case: Dict) -> Dict[str, float]:
    pre_text = structured_case.get("pre_decision_text", "")
    chain_metrics = evidence_chain_metrics(structured_case.get("source_span_pointers", []))
    return {
        "delay_event_count": float(len(structured_case.get("delay_events", []))),
        "procedure_cue_count": float(len(structured_case.get("procedural_compliance_cues", []))),
        "evidence_mention_count": float(len(structured_case.get("evidence_mentions", []))),
        "claim_count": float(len((structured_case.get("claims_defenses", {}) or {}).get("claims", []))),
        "defense_count": float(len((structured_case.get("claims_defenses", {}) or {}).get("defenses", []))),
        "pre_text_length": float(len(pre_text)),
        "role_coverage_rate": chain_metrics["role_coverage_rate"],
        "missing_role_rate": chain_metrics["missing_role_rate"],
        "evidence_sufficiency": evidence_sufficiency_score(structured_case),
        "potential_leakage_flag": float(structured_case.get("potential_leakage_flag", 0)),
    }


def rule_baseline_prediction(structured_case: Dict) -> str:
    text = structured_case.get("pre_decision_text", "")
    owner_score = sum(1 for kw in OWNER_CAUSE_KEYWORDS if kw in text)
    contractor_score = sum(1 for kw in CONTRACTOR_CAUSE_KEYWORDS if kw in text)
    evidence_score = len(structured_case.get("evidence_mentions", []))
    procedure_score = len(structured_case.get("procedural_compliance_cues", []))
    delay_score = len(structured_case.get("delay_events", []))
    if any(x in text for x in ["双方", "均有责任", "共同责任"]):
        return "partial"
    if contractor_score >= owner_score + 1 and (procedure_score == 0 or evidence_score <= 1):
        return "not_support"
    if owner_score >= contractor_score + 1 and evidence_score >= 2 and delay_score >= 2:
        return "support"
    if evidence_score >= 2 or procedure_score >= 2:
        return "partial"
    return "not_support"


def classify_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, object]:
    rep = classification_report(y_true, y_pred, labels=list(labels), output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(labels)).tolist()
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "per_class": rep,
        "confusion_matrix": cm,
    }


def bootstrap_ci(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str], rounds: int = 300, seed: int = 2026) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0
    vals = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        vals.append(f1_score(y_true[idx], y_pred[idx], labels=list(labels), average="macro", zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def maybe_kappa(df: pd.DataFrame, reviewer_col: str = "reviewer_id", case_col: str = "case_id", score_col: str = "consistency_score") -> Dict[str, float]:
    out = {"percent_agreement": 0.0, "cohen_kappa": 0.0}
    if df.empty or reviewer_col not in df.columns or case_col not in df.columns or score_col not in df.columns:
        return out
    pivot = df.pivot_table(index=case_col, columns=reviewer_col, values=score_col, aggfunc="first")
    if pivot.shape[1] < 2:
        return out
    cols = list(pivot.columns[:2])
    tmp = pivot.dropna(subset=cols)
    if tmp.empty:
        return out
    out["percent_agreement"] = float((tmp[cols[0]] == tmp[cols[1]]).mean())
    out["cohen_kappa"] = float(cohen_kappa_score(tmp[cols[0]], tmp[cols[1]]))
    return out


def export_table(df: pd.DataFrame, base_path: Path) -> None:
    ensure_dir(base_path.parent)
    df.to_csv(base_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    df.to_excel(base_path.with_suffix(".xlsx"), index=False)


def latest_run_dir(results_root: Path) -> Optional[Path]:
    runs = sorted([d for d in results_root.glob("final_eval_*") if d.is_dir()])
    return runs[-1] if runs else None


def json_dump(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def structured_signature_tokens(structured_case: Dict) -> List[str]:
    tokens: List[str] = []
    for key in ["delay_events", "procedural_compliance_cues", "evidence_mentions"]:
        for item in structured_case.get(key, []):
            text = str(item.get("text", ""))
            for kw_group in [DELAY_KEYWORDS, PROCEDURE_KEYWORDS, EVIDENCE_KEYWORDS]:
                for kw in kw_group:
                    if kw in text:
                        tokens.append(kw)
    return list(dict.fromkeys(tokens))


def reasoning_trace(structured_case: Dict, pred_label: str, resp_diag: Dict[str, object], retrieved_cases: List[str], evidence_chain: List[Dict[str, object]]) -> Dict[str, object]:
    issue = "工期延误责任与索赔支持条件"
    rule_basis = []
    if any(x.get("role_label") == "entitlement" and x.get("text") for x in evidence_chain):
        rule_basis.append("存在合同/索赔依据表述")
    if resp_diag.get("procedural_compliance_status") != "uncertain":
        rule_basis.append(f"程序履约状态={resp_diag.get('procedural_compliance_status')}")
    application_findings = []
    for role in ["causality", "impact_schedule_relevance", "documentation_integrity"]:
        hit = next((x for x in evidence_chain if x.get("role_label") == role and x.get("text")), None)
        if hit:
            application_findings.append({"role": role, "text": hit["text"][:180]})
    management_action = []
    if resp_diag.get("documentation_integrity_flag") == "incomplete":
        management_action.append("补强 contemporaneous records 与签证资料")
    if resp_diag.get("procedural_compliance_status") == "noncompliant":
        management_action.append("优先修复通知、报审、备案类程序缺口")
    if not management_action:
        management_action.append("维持证据归档并准备类案对照说明")
    high_dispute_flag = int(resp_diag.get("uncertainty_flag", 0) == 1 or evidence_sufficiency_score(structured_case) < 0.45)
    return {
        "case_id": structured_case.get("case_id"),
        "retrieved_cases": retrieved_cases,
        "issue_focus": issue,
        "rule_basis": rule_basis,
        "application_findings": application_findings,
        "evidence_citations": [x for x in evidence_chain if x.get("text")],
        "leakage_check": int(structured_case.get("potential_leakage_flag", 0) == 0),
        "responsibility_primary": resp_diag.get("primary_responsible_party"),
        "outcome_label": pred_label,
        "management_action": management_action,
        "high_dispute_flag": high_dispute_flag,
    }
