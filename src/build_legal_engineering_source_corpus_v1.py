import json
import random
from pathlib import Path


OUT_DIR = Path("data/legal_engineering_source_corpus_v1")
LEAKAGE = "No frozen test labels or post-decision test reasoning used."
INSTRUCTION = "根据问题和候选法律—工程知识块，提取与工期延误争议判断相关的内容。如果候选内容无关，请只输出 No relevant content。"

CATEGORIES = [
    "entitlement",
    "notice",
    "causality",
    "schedule_impact",
    "documentation",
    "concurrent_delay",
    "variation",
    "liquidated_damages",
    "delay_compensation",
    "burden_of_proof",
    "responsibility_allocation",
]

EVIDENCE_ROLES = {
    "entitlement": "ENT",
    "notice": "NOT",
    "causality": "CAU",
    "schedule_impact": "IMP",
    "documentation": "DOC",
    "concurrent_delay": "CAU",
    "variation": "NOT",
    "liquidated_damages": "ENT",
    "delay_compensation": "ENT",
    "burden_of_proof": "DOC",
    "responsibility_allocation": "RESP",
}

DOCS = [
    ("civil_code_contract_001", "law", "中华人民共和国民法典合同编：合同履行规则", "Civil Code Contract Book: Performance of Contracts", "全国人民代表大会", "2021-01-01", "Civil Code of the PRC, Contract Book, general performance rules", "合同履行、协作义务、履行抗辩"),
    ("civil_code_contract_002", "law", "中华人民共和国民法典合同编：违约责任规则", "Civil Code Contract Book: Liability for Breach", "全国人民代表大会", "2021-01-01", "Civil Code of the PRC, Contract Book, breach liability rules", "违约责任、继续履行、损害赔偿"),
    ("civil_code_contract_003", "law", "中华人民共和国民法典合同编：损失赔偿与可预见性", "Civil Code Contract Book: Damages and Foreseeability", "全国人民代表大会", "2021-01-01", "Civil Code of the PRC, Contract Book, damages rules", "损失赔偿、违约金调整、损失证明"),
    ("civil_code_contract_004", "law", "中华人民共和国民法典合同编：不可抗力与情势变化", "Civil Code Contract Book: Force Majeure and Change of Circumstances", "全国人民代表大会", "2021-01-01", "Civil Code of the PRC, force majeure and changed circumstances rules", "不可抗力、疫情、异常天气、风险分配"),
    ("civil_code_contract_005", "law", "中华人民共和国民法典合同编：建设工程合同一般规则", "Civil Code Construction Contract Rules", "全国人民代表大会", "2021-01-01", "Civil Code of the PRC, construction contract chapter", "建设工程合同、发包承包、验收结算"),
    ("construction_judicial_interp_001", "judicial_interpretation", "最高人民法院建设工程施工合同司法解释：合同效力与工程价款", "SPC Construction Contract Interpretation: Validity and Payment", "最高人民法院", "2021-01-01", "SPC interpretation on construction contract disputes, validity and payment issues", "合同效力、工程价款、结算依据"),
    ("construction_judicial_interp_002", "judicial_interpretation", "最高人民法院建设工程施工合同司法解释：质量、验收与修复", "SPC Construction Contract Interpretation: Quality and Acceptance", "最高人民法院", "2021-01-01", "SPC construction contract interpretation, quality and acceptance issues", "质量责任、竣工验收、修复费用"),
    ("construction_judicial_interp_003", "judicial_interpretation", "最高人民法院建设工程施工合同司法解释：逾期竣工与违约责任", "SPC Construction Contract Interpretation: Delay and Breach", "最高人民法院", "2021-01-01", "SPC construction contract interpretation, delay-related breach reasoning", "工期延误、逾期竣工、违约责任"),
    ("model_contract_2017_001", "model_contract", "建设工程施工合同示范文本 GF-2017-0201：工期与进度", "Model Construction Contract GF-2017-0201: Time and Progress", "住房和城乡建设部、国家工商行政管理总局", "2017-10-01", "Model Construction Contract GF-2017-0201, time and progress clauses", "开工、竣工、进度计划、工期顺延"),
    ("model_contract_2017_002", "model_contract", "建设工程施工合同示范文本 GF-2017-0201：变更、签证与索赔", "Model Construction Contract GF-2017-0201: Variation, Site Instruction and Claims", "住房和城乡建设部、国家工商行政管理总局", "2017-10-01", "Model Construction Contract GF-2017-0201, variation and claims clauses", "变更、签证、索赔通知、证据提交"),
    ("model_contract_2017_003", "model_contract", "建设工程施工合同示范文本 GF-2017-0201：暂停施工与复工", "Model Construction Contract GF-2017-0201: Suspension and Resumption", "住房和城乡建设部、国家工商行政管理总局", "2017-10-01", "Model Construction Contract GF-2017-0201, suspension and resumption clauses", "暂停施工、复工、发包人原因、承包人原因"),
    ("model_contract_2017_004", "model_contract", "建设工程施工合同示范文本 GF-2017-0201：违约责任与损失", "Model Construction Contract GF-2017-0201: Breach and Damages", "住房和城乡建设部、国家工商行政管理总局", "2017-10-01", "Model Construction Contract GF-2017-0201, breach and damages clauses", "违约责任、工期违约金、停窝工损失"),
    ("engineering_pm_001", "engineering_standard", "建设工程项目管理规范：进度控制", "Construction Project Management: Schedule Control", "工程管理通用规范来源", "", "General construction project management schedule-control principles", "进度计划、计划更新、进度控制"),
    ("engineering_pm_002", "engineering_standard", "建设工程项目管理规范：资料与记录管理", "Construction Project Management: Documentation Control", "工程管理通用规范来源", "", "General construction project management documentation-control principles", "施工日志、会议纪要、监理指令、往来函件"),
    ("engineering_delay_001", "expert_synthesized", "工期延误分析：关键路径与总工期影响", "Delay Analysis: Critical Path and Overall Completion Impact", "Expert synthesized legal-engineering reasoning", "", "general legal-engineering reasoning", "关键路径、总工期影响、延误归因"),
    ("engineering_delay_002", "expert_synthesized", "工期延误分析：同期延误与责任分配", "Delay Analysis: Concurrent Delay and Responsibility Allocation", "Expert synthesized legal-engineering reasoning", "", "general legal-engineering reasoning", "同期延误、共同责任、责任分配"),
    ("engineering_delay_003", "expert_synthesized", "工期索赔证据链：通知、签证、进度资料", "Delay Claim Evidence Chain: Notice, Site Records and Schedule Data", "Expert synthesized legal-engineering reasoning", "", "general legal-engineering reasoning", "索赔通知、签证、进度记录、证据链"),
    ("engineering_delay_004", "expert_synthesized", "工期索赔证据链：损失证明与计算", "Delay Claim Evidence Chain: Loss Proof and Quantification", "Expert synthesized legal-engineering reasoning", "", "general legal-engineering reasoning", "停窝工损失、机械闲置、人工窝工、费用证明"),
    ("public_rule_001", "expert_synthesized", "公开裁判规则归纳：发包人原因导致工期延误", "Public Adjudication Rule Synthesis: Employer-Caused Delay", "Expert synthesized from public adjudication reasoning", "", "general public adjudication rule synthesis without case-specific test reasoning", "发包人原因、图纸迟延、变更、付款迟延、场地交付"),
    ("public_rule_002", "expert_synthesized", "公开裁判规则归纳：承包人原因导致工期延误", "Public Adjudication Rule Synthesis: Contractor-Caused Delay", "Expert synthesized from public adjudication reasoning", "", "general public adjudication rule synthesis without case-specific test reasoning", "承包人原因、组织不力、质量返工、资料不全"),
    ("public_rule_003", "expert_synthesized", "公开裁判规则归纳：证据不足与举证责任", "Public Adjudication Rule Synthesis: Evidence Insufficiency and Burden of Proof", "Expert synthesized from public adjudication reasoning", "", "general public adjudication rule synthesis without case-specific test reasoning", "举证责任、证据不足、因果证明"),
    ("public_rule_004", "expert_synthesized", "公开裁判规则归纳：工期违约金调整", "Public Adjudication Rule Synthesis: Adjustment of Delay Liquidated Damages", "Expert synthesized from public adjudication reasoning", "", "general public adjudication rule synthesis without case-specific test reasoning", "违约金调整、过高过低、实际损失"),
    ("public_rule_005", "expert_synthesized", "公开裁判规则归纳：商品房逾期交付", "Public Adjudication Rule Synthesis: Late Delivery of Property", "Expert synthesized from public adjudication reasoning", "", "general public adjudication rule synthesis without case-specific test reasoning", "逾期交房、交付条件、竣工验收备案"),
    ("public_rule_006", "expert_synthesized", "公开裁判规则归纳：不可抗力、疫情与异常天气", "Public Adjudication Rule Synthesis: Force Majeure, Pandemic and Severe Weather", "Expert synthesized from public adjudication reasoning", "", "general public adjudication rule synthesis without case-specific test reasoning", "疫情、暴雨、不可抗力、顺延期间"),
]

TEMPLATES = {
    "entitlement": ("权利基础", "判断工期顺延、逾期竣工违约金或延误损失时，应先识别合同、法律或工程规则中是否存在可请求的权利基础。"),
    "notice": ("通知与签证", "主张工期顺延或延误损失通常需要及时通知、提交签证或索赔文件，并保留对方接收、监理确认或会议确认的记录。"),
    "causality": ("因果关系", "延误主张需要说明事件如何导致施工活动受阻，并把责任事件与关键施工活动、实际延误后果连接起来。"),
    "schedule_impact": ("进度影响", "仅有事件发生并不足够，还需要证明该事件影响关键路径或总工期，而不是只影响非关键作业。"),
    "documentation": ("证据资料", "施工日志、进度计划、监理通知、会议纪要、往来函件、签证单和结算资料共同构成延误证据链。"),
    "concurrent_delay": ("同期延误", "当发包人原因与承包人原因在同一期间共同影响工期时，应区分并发影响，避免把全部延误归于单方。"),
    "variation": ("变更与指令", "设计变更、工程量增加、现场指令或发包人要求调整施工顺序，可能构成顺延工期或费用补偿的触发事件。"),
    "liquidated_damages": ("工期违约金", "工期违约金判断需要结合合同约定、实际逾期期间、责任原因和是否存在调整事由。"),
    "delay_compensation": ("延误损失", "停窝工、机械闲置、管理费增加等损失需要有期间、数量、单价、因果关系和责任归属证明。"),
    "burden_of_proof": ("举证责任", "主张一方通常应证明延误事件、责任原因、通知程序、影响期间和损失数额，证据不足会削弱其主张。"),
    "responsibility_allocation": ("责任分配", "责任分配应结合合同义务、现场控制权、变更原因、付款与场地条件、施工组织和证据完整性综合判断。"),
}

TRIGGER_TERMS = {
    "entitlement": (["权利基础", "合同约定", "法定责任", "请求依据"], ["entitlement", "contractual basis", "legal basis"]),
    "notice": (["通知", "签证", "索赔意向", "报审", "监理确认"], ["notice", "site instruction", "claim notice", "engineer confirmation"]),
    "causality": (["因果关系", "导致", "影响施工", "责任事件"], ["causation", "causal link", "delay event"]),
    "schedule_impact": (["关键路径", "总工期", "进度计划", "实际进度"], ["critical path", "overall completion", "schedule impact"]),
    "documentation": (["施工日志", "会议纪要", "往来函件", "进度记录", "证据链"], ["site diary", "meeting minutes", "correspondence", "evidence chain"]),
    "concurrent_delay": (["同期延误", "共同原因", "并发影响", "责任交叉"], ["concurrent delay", "shared delay", "overlapping causes"]),
    "variation": (["设计变更", "工程变更", "现场指令", "工程量增加"], ["variation", "change order", "site instruction"]),
    "liquidated_damages": (["违约金", "逾期竣工", "逾期交付", "调整标准"], ["liquidated damages", "late completion", "late delivery"]),
    "delay_compensation": (["停窝工损失", "机械闲置", "人工窝工", "费用增加"], ["delay compensation", "idle labor", "idle equipment"]),
    "burden_of_proof": (["举证责任", "证据不足", "证明责任", "不能证明"], ["burden of proof", "insufficient evidence"]),
    "responsibility_allocation": (["责任分配", "发包人原因", "承包人原因", "共同责任"], ["responsibility allocation", "employer cause", "contractor cause"]),
}

QUESTIONS = {
    "entitlement": "哪些内容能够说明工期顺延、违约金或延误损失主张的权利基础？",
    "notice": "承包人主张工期顺延时，哪些内容与通知、签证和索赔程序有关？",
    "causality": "哪些内容能够帮助判断延误事件与工期后果之间的因果关系？",
    "schedule_impact": "哪些内容与关键路径、总工期影响或进度计划对比有关？",
    "documentation": "哪些内容可以作为工期延误争议中的证据链资料？",
    "concurrent_delay": "哪些内容与同期延误、共同原因或责任交叉有关？",
    "variation": "哪些内容与设计变更、工程变更、现场指令或签证有关？",
    "liquidated_damages": "哪些内容与逾期竣工、逾期交付或工期违约金有关？",
    "delay_compensation": "哪些内容与停窝工损失、机械闲置和延误费用证明有关？",
    "burden_of_proof": "哪些内容说明主张方应承担的举证责任和证据不足风险？",
    "responsibility_allocation": "哪些内容有助于区分发包人、承包人、共同责任或外部原因？",
}


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_docs():
    rows = []
    for source_id, source_type, cn, en, body, date, ref, scope in DOCS:
        rows.append({
            "source_id": source_id,
            "source_type": source_type,
            "source_title_cn": cn,
            "source_title_en": en,
            "issuing_body": body,
            "effective_date": date,
            "source_reference": ref,
            "scope": scope,
            "leakage_check": LEAKAGE,
        })
    return rows


def build_blocks(docs):
    blocks = []
    block_no = 1
    for d in docs:
        # six categories per source, rotating across the full taxonomy.
        start = (block_no - 1) % len(CATEGORIES)
        chosen = [CATEGORIES[(start + i) % len(CATEGORIES)] for i in range(6)]
        for cat in chosen:
            title, base_summary = TEMPLATES[cat]
            zh_terms, en_terms = TRIGGER_TERMS[cat]
            source_note = "法律—工程规则摘要（非具体条文原文）" if d["source_type"] != "expert_synthesized" else "专家归纳规则"
            original = (
                f"{source_note}：在“{d['source_title_cn']}”范围内，{base_summary}"
                f" 适用时应结合{d['scope']}等背景，检查是否存在明确的合同依据、程序记录、"
                f"进度影响和责任归属证据。"
            )
            plain_cn = f"{title}：{base_summary} 在案件分析中应只作为证据定位和规则摘要依据，不直接替代裁判结论。"
            plain_en = (
                f"{title}: {base_summary} In dispute analytics, this block supports evidence grounding "
                f"and rule summarization rather than direct outcome classification."
            )
            blocks.append({
                "block_id": f"legal_engineering_block_{block_no:03d}",
                "source_id": d["source_id"],
                "source_type": d["source_type"],
                "source_title_cn": d["source_title_cn"],
                "article_or_section": f"{title} / {cat}",
                "category": cat,
                "evidence_role": EVIDENCE_ROLES[cat],
                "original_text_cn": original,
                "plain_language_summary_cn": plain_cn,
                "plain_language_summary_en": plain_en,
                "trigger_terms_cn": zh_terms,
                "trigger_terms_en": en_terms,
                "source_reference": d["source_reference"],
                "leakage_check": LEAKAGE,
            })
            block_no += 1
    return blocks


def positive_sample(block, question_category):
    question = QUESTIONS[question_category]
    out = f"Summary: {block['plain_language_summary_cn']} 关键词：{', '.join(block['trigger_terms_cn'][:3])}。"
    return {
        "instruction": INSTRUCTION,
        "input": f"Question: {question}\nOriginal_Content: {block['original_text_cn']}",
        "output": out,
    }


def negative_sample(block, question_category):
    question = QUESTIONS[question_category]
    return {
        "instruction": INSTRUCTION,
        "input": f"Question: {question}\nOriginal_Content: {block['original_text_cn']}",
        "output": "No relevant content",
    }


def build_samples(blocks, n_total, neg_ratio, seed):
    rng = random.Random(seed)
    n_neg = round(n_total * neg_ratio)
    n_pos = n_total - n_neg
    rows = []
    for _ in range(n_pos):
        b = rng.choice(blocks)
        rows.append(positive_sample(b, b["category"]))
    for _ in range(n_neg):
        b = rng.choice(blocks)
        other_categories = [c for c in CATEGORIES if c != b["category"]]
        # Prefer a semantically distant category for robust noise filtering.
        if b["category"] in {"notice", "documentation", "burden_of_proof"}:
            candidates = ["liquidated_damages", "delay_compensation", "schedule_impact"]
        elif b["category"] in {"liquidated_damages", "delay_compensation"}:
            candidates = ["notice", "documentation", "concurrent_delay"]
        else:
            candidates = other_categories
        question_category = rng.choice([c for c in candidates if c != b["category"]])
        rows.append(negative_sample(b, question_category))
    rng.shuffle(rows)
    return rows


def readme_text(docs, blocks, train, val):
    neg_train = sum(1 for r in train if r["output"] == "No relevant content")
    neg_val = sum(1 for r in val if r["output"] == "No relevant content")
    return f"""# Legal-Engineering Source Corpus v1 for Jiang-style 1st-LoRA

## Purpose

This corpus supports DelayDispute / LEAP-BD by preparing a legal-engineering grounding dataset before outcome classification. It organizes legal rules, construction-contract clauses, public adjudication rule syntheses, and engineering-management knowledge into retrievable content blocks.

## Relationship to Jiang-style 1st-LoRA

The 1st-LoRA is not an outcome classifier. Its task is:

`Question + Original_Content -> Summary`

If the candidate content is irrelevant to the question, the model should output exactly:

`No relevant content`

This trains evidence grounding, legal-engineering summarization, and noise filtering before any support / partial_support / not_support prediction is attempted.

## Files

- `legal_engineering_source_docs_v1.jsonl`: source-level metadata. Rows: {len(docs)}.
- `legal_engineering_content_blocks_v1.jsonl`: retrievable law-contract-engineering blocks. Rows: {len(blocks)}.
- `first_lora_grounding_train_seed_v1.jsonl`: seed training samples for knowledge extraction. Rows: {len(train)}; negative rows: {neg_train} ({neg_train / len(train):.1%}).
- `first_lora_grounding_val_seed_v1.jsonl`: validation samples. Rows: {len(val)}; negative rows: {neg_val} ({neg_val / len(val):.1%}).

## Field Notes

`source_docs` records the source title, source type, issuing body, scope, and leakage check. `content_blocks` splits the source material into categories such as entitlement, notice, causality, schedule impact, documentation, concurrent delay, variation, liquidated damages, delay compensation, burden of proof, and responsibility allocation.

Legal blocks are written as legal-engineering summaries, not as fabricated statutory quotations. When exact statutory wording is not used, the block text explicitly indicates that it is a summary rather than a quoted article.

## Why 1st-LoRA Is Not an Outcome Classifier

This dataset deliberately avoids support / partial_support / not_support labels. It does not use frozen test500 labels, post-decision reasoning, or case-specific adjudicated conclusions. Outcome prediction is delegated to later stages.

## Intended Pipeline

legal-engineering source corpus
→ content block retrieval
→ 1st-LoRA evidence grounding / summary / noise filtering
→ Binary-1: accepted vs not_support
→ Binary-2: support vs partial_support
→ RAG verification / selective triage

## Leakage Discipline

All rows include: `{LEAKAGE}`
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = build_docs()
    blocks = build_blocks(docs)
    train = build_samples(blocks, 500, 0.30, 2026052901)
    val = build_samples(blocks, 100, 0.30, 2026052902)

    write_jsonl(OUT_DIR / "legal_engineering_source_docs_v1.jsonl", docs)
    write_jsonl(OUT_DIR / "legal_engineering_content_blocks_v1.jsonl", blocks)
    write_jsonl(OUT_DIR / "first_lora_grounding_train_seed_v1.jsonl", train)
    write_jsonl(OUT_DIR / "first_lora_grounding_val_seed_v1.jsonl", val)
    (OUT_DIR / "README_legal_engineering_source_corpus.md").write_text(
        readme_text(docs, blocks, train, val), encoding="utf-8"
    )

    manifest = {
        "source_docs": len(docs),
        "content_blocks": len(blocks),
        "train_seed": len(train),
        "val_seed": len(val),
        "train_negative_ratio": sum(1 for r in train if r["output"] == "No relevant content") / len(train),
        "val_negative_ratio": sum(1 for r in val if r["output"] == "No relevant content") / len(val),
        "leakage_check": LEAKAGE,
    }
    (OUT_DIR / "manifest_v1.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT_DIR.resolve()), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
