import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path("data/legal_engineering_source_corpus_v1")
LEAKAGE = "No frozen test labels or post-decision test reasoning used."
INSTRUCTION = "根据问题和候选法律—工程知识块，提取与工期延误争议判断相关的内容。如果候选内容无关，请只输出 No relevant content。"

CATEGORY_INFO = {
    "entitlement": {
        "question_ids": ["q_entitlement_basis", "q_contractual_right", "q_legal_basis"],
        "questions": [
            "哪些内容能够说明工期顺延、违约金或延误损失主张的权利基础？",
            "候选内容中是否包含可支持延误争议请求的合同或法律依据？",
            "请提取与延误主张权利来源有关的法律—工程规则。",
        ],
        "role": "ENT",
        "near": ["liquidated_damages", "delay_compensation", "responsibility_allocation"],
    },
    "notice": {
        "question_ids": ["q_notice_claim", "q_site_instruction", "q_procedural_compliance"],
        "questions": [
            "承包人主张工期顺延时，哪些内容与通知、签证和索赔程序有关？",
            "候选内容是否说明了通知、报审、签证或监理确认的程序要求？",
            "请提取与工期索赔程序合规有关的内容。",
        ],
        "role": "NOT",
        "near": ["documentation", "variation", "burden_of_proof"],
    },
    "causality": {
        "question_ids": ["q_delay_causation", "q_event_to_delay", "q_responsibility_event"],
        "questions": [
            "哪些内容能够帮助判断延误事件与工期后果之间的因果关系？",
            "请提取说明责任事件如何影响施工进展的规则。",
            "候选内容中哪些部分有助于连接延误原因和延误后果？",
        ],
        "role": "CAU",
        "near": ["schedule_impact", "concurrent_delay", "responsibility_allocation"],
    },
    "schedule_impact": {
        "question_ids": ["q_critical_path", "q_total_float", "q_schedule_comparison"],
        "questions": [
            "哪些内容与关键路径、总工期影响或进度计划对比有关？",
            "请提取可用于判断延误是否影响总工期的内容。",
            "候选内容是否说明了计划进度、实际进度和关键作业之间的关系？",
        ],
        "role": "IMP",
        "near": ["causality", "concurrent_delay", "documentation"],
    },
    "documentation": {
        "question_ids": ["q_evidence_chain", "q_project_records", "q_documentation_quality"],
        "questions": [
            "哪些内容可以作为工期延误争议中的证据链资料？",
            "请提取与施工日志、会议纪要、往来函件或进度记录有关的内容。",
            "候选内容是否有助于判断延误证据是否完整？",
        ],
        "role": "DOC",
        "near": ["notice", "burden_of_proof", "variation"],
    },
    "concurrent_delay": {
        "question_ids": ["q_concurrent_delay", "q_overlapping_causes", "q_shared_time_responsibility"],
        "questions": [
            "哪些内容与同期延误、共同原因或责任交叉有关？",
            "请提取有助于区分并发延误和单方延误的内容。",
            "候选内容是否说明多个责任原因同时影响工期的处理逻辑？",
        ],
        "role": "CAU",
        "near": ["causality", "schedule_impact", "responsibility_allocation"],
    },
    "variation": {
        "question_ids": ["q_variation_order", "q_change_instruction", "q_site_instruction_effect"],
        "questions": [
            "哪些内容与设计变更、工程变更、现场指令或签证有关？",
            "请提取说明变更指令如何影响工期或费用的内容。",
            "候选内容是否涉及变更、签证或发包人指令的管理要求？",
        ],
        "role": "NOT",
        "near": ["notice", "documentation", "entitlement"],
    },
    "liquidated_damages": {
        "question_ids": ["q_delay_ld", "q_late_completion_penalty", "q_ld_adjustment"],
        "questions": [
            "哪些内容与逾期竣工、逾期交付或工期违约金有关？",
            "请提取可用于判断工期违约金责任和调整的内容。",
            "候选内容是否说明违约金计算、调整或责任基础？",
        ],
        "role": "ENT",
        "near": ["entitlement", "delay_compensation", "responsibility_allocation"],
    },
    "delay_compensation": {
        "question_ids": ["q_delay_loss", "q_idle_cost", "q_loss_quantification"],
        "questions": [
            "哪些内容与停窝工损失、机械闲置和延误费用证明有关？",
            "请提取可用于计算或证明延误损失的内容。",
            "候选内容是否涉及人工、机械、管理费或其他延误费用的证明？",
        ],
        "role": "ENT",
        "near": ["liquidated_damages", "documentation", "burden_of_proof"],
    },
    "burden_of_proof": {
        "question_ids": ["q_burden_of_proof", "q_evidence_insufficiency", "q_proof_risk"],
        "questions": [
            "哪些内容说明主张方应承担的举证责任和证据不足风险？",
            "请提取与证明责任、证据不足或不能证明有关的内容。",
            "候选内容是否说明延误主张需要证明哪些要素？",
        ],
        "role": "DOC",
        "near": ["documentation", "notice", "causality"],
    },
    "responsibility_allocation": {
        "question_ids": ["q_responsibility_allocation", "q_owner_contractor_external", "q_fault_apportionment"],
        "questions": [
            "哪些内容有助于区分发包人、承包人、共同责任或外部原因？",
            "请提取用于判断工期延误责任分配的内容。",
            "候选内容是否说明责任归属、共同责任或外部风险的划分？",
        ],
        "role": "RESP",
        "near": ["causality", "concurrent_delay", "entitlement"],
    },
}

DISTANT = {
    "entitlement": ["documentation", "schedule_impact", "concurrent_delay"],
    "notice": ["liquidated_damages", "delay_compensation", "schedule_impact"],
    "causality": ["notice", "liquidated_damages", "documentation"],
    "schedule_impact": ["entitlement", "notice", "liquidated_damages"],
    "documentation": ["liquidated_damages", "schedule_impact", "responsibility_allocation"],
    "concurrent_delay": ["notice", "delay_compensation", "entitlement"],
    "variation": ["liquidated_damages", "concurrent_delay", "delay_compensation"],
    "liquidated_damages": ["notice", "schedule_impact", "documentation"],
    "delay_compensation": ["notice", "concurrent_delay", "variation"],
    "burden_of_proof": ["liquidated_damages", "schedule_impact", "variation"],
    "responsibility_allocation": ["notice", "delay_compensation", "documentation"],
}

SUMMARY_OPENERS = [
    "Summary: 该内容可用于定位{theme}，核心是{core}。来源范围为“{source_focus}”，应与{role_hint}一起核验。",
    "Summary: 与问题相关的部分在于{core}。在“{source_focus}”语境下，它提示分析者关注{theme}，但不能直接推出案件结果。",
    "Summary: 候选知识块强调{theme}，可帮助整理{role_hint}。关键词包括{terms}，适用时还需要结合项目记录和责任事件。",
    "Summary: 可提取的规则是：{core}。该规则主要服务于{theme}的证据归纳，而不是支持或驳回结论；来源为“{source_focus}”。",
    "Summary: 本块与问题的联系是{theme}；摘要要点为{core}，并应检查{role_hint}是否存在。触发词：{terms}。",
    "Summary: 该段提供了{theme}的判断线索。可概括为{core}，用于后续检索和噪声过滤；其来源范围是“{source_focus}”。",
    "Summary: 从“{source_focus}”可抽取的相关内容是{core}。对 1st-LoRA 来说，它用于识别{theme}及其证据角色。",
    "Summary: 该块不是裁判结论，而是关于{theme}的规则摘要。可保留的要点是{core}，并优先核对{terms}。",
    "Summary: 针对该问题，候选内容中的有效信息是{core}。它对应的证据核验对象为{role_hint}。",
    "Summary: 本段可作为{theme}的 grounding block。摘要为{core}；使用时应避免把规则摘要误当成个案结果。",
    "Summary: 可用于检索的知识点包括{terms}。这些内容共同指向{theme}，并由“{source_focus}”提供背景。",
    "Summary: 该候选块与问题相关，因为它说明了{core}。其作用是帮助模型从原文中筛出{role_hint}。"
]

ROLE_HINTS = {
    "ENT": "权利基础、合同依据或可请求事项",
    "NOT": "通知、签证、报审、监理确认等程序记录",
    "CAU": "责任事件、因果链和延误期间",
    "IMP": "进度计划、关键路径和总工期影响",
    "DOC": "施工日志、会议纪要、往来函件和证明责任",
    "RESP": "发包人、承包人、共同责任或外部原因",
}


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_positive(block, q_idx, variant_idx, split, sample_no):
    info = CATEGORY_INFO[block["category"]]
    question = info["questions"][q_idx % len(info["questions"])]
    question_id = info["question_ids"][q_idx % len(info["question_ids"])]
    theme = block["article_or_section"].split("/")[0].strip()
    core = block["plain_language_summary_cn"].replace("在案件分析中应只作为证据定位和规则摘要依据，不直接替代裁判结论。", "").strip()
    role_hint = ROLE_HINTS.get(block["evidence_role"], "相关证据角色")
    source_focus = block["source_title_cn"]
    terms = "、".join(block.get("trigger_terms_cn", [])[:3])
    template = SUMMARY_OPENERS[variant_idx % len(SUMMARY_OPENERS)]
    output = template.format(theme=theme, core=core, role_hint=role_hint, source_focus=source_focus, terms=terms)
    sample_id = f"{split}_pos_{sample_no:04d}"
    sample = {
        "instruction": INSTRUCTION,
        "input": f"Question: {question}\nOriginal_Content: {block['original_text_cn']}",
        "output": output,
    }
    meta = {
        "sample_id": sample_id,
        "question_id": question_id,
        "block_id": block["block_id"],
        "source_id": block["source_id"],
        "category": block["category"],
        "evidence_role": block["evidence_role"],
        "is_relevant": True,
        "negative_type": "",
        "leakage_check": LEAKAGE,
    }
    return sample, meta


def make_negative(block, question_category, negative_type, split, sample_no):
    info = CATEGORY_INFO[question_category]
    question_idx = sample_no % len(info["questions"])
    question = info["questions"][question_idx]
    question_id = info["question_ids"][question_idx]
    sample_id = f"{split}_neg_{sample_no:04d}"
    sample = {
        "instruction": INSTRUCTION,
        "input": f"Question: {question}\nOriginal_Content: {block['original_text_cn']}",
        "output": "No relevant content",
    }
    meta = {
        "sample_id": sample_id,
        "question_id": question_id,
        "block_id": block["block_id"],
        "source_id": block["source_id"],
        "category": block["category"],
        "evidence_role": block["evidence_role"],
        "is_relevant": False,
        "negative_type": negative_type,
        "leakage_check": LEAKAGE,
    }
    return sample, meta


def select_question_category(block, negative_type, rng):
    cat = block["category"]
    if negative_type == "random_negative":
        return rng.choice(DISTANT[cat])
    if negative_type == "near_negative":
        # Same broad topic family but different requested extraction target.
        return rng.choice([c for c in CATEGORY_INFO[cat]["near"] if c != cat])
    if negative_type == "hard_negative":
        # Share a trigger family but require a different evidence role.
        role = block["evidence_role"]
        candidates = [c for c, info in CATEGORY_INFO.items() if info["role"] != role and c in CATEGORY_INFO[cat]["near"] + DISTANT[cat]]
        if not candidates:
            candidates = [c for c, info in CATEGORY_INFO.items() if info["role"] != role]
        return rng.choice(candidates)
    raise ValueError(negative_type)


def build_split(blocks, n_total, split, seed, forbidden_inputs=None):
    rng = random.Random(seed)
    n_neg = round(n_total * 0.30)
    n_pos = n_total - n_neg
    rows, metas = [], []
    forbidden_inputs = set(forbidden_inputs or [])
    used_inputs = set(forbidden_inputs)
    own_inputs = set()
    pos_count = 0
    neg_count = 0

    # Positive samples: at most one unique question per block before cycling.
    candidates = []
    for block in blocks:
        for q_idx in range(3):
            for v_idx in range(4):
                candidates.append(("pos", block, q_idx, v_idx))
    rng.shuffle(candidates)
    for _, block, q_idx, v_idx in candidates:
        if pos_count >= n_pos:
            break
        sample, meta = make_positive(block, q_idx, v_idx + pos_count, split, pos_count + 1)
        if sample["input"] in used_inputs:
            continue
        used_inputs.add(sample["input"])
        own_inputs.add(sample["input"])
        rows.append(sample)
        metas.append(meta)
        pos_count += 1

    neg_types = ["random_negative", "near_negative", "hard_negative"]
    neg_candidates = []
    for block in blocks:
        for nt in neg_types:
            for _ in range(3):
                qc = select_question_category(block, nt, rng)
                neg_candidates.append((block, qc, nt))
    rng.shuffle(neg_candidates)
    for block, qc, nt in neg_candidates:
        if neg_count >= n_neg:
            break
        sample, meta = make_negative(block, qc, nt, split, neg_count + 1)
        if sample["input"] in used_inputs:
            continue
        # Guardrail: do not mark same-category content as irrelevant.
        if qc == block["category"]:
            continue
        used_inputs.add(sample["input"])
        own_inputs.add(sample["input"])
        rows.append(sample)
        metas.append(meta)
        neg_count += 1

    if len(rows) != n_total:
        raise RuntimeError(f"{split}: generated {len(rows)} not {n_total}; pos={pos_count}, neg={neg_count}")

    order = list(range(len(rows)))
    rng.shuffle(order)
    return [rows[i] for i in order], [metas[i] for i in order], own_inputs


def assert_no_overlap(train, val):
    train_inputs = {r["input"] for r in train}
    val_inputs = {r["input"] for r in val}
    overlap = train_inputs & val_inputs
    if overlap:
        raise RuntimeError(f"train/val duplicate input count: {len(overlap)}")


def update_readme():
    readme = ROOT / "README_legal_engineering_source_corpus.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    addendum = """

## v1.1 Revision Note

Version v1.1 keeps the same `legal_engineering_source_docs_v1.jsonl` and
`legal_engineering_content_blocks_v1.jsonl`, but regenerates the 1st-LoRA
grounding train/validation seeds with stricter quality controls:

- train/validation inputs are de-duplicated;
- every sample has a metadata row with `sample_id`, `question_id`, `block_id`,
  `source_id`, `category`, `evidence_role`, `is_relevant`, `negative_type`,
  and `leakage_check`;
- negative samples are divided into `random_negative`, `near_negative`, and
  `hard_negative`;
- positive summaries use varied question forms and varied summary expressions.

This dataset is an expert-synthesized legal-engineering grounding corpus. It is
not a statutory article corpus and should not be described as one. Some source
records refer to laws, judicial interpretations, model contracts, and
engineering standards, but the content blocks are legal-engineering summaries
and retrieval-oriented knowledge blocks rather than verbatim statutory articles.
If exact legal text is required, it must be collected and cited separately from
official sources.
"""
    if "## v1.1 Revision Note" not in text:
        readme.write_text(text.rstrip() + "\n" + addendum.lstrip(), encoding="utf-8")


def main():
    blocks = read_jsonl(ROOT / "legal_engineering_content_blocks_v1.jsonl")
    train, train_meta, train_inputs = build_split(blocks, 500, "train_v1_1", 202605291101)
    val, val_meta, _ = build_split(blocks, 100, "val_v1_1", 202605291102, forbidden_inputs=train_inputs)
    assert_no_overlap(train, val)

    write_jsonl(ROOT / "first_lora_grounding_train_seed_v1_1.jsonl", train)
    write_jsonl(ROOT / "first_lora_grounding_val_seed_v1_1.jsonl", val)
    write_jsonl(ROOT / "first_lora_grounding_train_seed_v1_1_meta.jsonl", train_meta)
    write_jsonl(ROOT / "first_lora_grounding_val_seed_v1_1_meta.jsonl", val_meta)
    update_readme()

    unique_outputs = len({r["output"] for r in train if r["output"] != "No relevant content"})
    manifest = {
        "train_rows": len(train),
        "val_rows": len(val),
        "train_negative_ratio": sum(1 for r in train if r["output"] == "No relevant content") / len(train),
        "val_negative_ratio": sum(1 for r in val if r["output"] == "No relevant content") / len(val),
        "train_unique_positive_outputs": unique_outputs,
        "train_negative_types": dict(Counter(m["negative_type"] for m in train_meta if not m["is_relevant"])),
        "val_negative_types": dict(Counter(m["negative_type"] for m in val_meta if not m["is_relevant"])),
        "leakage_check": LEAKAGE,
    }
    (ROOT / "manifest_v1_1.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(ROOT.resolve()), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
