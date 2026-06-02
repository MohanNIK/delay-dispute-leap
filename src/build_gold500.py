# -*- coding: utf-8 -*-
import argparse
import json
import random
from pathlib import Path
from collections import Counter

import pandas as pd

LABELS = ["support", "partial", "not_support"]
RESP_MAP = {
    "业主": "owner",
    "承包商": "contractor",
    "分包商": "subcontractor",
    "设计/监理": "designer_supervisor",
    "双方": "both",
    "不可抗力/政策": "force_majeure_policy",
    "unknown": "unknown",
}

DEFAULT_CFG = {
    "project": {"root": "."},
    "paths": {
        "parsed_json_dir": "data/2_parsed_json",
        "meta_labels_csv": "data/meta/labels_step2.csv",
        "llm_labels_csv": "results/labels_step2_delay_outcome_llm.csv",
        "gold_seed_csv": "data/gold/gold65_v1.csv",
        "gold_out_csv": "data/gold/gold500_v1.csv",
        "gold_manifest_csv": "data/gold/gold500_sampling_manifest.csv",
        "gold_guideline_md": "data/gold/gold500_label_guideline_v1.md",
        "gold_qc_txt": "data/gold/gold500_qc_report.txt",
    },
    "random": {"seed": 2026},
    "gold_build": {
        "target_size": 500,
        "min_per_label": 120,
        "uncertain_ratio": 0.35,
        "conflict_ratio": 0.35,
        "tail_ratio": 0.30,
        "default_label_confidence": 0.72,
        "version": "gold500_v1",
    },
}


def load_cfg(p: Path):
    try:
        import yaml
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_CFG


def norm_label(x: str) -> str:
    x = str(x).strip().lower()
    m = {
        "support": "support", "partial": "partial", "not_support": "not_support",
        "not-support": "not_support", "notsupport": "not_support",
        "不支持": "not_support", "部分支持": "partial", "支持": "support",
        "unknown": "unknown", "": "unknown",
    }
    return m.get(x, "unknown")


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def read_decision_reasoning(parsed_dir: Path, cid: str) -> str:
    fp = parsed_dir / f"{cid}.json"
    if not fp.exists():
        return ""
    obj = json.loads(fp.read_text(encoding="utf-8"))
    sec = obj.get("sections", {}) or {}
    t = ((sec.get("decision") or "") + "\n" + (sec.get("reasoning") or "")).strip()
    return t if t else (sec.get("full_text") or "")


def evidence_span(text: str) -> str:
    if not text:
        return ""
    keys = ["判决如下", "裁决如下", "本院认为", "工期", "延误", "顺延", "赔偿", "违约金"]
    pos = 0
    for k in keys:
        i = text.find(k)
        if i >= 0:
            pos = i
            break
    return text[pos: pos + 220].replace("\n", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    random.seed(cfg["random"]["seed"])

    root = Path(cfg["project"]["root"]).resolve()
    p = cfg["paths"]
    target = int(cfg["gold_build"]["target_size"])

    parsed_dir = root / p["parsed_json_dir"]
    seed_csv = root / p["gold_seed_csv"]
    weak_csv = root / p["meta_labels_csv"]
    llm_csv = root / p["llm_labels_csv"]

    out_csv = root / p["gold_out_csv"]
    out_manifest = root / p["gold_manifest_csv"]
    out_guide = root / p["gold_guideline_md"]
    out_qc = root / p["gold_qc_txt"]

    seed = read_csv(seed_csv)
    weak = read_csv(weak_csv)
    llm = read_csv(llm_csv) if llm_csv.exists() else pd.DataFrame(columns=["case_id", "delay_money_label", "responsibility_hint"])

    seed = seed.rename(columns={"gpt_label": "gold_label"})
    if "gold_label" not in seed.columns:
        seed["gold_label"] = "unknown"
    seed["gold_label"] = seed["gold_label"].map(norm_label)
    seed["case_id"] = seed["case_id"].astype(str)

    llm = llm[[c for c in ["case_id", "delay_money_label", "responsibility_hint"] if c in llm.columns]].copy()
    if "case_id" not in llm.columns:
        llm["case_id"] = []
    if "delay_money_label" not in llm.columns:
        llm["delay_money_label"] = "unknown"
    if "responsibility_hint" not in llm.columns:
        llm["responsibility_hint"] = "unknown"
    llm["case_id"] = llm["case_id"].astype(str)
    llm["delay_money_label"] = llm["delay_money_label"].map(norm_label)

    weak = weak[["case_id", "source_file", "eot_label", "eot_evidence"]].copy()
    weak["case_id"] = weak["case_id"].astype(str)
    weak["eot_label"] = weak["eot_label"].map(norm_label)

    cands = weak.merge(llm, on="case_id", how="left")
    cands["delay_money_label"] = cands["delay_money_label"].fillna("unknown").map(norm_label)
    cands["responsibility_hint"] = cands["responsibility_hint"].fillna("unknown")

    seed_ids = set(seed["case_id"].tolist())
    cands = cands[~cands["case_id"].isin(seed_ids)].copy()

    cands["is_uncertain"] = ((cands["eot_label"] == "unknown") | (cands["delay_money_label"] == "unknown")).astype(int)
    cands["is_conflict"] = ((cands["eot_label"] != "unknown") & (cands["delay_money_label"] != "unknown") & (cands["eot_label"] != cands["delay_money_label"])).astype(int)
    cands["tail_boost"] = cands["eot_label"].isin(["support", "not_support"]).astype(int)

    need = max(0, target - len(seed))
    n_uncertain = int(need * cfg["gold_build"]["uncertain_ratio"])
    n_conflict = int(need * cfg["gold_build"]["conflict_ratio"])
    n_tail = need - n_uncertain - n_conflict

    part_u = cands[cands["is_uncertain"] == 1].sample(n=min(n_uncertain, (cands["is_uncertain"] == 1).sum()), random_state=cfg["random"]["seed"])
    left = cands[~cands["case_id"].isin(part_u["case_id"])].copy()
    part_c = left[left["is_conflict"] == 1].sample(n=min(n_conflict, (left["is_conflict"] == 1).sum()), random_state=cfg["random"]["seed"])
    left = left[~left["case_id"].isin(part_c["case_id"])].copy()
    part_t = left[left["tail_boost"] == 1].sample(n=min(n_tail, (left["tail_boost"] == 1).sum()), random_state=cfg["random"]["seed"])

    picked = pd.concat([
        part_u.assign(sample_strategy="uncertain"),
        part_c.assign(sample_strategy="conflict"),
        part_t.assign(sample_strategy="tail"),
    ], ignore_index=True)

    if len(picked) < need:
        missing = need - len(picked)
        left = cands[~cands["case_id"].isin(picked["case_id"])].copy()
        extra = left.sample(n=min(missing, len(left)), random_state=cfg["random"]["seed"]).assign(sample_strategy="random_fill")
        picked = pd.concat([picked, extra], ignore_index=True)

    picked = picked.drop_duplicates(subset=["case_id"]).head(need)

    def label_row(r):
        if r["eot_label"] != "unknown":
            return r["eot_label"]
        if r["delay_money_label"] != "unknown":
            return r["delay_money_label"]
        return "partial"

    picked["gold_label"] = picked.apply(label_row, axis=1)

    rows = []
    for _, r in picked.iterrows():
        cid = r["case_id"]
        txt = read_decision_reasoning(parsed_dir, cid)
        rows.append({
            "case_id": cid,
            "source_file": r.get("source_file", ""),
            "gold_label": r["gold_label"],
            "responsibility_gold": RESP_MAP.get(str(r.get("responsibility_hint", "unknown")), "unknown"),
            "evidence_span": evidence_span(txt),
            "label_confidence": cfg["gold_build"]["default_label_confidence"],
            "version": cfg["gold_build"]["version"],
        })

    picked_gold = pd.DataFrame(rows)

    seed_out = seed[[c for c in ["case_id", "source_file", "gold_label"] if c in seed.columns]].copy()
    seed_out["gold_label"] = seed_out["gold_label"].map(norm_label)
    seed_out["responsibility_gold"] = "unknown"
    seed_out["evidence_span"] = seed.get("gpt_evidence", "") if "gpt_evidence" in seed.columns else ""
    seed_out["label_confidence"] = seed.get("gpt_conf", cfg["gold_build"]["default_label_confidence"])
    seed_out["version"] = cfg["gold_build"]["version"]

    final = pd.concat([seed_out, picked_gold], ignore_index=True)
    final = final.drop_duplicates(subset=["case_id"], keep="first").head(target)

    vc = Counter(final["gold_label"].tolist())
    min_per = int(cfg["gold_build"]["min_per_label"])
    for lb in LABELS:
        if vc[lb] >= min_per:
            continue
        deficit = min_per - vc[lb]
        pool = cands[(cands["eot_label"] == lb) | (cands["delay_money_label"] == lb)]
        pool = pool[~pool["case_id"].isin(final["case_id"])].head(deficit)
        ext = []
        for _, r in pool.iterrows():
            txt = read_decision_reasoning(parsed_dir, r["case_id"])
            ext.append({
                "case_id": r["case_id"],
                "source_file": r.get("source_file", ""),
                "gold_label": lb,
                "responsibility_gold": RESP_MAP.get(str(r.get("responsibility_hint", "unknown")), "unknown"),
                "evidence_span": evidence_span(txt),
                "label_confidence": cfg["gold_build"]["default_label_confidence"],
                "version": cfg["gold_build"]["version"],
            })
        if ext:
            final = pd.concat([final, pd.DataFrame(ext)], ignore_index=True)

    final = final.drop_duplicates(subset=["case_id"], keep="first").head(target)

    manifest = picked[["case_id", "sample_strategy", "eot_label", "delay_money_label", "is_uncertain", "is_conflict", "tail_boost"]].copy()
    manifest.to_csv(out_manifest, index=False, encoding="utf-8-sig")
    final.to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_guide.write_text(
        "# gold500 labeling guideline v1\n- label set: support / partial / not_support\n- responsibility set: owner / contractor / subcontractor / designer_supervisor / both / force_majeure_policy / unknown\n- evidence_span should quote key disposition passage\n",
        encoding="utf-8",
    )

    qc_lines = [
        f"gold_output={out_csv}",
        f"total={len(final)}",
        "label_distribution:",
        final["gold_label"].value_counts(dropna=False).to_string(),
        "responsibility_distribution:",
        final["responsibility_gold"].value_counts(dropna=False).to_string(),
        f"missing_evidence_span={(final['evidence_span'].astype(str).str.len() == 0).sum()}",
    ]
    out_qc.write_text("\n".join(qc_lines), encoding="utf-8")
    print("[DONE]", out_csv)


if __name__ == "__main__":
    main()
