# -*- coding: utf-8 -*-
"""Build a Chinese-lexical schedule-delay usable pool from raw full texts.

This is an automated candidate-pool view. It is stricter than the broad
construction support corpus but does not imply human validation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def u(text: str) -> str:
    return text.encode("ascii").decode("unicode_escape")


CONSTRUCTION_TERMS = [
    u(r"\u5efa\u8bbe\u5de5\u7a0b"),  # construction project
    u(r"\u65bd\u5de5\u5408\u540c"),  # construction contract
    u(r"\u5de5\u7a0b\u6b3e"),  # project payment
    u(r"\u53d1\u5305\u4eba"),  # owner
    u(r"\u627f\u5305\u4eba"),  # contractor
    u(r"\u65bd\u5de5"),  # construction
    u(r"\u7ae3\u5de5"),  # completion
    u(r"\u5de5\u7a0b\u65bd\u5de5"),  # engineering construction
]

DELAY_TERMS = [
    u(r"\u5de5\u671f"),  # schedule period
    u(r"\u5de5\u671f\u5ef6\u8bef"),
    u(r"\u903e\u671f\u7ae3\u5de5"),
    u(r"\u5de5\u671f\u987a\u5ef6"),
    u(r"\u987a\u5ef6\u5de5\u671f"),
    u(r"\u505c\u5de5"),
    u(r"\u7a9d\u5de5"),
    u(r"\u505c\u5de5\u7a9d\u5de5"),
    u(r"\u7ae3\u5de5\u9a8c\u6536"),
    u(r"\u5ef6\u671f\u4ea4\u4ed8"),
    u(r"\u5ef6\u8bef\u635f\u5931"),
    u(r"\u5de5\u671f\u7d22\u8d54"),
    u(r"\u65bd\u5de5\u8fdb\u5ea6"),
    u(r"\u5173\u952e\u7ebf\u8def"),
    u(r"\u8bef\u5de5"),
    u(r"\u5ef6\u671f"),
    u(r"\u5ef6\u8bef"),
]

EVIDENCE_TERMS = [
    u(r"\u7b7e\u8bc1"),
    u(r"\u65bd\u5de5\u65e5\u5fd7"),
    u(r"\u4f1a\u8bae\u7eaa\u8981"),
    u(r"\u5f00\u5de5\u4ee4"),
    u(r"\u7ae3\u5de5\u62a5\u544a"),
    u(r"\u9274\u5b9a\u610f\u89c1"),
    u(r"\u5de5\u671f\u9274\u5b9a"),
    u(r"\u5de5\u7a0b\u7ed3\u7b97"),
    u(r"\u7d22\u8d54"),
    u(r"\u8fdd\u7ea6\u91d1"),
]

DECISION_TERMS = [
    u(r"\u672c\u9662\u8ba4\u4e3a"),
    u(r"\u6cd5\u9662\u8ba4\u4e3a"),
    u(r"\u5224\u51b3\u5982\u4e0b"),
    u(r"\u88c1\u5224\u7ed3\u679c"),
    u(r"\u4e00\u5ba1"),
    u(r"\u4e8c\u5ba1"),
    u(r"\u6c11\u4e8b\u5224\u51b3\u4e66"),
]

PROCEDURAL_ONLY_TERMS = [
    u(r"\u7ba1\u8f96\u6743\u5f02\u8bae"),
    u(r"\u6267\u884c\u88c1\u5b9a"),
    u(r"\u64a4\u8bc9"),
    u(r"\u9a73\u56de\u8d77\u8bc9"),
]


def resolve(path: object) -> Path:
    p = Path(str(path or ""))
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False) if path.exists() else pd.DataFrame()


def count_terms(text: str, terms: Iterable[str]) -> int:
    return sum(1 for term in terms if term in text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-dir", default="data/1_raw_text/combined_delay_dispute_corpus_20260527")
    ap.add_argument("--max-text-chars", type=int, default=50000)
    args = ap.parse_args()

    combined_dir = resolve(args.combined_dir)
    manifest = read_csv(combined_dir / "combined_raw_text_manifest_dedup.csv")
    rows = []
    for idx, row in manifest.iterrows():
        raw_path = resolve(row.get("raw_text_path", ""))
        text = ""
        if raw_path.exists():
            try:
                text = raw_path.read_text(encoding="utf-8", errors="ignore")[: args.max_text_chars]
            except Exception:
                text = ""
        title = str(row.get("title", "") or "")
        probe = f"{title}\n{text}"
        construction_hits = count_terms(probe, CONSTRUCTION_TERMS)
        delay_hits = count_terms(probe, DELAY_TERMS)
        evidence_hits = count_terms(probe, EVIDENCE_TERMS)
        decision_hits = count_terms(probe, DECISION_TERMS)
        procedural_hits = count_terms(title + "\n" + text[:3000], PROCEDURAL_ONLY_TERMS)
        text_chars = int(pd.to_numeric(row.get("text_chars", 0), errors="coerce") or len(text))
        delay_core = delay_hits >= 2 or any(term in probe for term in DELAY_TERMS[:8])
        research_candidate = (
            text_chars >= 1200
            and construction_hits >= 1
            and delay_hits >= 1
            and decision_hits >= 1
            and procedural_hits == 0
        )
        usable = (
            text_chars >= 1800
            and construction_hits >= 1
            and delay_core
            and decision_hits >= 1
            and procedural_hits == 0
        )
        usable_relaxed = (
            text_chars >= 1500
            and construction_hits >= 1
            and delay_core
            and decision_hits >= 1
            and procedural_hits == 0
        )
        strong = usable and evidence_hits >= 1 and decision_hits >= 2
        rows.append(
            {
                "case_id": row.get("case_id", ""),
                "pageid": row.get("pageid", ""),
                "title": title,
                "raw_text_path": row.get("raw_text_path", ""),
                "source_batch": row.get("source_batch", ""),
                "text_sha256": row.get("text_sha256", ""),
                "text_chars": text_chars,
                "chinese_construction_hits": construction_hits,
                "chinese_delay_hits": delay_hits,
                "chinese_evidence_hits": evidence_hits,
                "chinese_decision_hits": decision_hits,
                "chinese_procedural_only_hits": procedural_hits,
                "chinese_delay_research_candidate_flag": int(research_candidate),
                "chinese_delay_usable_relaxed_flag": int(usable_relaxed),
                "chinese_delay_usable_flag": int(usable),
                "chinese_delay_strong_flag": int(strong),
            }
        )
        if idx and idx % 10000 == 0:
            print(json.dumps({"scored": int(idx), "usable_so_far": int(sum(r["chinese_delay_usable_flag"] for r in rows))}), flush=True)

    scored = pd.DataFrame(rows)
    research_candidate = scored[scored["chinese_delay_research_candidate_flag"].eq(1)].copy()
    usable_relaxed = scored[scored["chinese_delay_usable_relaxed_flag"].eq(1)].copy()
    usable = scored[scored["chinese_delay_usable_flag"].eq(1)].copy()
    strong = scored[scored["chinese_delay_strong_flag"].eq(1)].copy()
    scored.to_csv(combined_dir / "chinese_delay_rescored_manifest.csv", index=False, encoding="utf-8-sig")
    research_candidate.to_csv(combined_dir / "chinese_delay_research_candidate_manifest.csv", index=False, encoding="utf-8-sig")
    usable_relaxed.to_csv(combined_dir / "chinese_delay_usable_relaxed_manifest.csv", index=False, encoding="utf-8-sig")
    usable.to_csv(combined_dir / "chinese_delay_usable_manifest.csv", index=False, encoding="utf-8-sig")
    strong.to_csv(combined_dir / "chinese_delay_strong_manifest.csv", index=False, encoding="utf-8-sig")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifest": str(combined_dir / "combined_raw_text_manifest_dedup.csv"),
        "scored_rows": int(len(scored)),
        "chinese_delay_research_candidate_total": int(len(research_candidate)),
        "chinese_delay_usable_relaxed_total": int(len(usable_relaxed)),
        "chinese_delay_usable_total": int(len(usable)),
        "chinese_delay_strong_total": int(len(strong)),
        "note": "Automated Chinese lexical schedule-delay pools; research_candidate is broad, usable is stricter, strong is stricter still. Not human validation.",
    }
    (combined_dir / "chinese_delay_pool_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
