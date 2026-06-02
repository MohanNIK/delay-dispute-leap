# -*- coding: utf-8 -*-
"""Export collected public case CSV/JSONL rows into raw text files.

The project already stores the original 4,592 documents in data/1_raw_text.
For external public-document expansion, this script writes new cases into a
separate subfolder to avoid overwriting the original corpus while keeping all
raw-text assets under the same data/1_raw_text root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def safe_name(value: Any, max_len: int = 80) -> str:
    value = str(value or "")
    value = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", value, flags=re.UNICODE).strip("_")
    return value[:max_len] if value else "untitled"


def read_source(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    return pd.read_csv(path, encoding="utf-8-sig")


def first_existing(row: pd.Series, names: Iterable[str]) -> str:
    for name in names:
        if name in row and pd.notna(row[name]) and str(row[name]).strip():
            return str(row[name])
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV/JSONL file containing raw_text/full text")
    ap.add_argument("--out-dir", required=True, help="Output directory under data/1_raw_text")
    ap.add_argument("--prefix", default="mimo")
    ap.add_argument("--text-cols", default="raw_text,text,full_text,content")
    ap.add_argument("--dedupe", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    txt_dir = out_dir / "txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    df = read_source(in_path)
    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]

    records: List[Dict[str, Any]] = []
    seen_hashes = set()
    for idx, row in df.iterrows():
        text = first_existing(row, text_cols)
        if not text:
            continue
        text_hash = sha256_text(text)
        if args.dedupe and text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)

        pageid = first_existing(row, ["pageid", "id", "case_id"]) or str(idx + 1)
        title = first_existing(row, ["title", "case_title", "name"])
        case_id = f"{args.prefix}_{safe_name(pageid, 40)}_{text_hash[:12]}"
        txt_path = txt_dir / f"{case_id}.txt"
        txt_path.write_text(text, encoding="utf-8")

        rec = {
            "case_id": case_id,
            "source_row_index": int(idx),
            "source_file": str(in_path),
            "pageid": pageid,
            "title": title,
            "source_url": first_existing(row, ["source_url", "url"]),
            "case_year": first_existing(row, ["case_year", "year"]),
            "text_chars": len(text),
            "text_sha256": text_hash,
            "raw_text_path": str(txt_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        for col in [
            "usable_tier",
            "usage_scope",
            "mimo_status",
            "mimo_accept_flag",
            "strict_info_complete_flag",
            "overall_completeness_score",
            "evidence_completeness",
            "pre_decision_facts_sufficient",
            "project_management_relevance",
        ]:
            if col in row:
                rec[col] = row[col]
        records.append(rec)

    manifest = pd.DataFrame(records)
    manifest_path = out_dir / "raw_text_manifest.csv"
    jsonl_path = out_dir / "raw_text_manifest.jsonl"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(in_path),
        "out_dir": str(out_dir),
        "txt_dir": str(txt_dir),
        "exported_text_files": int(len(records)),
        "dedupe": bool(args.dedupe),
        "manifest": str(manifest_path),
        "manifest_jsonl": str(jsonl_path),
    }
    (out_dir / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
