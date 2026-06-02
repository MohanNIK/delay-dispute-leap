# -*- coding: utf-8 -*-
"""Stream-fetch public MediaWiki pages into sharded raw-text files.

This avoids loading tens of thousands of full texts into one in-memory
DataFrame. It is intended for the 100k expansion corpus under data/1_raw_text.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import requests

from collect_public_cases_mimo import (
    load_json_config,
    local_quality_score,
    mediawiki_get,
    sha256_text,
    strip_wikitext,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def append_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")


def chunks(items: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def normalize_pageid(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        return str(int(float(text)))
    except Exception:
        return text


def fetch_batch(cfg: Dict[str, Any], session: requests.Session, pageids: List[str]) -> List[Dict[str, Any]]:
    src = cfg["source"]
    api_url = src["api_url"]
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions|info",
        "rvprop": "content",
        "rvslots": "main",
        "inprop": "url",
        "pageids": "|".join(pageids),
    }
    data = mediawiki_get(session, api_url, params, timeout=90)
    pages = data.get("query", {}).get("pages", {})
    rows: List[Dict[str, Any]] = []
    min_chars = int(cfg["quality"].get("min_text_chars", 2500))
    for pageid, page in pages.items():
        title = page.get("title", "")
        revs = page.get("revisions", []) or []
        raw = ""
        if revs:
            slots = revs[0].get("slots", {})
            raw = slots.get("main", {}).get("*") or revs[0].get("*") or ""
        text = strip_wikitext(raw)
        if not text:
            rows.append({"pageid": str(pageid), "title": title, "fetch_status": "empty_text", "raw_text": ""})
            continue
        quality = local_quality_score(title, text, min_chars)
        rows.append(
            {
                "pageid": str(pageid),
                "title": title,
                "source_url": page.get("fullurl", ""),
                "fetch_status": "ok",
                "raw_text": text,
                "text_sha256": sha256_text(text),
                **quality,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/research_public_case_collection_mimo_100k_rawtext.json")
    ap.add_argument("--collect-dir", default="data/1_raw_text/mimo_public_100k_collect_20260527")
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--sleep-sec", type=float, default=0.15)
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument(
        "--min-search-hit-size",
        type=int,
        default=None,
        help="Override config quality.min_search_hit_size when fetching from a prepared queue.",
    )
    args = ap.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    cfg = load_json_config(cfg_path)
    collect_dir = PROJECT_ROOT / args.collect_dir
    collect_dir.mkdir(parents=True, exist_ok=True)
    hits_path = collect_dir / "search_hits.csv"
    if not hits_path.exists():
        inc = collect_dir / "search_hits_incremental.csv"
        if inc.exists():
            hits_path = inc
        else:
            raise SystemExit(f"missing search hits: {hits_path}")
    hits = read_csv(hits_path)
    if "pageid" not in hits.columns:
        raise SystemExit("search hits missing pageid")

    manifest_path = collect_dir / "raw_text_manifest_incremental.csv"
    audit_path = collect_dir / "local_quality_audit_incremental.csv"
    raw_csv_path = collect_dir / "raw_public_cases_incremental.csv"
    done = set()
    if manifest_path.exists():
        old = read_csv(manifest_path)
        if "pageid" in old.columns:
            done = set(old["pageid"].dropna().astype(str))

    if "api_status" in hits.columns:
        hits = hits[hits["api_status"].astype(str).eq("search_hit")].copy()
    if "search_size" in hits.columns:
        min_hit_size = int(args.min_search_hit_size if args.min_search_hit_size is not None else cfg["quality"].get("min_search_hit_size", 2500))
        hits = hits[pd.to_numeric(hits["search_size"], errors="coerce").fillna(0).ge(min_hit_size)].copy()
    pageids = [normalize_pageid(x) for x in hits["pageid"].dropna().tolist()]
    pageids = [x for x in pageids if x and x not in done]
    if args.max_pages > 0:
        pageids = pageids[: args.max_pages]

    session = requests.Session()
    session.headers.update({"User-Agent": cfg["source"].get("user_agent", "DelayDisputeCopilotResearch/1.0")})

    exported = len(done)
    batch_size = 50
    raw_rows_buffer: List[Dict[str, Any]] = []
    for batch in chunks(pageids, batch_size):
        rows = fetch_batch(cfg, session, batch)
        manifest_rows: List[Dict[str, Any]] = []
        audit_rows: List[Dict[str, Any]] = []
        for row in rows:
            raw_text = row.pop("raw_text", "")
            if row.get("fetch_status") == "ok" and raw_text:
                text_hash = row["text_sha256"]
                shard_idx = exported // args.shard_size
                shard_dir = collect_dir / "txt_shards" / f"shard_{shard_idx:04d}"
                shard_dir.mkdir(parents=True, exist_ok=True)
                case_id = f"mimo100k_{row['pageid']}_{text_hash[:12]}"
                txt_path = shard_dir / f"{case_id}.txt"
                txt_path.write_text(raw_text, encoding="utf-8")
                row["case_id"] = case_id
                row["raw_text_path"] = str(txt_path)
                row["text_chars"] = len(raw_text)
                exported += 1
                raw_rows_buffer.append({**row, "raw_text": raw_text})
            manifest_rows.append({k: v for k, v in row.items() if k != "raw_text"})
            audit_rows.append({k: v for k, v in row.items() if k not in {"raw_text_path"}})
        append_csv(manifest_path, manifest_rows)
        append_csv(audit_path, audit_rows)
        if len(raw_rows_buffer) >= 500:
            append_csv(raw_csv_path, raw_rows_buffer)
            raw_rows_buffer = []
        if exported % 500 == 0:
            print(json.dumps({"exported_text_files": exported, "remaining": len(pageids) - exported}, ensure_ascii=False), flush=True)
        time.sleep(args.sleep_sec)

    append_csv(raw_csv_path, raw_rows_buffer)
    manifest = read_csv(manifest_path)
    if not manifest.empty:
        manifest.drop_duplicates("text_sha256", keep="first").to_csv(collect_dir / "raw_text_manifest.csv", index=False, encoding="utf-8-sig")
    audit = read_csv(audit_path)
    if not audit.empty:
        audit.drop_duplicates("pageid", keep="last").to_csv(collect_dir / "local_quality_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "status": "complete",
        "search_hits_source": str(hits_path),
        "manifest_rows": int(len(manifest)) if not manifest.empty else 0,
        "exported_ok_text_files": int((manifest.get("fetch_status", pd.Series(dtype=str)) == "ok").sum()) if not manifest.empty else 0,
        "collect_dir": str(collect_dir),
    }
    (collect_dir / "stream_fetch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
