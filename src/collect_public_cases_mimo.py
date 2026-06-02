# -*- coding: utf-8 -*-
"""Collect public construction-delay dispute candidates and screen them with Mimo.

This collector uses publicly accessible MediaWiki API pages from Chinese
Wikisource as a document source. Mimo/OpenAI-compatible API is used only for
quality screening and dispute relevance assessment; it does not fabricate cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]


CONSTRUCTION_TERMS = [
    "????",
    "????",
    "????",
    "???",
    "???",
    "???",
    "??",
    "??",
    "??",
    "???",
    "??",
    "???",
]
DELAY_TERMS = [
    "????",
    "????",
    "????",
    "????",
    "????",
    "????",
    "????",
    "??",
    "??",
    "????",
    "???",
    "????",
    "?????",
    "??",
    "??",
    "??",
]
EVIDENCE_TERMS = [
    "??",
    "??",
    "??",
    "????",
    "????",
    "????",
    "????",
    "??",
    "??",
    "????",
    "???",
    "??",
    "?????",
    "??",
]
DECISION_TERMS = ["????", "????", "????", "????", "??", "??", "??", "??"]
PROCEDURAL_TERMS = ["?????", "????", "?????", "????", "????", "????"]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def compact_for_mimo(title: str, text: str, max_chars: int) -> str:
    text = clean_ws(text)
    if len(text) <= max_chars:
        return text
    # Keep document beginning, decision area, and delay/evidence-heavy snippets.
    chunks: List[str] = [text[: int(max_chars * 0.28)]]
    for marker in ["本院认为", "法院认为", "判决如下", "裁判结果"]:
        pos = text.find(marker)
        if pos >= 0:
            chunks.append(text[max(0, pos - 1200) : pos + 2200])
            break
    hits: List[str] = []
    for term in DELAY_TERMS + EVIDENCE_TERMS:
        pos = text.find(term)
        if pos >= 0:
            hits.append(text[max(0, pos - 450) : pos + 900])
        if sum(len(x) for x in hits) > int(max_chars * 0.35):
            break
    chunks.extend(hits)
    chunks.append(text[-int(max_chars * 0.14) :])
    out = clean_ws("\n".join(chunks))
    return out[:max_chars]


def count_hits(text: str, terms: Sequence[str]) -> int:
    return sum(1 for t in terms if t in text)


def infer_year(text: str, title: str = "") -> str:
    m = re.search(r"[（(](20\d{2}|19\d{2})[）)]", title)
    if m:
        return m.group(1)
    m = re.search(r"(20\d{2}|19\d{2})年", text[:3000])
    return m.group(1) if m else ""


def local_quality_score(title: str, text: str, min_text_chars: int) -> Dict[str, Any]:
    title = str(title or "")
    text = str(text or "")
    text_len = len(text)
    construction_hits = count_hits(title + text[:10000], CONSTRUCTION_TERMS)
    delay_hits = count_hits(title + text[:20000], DELAY_TERMS)
    evidence_hits = count_hits(text[:30000], EVIDENCE_TERMS)
    decision_hits = count_hits(text, DECISION_TERMS)
    procedural_hits = count_hits(title + text[:5000], PROCEDURAL_TERMS)
    judgment_title = bool(("判决书" in title) and ("裁定书" not in title))
    substantive = text_len >= min_text_chars and decision_hits >= 2
    procedural_only = procedural_hits >= 1 and delay_hits == 0
    score = 0.0
    score += min(construction_hits, 4) * 0.12
    score += min(delay_hits, 4) * 0.16
    score += min(evidence_hits, 5) * 0.05
    score += min(decision_hits, 4) * 0.06
    score += 0.12 if judgment_title else 0.0
    score += 0.12 if substantive else 0.0
    score -= 0.25 if procedural_only else 0.0
    score = max(0.0, min(1.0, score))
    return {
        "text_chars": text_len,
        "case_year": infer_year(text, title),
        "construction_hits": construction_hits,
        "delay_hits": delay_hits,
        "evidence_hits": evidence_hits,
        "decision_hits": decision_hits,
        "procedural_hits": procedural_hits,
        "judgment_title_flag": int(judgment_title),
        "substantive_decision_flag": int(substantive),
        "procedural_only_flag": int(procedural_only),
        "local_quality_score": round(score, 4),
    }


def mediawiki_get(session: requests.Session, api_url: str, params: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            resp = session.get(api_url, params=params, timeout=timeout)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else min(90.0, 8.0 * (attempt + 1))
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_error = exc
            time.sleep(min(60.0, 3.0 * (attempt + 1)))
    raise last_error if last_error else RuntimeError("MediaWiki request failed")


def collect_search_hits(cfg: Dict[str, Any], out_dir: Path, max_total: int) -> pd.DataFrame:
    src = cfg["source"]
    api_url = src["api_url"]
    limit = int(src.get("search_limit_per_call", 50))
    sleep_sec = float(src.get("sleep_between_requests_sec", 0.2))
    session = requests.Session()
    session.headers.update({"User-Agent": src.get("user_agent", "DelayDisputeCopilotResearch/1.0")})
    inc_path = out_dir / "search_hits_incremental.csv"
    rows: List[Dict[str, Any]] = []
    seen_pageids = set()
    if inc_path.exists():
        try:
            existing = pd.read_csv(inc_path, encoding="utf-8-sig")
            rows = existing.to_dict("records")
            if "pageid" in existing.columns:
                seen_pageids = set(existing["pageid"].dropna().astype(str).tolist())
        except Exception:
            rows = []
            seen_pageids = set()
    for query in src["queries"]:
        offset = 0
        while len(seen_pageids) < max_total:
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "srnamespace": 0,
                "sroffset": offset,
            }
            try:
                obj = mediawiki_get(session, api_url, params)
            except Exception as exc:
                rows.append({"query": query, "pageid": "", "title": "", "api_status": "search_error", "error": str(exc)[:500]})
                break
            hits = obj.get("query", {}).get("search", [])
            if not hits:
                break
            for item in hits:
                pageid = str(item.get("pageid", ""))
                if not pageid or pageid in seen_pageids:
                    continue
                seen_pageids.add(pageid)
                rows.append(
                    {
                        "query": query,
                        "pageid": pageid,
                        "title": item.get("title", ""),
                        "search_size": item.get("size", 0),
                        "wordcount": item.get("wordcount", 0),
                        "snippet": re.sub(r"<[^>]+>", "", item.get("snippet", "")),
                        "source_url": f"https://zh.wikisource.org/wiki/{requests.utils.quote(str(item.get('title', '')).replace(' ', '_'))}",
                        "api_status": "search_hit",
                    }
                )
                if len(seen_pageids) >= max_total:
                    break
            pd.DataFrame(rows).to_csv(out_dir / "search_hits_incremental.csv", index=False, encoding="utf-8-sig")
            cont = obj.get("continue", {})
            if "sroffset" not in cont:
                break
            offset = int(cont["sroffset"])
            time.sleep(sleep_sec)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "search_hits.csv", index=False, encoding="utf-8-sig")
    return df


def strip_wikitext(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\{\{Header/裁判文书[\s\S]*?\}\}\s*", "", text, count=1)
    text = re.sub(r"\{\{[^{}]{0,200}\}\}", " ", text)
    text = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"'{2,}", "", text)
    return re.sub(r"\s+", "\n", text).strip()


def fetch_revision_batch(session: requests.Session, api_url: str, pageids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch full wikitext for multiple pages.

    The extracts API only returns one whole article per request. Revisions can
    return multiple full page contents, which is needed for large-scale public
    document collection.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions|info",
        "rvprop": "content",
        "rvslots": "main",
        "inprop": "url",
        "pageids": "|".join(pageids),
    }
    obj = mediawiki_get(session, api_url, params, timeout=90)
    return obj.get("query", {}).get("pages", {})


def fetch_extract_single(session: requests.Session, api_url: str, pageid: str) -> Dict[str, Any]:
    """Fetch one full extract.

    MediaWiki lowers exlimit to 1 for whole-article extracts. Fetching one page
    at a time avoids silently receiving empty extracts for most pageids.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "explaintext": 1,
        "exsectionformat": "plain",
        "inprop": "url",
        "pageids": str(pageid),
    }
    obj = mediawiki_get(session, api_url, params, timeout=90)
    pages = obj.get("query", {}).get("pages", {})
    return next(iter(pages.values())) if pages else {}


def fetch_extracts(cfg: Dict[str, Any], hits: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    src = cfg["source"]
    api_url = src["api_url"]
    sleep_sec = float(src.get("sleep_between_requests_sec", 0.2))
    min_hit_size = int(cfg["quality"].get("min_search_hit_size", 3000))
    if "search_size" in hits.columns:
        search_size = pd.to_numeric(hits["search_size"], errors="coerce").fillna(0)
    else:
        search_size = pd.Series([0] * len(hits), index=hits.index)
    candidates = hits[(hits["api_status"].eq("search_hit")) & (search_size >= min_hit_size)].copy()
    pageids = candidates["pageid"].astype(str).drop_duplicates().tolist()
    session = requests.Session()
    session.headers.update({"User-Agent": src.get("user_agent", "DelayDisputeCopilotResearch/1.0")})
    rows: List[Dict[str, Any]] = []
    for i in range(0, len(pageids), 50):
        batch = pageids[i : i + 50]
        try:
            pages = fetch_revision_batch(session, api_url, [str(x) for x in batch])
            for pageid, page in pages.items():
                title = page.get("title", "")
                rev = (page.get("revisions") or [{}])[0]
                slot = (rev.get("slots") or {}).get("main", {})
                content = slot.get("*") or slot.get("content") or ""
                text = strip_wikitext(content)
                rows.append(
                    {
                        "pageid": str(pageid),
                        "title": title,
                        "source_url": page.get("fullurl", f"https://zh.wikisource.org/wiki/{requests.utils.quote(str(title).replace(' ', '_'))}"),
                        "raw_text": text,
                        "text_sha256": sha256_text(text),
                        "fetch_status": "ok" if text else "empty_revision",
                    }
                )
        except Exception as exc:
            for pageid in batch:
                rows.append({"pageid": str(pageid), "title": "", "source_url": "", "raw_text": "", "text_sha256": "", "fetch_status": "fetch_error", "error": str(exc)[:500]})
        done = min(i + len(batch), len(pageids))
        if done % 250 == 0 or done == len(pageids):
            pd.DataFrame(rows).to_csv(out_dir / "raw_public_cases_incremental.csv", index=False, encoding="utf-8-sig")
            print(f"FETCHED_REVISIONS {done}/{len(pageids)}", flush=True)
        time.sleep(sleep_sec)
    df = pd.DataFrame(rows)
    df = df.drop_duplicates("text_sha256").reset_index(drop=True)
    df.to_csv(out_dir / "raw_public_cases.csv", index=False, encoding="utf-8-sig")
    return df


def local_prefilter(cfg: Dict[str, Any], raw: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    min_text = int(cfg["quality"].get("min_text_chars", 2500))
    threshold = float(cfg["quality"].get("local_candidate_threshold", 0.55))
    rows = []
    for _, row in raw.iterrows():
        q = local_quality_score(str(row.get("title", "")), str(row.get("raw_text", "")), min_text)
        keep = q["local_quality_score"] >= threshold
        if cfg["quality"].get("require_judgment_title", True) and not q["judgment_title_flag"]:
            keep = False
        if cfg["quality"].get("drop_procedural_only", True) and q["procedural_only_flag"]:
            keep = False
        if q["text_chars"] < min_text:
            keep = False
        rows.append({**row.to_dict(), **q, "local_keep_flag": int(keep)})
    if not rows:
        df = pd.DataFrame(columns=list(raw.columns) + [
            "text_chars",
            "case_year",
            "construction_hits",
            "delay_hits",
            "evidence_hits",
            "decision_hits",
            "procedural_hits",
            "judgment_title_flag",
            "substantive_decision_flag",
            "procedural_only_flag",
            "local_quality_score",
            "local_keep_flag",
        ])
        df.to_csv(out_dir / "local_quality_audit.csv", index=False, encoding="utf-8-sig")
        df.to_csv(out_dir / "candidate_pool_prefilter.csv", index=False, encoding="utf-8-sig")
        return df
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "local_quality_audit.csv", index=False, encoding="utf-8-sig")
    kept = df[df["local_keep_flag"].eq(1)].copy().reset_index(drop=True)
    kept.to_csv(out_dir / "candidate_pool_prefilter.csv", index=False, encoding="utf-8-sig")
    return kept


def get_mimo_settings(cfg: Dict[str, Any]) -> Tuple[str, str, str]:
    mc = cfg["mimo"]
    base_url = os.getenv(str(mc.get("base_url_env", "MIMO_OPENAI_BASE_URL")), "").strip().rstrip("/")
    api_key = os.getenv(str(mc.get("api_key_env", "MIMO_API_KEY")), "").strip()
    model = str(mc.get("model_name", "gpt-5.5"))
    return base_url, api_key, model


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def build_mimo_messages(row: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    text = compact_for_mimo(str(row.get("title", "")), str(row.get("raw_text", "")), int(cfg["mimo"].get("max_chars_for_screening", 8000)))
    payload = {
        "task": "screen_public_case_for_delay_dispute_database",
        "important": [
            "Do not invent facts.",
            "Assess whether this public document is complete enough for a construction schedule-delay dispute corpus.",
            "Reject procedural-only, incomplete, non-construction, non-delay, or unclear documents.",
            "This is candidate collection, not human gold labeling.",
        ],
        "required_json": {
            "case_related_to_construction": "boolean",
            "schedule_delay_material_issue": "boolean",
            "substantive_facts_available": "boolean",
            "adjudicated_outcome_available": "boolean",
            "procedural_only": "boolean",
            "pre_decision_facts_sufficient": "float 0-1",
            "evidence_completeness": "float 0-1",
            "overall_completeness_score": "float 0-1",
            "recommended_bucket": "accept|rag_only|reject",
            "project_management_relevance": "float 0-1",
            "main_delay_terms": ["short terms"],
            "evidence_types_found": ["notice", "visa", "schedule", "site_log", "meeting_minutes", "expert_opinion", "correspondence", "other"],
            "reason_short": "short Chinese reason",
        },
        "title": row.get("title", ""),
        "local_quality_score": row.get("local_quality_score", ""),
        "document_text": text,
    }
    prompt = json.dumps(payload, ensure_ascii=False)
    return [
        {"role": "system", "content": "Return exactly one valid JSON object. No Markdown."},
        {"role": "user", "content": prompt},
    ], prompt


def call_mimo(base_url: str, api_key: str, model: str, messages: List[Dict[str, str]], cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(cfg["mimo"].get("temperature", 0.0)),
        "max_tokens": int(cfg["mimo"].get("max_tokens", 900)),
    }
    if bool(cfg["mimo"].get("disable_thinking", False)):
        payload["thinking"] = {"type": "disabled"}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=int(cfg["mimo"].get("timeout_sec", 120)))
    resp.raise_for_status()
    obj = resp.json()
    return obj["choices"][0]["message"]["content"], obj.get("usage", {})


def screen_one(row: Dict[str, Any], cfg: Dict[str, Any], base_url: str, api_key: str, model: str) -> Dict[str, Any]:
    started = time.time()
    messages, prompt = build_mimo_messages(row, cfg)
    last_error = ""
    for attempt in range(int(cfg["mimo"].get("retries", 1)) + 1):
        try:
            raw, usage = call_mimo(base_url, api_key, model, messages, cfg)
            parsed = extract_json_object(raw)
            score = float(parsed.get("overall_completeness_score", 0.0) or 0.0)
            accept_threshold = float(cfg["quality"].get("mimo_accept_threshold", 0.72))
            bucket = str(parsed.get("recommended_bucket", "reject")).strip()
            reason_short = str(parsed.get("reason_short", ""))
            incomplete_risk = bool(re.search(r"不完整|不足|缺失|不充分|不清楚|无法|不明确", reason_short))
            evidence_completeness = float(parsed.get("evidence_completeness", 0.0) or 0.0)
            pre_decision_sufficiency = float(parsed.get("pre_decision_facts_sufficient", 0.0) or 0.0)
            accepted = (
                bucket == "accept"
                and score >= accept_threshold
                and evidence_completeness >= 0.65
                and pre_decision_sufficiency >= 0.65
                and bool(parsed.get("case_related_to_construction", False))
                and bool(parsed.get("schedule_delay_material_issue", False))
                and bool(parsed.get("substantive_facts_available", False))
                and bool(parsed.get("adjudicated_outcome_available", False))
                and not bool(parsed.get("procedural_only", True))
                and not incomplete_risk
            )
            return {
                "pageid": row.get("pageid", ""),
                "title": row.get("title", ""),
                "source_url": row.get("source_url", ""),
                "mimo_status": "ok",
                "model_name": model,
                "attempts": attempt + 1,
                "latency_sec": round(time.time() - started, 4),
                "prompt_sha256": sha256_text(prompt),
                "prompt_chars": len(prompt),
                "mimo_accept_flag": int(accepted),
                "case_related_to_construction": int(bool(parsed.get("case_related_to_construction", False))),
                "schedule_delay_material_issue": int(bool(parsed.get("schedule_delay_material_issue", False))),
                "substantive_facts_available": int(bool(parsed.get("substantive_facts_available", False))),
                "adjudicated_outcome_available": int(bool(parsed.get("adjudicated_outcome_available", False))),
                "procedural_only": int(bool(parsed.get("procedural_only", True))),
                "pre_decision_facts_sufficient": pre_decision_sufficiency,
                "evidence_completeness": evidence_completeness,
                "overall_completeness_score": score,
                "recommended_bucket": bucket,
                "project_management_relevance": parsed.get("project_management_relevance", 0),
                "main_delay_terms": json.dumps(parsed.get("main_delay_terms", []), ensure_ascii=False),
                "evidence_types_found": json.dumps(parsed.get("evidence_types_found", []), ensure_ascii=False),
                "reason_short": str(parsed.get("reason_short", ""))[:500],
                "usage_json": json.dumps(usage, ensure_ascii=False),
                "raw_response": raw,
            }
        except Exception as exc:
            last_error = str(exc)[:1000]
            if attempt < int(cfg["mimo"].get("retries", 1)):
                # Mimo can return burst-rate 429s. Back off long enough that
                # retry attempts do not amplify the throttle.
                if "429" in last_error or "Too Many Requests" in last_error:
                    time.sleep(min(45.0, 8.0 * (attempt + 1)))
                else:
                    time.sleep(1 + attempt)
    return {
        "pageid": row.get("pageid", ""),
        "title": row.get("title", ""),
        "source_url": row.get("source_url", ""),
        "mimo_status": "api_error",
        "model_name": model,
        "latency_sec": round(time.time() - started, 4),
        "error": last_error,
    }


def run_mimo_screening(cfg: Dict[str, Any], pool: pd.DataFrame, out_dir: Path, max_screen_cases: int, resume: bool) -> pd.DataFrame:
    base_url, api_key, model = get_mimo_settings(cfg)
    if not base_url or not api_key:
        pd.DataFrame([{"mimo_status": "api_unavailable", "error": "missing MIMO base_url or api_key"}]).to_csv(out_dir / "mimo_screening_results.csv", index=False, encoding="utf-8-sig")
        return pd.DataFrame()
    existing = pd.DataFrame()
    out_path = out_dir / "mimo_screening_results.csv"
    done = set()
    if resume and out_path.exists():
        existing = pd.read_csv(out_path, encoding="utf-8-sig")
        done = set(existing.get("pageid", pd.Series(dtype=str)).astype(str).tolist())
    todo = pool[~pool["pageid"].astype(str).isin(done)].copy()
    if max_screen_cases > 0:
        todo = todo.head(max_screen_cases)
    rows = existing.to_dict("records") if not existing.empty else []
    workers = int(cfg["mimo"].get("workers", 8))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(screen_one, row.to_dict(), cfg, base_url, api_key, model) for _, row in todo.iterrows()]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            rows.append(rec)
            if i % 20 == 0 or i == len(futs):
                pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
                print(f"MIMO_SCREENED {len(rows)} total, latest_status={rec.get('mimo_status')}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def export_accepted(pool: pd.DataFrame, screening: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if screening.empty:
        accepted = pool.copy()
        accepted["mimo_accept_flag"] = 0
        accepted.to_csv(out_dir / "mimo_delay_dispute_candidates.csv", index=False, encoding="utf-8-sig")
        return accepted
    cols = [
        "pageid",
        "mimo_status",
        "model_name",
        "mimo_accept_flag",
        "case_related_to_construction",
        "schedule_delay_material_issue",
        "substantive_facts_available",
        "adjudicated_outcome_available",
        "procedural_only",
        "pre_decision_facts_sufficient",
        "evidence_completeness",
        "overall_completeness_score",
        "recommended_bucket",
        "project_management_relevance",
        "main_delay_terms",
        "evidence_types_found",
        "reason_short",
    ]
    merged = pool.merge(screening[[c for c in cols if c in screening.columns]], on="pageid", how="left")
    if "mimo_accept_flag" in merged.columns:
        accept_flag = pd.to_numeric(merged["mimo_accept_flag"], errors="coerce").fillna(0).astype(int)
    else:
        accept_flag = pd.Series([0] * len(merged), index=merged.index)
    accepted = merged[accept_flag.eq(1)].copy()
    accepted.to_csv(out_dir / "mimo_delay_dispute_candidates.csv", index=False, encoding="utf-8-sig")
    # Export a text-free metadata version and a raw-text JSONL for downstream parsing.
    meta_cols = [c for c in accepted.columns if c != "raw_text"]
    accepted[meta_cols].to_csv(out_dir / "mimo_delay_dispute_candidates_manifest.csv", index=False, encoding="utf-8-sig")
    with (out_dir / "mimo_delay_dispute_candidates_raw.jsonl").open("w", encoding="utf-8") as fh:
        for _, row in accepted.iterrows():
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    return accepted


def write_run_manifest(cfg_path: Path, cfg: Dict[str, Any], out_dir: Path, stats: Dict[str, Any]) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config_path": str(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "source_provider": cfg["source"].get("provider"),
        "source_api_url": cfg["source"].get("api_url"),
        "mimo_base_url_env": cfg["mimo"].get("base_url_env"),
        "mimo_api_key_env": cfg["mimo"].get("api_key_env"),
        "mimo_api_key_present": bool(os.getenv(str(cfg["mimo"].get("api_key_env", "MIMO_API_KEY")), "").strip()),
        "mimo_model_name": cfg["mimo"].get("model_name"),
        "stats": stats,
        "artifact_hashes": {},
    }
    for name in [
        "search_hits.csv",
        "raw_public_cases.csv",
        "local_quality_audit.csv",
        "candidate_pool_prefilter.csv",
        "mimo_screening_results.csv",
        "mimo_delay_dispute_candidates.csv",
        "mimo_delay_dispute_candidates_manifest.csv",
        "mimo_delay_dispute_candidates_raw.jsonl",
    ]:
        path = out_dir / name
        manifest["artifact_hashes"][name] = sha256_file(path)
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/research_public_case_collection_mimo.json")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--max-total-candidates", type=int, default=0)
    ap.add_argument("--max-screen-cases", type=int, default=0, help="0 means screen all locally kept candidates")
    ap.add_argument("--skip-mimo", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--search-only", action="store_true", help="Only collect/resume search hits, then stop before full-text download")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    cfg = load_json_config(cfg_path)
    max_total = args.max_total_candidates or int(cfg["source"].get("max_total_candidates", 30000))
    out_root = PROJECT_ROOT / cfg["paths"].get("output_root", "data/external_mimo_candidates")
    out_dir = Path(args.out_dir) if args.out_dir else out_root / f"wikisource_mimo_{now_stamp()}"
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    ensure_dir(out_dir)

    hits_path = out_dir / "search_hits.csv"
    raw_path = out_dir / "raw_public_cases.csv"
    pool_path = out_dir / "candidate_pool_prefilter.csv"
    hits_inc_path = out_dir / "search_hits_incremental.csv"
    if args.resume and hits_path.exists():
        hits = pd.read_csv(hits_path, encoding="utf-8-sig")
    elif args.resume and hits_inc_path.exists():
        # Treat the incremental file as a recoverable search state.
        hits = collect_search_hits(cfg, out_dir, max_total)
    else:
        hits = collect_search_hits(cfg, out_dir, max_total)
    if args.search_only:
        stats = {
            "search_hits": int(len(hits)),
            "raw_public_cases": 0,
            "local_prefilter_kept": 0,
            "mimo_screened": 0,
            "mimo_accepted": 0,
            "out_dir": str(out_dir),
            "mode": "search_only",
        }
        write_run_manifest(cfg_path, cfg, out_dir, stats)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return
    if args.resume and raw_path.exists():
        raw = pd.read_csv(raw_path, encoding="utf-8-sig")
        # If search was extended after raw extraction, fetch only missing pageids.
        if "pageid" in hits.columns and "pageid" in raw.columns:
            have = set(raw["pageid"].dropna().astype(str))
            missing_hits = hits[~hits["pageid"].astype(str).isin(have)].copy()
            if not missing_hits.empty:
                missing_raw = fetch_extracts(cfg, missing_hits, out_dir)
                raw = pd.concat([raw, missing_raw], ignore_index=True)
                raw = raw.drop_duplicates("text_sha256").reset_index(drop=True)
                raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    else:
        raw = fetch_extracts(cfg, hits, out_dir)
    if args.resume and pool_path.exists():
        pool = pd.read_csv(pool_path, encoding="utf-8-sig")
        if "pageid" in raw.columns and "pageid" in pool.columns:
            raw_ids = set(raw["pageid"].dropna().astype(str))
            pool_ids = set(pool["pageid"].dropna().astype(str))
            if not raw_ids.issubset(pool_ids):
                pool = local_prefilter(cfg, raw, out_dir)
    else:
        pool = local_prefilter(cfg, raw, out_dir)

    if args.collect_only or args.skip_mimo or not bool(cfg["mimo"].get("enabled", True)):
        screening = pd.DataFrame()
    else:
        screening = run_mimo_screening(cfg, pool, out_dir, args.max_screen_cases, resume=args.resume)
    accepted = export_accepted(pool, screening, out_dir)
    stats = {
        "search_hits": int(len(hits)),
        "raw_public_cases": int(len(raw)),
        "local_prefilter_kept": int(len(pool)),
        "mimo_screened": int(len(screening)) if not screening.empty else 0,
        "mimo_accepted": int(len(accepted[accepted.get("mimo_accept_flag", 0).fillna(0).astype(int).eq(1)])) if "mimo_accept_flag" in accepted.columns else 0,
        "out_dir": str(out_dir),
    }
    write_run_manifest(cfg_path, cfg, out_dir, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
