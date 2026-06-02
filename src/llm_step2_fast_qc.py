# -*- coding: utf-8 -*-
"""
LLM Step2 (Qwen): Delay-dispute outcome labeling (FAST + QC)
INPUT : data/2_parsed_json/*.json
TEXT  : decision+reasoning preferred, fallback full_text
OUTPUT:
  - results/llm_case_json/{case_id}.json
  - results/labels_step2_delay_outcome_llm.csv
  - results/llm_errors.csv
  - results/qc_report_delay_outcome.txt
  - results/qc_flags_delay_outcome.csv

Speed-ups:
- ThreadPoolExecutor concurrency
- shorter prompt text via decision-tail + keyword windows + sentence selection
- resume mode
"""

import os
import json
import time
import argparse
import random
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = PROJECT_ROOT / "data" / "2_parsed_json"
RESULTS_DIR = PROJECT_ROOT / "results"
CASE_JSON_DIR = RESULTS_DIR / "llm_case_json"


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


# =========================
# Qwen Client (OpenAI-compatible)
# =========================
class QwenClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 120):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 700) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# =========================
# Prompt
# =========================
SYSTEM_MSG = "你是建设工程合同纠纷裁判文书分析专家。只输出 JSON。"

SCHEMA_NOTE = """
你必须只输出 JSON，不要解释，不要 Markdown。

Fields:
- delay_money_label: support | partial | not_support | unknown
- responsibility_hint: 业主 | 承包商 | 分包商 | 设计/监理 | 双方 | 不可抗力/政策 | unknown
- money_items: [
    {
      item_type: 逾期违约金 | 延期损失 | 窝工费 | 赶工费 | 停工损失 | 利息 | 其他,
      label: support | partial | not_support | unknown,
      amount: 金额字符串或 null,
      evidence: 原文关键句，最多 120 字
    }
  ]
- key_reason: 最多 120 字，总结裁判理由，聚焦“为何支持/不支持/部分支持延期相关金钱请求”。

Decision rules:
1. Only consider money disputes caused by delay, overdue completion, schedule extension, suspension, liquidated damages, delay loss, idle cost, acceleration cost, or interest.
2. Ordinary progress payment, settlement payment, quality bond, or repair bond is not delay_money unless explicitly linked to schedule delay.
3. If the ruling cannot be connected to delay_money, return unknown.
""".strip()


def build_messages(text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {
            "role": "user",
            "content": f"""
璇峰熀浜庝互涓嬫枃鏈紝鎶藉彇鈥滃欢璇浉鍏崇殑閲戦挶浜夎鍙婂叾瑁佸垽缁撴灉鈥濄€?

銆愭枃鏈紑濮嬨€?
{text}
銆愭枃鏈粨鏉熴€?

{SCHEMA_NOTE}
""".strip(),
        },
    ]


# =========================
# Text selection (shorter input)
# =========================
SENT_SPLIT = re.compile(r"[。！？；\n]+")
DECISION_ANCHOR = re.compile(r"(判决如下|裁决如下|裁定如下|本院判决如下|本院裁定如下|仲裁裁决如下)")

MONEY_KWS = [
    "违约金", "赔偿", "补偿", "损失", "延期损失", "延误损失", "停工损失", "窝工费", "赶工费",
    "机械停置费", "利息", "逾期", "迟延", "延误", "延期", "工期", "顺延", "展期"
]
EXCLUDE_KWS = ["工程款", "工程价款", "结算", "价款", "质保金", "保修金", "进度款"]

def normalize_text(t: str) -> str:
    t = (t or "").replace("\u3000", " ").replace("\t", " ").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def choose_text(sections: Dict[str, str]) -> Tuple[str, str]:
    decision = normalize_text(sections.get("decision") or "")
    reasoning = normalize_text(sections.get("reasoning") or "")
    full_text = normalize_text(sections.get("full_text") or "")

    if decision or reasoning:
        return (decision + "\n" + reasoning).strip(), "decision+reasoning"
    return full_text, "full_text_only"

def extract_decision_tail(text: str, tail_chars: int = 1800) -> str:
    if not text:
        return ""
    m = DECISION_ANCHOR.search(text)
    if m:
        seg = text[m.start():].strip()
        return seg[:tail_chars]
    return text[-tail_chars:] if len(text) > tail_chars else text

def sentence_pool(text: str) -> List[str]:
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s and s.strip()]
    return [s for s in sents if len(s) >= 6]

def build_short_text(base_text: str, max_chars: int = 2600) -> str:
    """
    Build shorter context for LLM:
    - decision tail (strong)
    - sentences containing MONEY_KWS but not only EXCLUDE_KWS
    - capped to max_chars
    """
    base_text = normalize_text(base_text)
    if len(base_text) <= max_chars:
        return base_text

    tail = extract_decision_tail(base_text, tail_chars=1600)

    sents = sentence_pool(base_text)
    picked = []
    for s in sents:
        if any(k in s for k in MONEY_KWS):
            if any(ek in s for ek in EXCLUDE_KWS) and not any(k in s for k in ["违约金", "赔偿", "补偿", "损失", "利息", "窝工", "赶工", "停工"]):
                continue
            picked.append(s)
        if len(picked) >= 18:
            break

    short = "\n".join(["【裁判结论尾段】", tail, "【与延误金钱请求相关的句子】"] + picked)
    return short[:max_chars]


# =========================
# Robust JSON parse
# =========================
def parse_json_only(text: str) -> Dict[str, Any]:
    """
    Robust parse:
    - strip code fences
    - find outermost {...}
    """
    t = (text or "").strip()

    # remove ```json ... ```
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    if "{" not in t or "}" not in t:
        raise ValueError("No JSON found in model output")

    start = t.find("{")
    end = t.rfind("}")
    if end <= start:
        raise ValueError("Malformed JSON bounds")

    candidate = t[start:end+1]
    return json.loads(candidate)


# =========================
# QC (quality check)
# =========================
SUP_KWS = ["予以支持", "应予支持", "判令", "应当支付", "承担违约责任", "准许", "支持"]
NOT_KWS = ["不予支持", "驳回", "不支持", "无依据", "不予认可", "不予采纳"]
PAR_KWS = ["部分支持", "予以部分支持", "其余驳回", "酌定", "范围内", "支持.*部分", "仅支持"]

def kw_hit(text: str, kws: List[str]) -> bool:
    if not text:
        return False
    for k in kws:
        if re.search(k, text):
            return True
    return False

def qc_flags_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flag suspicious combinations based on evidence fields.
    """
    label = row.get("delay_money_label", "unknown")
    key_reason = row.get("key_reason", "")
    items = row.get("money_items", [])
    ev_list = []
    if isinstance(items, list):
        for it in items[:6]:
            if isinstance(it, dict):
                ev = it.get("evidence", "")
                if ev:
                    ev_list.append(str(ev))
    evidence_blob = " ".join(ev_list)[:2000]

    flags = []
    if label == "support" and (kw_hit(evidence_blob, NOT_KWS) or kw_hit(key_reason, NOT_KWS)):
        flags.append("support_but_not_keywords")
    if label == "not_support" and (kw_hit(evidence_blob, SUP_KWS) or kw_hit(key_reason, SUP_KWS)):
        flags.append("not_support_but_support_keywords")
    if label == "partial" and not (kw_hit(evidence_blob, PAR_KWS) or kw_hit(key_reason, PAR_KWS)):
        flags.append("partial_without_partial_markers")
    if label == "unknown" and (kw_hit(evidence_blob, SUP_KWS) or kw_hit(evidence_blob, NOT_KWS)):
        flags.append("unknown_but_has_disposition_keywords")

    # item-level contradictions
    if isinstance(items, list) and items:
        for it in items[:8]:
            if not isinstance(it, dict):
                continue
            ilab = it.get("label", "unknown")
            iev = str(it.get("evidence", ""))
            if ilab == "support" and kw_hit(iev, NOT_KWS):
                flags.append("item_support_but_not_kw")
            if ilab == "not_support" and kw_hit(iev, SUP_KWS):
                flags.append("item_not_support_but_support_kw")

    return {
        "case_id": row.get("case_id", ""),
        "delay_money_label": label,
        "responsibility_hint": row.get("responsibility_hint", ""),
        "n_items": row.get("n_items", 0),
        "flags": ";".join(sorted(set(flags))),
        "flag_count": len(set(flags)),
    }


def write_qc_report(df_rows: pd.DataFrame, case_json_dir: Path, out_txt: Path, out_flags_csv: Path):
    # reload per-case json to get evidence/key_reason for QC
    enriched = []
    for cid in df_rows["case_id"].astype(str).tolist():
        fp = case_json_dir / f"{cid}.json"
        if not fp.exists():
            continue
        obj = json.loads(fp.read_text(encoding="utf-8"))
        enriched.append({
            "case_id": cid,
            "delay_money_label": obj.get("delay_money_label", "unknown"),
            "responsibility_hint": obj.get("responsibility_hint", "unknown"),
            "money_items": obj.get("money_items", []),
            "key_reason": obj.get("key_reason", ""),
            "text_source": obj.get("text_source", ""),
        })
    if not enriched:
        out_txt.write_text("QC: no case json found.\n", encoding="utf-8")
        return

    base_df = pd.DataFrame(enriched)
    # summary
    lines = []
    lines.append(f"QC REPORT @ {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Total labeled cases: {len(base_df)}\n")

    lines.append("=== delay_money_label distribution ===")
    lines.append(base_df["delay_money_label"].value_counts(dropna=False).to_string())
    lines.append("")

    lines.append("=== responsibility_hint distribution ===")
    lines.append(base_df["responsibility_hint"].value_counts(dropna=False).to_string())
    lines.append("")

    # flags
    flags_rows = []
    for r in enriched:
        flags_rows.append(qc_flags_row({
            "case_id": r["case_id"],
            "delay_money_label": r["delay_money_label"],
            "responsibility_hint": r["responsibility_hint"],
            "n_items": len(r.get("money_items") or []) if isinstance(r.get("money_items"), list) else 0,
            "money_items": r.get("money_items", []),
            "key_reason": r.get("key_reason", ""),
        }))
    flags_df = pd.DataFrame(flags_rows)
    flags_df.to_csv(out_flags_csv, index=False, encoding="utf-8-sig")

    lines.append("=== QC flags summary ===")
    lines.append(flags_df["flag_count"].value_counts(dropna=False).to_string())
    lines.append("")

    top_flagged = flags_df.sort_values(["flag_count"], ascending=False).head(30)
    lines.append("=== Top flagged (up to 30) ===")
    lines.append(top_flagged.to_string(index=False))
    lines.append("")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================
# Worker
# =========================
def process_one(
    client: QwenClient,
    fp: Path,
    resume: bool,
    sleep: float,
    max_chars: int,
    short_mode: bool,
    max_tokens: int,
    max_retries: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, str]]]:

    try:
        obj = json.loads(fp.read_text(encoding="utf-8"))
        case_id = obj.get("case_id", fp.stem)
        out_fp = CASE_JSON_DIR / f"{case_id}.json"

        if resume and out_fp.exists():
            return None, None

        text, source_note = choose_text(obj.get("sections", {}) or {})
        if len(text) < 200:
            return None, {"file": fp.name, "error": "Text too short"}

        if short_mode:
            text_for_llm = build_short_text(text, max_chars=max_chars)
            source_note = source_note + "|short"
        else:
            text_for_llm = text[:max_chars]

        messages = build_messages(text_for_llm)

        last_err = None
        for attempt in range(max_retries + 1):
            try:
                raw = client.chat(messages, temperature=0.0, max_tokens=max_tokens)
                parsed = parse_json_only(raw)
                out = {
                    "case_id": case_id,
                    "text_source": source_note,
                    "delay_money_label": parsed.get("delay_money_label", "unknown"),
                    "responsibility_hint": parsed.get("responsibility_hint", "unknown"),
                    "money_items": parsed.get("money_items", []),
                    "key_reason": parsed.get("key_reason", ""),
                    "llm_model": DEFAULT_MODEL,
                    "labeled_at": datetime.now().isoformat(timespec="seconds"),
                }
                out_fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

                if sleep > 0:
                    time.sleep(sleep + random.random() * sleep * 0.3)

                row = {
                    "case_id": case_id,
                    "delay_money_label": out["delay_money_label"],
                    "responsibility_hint": out["responsibility_hint"],
                    "n_items": len(out["money_items"]) if isinstance(out["money_items"], list) else 0,
                    "text_source": source_note,
                }
                return row, None

            except Exception as e:
                last_err = str(e)
                # backoff
                time.sleep(0.8 * (attempt + 1))

        return None, {"file": fp.name, "error": f"retry_failed: {last_err}"}

    except Exception as e:
        return None, {"file": fp.name, "error": str(e)}


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="skip processed cases")
    ap.add_argument("--sleep", type=float, default=0.05, help="sleep per case (each thread) to avoid rate-limit")
    ap.add_argument("--max_docs", type=int, default=0, help="0 means all")
    ap.add_argument("--workers", type=int, default=8, help="concurrent workers")
    ap.add_argument("--short_mode", action="store_true", help="use shorter text selection")
    ap.add_argument("--max_chars", type=int, default=2600)
    ap.add_argument("--max_tokens", type=int, default=700)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    # =========================
# API Key (debug fallback)
# =========================
# 鉁?鎺ㄨ崘锛氫紭鍏堢敤鐜鍙橀噺锛堟洿瀹夊叏锛?
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

# 鉁?浠呯敤浜庢湰鏈鸿皟璇曪細濡傛灉浣犱笉鎯抽厤鐜鍙橀噺锛屽氨鍦ㄨ繖閲屼复鏃跺～涓€涓?
    DEBUG_API_KEY = ""

    if not api_key:
        api_key = DEBUG_API_KEY.strip()

    if not api_key or api_key.startswith("sk-xxxx"):
        raise RuntimeError("Missing API key. Set DASHSCOPE_API_KEY before running this script.")


    RESULTS_DIR.mkdir(exist_ok=True)
    CASE_JSON_DIR.mkdir(exist_ok=True)

    client = QwenClient(api_key, DEFAULT_BASE_URL, DEFAULT_MODEL, timeout=args.timeout)

    files = sorted(JSON_DIR.glob("*.json"))
    if args.max_docs > 0:
        files = files[: args.max_docs]

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    desc = f"LLM Step2 FAST (workers={args.workers}, short={args.short_mode})"
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for fp in files:
            futs.append(ex.submit(
                process_one,
                client, fp,
                args.resume, args.sleep,
                args.max_chars, args.short_mode,
                args.max_tokens, args.retries
            ))

        for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
            row, err = fut.result()
            if row:
                rows.append(row)
            if err:
                errors.append(err)

    # Save main CSV
    out_csv = RESULTS_DIR / "labels_step2_delay_outcome_llm.csv"
    pd.DataFrame(rows).sort_values("case_id").to_csv(out_csv, index=False, encoding="utf-8-sig")

    # Save errors
    if errors:
        pd.DataFrame(errors).to_csv(RESULTS_DIR / "llm_errors.csv", index=False, encoding="utf-8-sig")

    # QC report
    qc_txt = RESULTS_DIR / "qc_report_delay_outcome.txt"
    qc_flags = RESULTS_DIR / "qc_flags_delay_outcome.csv"
    if rows:
        write_qc_report(pd.DataFrame(rows), CASE_JSON_DIR, qc_txt, qc_flags)

    print("\nDONE.")
    print(f"Saved: {out_csv}")
    if errors:
        print(f"Errors: {RESULTS_DIR / 'llm_errors.csv'}")
    if rows:
        print(f"QC: {qc_txt}")
        print(f"QC flags: {qc_flags}")


if __name__ == "__main__":
    main()

