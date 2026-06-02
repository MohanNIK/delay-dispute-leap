# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

# =========================
# 你只需要确认这里
# =========================
PROJECT_ROOT = Path(r"C:\Users\1\Desktop\python论文\delay_dispute_madra")
HOLDOUT_DIR = PROJECT_ROOT / "results" / "holdout_20251225_234433"
JSON_DIR = PROJECT_ROOT / "data" / "2_parsed_json"

# 你给的 20 个错例
SUPPORT_TO_PARTIAL = [
    "bc1d12630435", "9e8a999d659d", "cbb45dfb52b2", "c0520381833e", "619310f5a899",
    "21c33fc77483", "509c0a3794fa", "d52b9ac54510", "6bdd327bbc68", "c677b8c84eeb"
]
NOTSUPPORT_TO_PARTIAL = [
    "c2d48e2dc634", "a1244f46d768", "764f4cbb805e", "bb428d73456c", "1aa5e0ae9980",
    "97838e7f77f5", "558d9c130ee3", "d0c2147f5bdd", "e808fe489fe2", "551c0bd9f7e6"
]

OUT_XLSX = HOLDOUT_DIR / "top20_errors_with_snippets.xlsx"

# =========================
# 片段抽取配置
# =========================
# 优先关键词：尽量把片段截取到“工期顺延”判定语句附近
KW_PRIMARY = [
    "工期顺延", "顺延工期", "展期", "延期", "延长工期", "工期调整", "调整工期",
    "顺延天数", "延期天数", "顺延至", "延期至",
    "逾期完工", "逾期竣工", "竣工日期", "交工日期", "完工日期",
    "不予顺延", "不予支持", "不支持", "驳回", "部分支持", "予以支持", "应予支持", "准许",
    "其余请求驳回", "其他请求驳回"
]
KW_FALLBACK = ["工期", "延误", "竣工", "完工", "顺延", "延期", "驳回", "支持"]

# 句子分割（用于 fallback）
SENT_SPLIT = re.compile(r"[。！？；\n]+")


def norm(text: str) -> str:
    t = (text or "")
    t = t.replace("\u3000", " ").replace("\t", " ").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def load_dr_text(case_id: str) -> Tuple[str, str]:
    """return (decision+reasoning or fallback, source_note)"""
    fp = JSON_DIR / f"{case_id}.json"
    obj = json.loads(fp.read_text(encoding="utf-8"))
    sec = obj.get("sections", {}) or {}
    decision = norm(sec.get("decision", ""))
    reasoning = norm(sec.get("reasoning", ""))
    full_text = norm(sec.get("full_text", ""))
    dr = norm((decision + "\n" + reasoning).strip())
    if dr:
        return dr, "decision+reasoning"
    return full_text, "full_text_only"


def best_window(text: str, keywords: List[str], clip_len: int = 380) -> Optional[str]:
    """找到最靠前的关键词位置，截取一段 window"""
    for kw in keywords:
        idx = text.find(kw)
        if idx != -1:
            # 以关键词为中心截取
            half = clip_len // 2
            s = max(0, idx - half)
            e = min(len(text), idx + half)
            snippet = text[s:e].replace("\n", " ").strip()
            return snippet + ("..." if e < len(text) else "")
    return None


def clip_snippet(text: str, min_len: int = 200, max_len: int = 420) -> str:
    t = norm(text).replace("\n", " ")
    if not t:
        return ""

    # 1) primary keyword window
    snip = best_window(t, KW_PRIMARY, clip_len=380)
    if snip and len(snip) >= min_len:
        return snip[:max_len] + ("..." if len(snip) > max_len else "")

    # 2) fallback keyword window
    snip = best_window(t, KW_FALLBACK, clip_len=380)
    if snip and len(snip) >= min_len:
        return snip[:max_len] + ("..." if len(snip) > max_len else "")

    # 3) sentence fallback: 拼接若干句直到长度够
    sents = [s.strip() for s in SENT_SPLIT.split(t) if s and s.strip()]
    buf = ""
    for s in sents[:30]:
        if len(buf) < max_len:
            buf = (buf + "。" + s) if buf else s
        if len(buf) >= min_len:
            break
    if len(buf) > max_len:
        buf = buf[:max_len] + "..."
    return buf


def main():
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    def add_group(case_ids: List[str], group_name: str):
        for cid in case_ids:
            text, source_note = load_dr_text(cid)
            snippet = clip_snippet(text, min_len=200, max_len=420)
            rows.append({
                "group": group_name,
                "case_id": cid,
                "text_source_note": source_note,
                "snippet_200_420": snippet
            })

    add_group(SUPPORT_TO_PARTIAL, "support->partial")
    add_group(NOTSUPPORT_TO_PARTIAL, "not_support->partial")

    df = pd.DataFrame(rows)
    df.to_excel(OUT_XLSX, index=False)
    print(f"[SAVED] {OUT_XLSX}")
    print(df.groupby("group")["case_id"].count().to_string())


if __name__ == "__main__":
    main()
