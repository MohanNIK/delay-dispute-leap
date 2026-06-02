# -*- coding: utf-8 -*-
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\1\Desktop\python论文\delay_dispute_madra")
HOLDOUT_CSV = PROJECT_ROOT / "results" / "holdout_output.csv"   # <- 如果名字不同，改这里
JSON_DIR = PROJECT_ROOT / "data" / "2_parsed_json"
OUT_CSV = PROJECT_ROOT / "results" / "holdout_top20_errors_with_snippet.csv"

def load_dr(case_id: str) -> str:
    fp = JSON_DIR / f"{case_id}.json"
    obj = json.loads(fp.read_text(encoding="utf-8"))
    sec = obj.get("sections", {}) or {}
    decision = (sec.get("decision") or "").strip()
    reasoning = (sec.get("reasoning") or "").strip()
    text = (decision + "\n" + reasoning).strip()
    if not text:
        text = (sec.get("full_text") or "").strip()
    return text

def clip(text: str, n=380) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t[:n] + ("..." if len(t) > n else "")

def main():
    df = pd.read_csv(HOLDOUT_CSV, encoding="utf-8-sig")
    # 自动猜列名
    true_col = "y_true" if "y_true" in df.columns else "gold"
    pred_col = "y_pred" if "y_pred" in df.columns else "pred"

    sp2pa = df[(df[true_col] == "support") & (df[pred_col] == "partial")].head(10)
    ns2pa = df[(df[true_col] == "not_support") & (df[pred_col] == "partial")].head(10)

    pick = pd.concat([sp2pa, ns2pa], axis=0).copy()
    snippets = []
    for cid in pick["case_id"].astype(str).tolist():
        dr = load_dr(cid)
        snippets.append(clip(dr, 380))
    pick["decision_reasoning_snippet"] = snippets

    pick.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {OUT_CSV}")
    print("Rows:", len(pick))

if __name__ == "__main__":
    main()
