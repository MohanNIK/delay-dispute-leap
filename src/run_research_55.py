# -*- coding: utf-8 -*-
"""One-command runner for the audit-first MMEC-PAESC 5.5 branch."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=PROJECT_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/research_v2_55.yaml")
    ap.add_argument("--style", default="config/figure_style_sci.yaml")
    ap.add_argument("--use-api", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--skip-manuscripts", action="store_true")
    ap.add_argument("--max-cases", type=int, default=0)
    args = ap.parse_args()

    py = sys.executable
    extract_cmd = [py, "src/llm55_mechanism_extraction.py", "--config", args.config]
    if args.use_api:
        extract_cmd.append("--use-api")
    if args.max_cases > 0:
        extract_cmd.extend(["--max-cases", str(args.max_cases)])
    run(extract_cmd)

    run([py, "src/final_eval_55.py", "--config", args.config])
    run([py, "src/run_forensic_audit_55.py", "--config", args.config])

    if not args.skip_figures:
        run([py, "src/make_paper_figures.py", "--config", args.config, "--style", args.style])
    if not args.skip_manuscripts:
        run([py, "src/build_bilingual_manuscripts_55.py", "--config", args.config])

    print("[DONE] MMEC-PAESC 5.5 audit-first rerun completed")


if __name__ == "__main__":
    main()
