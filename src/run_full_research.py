# -*- coding: utf-8 -*-
"""One-command orchestration for the IEEE-TEM-oriented DelayDispute Copilot pipeline."""

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
    ap.add_argument("--config", type=str, default="config/research_v1.yaml")
    ap.add_argument("--style", type=str, default="config/figure_style_sci.yaml")
    ap.add_argument("--skip_parse", action="store_true")
    args = ap.parse_args()

    py = sys.executable

    if not args.skip_parse:
        run([py, "src/pipeline_all_in_one.py", "--config", args.config, "--stage", "enrich"])
        run([py, "src/pipeline_all_in_one.py", "--config", args.config, "--stage", "prepare_labels"])

    run([py, "src/build_candidate_gold.py", "--config", args.config])
    run([py, "src/final_eval.py", "--config", args.config])
    run([py, "src/run_ablation.py", "--config", args.config])
    run([py, "src/error_analysis.py", "--config", args.config])
    run([py, "src/make_paper_figures.py", "--config", args.config, "--style", args.style])
    print("[DONE] full IEEE-TEM-oriented pipeline completed")


if __name__ == "__main__":
    main()
