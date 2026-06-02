# delay-dispute-leap

`delay-dispute-leap` is a research-oriented toolkit for reproducible experiments on construction delay-dispute outcome prediction and responsibility diagnosis. It packages the parts of a larger PhD research workflow that are safe to publish: evaluation utilities, audit helpers, figure generation, lightweight configs, and a small smoke-test path.

## Scope

- Pre-decision text processing and audit helpers for delay-dispute analysis
- Candidate benchmark and evaluation scripts for outcome prediction experiments
- Reproducible reporting assets and example tables/figures
- Public-safe sample artifacts only; no full raw corpus or private case archive

## Repository Layout

- `src/`: research scripts and utilities
- `config/`: public-safe configuration templates
- `tests/`: preserved test entrypoints from the local research workflow
- `docs/assets/`: sample figures and tables from non-sensitive experiment outputs
- `examples/`: toy inputs and sample prediction rows
- `smoke_test.py`: lightweight offline sanity check

## Quick Start

```powershell
python -m pip install -r requirements.txt
python smoke_test.py
```

The smoke test does not call external APIs. API-backed scripts require `DASHSCOPE_API_KEY` to be set explicitly in your environment.

## Included Public Artifacts

- `docs/assets/fig3_outcome_prediction_comparison.png`
- `docs/assets/fig4_confusion_matrix.png`
- `docs/assets/table_main_results.csv`
- `docs/assets/table_responsibility_results.csv`
- `examples/toy_case_record.json`
- `examples/sample_predictions.csv`

## Notes

- This repo intentionally excludes the full raw-text corpus, large result caches, and unpublished manuscript materials.
- Any script that uses a live LLM endpoint now requires an explicit environment variable. No debug key is bundled.
- For the multi-agent responsibility-attribution layer built on top of this workflow, see the sister repo `madra-delay-attribution`.

## Citation

If this repository helps your work, cite the software record in `CITATION.cff`.
