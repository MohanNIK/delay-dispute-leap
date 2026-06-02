# Reproduce The IEEE-TEM Research Pipeline

All commands below assume the project root is:
`C:\Users\pig\Desktop\python论文\delay_dispute_madra`

## 1. Install dependencies
```powershell
src\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Refresh leakage-aware structured cases
```powershell
src\.venv\Scripts\python.exe src\pipeline_all_in_one.py --config config\research_v1.yaml --stage enrich --overwrite
src\.venv\Scripts\python.exe src\pipeline_all_in_one.py --config config\research_v1.yaml --stage prepare_labels
```

## 3. Build candidate benchmarks and review package
```powershell
src\.venv\Scripts\python.exe src\build_candidate_gold.py --config config\research_v1.yaml
```

## 4. Run the unified evaluation
```powershell
src\.venv\Scripts\python.exe src\final_eval.py --config config\research_v1.yaml
```

## 5. Run ablation and error analysis on the latest evaluation run
```powershell
src\.venv\Scripts\python.exe src\run_ablation.py --config config\research_v1.yaml
src\.venv\Scripts\python.exe src\error_analysis.py --config config\research_v1.yaml
```

## 6. Generate figures, tables, text assets, and Excel packs
```powershell
src\.venv\Scripts\python.exe src\make_paper_figures.py --config config\research_v1.yaml --style config\figure_style_sci.yaml
```

## 7. One-command run
```powershell
src\.venv\Scripts\python.exe src\run_full_research.py --config config\research_v1.yaml --style config\figure_style_sci.yaml
```

## Main outputs
- `data/gold/candidate_gold_strict_v1.csv`
- `data/gold/candidate_gold_extended_v1.csv`
- `data/review/audit_subset_cases.csv`
- `results/final_eval_<timestamp>/metrics_main.json`
- `results/final_eval_<timestamp>/predictions_main.csv`
- `results/final_eval_<timestamp>/responsibility_eval.csv`
- `results/final_eval_<timestamp>/evidence_chain_eval.csv`
- `results/final_eval_<timestamp>/ablation_results.csv`
- `results/final_eval_<timestamp>/error_analysis.csv`
- `paper_assets/figures/`
- `paper_assets/figure_data/`
- `paper_assets/tables/`
- `paper_assets/paper_data_pack.xlsx`

## Scientific caution
- `candidate_gold_*` are machine-assisted candidate benchmarks, not human gold.
- Responsibility and evidence-chain outputs are audit-ready and uncertainty-aware, but not yet fully human-validated.
