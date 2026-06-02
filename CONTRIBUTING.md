# Contributing

This repository is released as a compact public research package. Contributions should preserve reproducibility, data hygiene, and clear provenance.

## Local Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python smoke_test.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Scope

- Keep the public dataset limited to toy or anonymized examples.
- Do not commit raw private case text, credentials, or machine-specific paths.
- Prefer small, reviewable pull requests with a clear research or maintenance purpose.

## Validation

Before opening a pull request, run:

```bash
python smoke_test.py
```

If you change evaluation or plotting code, include a short note describing the expected effect on outputs.
