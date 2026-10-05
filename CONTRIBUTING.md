# Contributing

Thanks for your interest. This is an academic research repository studying
stochastic volatility in prediction markets, so contributions are most useful
when they improve **correctness, reproducibility, or clarity**.

## Ground rules

- **Never commit a credential.** Read [SECURITY.md](SECURITY.md) before your
  first commit. API keys live in environment variables, never in the repo.
- **Never commit regenerated data noise.** `data/` is version-controlled on
  purpose so results are reproducible. Only commit a data change when the
  numbers actually changed, and say *why* in the commit message.
- **Explain the finance, not just the code.** A change to a calibration routine
  should say what it does to the estimates.

## Local setup

The pipeline needs **Python 3.10 or newer** (the reference environment is
Python 3.11).

```bash
# 1. Clone
git clone https://github.com/MilindCodes/prediction-market-research-v2.git
cd prediction-market-research-v2

# 2. Create the environment (conda shown; venv works too)
conda create -n pmr python=3.11 -y
conda activate pmr

# 3. Install runtime + development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Confirm everything is wired up
python run_pipeline.py check
```

`check` verifies your Python version, confirms every package imports, reports
whether your API credentials are visible, and creates the `data/` subdirectories.

### API credentials

Only Kalshi needs a key. Polymarket's Gamma API is public.

```bash
export KALSHI_KEY_FILE=~/.config/kalshi/kalshi-api-key.pem   # NOT in this repo
export KALSHI_KEY_ID=<your-key-id>
```

Keep the `.pem` outside the repository directory and `chmod 600` it.

## Running things

```bash
python run_pipeline.py help   # data collection → calibration → results
python run_smm.py help        # Section 4: Bates SMM estimation
```

## Before you open a pull request

```bash
# Lint — must be clean
ruff check .

# Format — rewrites files in place
ruff format .

# Tests
pytest -q
```

CI runs the same three commands on every push and pull request. A red build
will not be merged.

## Pull request guidelines

1. **Branch off `main`.** Use a descriptive name: `fix-heston-warm-start`,
   not `patch-1`.
2. **One concern per PR.** A refactor and a bug fix belong in separate PRs.
3. **Write a commit message that says why.** The subject line is imperative and
   under ~72 characters; the body explains the reasoning.
4. **Fill in the PR template.** It asks what changed, how you verified it, and
   whether any numbers in the paper move.
5. **If results change, show them.** Paste the before/after for the affected
   metric. A silent change to an estimate is the hardest kind of bug to catch.
6. **Keep diffs reviewable.** If a PR exceeds ~400 changed lines of source,
   consider splitting it.

## Reporting bugs and requesting features

Use the issue templates in
[`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE). For a bug, the single most
useful thing you can provide is the exact command you ran and the full
traceback.

## Code style

Enforced by `ruff` and [`.editorconfig`](.editorconfig):

- 4-space indentation, LF line endings, UTF-8, trailing newline.
- Maximum line length 100.
- Import order: standard library, then third-party, then local (`config`,
  `src.*`, `analysis.*`), each group separated by a blank line.
- Public functions and classes get a docstring. NumPy style is used throughout
  `src/models/` — match it.
- Type hints on new code. Every module starts with
  `from __future__ import annotations` so modern syntax works.
