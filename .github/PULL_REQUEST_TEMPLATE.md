## What does this change?

<!-- One paragraph. What is different after this PR that wasn't before? -->

## Why?

<!-- The reasoning. Link the issue it closes: "Closes #123" -->

## Type of change

- [ ] Bug fix (does not change existing behaviour except to correct it)
- [ ] New feature (model, data source, diagnostic, or CLI step)
- [ ] Refactor / cleanup (no behaviour change intended)
- [ ] Documentation
- [ ] Tooling, CI, or dependencies

## Does this change any numbers?

**This is the most important question in this template.**

- [ ] No — no estimate, figure, or exported CSV changes
- [ ] Yes — and I have shown the before/after below

<!-- If yes, paste the affected metric before and after. For example:

     | Contract | Metric | Before | After |
     | -------- | ------ | ------ | ----- |
     | ...      | QLIKE  | 0.0412 | 0.0398 |

     A change to an estimate that isn't called out is the hardest kind of bug
     to catch after the fact. -->

## How did you verify this?

<!-- Be specific. "Ran the pipeline" is not enough — which step, on what data? -->

- [ ] `ruff check .` is clean
- [ ] `ruff format --check .` is clean
- [ ] `pytest -q` passes
- [ ] I ran the affected pipeline step end to end
- [ ] I checked the figures in `data/processed/figures/` still render correctly

## Checklist

- [ ] No credentials, API keys, or `.pem` contents are in this diff
- [ ] Any new credential path is added to `.gitignore` **before** the file exists
- [ ] New public functions and classes have docstrings
- [ ] `requirements.txt` updated if I added a runtime dependency
- [ ] Commit messages explain *why*, not just *what*
- [ ] Data files under `data/` are committed only because the numbers genuinely changed

## Anything reviewers should look at closely?

<!-- Point at the risky part. -->
