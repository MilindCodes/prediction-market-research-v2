---
name: Bug report
about: Something in the pipeline crashes, or produces a result you believe is wrong
title: "[Bug] "
labels: bug
assignees: ''
---

<!--
SECURITY: do not report a vulnerability here. See SECURITY.md.
CREDENTIALS: scrub API keys, key IDs, and .pem contents from anything you paste.
-->

## What happened

<!-- One or two sentences. -->

## What you expected instead

<!-- Especially important for numerical bugs: what value did you expect, and why? -->

## Exact command you ran

```bash
# e.g. python run_pipeline.py backtest-bates
```

## Full output / traceback

<details>
<summary>Traceback</summary>

```text
paste here
```

</details>

## Is this a crash or a wrong number?

- [ ] Crash / exception
- [ ] Runs fine but the output looks wrong
- [ ] Runs fine but is unusably slow
- [ ] Documentation is wrong or missing

## If the output looks wrong

<!-- Which contract(s)? Which parameter? What value did you get vs. expect?
     Attaching the relevant row from data/exports/*.csv is very helpful. -->

## Environment

- OS and version:
- Python version (`python --version`):
- How you installed dependencies (conda / venv / other):
- Commit or branch you are on (`git rev-parse --short HEAD`):
- Did `python run_pipeline.py check` pass?

## Anything else

<!-- Recent changes, whether it used to work, upstream API changes, etc. -->
