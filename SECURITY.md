# Security Policy

## Supported versions

This is a single-branch academic research repository. Only the current `main`
branch is maintained; there are no tagged releases or backported fixes.

| Branch | Supported |
| ------ | --------- |
| `main` | ✅ |
| Anything else | ❌ |

## Reporting a vulnerability

**Please do not open a public GitHub issue for a security problem.**

Report it privately instead, using either of these:

1. **GitHub private vulnerability reporting** (preferred) — go to the
   [Security tab](https://github.com/MilindCodes/prediction-market-research-v2/security)
   and choose **Report a vulnerability**.
2. **Email** the maintainer directly at the address on the
   [MilindCodes GitHub profile](https://github.com/MilindCodes).

Please include:

- What the problem is and where in the repository it lives (file and line).
- How to reproduce it.
- What an attacker could do with it.

**Response times.** This project is maintained by one person alongside other
work. Expect an acknowledgement within 7 days and an assessment within 30 days.
Please give the maintainer a reasonable window to fix the issue before
disclosing it publicly.

## Credentials and this repository

This project talks to the **Kalshi** and **Polymarket** APIs, so it handles API
credentials. The rules are:

- **Credentials are read from environment variables only** — see
  [`config.py`](config.py). Nothing is hardcoded.
- **Never commit a key.** `.gitignore` blocks `*.pem`, `*.key`, `.env`,
  `id_rsa*`, `*credentials*.json` and the two `researchproject*.txt` files.
  If you introduce a new credential file, **add it to `.gitignore` first, then
  create the file.**
- **Store keys outside the repository.** The recommended location is
  `~/.config/kalshi/` (mode `600`), not the project directory. Point
  `KALSHI_KEY_FILE` at it.
- **Never paste a key into an issue, a pull request, or a commit message.**

### If a key is committed by accident

Deleting the file in a later commit is **not sufficient** — the key stays
readable in the repository's history for anyone who clones it.

Do this, in order:

1. **Revoke the key at the provider immediately.** For Kalshi:
   kalshi.com → Settings → API Keys → revoke, then issue a new one.
   Treat the old key as compromised from the moment it was pushed.
2. Remove the file from history with
   [`git filter-repo`](https://github.com/newren/git-filter-repo) and
   force-push, or delete and recreate the repository.
3. Rotate anything else that key could reach.

Step 1 is the one that actually protects you. Steps 2 and 3 are cleanup.
