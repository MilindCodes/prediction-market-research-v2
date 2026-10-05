# Data Collection Protocol

> Converted from `README.docx`, which was a binary file excluded by
> `.gitignore` and therefore invisible on GitHub. The original checklist is
> preserved below; verified values have been filled in from `config.py` and the
> committed artefacts. **Items marked `TODO` still need your input** — they
> cannot be recovered from the code.

**Purpose:** document everything about the data pull so the results in the
paper are reproducible.

---

## 1. Required disclosures

The checklist from the original document, with current status:

| Item | Status |
| --- | --- |
| Pull date | `TODO` — see [Pull log](#5-pull-log) |
| API version / endpoint used | ✅ [Section 2](#2-sources-and-endpoints) |
| Date range of contracts included | ✅ [Section 3](#3-sample-window) |
| Filters applied | ✅ [Section 4](#4-filters) |
| Number of contracts and observations per dataset | ✅ [Section 6](#6-sample-size) |
| Handling of missing data / gaps in trade history | `TODO` — [Section 7](#7-missing-data) |
| Construction of hourly prices from tick data | `TODO` — [Section 8](#8-price-construction) |

---

## 2. Sources and endpoints

| Venue | API | Auth | Implementation |
| --- | --- | --- | --- |
| Polymarket | Gamma API — market metadata, volume, resolution, UMA dispute status | None (public) | `src/polymarket/gamma_client.py` |
| Polymarket | CLOB API `/prices-history` — price history | None | `src/polymarket/client.py`, `src/polymarket/trades.py` |
| Kalshi | Elections API | RSA-signed request or Bearer token | `src/kalshi/client.py`, `src/kalshi/trades.py` |

**Polymarket granularity.** The CLOB `/prices-history` endpoint only retains
**daily** bars (`fidelity=1440`) for closed markets, so calibration uses
`DAILY_DT = 1/365.25` rather than the hourly step.

**Polymarket token IDs.** The endpoint requires the **YES-outcome token ID**
from `clobTokenIds`, *not* the `conditionId`. The `conditionId` column in
`catalog_full.parquet` is not reliable for this — resolve identifiers through
the CLOB API instead.

---

## 3. Sample window

The two venues cover different periods deliberately: Polymarket has full
2022–2024 history, while the Kalshi Elections API only reaches back to
roughly 2025.

| Venue | Start | End | `config.py` |
| --- | --- | --- | --- |
| Polymarket | 2022-01-01 | 2024-12-31 | `POLYMARKET_DATE_START` / `_END` |
| Kalshi | 2025-01-01 | 2026-12-31 | `KALSHI_DATE_START` / `_END` |

---

## 4. Filters

### From the original document

- **Minimum 500 trades** — enough tick data to construct a meaningful series.
  (`config.MIN_TRADE_COUNT = 500`)
- **Minimum 14 days between listing and resolution** — ensures enough
  time-series length. (`config.MIN_DURATION_DAYS = 14`)
- **Exclude contracts that resolved within 48 hours of listing** — outcomes
  were effectively already known.

> [!IMPORTANT]
> The 48-hour rule is stated in the original protocol but there is **no
> `MIN_HOURS`-style constant in `config.py`**. Either it is enforced inside the
> catalog filters, or it was never implemented. **`TODO`: confirm which, and
> promote it to a named constant either way.**

### Topic filters

Markets are bucketed by keyword lists in `config.py`:

- `FED_KEYWORDS` — FOMC, federal reserve, fed funds, rate hike/cut, basis
  point, Powell, …
- `CPI_KEYWORDS` — CPI, core CPI, inflation, PCE, PPI, GDP, nonfarm payroll,
  unemployment rate, retail sales, durable goods
- `ECONOMIC_KEYWORDS` = `FED_KEYWORDS + CPI_KEYWORDS`
- `POLITICAL_KEYWORDS` — election, president, senate, congress, …

Phrases are deliberately specific: bare `"fed"` matches "federal student
loans", and bare `"rate"` matches "approval rate".

### Numerical treatment

Prices are clipped to `[LOG_ODDS_CLIP_LO, LOG_ODDS_CLIP_HI] = [0.02, 0.98]`
before the log-odds transform, so `log(p / (1 - p))` stays finite at the
boundaries.

---

## 5. Pull log

`TODO` — record each pull as it happens. Nothing in the repository records
when the APIs were actually queried.

| Date pulled | Venue | Step | Contracts returned | Notes |
| --- | --- | --- | --- | --- |
| | | | | |

---

## 6. Sample size

Counts verified from the committed artefacts:

| Stage | Count |
| --- | --- |
| Polymarket contracts surfaced for manual review | 34,505 |
| Polymarket contracts with committed price history | 39 |
| Contracts in the fitted corpus (`model_comparison.csv`) | 39 (all Polymarket) |
| Total price observations across the corpus | 3,705 |
| Observations per contract | min 16 · median 41 · max 272 |
| Contracts with converged Heston / Bates fits | 39 / 39 |

> [!NOTE]
> The corpus is currently **Fed-only** and contains **no Kalshi contracts**.
> `TODO`: document why Kalshi contracts were excluded from the final corpus,
> and note that the median contract carries only ~41 daily observations — a
> constraint worth stating explicitly when reporting 8-parameter Bates fits.

---

## 7. Missing data

`TODO` — document the actual handling. Specify:

- How gaps in the price series are detected.
- Whether gaps are forward-filled, interpolated, or left as gaps.
- What happens to a contract whose history has a large hole.
- How the `near_resolution_frac` column in `model_comparison.csv` is used.

`src/cleaning.py` writes quality flags for degenerate fits — describe those
flags and the threshold for each.

---

## 8. Price construction

`TODO` — this is the single most important undocumented choice. State whether
each bar is:

- a **VWAP** over the interval,
- the **last trade** in the interval,
- the **midpoint** of the book, or
- whatever the venue returns unmodified.

For Polymarket the answer is constrained by the CLOB `/prices-history`
endpoint, which returns pre-aggregated daily bars — so record what
*Polymarket* aggregates, not what this code does. Note also that volume fields
are not returned by that endpoint and are set to `0`.

---

## 9. Reproducing a pull

```bash
python run_pipeline.py check
python run_pipeline.py catalog-polymarket   # no API key required
python run_pipeline.py trades-sample        # ~5-10 min
python run_pipeline.py backtest-implied-vol
python run_pipeline.py backtest-bs
python run_pipeline.py backtest-heston
python run_pipeline.py backtest-bates
python run_pipeline.py compare
python run_pipeline.py export
```

Because `data/` is version-controlled, `git status` after a re-pull is itself
the reproducibility check: **if the numbers did not change, the diff is empty.**
