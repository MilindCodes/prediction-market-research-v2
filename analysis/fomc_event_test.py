"""Model-free FOMC-day jump test on the daily Fed panel (§4, jump evidence).

If Fed-rate prices jump, the jumps should land on the days the FOMC speaks.
This checks that directly, with no model in between: every daily increment
is labelled by whether it spans an FOMC statement, and FOMC-day moves are
compared with every other day's.

DATING THE INCREMENTS.  `t` in smm_panel.parquet is a step counter, not a
date.  Day 0 of a contract is its first raw bar (data/raw/polymarket/<id>)
floored to the day, and row t = k is the move from bar k-1 to bar k, so it
ends on day 0 + k.  Statements come out at 18:00/19:00 UTC (config.FOMC_DATES)
and the bars are stamped within a minute of 00:00 UTC, so a meeting on day A
is the increment that ends on day A + 1.  _dated_panel() rebuilds every price
from the raw bars on that calendar and refuses to run if any row disagrees
with the panel.

GROUPS.  mid_life_fomc: FOMC increment during the contract's life (the jump
evidence).  own_resolution_fomc: FOMC increment in the contract's last 3
increments (settlement, not a jump).  non_fomc: everything else.
Significance is a permutation test: within each contract the mid-life FOMC
label is reshuffled over that contract's mid-life increments, keeping the
per-contract count, and p = (1 + #{shuffled >= observed}) / (1 + N_PERM).

VARIANTS.  "spec" is exactly the definition above.  Checking it turned up two
data problems, so two stricter variants are reported beside it:
  own_meeting  also counts as own-resolution any FOMC increment on or after
               the contract's own meeting (from end_date in
               polymarket_contract_identities.csv).  Three May-2023 contracts
               stay listed 13 days after their meeting, so "last 3
               increments" misses their settlement move.
  clean        own_meeting, plus a separate "excluded" group for increments in
               the first 3 days of a contract (opening price discovery) or
               with either endpoint at exactly 0.500, the feed's placeholder
               price (the May-2023 trio reset to 0.500 after their meeting).
Top-1% / top-5% moves are always ranked over the whole panel, so the shares
sum to 1 across groups within a variant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

RAW = config.DATA_DIR / "raw" / "polymarket"
PROCESSED = config.DATA_DIR / "processed"
EXPORTS = config.DATA_DIR / "exports"

RES_WINDOW = 3          # own-resolution window: the contract's last 3 increments
LISTING_WINDOW = 3      # "clean": first 3 increments are opening price discovery
PLACEHOLDER = 0.5       # "clean": the feed's default / reset price
OWN_MEETING_SLACK = 3   # end_date sits 0-2 days before the meeting it resolves on
N_PERM = 5000
SEED = 20261021
TOP_Q = (0.01, 0.05)
VARIANTS = ("spec", "own_meeting", "clean")
GROUPS = ("mid_life_fomc", "own_resolution_fomc", "non_fomc", "excluded")

# Riley's quick check, printed beside our numbers for comparison.
PRELIM = {"n_fomc": 76, "n_contracts": 30, "mean_mid": 0.48, "mean_other": 0.11,
          "top1_share": 0.14, "fomc_share": 0.018, "perm_p": 0.0002}


def _announcement_days() -> pd.DatetimeIndex:
    """End day of each statement's increment: statement date + 1."""
    return pd.DatetimeIndex(pd.to_datetime(config.FOMC_DATES, utc=True)).floor("D") \
        + pd.Timedelta(days=1)


def _own_meeting_end_day(end_date: str, statements: pd.DatetimeIndex) -> pd.Timestamp:
    """End day of the increment carrying the meeting a contract resolves on.

    end_date sits 0-2 days before that meeting.  "N cuts in 2024" contracts end
    after the last 2024 meeting, so fall back to the last meeting before it.
    """
    end = pd.Timestamp(end_date).tz_convert("UTC").floor("D")
    days = statements.floor("D")
    after = days[days >= end - pd.Timedelta(days=OWN_MEETING_SLACK)]
    own = after.min() if len(after) else days[days <= end].max()
    return own + pd.Timedelta(days=1)


def _dated_panel() -> pd.DataFrame:
    """The daily panel with each increment's end day and both raw endpoints."""
    panel = pd.read_parquet(PROCESSED / "smm_panel.parquet")
    end_dates = pd.read_csv(EXPORTS / "polymarket_contract_identities.csv") \
        .set_index("conditionId")["end_date"]
    statements = pd.DatetimeIndex(pd.to_datetime(config.FOMC_DATES, utc=True))

    parts = []
    for cid, g in panel.groupby("contract_id", sort=False):
        raw = pd.read_parquet(RAW / f"{cid}.parquet")
        days = pd.to_datetime(raw["timestamp"], utc=True).dt.floor("D")
        day0 = days.min()
        # Same grid as SMMPanelBuilder: one bar per day, forward-filled.
        close = pd.Series(raw["close"].to_numpy(float), index=days).groupby(level=0).last()
        close = close.reindex(pd.date_range(day0, close.index.max(), freq="D")).ffill()

        g = g.copy()
        t = g["t"].to_numpy()
        g["end_day"] = day0 + pd.to_timedelta(t, unit="D")
        g["p_start"] = close.to_numpy()[t - 1]
        g["p_end"] = close.to_numpy()[t]
        if not np.allclose(g["p_end"], g["p_raw"], rtol=0, atol=1e-12):
            raise AssertionError(f"{cid}: day 0 + t does not reproduce the panel's prices")
        g["own_meeting_day"] = _own_meeting_end_day(end_dates[cid], statements)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _label(d: pd.DataFrame, variant: str) -> tuple[pd.Series, pd.Series]:
    """(group per increment, permutation-pool mask) under one variant."""
    fomc = d["end_day"].isin(_announcement_days())
    t_max = d.groupby("contract_id")["t"].transform("max")
    res = d["t"] > t_max - RES_WINDOW
    if variant in ("own_meeting", "clean"):
        res |= d["end_day"] >= d["own_meeting_day"]
    excluded = pd.Series(False, index=d.index)
    if variant == "clean":
        excluded = (d["t"] <= LISTING_WINDOW) \
            | np.isclose(d["p_start"], PLACEHOLDER, rtol=0, atol=1e-12) \
            | np.isclose(d["p_end"], PLACEHOLDER, rtol=0, atol=1e-12)
        excluded &= ~(fomc & res)        # settlement stays settlement

    group = pd.Series("non_fomc", index=d.index)
    group[fomc & ~res] = "mid_life_fomc"
    group[fomc & res] = "own_resolution_fomc"
    group[excluded] = "excluded"
    pool = ~res & ~excluded
    return group, pool


def _top_mask(absdx: np.ndarray, q: float) -> np.ndarray:
    """The ceil(q·n) largest moves (ties broken by panel order)."""
    k = int(np.ceil(q * len(absdx)))
    mask = np.zeros(len(absdx), dtype=bool)
    mask[np.argsort(-absdx, kind="stable")[:k]] = True
    return mask


def _permutation_p(d, group, pool, absdx, tops, rng) -> dict:
    """Shuffle the mid-life FOMC label within each contract's mid-life pool."""
    stats_obs = {"mean_abs_dx": 0.0, **{f"top{round(q * 100)}": 0.0 for q in TOP_Q}}
    stats_perm = {k: np.zeros(N_PERM) for k in stats_obs}
    n_lab = 0
    for _, idx in d[pool].groupby("contract_id", sort=False).groups.items():
        idx = d.index.get_indexer(idx)
        m = int((group.to_numpy()[idx] == "mid_life_fomc").sum())
        if m == 0:
            continue
        n_lab += m
        labelled = group.to_numpy()[idx] == "mid_life_fomc"
        # m distinct positions per shuffle: argsort of uniform keys
        pick = np.argsort(rng.random((N_PERM, len(idx))), axis=1)[:, :m]
        stats_obs["mean_abs_dx"] += absdx[idx][labelled].sum()
        stats_perm["mean_abs_dx"] += absdx[idx][pick].sum(axis=1)
        for q, top in zip(TOP_Q, tops, strict=True):
            key = f"top{round(q * 100)}"
            stats_obs[key] += top[idx][labelled].sum()
            stats_perm[key] += top[idx][pick].sum(axis=1)
    if n_lab == 0:
        return {f"perm_p_{k}": np.nan for k in stats_obs}
    return {f"perm_p_{k}": (1 + np.sum(stats_perm[k] >= stats_obs[k] - 1e-12)) / (1 + N_PERM)
            for k in stats_obs}


def run(verbose: bool = True) -> pd.DataFrame:
    d = _dated_panel()
    absdx = d["delta_X"].abs().to_numpy()
    tops = [_top_mask(absdx, q) for q in TOP_Q]
    n = len(d)

    rows = []
    for variant in VARIANTS:
        group, pool = _label(d, variant)
        perm = _permutation_p(d, group, pool, absdx, tops,
                              np.random.default_rng([SEED, VARIANTS.index(variant)]))
        for name in GROUPS:
            m = (group == name).to_numpy()
            if variant != "clean" and name == "excluded":
                continue
            row = {
                "variant": variant, "group": name,
                "n_increments": int(m.sum()),
                "n_contracts": int(d.loc[m, "contract_id"].nunique()),
                "share_of_increments": m.sum() / n,
                "mean_abs_dx": absdx[m].mean() if m.any() else np.nan,
                "median_abs_dx": np.median(absdx[m]) if m.any() else np.nan,
            }
            for q, top in zip(TOP_Q, tops, strict=True):
                key = f"top{round(q * 100)}"
                row[f"n_{key}"] = int((top & m).sum())
                row[f"share_{key}"] = (top & m).sum() / top.sum()
            row.update(perm if name == "mid_life_fomc"
                       else dict.fromkeys(perm, np.nan))
            row.update({"n_panel": n, "n_perm": N_PERM, "seed": SEED})
            rows.append(row)
    out = pd.DataFrame(rows)

    if verbose:
        _report(d, out)
    path = EXPORTS / "fomc_event_test.csv"
    out.to_csv(path, index=False, float_format="%.6f")
    if verbose:
        print(f"\n  Saved: {path}")
    return out


def _report(d: pd.DataFrame, out: pd.DataFrame) -> None:
    print("\n=== FOMC event test: |dX| on announcement increments (daily Fed panel) ===")
    print(f"  {len(d):,} increments, {d.contract_id.nunique()} contracts, "
          f"{N_PERM:,} within-contract shuffles, seed {SEED}")
    cols = ["group", "n_increments", "n_contracts", "share_of_increments", "mean_abs_dx",
            "median_abs_dx", "share_top1", "share_top5", "perm_p_mean_abs_dx",
            "perm_p_top1"]
    for variant, g in out.groupby("variant", sort=False):
        print(f"\n  [{variant}]")
        print("    " + g[cols].to_string(index=False, float_format="{:.4f}".format)
              .replace("\n", "\n    "))

    spec = out[out.variant == "spec"].set_index("group")
    n_fomc = spec.loc[["mid_life_fomc", "own_resolution_fomc"], "n_increments"].sum()
    fomc = d["end_day"].isin(_announcement_days())
    print("\n  Against the preliminary check (spec variant):")
    print(f"    FOMC increments    {n_fomc} across {d.loc[fomc, 'contract_id'].nunique()} "
          f"contracts   (prelim {PRELIM['n_fomc']} / {PRELIM['n_contracts']})")
    print(f"    mean |dX|          mid-life {spec.loc['mid_life_fomc', 'mean_abs_dx']:.3f} vs "
          f"non-FOMC {spec.loc['non_fomc', 'mean_abs_dx']:.3f}   "
          f"(prelim {PRELIM['mean_mid']} vs {PRELIM['mean_other']})")
    top1 = spec.loc[["mid_life_fomc", "own_resolution_fomc"], "share_top1"].sum()
    print(f"    top-1% on FOMC     {top1:.1%} of top moves; FOMC = {n_fomc / len(d):.1%} "
          f"of increments   (prelim {PRELIM['top1_share']:.0%} / {PRELIM['fomc_share']:.1%})")
    print(f"    permutation p      {spec.loc['mid_life_fomc', 'perm_p_mean_abs_dx']:.4f}"
          f"   (prelim {PRELIM['perm_p']})")


if __name__ == "__main__":
    run()
