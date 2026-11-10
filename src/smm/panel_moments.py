"""
§4.3 (v2)  Within-contract moments, the panel-shaped simulator and the
observation process.

Why this module exists
----------------------
The v1 estimator computed its shape moments on one long concatenated series
and simulated one long path.  Three things go wrong with that:

1. Contracts have very different variances (per-contract SD ranges from 0.07
   to 1.09 on the daily panel).  A mixture of contracts with different
   variances is fat-tailed and has autocorrelated squares even if no single
   contract is — so between-contract differences were read as jumps and
   volatility clustering.
2. 36.6% of the daily increments are exact zeros.  They come from how prices
   are observed (a 0.001/0.0005 tick, and the [0.02, 0.98] clip), not from
   the dynamics, and the simulator had no way to produce them.
3. The weighting matrix came from an iid bootstrap of increments, which
   destroys the serial dependence the ACF moments measure.

So here every shape moment is computed WITHIN a contract and then pooled, by
one function that both the real and the simulated side call; the simulator
produces a panel shaped like the real one (same contracts, same lengths, same
starting prices) and pushes it through the same clip and tick; and W comes
from resampling whole contracts.

The five moments
----------------
For contract c with n_c increments d (demeaned within the contract):

  [0] Var(ΔX)            pooled:  Σ_c Σ d² / (N − 1)              (as in v1)
  [1] Kurt(ΔX)           Σ_c n_c·kurt_c / Σ_c n_c,  kurt_c = m4_c/m2_c² − 3.
                         This IS the excess kurtosis of increments
                         standardised by each contract's own SD and pooled —
                         the two definitions coincide (ddof = 0).
  [2] ACF(ΔX², lag 1)    Σ_c (n_c−1)·acf_c(1) / Σ_c (n_c−1)
  [3] ACF(ΔX², lag 2)    Σ_c (n_c−2)·acf_c(2) / Σ_c (n_c−2)
  [4] 95th pct |ΔX|      Σ_c n_c·q_c / Σ_c n_c with q_c the 95th percentile of
                         |d| within the contract — divided by the contract's
                         own SD when q95="standardized" (default), in raw
                         log-odds units when q95="raw".  Raw is not scale
                         free: averaging q_c ∝ σ_c against a POOLED variance
                         leaves a Jensen gap (E[σ_c]/√E[σ_c²] = 0.82 on the
                         daily panel) that looks like "fat tails given the
                         variance" — exactly the between-contract effect this
                         module is meant to remove.

  [5] Zero share         (observation="tick_stale" only)  share of grid steps
                         on which the observed price did not change,
                         Σ_c zeros_c / Σ_c steps_c.

A contract enters a shape moment only if that moment is defined for it
(≥ MIN_SHAPE_N increments, non-zero variance; for the ACF, non-constant
squares).  The same rule applies to every simulated contract.

The simulated moment vector is the AVERAGE over S = n_sim_multiplier
replicate panels of the same statistic, so it estimates E[m(panel)] under the
model including the small-sample bias of kurtosis on short contracts, and
Var(m_sim) = Var(m_real)/S exactly — which is what the J scale assumes.

The observation process
-----------------------
"tick"        latent log-odds → price → clip → round to the tick → logit →
              difference.  Zeros arise when the price stays inside one tick
              or sits on the clip.
"tick_stale"  the same, plus a stale-price rule: on each step, with
              probability q, the observed price does not update (the latent
              one keeps moving, so the next update catches up in one larger
              increment).  This is the fallback the tick process needs on
              this data: a daily move in log-odds is many ticks wide, so
              rounding alone yields ~2% zeros against the 21% repeated
              prices in the daily panel.  The zero share enters as a sixth
              moment and q is CONCENTRATED OUT: for every parameter vector
              the simulator solves for the q that reproduces the real zero
              share (q = 0 if the model's own zeros already exceed it, in
              which case the sixth moment penalises the excess).  The
              latent paths do not depend on q, so this costs one cheap
              extra pass, and it spends one degree of freedom.

Numba is used when installed (≈200× faster) and is optional: without it the
same code runs as plain Python/NumPy and gives the same numbers.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:                                    # optional accelerator
    import numba

    def _jit(fn):
        return numba.njit(cache=True)(fn)

    HAVE_NUMBA = True
except ImportError:                     # pragma: no cover - exercised in CI
    def _jit(fn):
        return fn

    HAVE_NUMBA = False


# Per-contract sufficient statistics (one row per contract, or per simulated
# contract path).
I_N, I_SS, I_M4, I_ACF1, I_ACF2, I_Q95, I_ZCLIP, I_ZROUND, I_STEPS = range(9)
K_STATS = 9

MIN_SHAPE_N = 5          # increments a contract needs to enter a shape moment
DEFAULT_BURN_IN = 50
OBS_NONE, OBS_TICK = 0, 1
_X_CAP = 50.0            # |log-odds| cap inside expit (avoids exp overflow)


# ---------------------------------------------------------------------------
# Per-contract statistics — the ONE function both sides call
# ---------------------------------------------------------------------------

@_jit
def _acf_sq(x, lag):
    """Pearson correlation of x[:-lag] with x[lag:] (np.corrcoef convention)."""
    n = x.shape[0] - lag
    if n < 3:
        return np.nan
    a = x[:n]
    b = x[lag:]
    ma = a.mean()
    mb = b.mean()
    da = a - ma
    db = b - mb
    va = (da * da).sum()
    vb = (db * db).sum()
    # Constant squares (e.g. an all-zero stretch): correlation undefined.
    if va <= 1e-18 * n * ma * ma or vb <= 1e-18 * n * mb * mb:
        return np.nan
    return (da * db).sum() / math.sqrt(va * vb)


@_jit
def contract_stats(d, out):
    """Fill `out` (length K_STATS) with one contract's statistics.

    `d` is the contract's increments as observed (not demeaned); the mean is
    removed here, within the contract.  Slots I_ZCLIP / I_ZROUND are left
    and I_STEPS are left untouched — the caller knows how the zeros arose
    and how many grid steps the contract spans.
    """
    n = d.shape[0]
    out[I_N] = n
    out[I_SS] = 0.0
    out[I_M4] = 0.0
    out[I_ACF1] = np.nan
    out[I_ACF2] = np.nan
    out[I_Q95] = np.nan
    if n < 2:
        return
    dm = d - d.mean()
    d2 = dm * dm
    out[I_SS] = d2.sum()
    out[I_M4] = (d2 * d2).sum()
    out[I_ACF1] = _acf_sq(d2, 1)
    out[I_ACF2] = _acf_sq(d2, 2)
    a = np.sort(np.abs(dm))
    pos = 0.95 * (n - 1)                 # numpy's default linear interpolation
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    out[I_Q95] = a[lo] + (pos - lo) * (a[hi] - a[lo])


def _wavg(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Weighted mean over the last axis, restricted to `mask`; NaN if empty."""
    w = np.where(mask, weights, 0.0)
    v = np.where(mask, values, 0.0)
    tot = w.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, (w * v).sum(axis=-1) / tot, np.nan)


def pool_moments(stats: np.ndarray, q95: str = "standardized",
                 zero_share: bool = False) -> np.ndarray:
    """Per-contract statistics (..., C, K_STATS) → pooled moments (..., 5 or 6).

    zero_share=True appends the share of grid steps with no price change.
    """
    n = stats[..., I_N]
    ss = stats[..., I_SS]
    m4 = stats[..., I_M4]
    has = n >= 2
    big_n = np.where(has, n, 0.0).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.where(big_n > 1,
                       np.where(has, ss, 0.0).sum(axis=-1) / (big_n - 1.0), np.nan)
        shape_ok = (n >= MIN_SHAPE_N) & (ss > 0)
        ss_safe = np.where(shape_ok, ss, 1.0)
        kurt_c = n * m4 / (ss_safe * ss_safe) - 3.0
        sd_c = np.sqrt(ss_safe / np.maximum(n - 1.0, 1.0))
    kurt = _wavg(kurt_c, n, shape_ok)
    a1, a2 = stats[..., I_ACF1], stats[..., I_ACF2]
    acf1 = _wavg(a1, n - 1.0, shape_ok & np.isfinite(a1))
    acf2 = _wavg(a2, n - 2.0, shape_ok & np.isfinite(a2))
    q = stats[..., I_Q95]
    if q95 == "standardized":
        tail = _wavg(q / sd_c, n, shape_ok)
    elif q95 == "raw":
        tail = _wavg(q, n, (n >= MIN_SHAPE_N) & np.isfinite(q))
    else:
        raise ValueError(f"q95 must be 'standardized' or 'raw', got {q95!r}")
    cols = [var, kurt, acf1, acf2, tail]
    if zero_share:
        steps = stats[..., I_STEPS].sum(axis=-1)
        zeros = (stats[..., I_ZCLIP] + stats[..., I_ZROUND]).sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cols.append(np.where(steps > 0, zeros / steps, np.nan))
    return np.stack(cols, axis=-1)


def mean_over_replicates(moments: np.ndarray) -> np.ndarray:
    """(S, 5) per-replicate moments → (5,) simulated moment vector."""
    if moments.shape[0] == 1:
        return moments[0]
    out = np.full(moments.shape[1], np.nan)
    for j in range(moments.shape[1]):
        col = moments[:, j]
        ok = np.isfinite(col)
        if ok.any():
            out[j] = col[ok].mean()
    return out


# ---------------------------------------------------------------------------
# The real panel's shape
# ---------------------------------------------------------------------------

def panel_shape(panel: pd.DataFrame, clip: tuple[float, float]) -> dict:
    """Everything the simulator needs to produce a panel shaped like `panel`.

    Returns contract ids, increments per contract `n`, the calendar span of
    each contract in base-grid steps `span` (= n under calendar sampling,
    Σ dt_steps under revision sampling), the latent starting log-odds `x0`,
    the per-contract statistics `stats` (C, K_STATS) and whether the panel
    is revision-sampled.

    x0 is the contract's first observed log-odds, X₁ − ΔX₁.  When that first
    price sits on the clip and the next raw price is beyond it, the raw
    price is used so the latent path starts where the market actually was.
    Panels without price columns (increments only) start at p = 0.5.
    """
    lo, hi = clip
    ids, n_list, span_list, x0_list, rows = [], [], [], [], []
    has_x = "X" in panel.columns
    has_p = "p" in panel.columns
    has_raw = "p_raw" in panel.columns
    has_dt = "dt_steps" in panel.columns
    revision = bool(has_dt and (panel["dt_steps"].dropna() > 1).any())
    eps = 1e-9
    for cid, grp in panel.groupby("contract_id", sort=False):
        grp = grp[grp["delta_X"].notna()]
        d = grp["delta_X"].to_numpy(dtype=float)
        if len(d) < 2:
            continue
        row = np.zeros(K_STATS)
        contract_stats(d, row)
        if has_p:
            p = grp["p"].to_numpy(dtype=float)
            at_clip = (p <= lo + eps) | (p >= hi - eps)
            row[I_ZCLIP] = float(((d == 0.0) & at_clip).sum())
            row[I_ZROUND] = float(((d == 0.0) & ~at_clip).sum())
        else:
            row[I_ZROUND] = float((d == 0.0).sum())
        x0 = 0.0
        if has_x:
            x0 = float(grp["X"].iloc[0] - d[0])
            p0 = 1.0 / (1.0 + math.exp(-x0))
            if has_raw and (p0 <= lo + eps or p0 >= hi - eps):
                raw = float(grp["p_raw"].iloc[0])
                if 0.0 < raw < 1.0 and (raw < lo - eps or raw > hi + eps):
                    x0 = math.log(raw / (1.0 - raw))
        span = int(round(float(grp["dt_steps"].sum()))) if revision else len(d)
        span = max(span, len(d))
        row[I_STEPS] = span
        if revision:
            # Collapsed away by the sampling; counted from the calendar span
            # (clip vs other is not recoverable, so all are filed as "other").
            row[I_ZCLIP] = 0.0
            row[I_ZROUND] = float(span - len(d))
        ids.append(cid)
        n_list.append(len(d))
        span_list.append(span)
        x0_list.append(x0)
        rows.append(row)
    if not rows:
        raise ValueError("panel has no contract with at least 2 increments")
    return {
        "ids": ids,
        "n": np.asarray(n_list, dtype=np.int64),
        "span": np.asarray(span_list, dtype=np.int64),
        "x0": np.asarray(x0_list, dtype=float),
        "stats": np.vstack(rows),
        "revision": revision,
    }


def zero_shares(stats: np.ndarray) -> dict:
    """Shares of grid steps with no price change (any leading dims).

    "clip": the price sits on the clip on both days; "other": every other
    repeated price (rounding inside one tick, or a stale quote).
    """
    n = max(float(stats[..., I_STEPS].sum()), 1.0)
    zc = float(stats[..., I_ZCLIP].sum())
    zr = float(stats[..., I_ZROUND].sum())
    return {"zero_share": (zc + zr) / n, "clip_zero_share": zc / n,
            "other_zero_share": zr / n}


def contract_bootstrap_weights(
    stats: np.ndarray, n_bootstrap: int, rng: np.random.Generator,
    q95: str = "standardized", zero_share: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """W = diag(1 / variance) from resampling WHOLE contracts with replacement.

    Resampling contracts (not increments) keeps each contract's serial
    dependence intact, so the ACF moments get an honest variance.  Returns
    (W, variances).
    """
    n_c = stats.shape[0]
    idx = rng.integers(0, n_c, size=(n_bootstrap, n_c))
    draws = pool_moments(stats[idx], q95=q95, zero_share=zero_share)   # (B, 5|6)
    variances = np.nanvar(draws, axis=0, ddof=1)
    variances = np.where(np.isfinite(variances) & (variances > 1e-30), variances, 1.0)
    return np.diag(1.0 / variances), variances


# ---------------------------------------------------------------------------
# Simulator: latent path → observation process → observed increments
# ---------------------------------------------------------------------------

@_jit
def _tick_index(x, tick, k_lo, k_hi):
    xc = min(max(x, -_X_CAP), _X_CAP)
    p = 1.0 / (1.0 + math.exp(-xc))
    k = int(math.floor(p / tick + 0.5))
    if k < k_lo:
        k = k_lo
    elif k > k_hi:
        k = k_hi
    return k


@_jit
def _latent_path(sigma_v, sigma_J, kappa, theta, lam, n, burn,
                 dw_x, dw_v, u_jump, z_jump, pos, dx_out):
    """Latent increments of one contract (the v1 Euler scheme, Δt = 1, full
    truncation).  The variance runs `burn` steps from v = θ before the first
    increment.  Uses randoms[pos : pos + burn + n]; writes dx_out[:n]."""
    v = theta
    for i in range(burn):
        v_pos = max(v, 0.0)
        sv = math.sqrt(v_pos)
        v = v_pos + kappa * (theta - v_pos) + sigma_v * sv * dw_v[pos + i]
        v = max(v, 0.0)
    pos += burn
    for i in range(n):
        v_pos = max(v, 0.0)
        sv = math.sqrt(v_pos)
        dx = sv * dw_x[pos + i]
        if u_jump[pos + i] < lam:
            dx += sigma_J * z_jump[pos + i]
        dx_out[i] = dx
        v = v_pos + kappa * (theta - v_pos) + sigma_v * sv * dw_v[pos + i]
        v = max(v, 0.0)


@_jit
def _latent_ticks(dx, n, x0, tick, k_lo, k_hi, k_out):
    """Latent log-odds path → the tick index of its clipped, rounded price."""
    x = x0
    for i in range(n):
        x += dx[i]
        k_out[i] = _tick_index(x, tick, k_lo, k_hi)


@_jit
def _observe(k_lat, n, k0, u_stale, upos, stale_q, tick, k_lo, k_hi, collapse,
             d_out, k_out, gap_out):
    """Observed increments from latent tick indices.

    On each step the observed price updates to the latent one, unless
    (probability stale_q) the quote is stale and it stays put.  An increment
    is exactly 0 when the observed tick index does not change; it is a "clip
    zero" if that index is a clip bound.  collapse drops the zeros
    (revision-time sampling); gap_out is then the grid steps each surviving
    increment spans.  Returns (m, n_clip_zeros, n_other_zeros).
    """
    k_prev = k0
    m = 0
    zc = 0
    zr = 0
    gap = 0
    for i in range(n):
        gap += 1
        k = k_lat[i]
        if stale_q > 0.0 and u_stale[upos + i] < stale_q:
            k = k_prev
        if k == k_prev:
            if k == k_lo or k == k_hi:
                zc += 1
            else:
                zr += 1
            if not collapse:
                d_out[m] = 0.0
                k_out[m] = k
                gap_out[m] = 1
                m += 1
        else:
            pa = k * tick
            pb = k_prev * tick
            d_out[m] = math.log(pa / (1.0 - pa)) - math.log(pb / (1.0 - pb))
            k_out[m] = k
            gap_out[m] = gap if collapse else 1
            m += 1
            k_prev = k
            gap = 0
    return m, zc, zr


@_jit
def _count_zeros(k_all, k0_all, starts, lens, ustarts, u_stale, stale_q):
    """Unchanged-price steps over all simulated paths at a given stale_q."""
    zeros = 0
    for j in range(starts.shape[0]):
        k_prev = k0_all[j]
        a = starts[j]
        u = ustarts[j]
        for i in range(lens[j]):
            k = k_all[a + i]
            if stale_q > 0.0 and u_stale[u + i] < stale_q:
                k = k_prev
            if k == k_prev:
                zeros += 1
            k_prev = k
    return zeros


@_jit
def _solve_stale_q(k_all, k0_all, starts, lens, ustarts, u_stale, target, total):
    """The stale-quote probability at which the simulated share of unchanged
    prices equals `target` (0 if the model's own zeros already exceed it).

    Bisection on a fixed bracket: deterministic in the parameters, and each
    step only re-reads the stored latent ticks.
    """
    if _count_zeros(k_all, k0_all, starts, lens, ustarts, u_stale, 0.0) >= target * total:
        return 0.0
    lo = 0.0
    hi = 0.99
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        z = _count_zeros(k_all, k0_all, starts, lens, ustarts, u_stale, mid)
        if z >= target * total:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@_jit
def _simulate_panel_stats(sigma_v, sigma_J, kappa, theta, lam, n_steps, x0, n_rep, burn,
                          dw_x, dw_v, u_jump, z_jump, u_stale,
                          obs, tick, k_lo, k_hi, stale_q, zero_target, collapse, out):
    """Simulate n_rep replicate panels; write per-contract stats to out[s, c].

    zero_target ≥ 0 (with obs = OBS_TICK) concentrates stale_q out: it is
    solved so the simulated zero share equals zero_target.  Returns the
    stale_q used.
    """
    n_c = n_steps.shape[0]
    max_n = 0
    total = 0
    for c in range(n_c):
        if n_steps[c] > max_n:
            max_n = n_steps[c]
        total += n_steps[c]
    total *= n_rep
    dx = np.empty(max_n)
    d = np.empty(max_n)
    k_buf = np.empty(max_n, dtype=np.int64)
    gap_buf = np.empty(max_n, dtype=np.int64)

    if obs == OBS_NONE:
        pos = 0
        for c in range(n_c):
            n = n_steps[c]
            for s in range(n_rep):
                _latent_path(sigma_v, sigma_J, kappa, theta, lam, n, burn,
                             dw_x, dw_v, u_jump, z_jump, pos, dx)
                pos += burn + n
                contract_stats(dx[:n], out[s, c])
                out[s, c, I_ZCLIP] = 0.0
                out[s, c, I_ZROUND] = 0.0
                out[s, c, I_STEPS] = n
        return 0.0

    n_paths = n_c * n_rep
    k_all = np.empty(total, dtype=np.int64)
    k0_all = np.empty(n_paths, dtype=np.int64)
    starts = np.empty(n_paths, dtype=np.int64)
    lens = np.empty(n_paths, dtype=np.int64)
    ustarts = np.empty(n_paths, dtype=np.int64)
    pos = 0
    a = 0
    j = 0
    for c in range(n_c):
        n = n_steps[c]
        k0 = _tick_index(x0[c], tick, k_lo, k_hi)
        for s in range(n_rep):
            _latent_path(sigma_v, sigma_J, kappa, theta, lam, n, burn,
                         dw_x, dw_v, u_jump, z_jump, pos, dx)
            _latent_ticks(dx, n, x0[c], tick, k_lo, k_hi, k_all[a:a + n])
            k0_all[j] = k0
            starts[j] = a
            lens[j] = n
            ustarts[j] = pos + burn
            pos += burn + n
            a += n
            j += 1

    q = stale_q
    if zero_target >= 0.0:
        q = _solve_stale_q(k_all, k0_all, starts, lens, ustarts, u_stale,
                           zero_target, total)

    j = 0
    for c in range(n_c):
        for s in range(n_rep):
            n = lens[j]
            m, zc, zr = _observe(k_all[starts[j]:starts[j] + n], n, k0_all[j], u_stale,
                                 ustarts[j], q, tick, k_lo, k_hi, collapse,
                                 d, k_buf, gap_buf)
            contract_stats(d[:m], out[s, c])
            out[s, c, I_ZCLIP] = zc
            out[s, c, I_ZROUND] = zr
            out[s, c, I_STEPS] = n
            j += 1
    return q


# ---------------------------------------------------------------------------
# Python-facing wrappers
# ---------------------------------------------------------------------------

def tick_bounds(tick: float, clip: tuple[float, float]) -> tuple[int, int]:
    """Clip bounds as tick indices."""
    return int(round(clip[0] / tick)), int(round(clip[1] / tick))


def draw_panel_randoms(n_steps: np.ndarray, n_rep: int, burn: int,
                       rng: np.random.Generator, rho: float = 0.0) -> dict:
    """Common random numbers for n_rep replicate panels, drawn once.

    Laid out contract by contract, replicate by replicate, each block
    `burn + n_steps[c]` long.  Reused on every objective evaluation so the
    objective is a deterministic function of the parameters.
    """
    total = int(n_rep * (int(n_steps.sum()) + burn * len(n_steps)))
    dw_x = rng.standard_normal(total)
    dw_v_indep = rng.standard_normal(total)
    u_jump = rng.uniform(size=total)
    z_jump = rng.standard_normal(total)
    u_stale = rng.uniform(size=total)
    dw_v = rho * dw_x + math.sqrt(max(1.0 - rho * rho, 0.0)) * dw_v_indep
    return {"kind": "panel", "dw_x": dw_x, "dw_v": dw_v, "u_jump": u_jump,
            "z_jump": z_jump, "u_stale": u_stale,
            "n_steps": np.asarray(n_steps, dtype=np.int64),
            "n_rep": int(n_rep), "burn": int(burn)}


def simulate_panel_stats(sigma_v: float, sigma_J: float, kappa: float, theta: float,
                         lambda_: float, randoms: dict, x0: np.ndarray,
                         observation: str = "none", tick: float = 0.001,
                         clip: tuple[float, float] = (0.02, 0.98),
                         collapse: bool = False, stale_q: float = 0.0,
                         zero_target: "float | None" = None,
                         ) -> tuple[np.ndarray, float]:
    """Per-contract statistics of n_rep simulated panels.

    Returns (stats, stale_q) with stats of shape (n_rep, C, K_STATS).  With
    observation="tick_stale" and a zero_target, stale_q is solved to match
    it and returned; otherwise the stale_q passed in is used as is.
    """
    n_steps = randoms["n_steps"]
    out = np.zeros((randoms["n_rep"], len(n_steps), K_STATS))
    obs = OBS_NONE if observation == "none" else OBS_TICK
    k_lo, k_hi = tick_bounds(tick, clip)
    stale = observation == "tick_stale"
    q = _simulate_panel_stats(
        float(sigma_v), float(sigma_J), float(kappa), float(theta), float(lambda_),
        n_steps, np.asarray(x0, dtype=float), randoms["n_rep"], randoms["burn"],
        randoms["dw_x"], randoms["dw_v"], randoms["u_jump"], randoms["z_jump"],
        randoms["u_stale"], obs, float(tick), k_lo, k_hi,
        float(stale_q) if stale else 0.0,
        float(zero_target) if (stale and zero_target is not None) else -1.0,
        bool(collapse and obs == OBS_TICK), out)
    return out, float(q)


def simulate_observed_panel(
    params: dict, ids: list, n_steps: np.ndarray, x0: np.ndarray,
    rng: np.random.Generator, observation: str = "tick_stale", tick: float = 0.001,
    clip: tuple[float, float] = (0.02, 0.98), collapse: bool = False,
    burn: int = DEFAULT_BURN_IN,
) -> pd.DataFrame:
    """One fake panel from known parameters, through the observation process.

    `params` holds the LATENT data-generating values: sigma_v, sigma_J, kappa,
    theta, lambda_, and (for "tick_stale") stale_q.  The output has the real
    panel's schema (contract_id, t, p_raw, p, X, delta_X, dt_steps), so it
    can be fed to the estimator exactly like real data.  Used for the
    null/power validation and for the parametric bootstrap.
    """
    obs = OBS_NONE if observation == "none" else OBS_TICK
    do_collapse = bool(collapse and obs == OBS_TICK)
    stale_q = float(params.get("stale_q", 0.0)) if observation == "tick_stale" else 0.0
    k_lo, k_hi = tick_bounds(tick, clip)
    parts = []
    for cid, n, start in zip(ids, n_steps, x0):
        n = int(n)
        m_tot = n + burn
        dw_x = rng.standard_normal(m_tot)
        dw_v = rng.standard_normal(m_tot)
        u_jump = rng.uniform(size=m_tot)
        z_jump = rng.standard_normal(m_tot)
        u_stale = rng.uniform(size=m_tot)
        dx = np.empty(n)
        _latent_path(float(params["sigma_v"]), float(params["sigma_J"]),
                     float(params["kappa"]), float(params["theta"]),
                     float(params["lambda_"]), n, burn,
                     dw_x, dw_v, u_jump, z_jump, 0, dx)
        if obs == OBS_NONE:
            d, gap = dx, np.ones(n)
            x_obs = float(start) + np.cumsum(d)
            p = np.clip(1.0 / (1.0 + np.exp(-np.clip(x_obs, -_X_CAP, _X_CAP))), *clip)
        else:
            k_lat = np.empty(n, dtype=np.int64)
            _latent_ticks(dx, n, float(start), float(tick), k_lo, k_hi, k_lat)
            d = np.empty(n)
            k_buf = np.empty(n, dtype=np.int64)
            gap = np.empty(n, dtype=np.int64)
            m, _, _ = _observe(k_lat, n, _tick_index(float(start), float(tick), k_lo, k_hi),
                               u_stale, burn, stale_q, float(tick), k_lo, k_hi,
                               do_collapse, d, k_buf, gap)
            d, gap = d[:m], gap[:m].astype(float)
            p = k_buf[:m] * tick
            x_obs = np.log(p / (1.0 - p))
        parts.append(pd.DataFrame({
            "contract_id": cid, "t": np.arange(1, len(d) + 1), "p_raw": p, "p": p,
            "X": x_obs, "delta_X": d, "dt_steps": gap,
        }))
    return pd.concat(parts, ignore_index=True)
