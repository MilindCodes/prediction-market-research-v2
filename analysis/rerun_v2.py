"""Rerun of §4 under the corrected (v2) SMM estimator, and its validation.

Everything here writes under data/processed/v2/ and data/exports/v2/ (or a
sub-folder of them for a variant).  The v1 CSVs the first draft cites are
never touched; they stay reproducible from the tag v1-paper-draft.

    python -m analysis.rerun_v2 ladders                 # full ladder, all six panels
    python -m analysis.rerun_v2 sensitivities           # truncation + bucketing
    python -m analysis.rerun_v2 bootstrap               # ΔJ bootstrap, all six panels
    python -m analysis.rerun_v2 validate                # null / power / recovery CSVs
    python -m analysis.rerun_v2 compare                 # old-vs-new tables

    python -m analysis.rerun_v2 ladders --variant scaling_only     # Step 1 only
    python -m analysis.rerun_v2 bootstrap --specs D/calendar --reps 40

Variants (`--variant`) switch individual fixes on or off so each one's effect
can be read off directly; see VARIANTS.  "v2" is the estimator the paper uses.

The bootstrap
-------------
For each panel the null is that panel's fitted Heston model (no jumps),
simulated as a panel shaped like the real one — same contracts, lengths and
starting prices — and pushed through the SAME observation process the
estimator assumes (clip, tick, stale quotes at the fitted probability,
revision sampling where the panel is revision-sampled).  So the zeros are in
the null by construction.  Each fake panel goes through the full ladder and
p = (#{simulated ΔJ ≥ real ΔJ} + 1) / (N + 1).  `--power` panels are drawn
from the fitted Bates model instead.

The D/calendar null runs double as the size check of the χ² test (how often
does χ²(1) at 5% select jumps when there are none?), the D/calendar Bates
runs as the power check and the parameter-recovery exercise.

Replications are seeded from (SEED, variant, spec, null model, rep): results
do not depend on worker count or order, and an interrupted run resumes.
Numba is needed in practice (pip install numba): a ladder takes ~90 s with
it and hours without.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.smm import panel_moments as pm
from src.smm.bates_smm import BatesSMM
from src.smm.nested_ladder import (
    NestedLadder,
    bucketing_analysis,
    processed_dir,
    sampling_rows,
    truncation_sensitivity,
)
from src.smm.panel import SAMPLING_SPEC_LABELS, load_sampling_panel

SEED = 20261110
KAPPA_INIT = 5.0
MAIN_RUN = {"n_sim_multiplier": 20, "n_bootstrap": 400, "n_restarts": 3}

SPECS = SAMPLING_SPEC_LABELS
NULL_MODELS = ("Heston", "Bates", "ConstantVol", "Merton")    # append only (seed)
REPS = {"D/calendar": 100, "D/revision": 100}       # others: DEFAULT_REPS
DEFAULT_REPS = 50
POWER_SPEC, POWER_REPS = "D/calendar", 100
CV_NULL_REPS = 50            # constant-vol panels: size of the "need SV?" test
MERTON_NULL_REPS = 50        # jump-only panels: size of "need SV, given jumps?"
_SUFFIX = {"Heston": "", "Bates": "_bates_power", "ConstantVol": "_constantvol_null",
           "Merton": "_merton_null"}
MAX_POOL_RESTARTS = 20

# name → (BatesSMM options on top of MAIN_RUN, start from legacy?).  Order is
# part of the seed: append, never reorder.
VARIANTS: dict[str, tuple[dict, bool]] = {
    # The estimator the paper uses: all three fixes, stale-quote observation
    # process, variance-targeted θ.
    "v2": ({}, False),
    # Step 1 only — v1 with the J / SE scaling corrected.
    "scaling_only": ({"j_scaling": "corrected"}, True),
    # Steps 1 + 2 — within-contract moments and contract-bootstrap W, no
    # observation process, v1 θ plug-in.
    "within_contract": ({"observation": "none", "theta_mode": "truncated"}, False),
    # Steps 1 + 2 + the tick/clip process as first specified (no stale quotes).
    "tick_only": ({"observation": "tick", "theta_mode": "truncated"}, False),
    # All three fixes with the v1 θ plug-in (truncated variance).
    "theta_truncated": ({"theta_mode": "truncated"}, False),
    # Main estimator, tail moment in raw log-odds units instead of contract SDs.
    "q95_raw": ({"q95": "raw"}, False),
    # Main estimator at the finer tick.
    "tick_0005": ({"tick": 0.0005}, False),
    # Main estimator on panels without each contract's FIRST increment.  Ten
    # of the 39 daily contracts open within 0.02 of 0.50 (a listing
    # placeholder) and move to their real level on day one; those 39
    # increments are 0.9% of the sample but 19% of the pooled fourth moment.
    "drop_opening": ({}, False),
}


def make_calibrator(variant: str = "v2", **overrides) -> BatesSMM:
    opts, from_legacy = VARIANTS[variant]
    kwargs = {**MAIN_RUN, **opts,
              "output_subdir": "v2" if variant == "v2" else f"v2/{variant}", **overrides}
    return BatesSMM.legacy(**kwargs) if from_legacy else BatesSMM(**kwargs)


def exports_dir(variant: str) -> Path:
    out = config.DATA_DIR / "exports" / make_calibrator(variant).output_subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def _quiet_panel(spec: str, variant: str = "v2") -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):     # the builder prints
        panel = load_sampling_panel(spec)
    if variant == "drop_opening":
        first = panel.groupby("contract_id", sort=False).cumcount() == 0
        panel = panel[~first].reset_index(drop=True)
    return panel


def _quiet_workers() -> None:
    # np.corrcoef on a zero-variance trial path (legacy pooled moments only).
    warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)


def _slug(spec: str) -> str:
    return spec.lower().replace("/", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# ladders — the full ladder on every sampling panel
# ---------------------------------------------------------------------------

def _ladder_one(task: tuple[str, str]) -> dict:
    variant, spec = task
    _quiet_workers()
    cal = make_calibrator(variant)
    if cal.moments == "pooled" and pm.HAVE_NUMBA:
        # The v1 long-path simulator is pure Python (~5 min a ladder); the
        # v1 bootstrap's Numba copy is bit-for-bit identical to it.
        from analysis import bootstrap_diffj
        bootstrap_diffj._install()
    panel = _quiet_panel(spec, variant)
    freq, sampling = spec.split("/")
    ladder = NestedLadder(calibrator=cal, kappa_init=KAPPA_INIT)
    main = spec == "D/calendar"      # the §4.4 headline fit
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        lr = ladder.run_selection(
            panel, verbose=False,
            save_as="smm_ladder_results_main.csv" if main else None,
            moment_table_as="smm_moment_table_main.csv" if main else None)
    row, full = sampling_rows(freq, sampling, panel, lr, cal, ladder.last_cache)
    row["seconds"] = round(time.time() - t0, 1)
    return {"spec": spec, "row": row, "full": full}


def run_ladders(variant: str, specs: list[str], workers: int) -> pd.DataFrame:
    import multiprocessing as mp
    out_dir = processed_dir(make_calibrator(variant))
    results = {}
    with ProcessPoolExecutor(min(workers, len(specs)),
                             mp_context=mp.get_context("spawn")) as ex:
        futures = {ex.submit(_ladder_one, (variant, s)): s for s in specs}
        for fut in as_completed(futures):
            res = fut.result()
            results[res["spec"]] = res
            r = res["row"]
            print(f"  {res['spec']:<20} J(H)={r['j_heston']:8.2f} J(B)={r['j_bates']:8.2f} "
                  f"ΔJ(SV)={r['diffj_sv']:7.2f} ΔJ(jumps)={r['diffj_jumps']:7.2f} "
                  f"[{r['seconds']:.0f}s]", flush=True)
    done = [s for s in SPECS if s in results]
    summary = pd.DataFrame([results[s]["row"] for s in done])
    full = pd.DataFrame([r for s in done for r in results[s]["full"]])
    # Merge with rows from specs fitted in an earlier call.
    for name, df in (("smm_sampling_sensitivity.csv", summary),
                     ("smm_sampling_ladder_full.csv", full)):
        path = out_dir / name
        if path.exists():
            old = pd.read_csv(path)
            df = pd.concat([old[~old["spec"].isin(done)], df], ignore_index=True)
        order = {s: i for i, s in enumerate(SPECS)}
        df = df.sort_values("spec", key=lambda c: c.map(order), kind="stable")
        df.to_csv(path, index=False, float_format="%.6f")
        print(f"Saved: {path}")
    return summary


def run_sensitivities(variant: str) -> None:
    cal = make_calibrator(variant)
    panel = _quiet_panel("D/calendar", variant)
    truncation_sensitivity(panel, cal, kappa=KAPPA_INIT)
    bucketing_analysis(panel, cal, kappa=KAPPA_INIT, n_buckets=3)


# ---------------------------------------------------------------------------
# bootstrap — one replication (runs in a worker)
# ---------------------------------------------------------------------------

_CACHE: dict = {}
_GUARD = re.compile(r"J_restricted < J_full: (\w+) J=(\S+) vs (\w+) J=(\S+) \(ΔJ=(\S+)\)")


def ladder_csv(variant: str) -> Path:
    return processed_dir(make_calibrator(variant)) / "smm_sampling_ladder_full.csv"


def ladder_row(variant: str, spec: str, model: str) -> pd.Series:
    df = pd.read_csv(ladder_csv(variant))
    row = df[(df["spec"] == spec) & (df["model"] == model)]
    if len(row) != 1:
        raise KeyError(f"{spec}/{model} not in {ladder_csv(variant)} — run `ladders` first")
    return row.iloc[0]


def real_diff_j_sv(variant: str, spec: str) -> float:
    return float(ladder_row(variant, spec, "ConstantVol")["j_stat"]
                 - ladder_row(variant, spec, "Heston")["j_stat"])


def real_diff_j(variant: str, spec: str) -> float:
    return float(ladder_row(variant, spec, "Heston")["j_stat"]
                 - ladder_row(variant, spec, "Bates")["j_stat"])


def dgp_params(variant: str, spec: str, null_model: str) -> dict:
    """LATENT data-generating parameters of one fitted rung of the ladder
    (Heston and ConstantVol have no jumps; Bates and Merton do), with the
    stale-quote probability that rung solved for."""
    cal = make_calibrator(variant)
    r = ladder_row(variant, spec, null_model)
    jumps = null_model in ("Bates", "Merton")
    sigma_j = float(r["sigma_J"]) if jumps else 0.0
    lam = float(r["lambda"]) if jumps else 0.0
    stale = float(r["stale_q"]) if "stale_q" in r and np.isfinite(r["stale_q"]) else 0.0
    return {"sigma_v": float(r["sigma_v"]), "kappa": float(r["kappa"]),
            "theta": cal.latent_theta(float(r["theta"]), lam, sigma_j),
            "sigma_J": sigma_j, "lambda_": lam, "stale_q": stale}


def _shape(variant: str, spec: str) -> dict:
    key = (variant, spec)
    if key not in _CACHE:
        cal = make_calibrator(variant)
        shape = pm.panel_shape(_quiet_panel(spec, variant), cal.clip)
        collapse = bool(shape["revision"] and cal.observation != "none")
        shape["collapse"] = collapse
        shape["n_steps"] = shape["span"] if collapse else shape["n"]
        _CACHE[key] = shape
    return _CACHE[key]


def _seed(variant: str, spec: str, null_model: str, rep: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([SEED, list(VARIANTS).index(variant), SPECS.index(spec),
                                   NULL_MODELS.index(null_model), rep])


def simulate_fake_panel(variant: str, spec: str, null_model: str, rep: int) -> pd.DataFrame:
    cal = make_calibrator(variant)
    shape = _shape(variant, spec)
    return pm.simulate_observed_panel(
        dgp_params(variant, spec, null_model), shape["ids"], shape["n_steps"], shape["x0"],
        np.random.default_rng(_seed(variant, spec, null_model, rep)),
        observation=cal.observation, tick=cal.tick, clip=cal.clip,
        collapse=shape["collapse"], burn=cal.burn_in)


def run_rep(task: tuple[str, str, str, int]) -> dict:
    variant, spec, null_model, rep = task
    _quiet_workers()
    cal = make_calibrator(variant)
    params = dgp_params(variant, spec, null_model)
    t0, c0 = time.time(), time.process_time()
    row = {"variant": variant, "spec": spec, "null_model": null_model, "rep": rep,
           **{f"dgp_{k.rstrip('_')}": v for k, v in params.items()}, "error": ""}
    try:
        panel = simulate_fake_panel(variant, spec, null_model, rep)
        dx = panel["delta_X"].to_numpy()
        row.update({"n_contracts": int(panel["contract_id"].nunique()),
                    "n_increments": len(dx),
                    "zero_frac": float(np.mean(dx == 0.0)) if len(dx) else np.nan})
        ladder = NestedLadder(calibrator=cal, kappa_init=KAPPA_INIT)
        with contextlib.redirect_stdout(io.StringIO()):
            lr = ladder.run_selection(panel, verbose=False, save_as=None,
                                      moment_table_as=None)
        row.update({
            "theta_hat": lr.bates.theta, "lambda_hat": lr.bates.lambda_,
            "j_cv": lr.constant_vol.j_stat, "j_heston": lr.heston.j_stat,
            "j_merton": lr.merton.j_stat, "j_bates": lr.bates.j_stat,
            "p_j_heston": lr.heston.j_pvalue, "p_j_bates": lr.bates.j_pvalue,
            "diff_j_sv": lr.test_sv.diff_j, "p_chi2_sv": lr.test_sv.p_value,
            "diff_j": lr.test_jumps.diff_j, "p_chi2_jumps": lr.test_jumps.p_value,
            "heston_sigma_v": lr.heston.sigma_v, "heston_kappa": lr.heston.kappa,
            "merton_sigma_J": lr.merton.sigma_J,
            "bates_sigma_v": lr.bates.sigma_v, "bates_sigma_J": lr.bates.sigma_J,
            "bates_kappa": lr.bates.kappa,
            "heston_stale_q": lr.heston.stale_q, "bates_stale_q": lr.bates.stale_q,
        })
    except AssertionError as e:          # diff_j_test's containment guard
        row["error"] = str(e).splitlines()[0]
        m = _GUARD.search(str(e))
        if m and (m.group(1), m.group(3)) == ("Heston", "Bates"):
            row.update({"j_heston": float(m.group(2)), "j_bates": float(m.group(4)),
                        "diff_j": float(m.group(5))})
    except Exception as e:               # anything else: keep the run, say why
        row["error"] = f"{type(e).__name__}: {e}"
    row["seconds"] = round(time.time() - t0, 1)
    row["cpu_seconds"] = round(time.process_time() - c0, 1)
    return row


# ---------------------------------------------------------------------------
# bootstrap — driver
# ---------------------------------------------------------------------------

def _out_path(variant: str, spec: str, null_model: str) -> Path:
    return exports_dir(variant) / f"bootstrap_diffj_{_slug(spec)}{_SUFFIX[null_model]}.csv"


def _load_done(variant: str, spec: str, null_model: str) -> pd.DataFrame:
    path = _out_path(variant, spec, null_model)
    return pd.read_csv(path, keep_default_na=False, na_values=[""]) if path.exists() \
        else pd.DataFrame()


def _write(variant: str, spec: str, null_model: str, rows: list[dict]) -> None:
    df = pd.DataFrame(rows).sort_values("rep").reset_index(drop=True)
    df["real_diff_j"] = real_diff_j(variant, spec)
    df["exceeds_real"] = df["diff_j"] >= df["real_diff_j"]
    df.to_csv(_out_path(variant, spec, null_model), index=False, float_format="%.6f")


def summarise(variant: str, spec: str, null_model: str, df: pd.DataFrame) -> dict:
    ok = df[df["error"].fillna("") == ""]
    n_err = len(df) - len(ok)
    real = real_diff_j(variant, spec)
    ge = int((ok["diff_j"] >= real).sum())
    q = ok["diff_j"].quantile([0.5, 0.9, 0.95, 0.99]) if len(ok) else pd.Series(dtype=float)
    chi2_p_real = float(1.0 - stats.chi2.cdf(max(real, 0.0), df=1))
    return {
        "spec": spec, "null_model": null_model, "n_reps": len(df), "n_ok": len(ok),
        "n_errors": n_err, "real_diff_j": real, "chi2_p_real": chi2_p_real,
        "n_ge_real": ge,
        "p_value": (ge + 1) / (len(ok) + 1),
        "p_conservative": (ge + n_err + 1) / (len(df) + 1),
        "sim_median": q.get(0.5, np.nan), "sim_q90": q.get(0.9, np.nan),
        "sim_q95": q.get(0.95, np.nan), "sim_q99": q.get(0.99, np.nan),
        "sim_max": ok["diff_j"].max() if len(ok) else np.nan,
        "chi2_rejects_share": float((ok["p_chi2_jumps"] < 0.05).mean()) if len(ok) else np.nan,
    }


def _jobs(variant: str, specs: list[str], reps: int | None, power: bool) -> list[tuple]:
    jobs = [(s, "Heston", reps or REPS.get(s, DEFAULT_REPS)) for s in specs]
    if power and POWER_SPEC in specs:
        jobs.append((POWER_SPEC, "Bates", reps or POWER_REPS))
        jobs.append((POWER_SPEC, "ConstantVol", reps or CV_NULL_REPS))
        jobs.append((POWER_SPEC, "Merton", reps or MERTON_NULL_REPS))
    return jobs


def _write_summary(variant: str, jobs: list[tuple]) -> pd.DataFrame:
    path = exports_dir(variant) / "bootstrap_diffj_summary.csv"
    rows = []
    for spec in SPECS:                       # every finished file, not just this call's
        for null_model in NULL_MODELS:
            df = _load_done(variant, spec, null_model)
            if len(df):
                rows.append(summarise(variant, spec, null_model, df))
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False, float_format="%.6f")
    return out


def run_bootstrap(variant: str, specs: list[str], reps: int | None, power: bool,
                  workers: int) -> pd.DataFrame:
    import multiprocessing as mp
    if not pm.HAVE_NUMBA:
        print("WARNING: numba is not installed — each ladder will take hours, not seconds.")
    jobs = _jobs(variant, specs, reps, power)
    rows: dict[tuple[str, str], list[dict]] = {}
    tasks = []
    for spec, null_model, n in jobs:
        done = _load_done(variant, spec, null_model)
        done = done[done["rep"] < n] if len(done) else done
        rows[(spec, null_model)] = done.to_dict("records")
        have = set(done["rep"]) if len(done) else set()
        tasks += [(variant, spec, null_model, r) for r in range(n) if r not in have]
        print(f"  {spec:<20} null={null_model:<7} reps={n:>4}  done={len(have):>4}  "
              f"real ΔJ={real_diff_j(variant, spec):,.2f}")
    order = {job[:2]: i for i, job in enumerate(jobs)}
    tasks.sort(key=lambda t: (t[3], order[t[1:3]]))      # round-robin across specs
    print(f"{len(tasks)} replications to run on {workers} workers.", flush=True)

    t0, i, pending = time.time(), 0, list(tasks)
    for attempt in range(MAX_POOL_RESTARTS + 1):
        if not pending:
            break
        try:
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
                futures = {ex.submit(run_rep, t): t for t in pending}
                for fut in as_completed(futures):
                    row = fut.result()
                    pending.remove(futures[fut])
                    i += 1
                    key = (row["spec"], row["null_model"])
                    rows[key].append(row)
                    _write(variant, *key, rows[key])
                    flag = f"  ERROR {row['error'][:60]}" if row.get("error") else ""
                    print(f"  [{i:>4}/{len(tasks)}] {key[0]:<20} {key[1]:<7} "
                          f"rep {row['rep']:>3}  ΔJ={row.get('diff_j', np.nan):>9.2f}  "
                          f"cpu {row['cpu_seconds']:>6.1f}s  "
                          f"elapsed {(time.time() - t0) / 60:6.1f} min{flag}", flush=True)
                    if i % 10 == 0:
                        _write_summary(variant, jobs)
            break
        except BrokenProcessPool:
            if attempt == MAX_POOL_RESTARTS:
                raise
            print(f"  A worker died; restarting the pool for the {len(pending)} "
                  "remaining replications.", flush=True)

    summary = _write_summary(variant, jobs)
    print("\n=== Bootstrap ΔJ (jumps) — p = (#{sim ΔJ ≥ real} + 1) / (N + 1) ===")
    print(summary.to_string(index=False, float_format="{:,.4f}".format))
    return summary


# ---------------------------------------------------------------------------
# validate — null test, power test, recovery (Step 4)
# ---------------------------------------------------------------------------

def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    z = 1.959964
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def run_validate(variant: str, spec: str = POWER_SPEC) -> dict[str, pd.DataFrame]:
    """Size and power of the "need jumps?" test and parameter recovery, from
    the bootstrap replications of one panel (default: the daily panel)."""
    out_dir = exports_dir(variant)
    null = _load_done(variant, spec, "Heston")
    alt = _load_done(variant, spec, "Bates")
    cv = _load_done(variant, spec, "ConstantVol")
    mer = _load_done(variant, spec, "Merton")

    def _ok(df):
        if not len(df):
            return df
        df = df[df["error"].fillna("") == ""].copy()
        # "need SV, given jumps?" — Merton → Bates, two extra parameters
        df["diff_j_sv_jumps"] = df["j_merton"] - df["j_bates"]
        df["p_chi2_sv_jumps"] = stats.chi2.sf(df["diff_j_sv_jumps"].clip(lower=0.0), 2)
        df["p_j_cv"] = stats.chi2.sf(df["j_cv"], 5)
        return df

    null_ok, alt_ok, cv_ok, mer_ok = _ok(null), _ok(alt), _ok(cv), _ok(mer)
    out: dict[str, pd.DataFrame] = {}
    alphas = (0.10, 0.05, 0.01)
    real = {"diff_j": real_diff_j(variant, spec), "diff_j_sv": real_diff_j_sv(variant, spec),
            "j_heston": float(ladder_row(variant, spec, "Heston")["j_stat"]),
            "j_bates": float(ladder_row(variant, spec, "Bates")["j_stat"]),
            "j_cv": float(ladder_row(variant, spec, "ConstantVol")["j_stat"]),
            "diff_j_sv_jumps": float(ladder_row(variant, spec, "Merton")["j_stat"]
                                     - ladder_row(variant, spec, "Bates")["j_stat"])}

    def _size_rows(dgp, df, n_all, test, pcol, stat, dof, nominal=True):
        rows = []
        n = len(df)
        if not n or pcol not in df:
            return rows
        ge = int((df[stat] >= real[stat]).sum())
        for alpha in alphas:
            k = int((df[pcol] < alpha).sum())
            lo, hi = _wilson(k, n)
            rows.append({
                "variant": variant, "spec": spec, "dgp": dgp, "test": test,
                "alpha": alpha, "n_runs": n, "n_reject": k, "reject_share": k / n,
                "ci95_lo": lo, "ci95_hi": hi,
                "nominal": alpha if nominal else np.nan,
                "chi2_critical": float(stats.chi2.ppf(1 - alpha, dof)),
                "simulated_critical": float(df[stat].quantile(1 - alpha)),
                "stat_mean": float(df[stat].mean()), "stat_median": float(df[stat].median()),
                "real_stat": real[stat], "bootstrap_p_real": (ge + 1) / (n + 1),
                "n_errors": n_all - n,
            })
        return rows

    # --- null tests: the restriction being tested is TRUE in the simulated data
    rows = (
        _size_rows("Heston (no jumps)", null_ok, len(null), "need jumps? ΔJ Heston→Bates",
                   "p_chi2_jumps", "diff_j", 1)
        + _size_rows("ConstantVol (no SV, no jumps)", cv_ok, len(cv),
                     "need SV? ΔJ ConstantVol→Heston", "p_chi2_sv", "diff_j_sv", 2)
        + _size_rows("ConstantVol (no SV, no jumps)", cv_ok, len(cv),
                     "need SV, given jumps? ΔJ Merton→Bates",
                     "p_chi2_sv_jumps", "diff_j_sv_jumps", 2)
        + _size_rows("Merton (jumps, no SV)", mer_ok, len(mer),
                     "need SV, given jumps? ΔJ Merton→Bates",
                     "p_chi2_sv_jumps", "diff_j_sv_jumps", 2)
        + _size_rows("ConstantVol (no SV, no jumps)", cv_ok, len(cv),
                     "overall fit: J(ConstantVol)", "p_j_cv", "j_cv", 5)
        + _size_rows("Heston (no jumps)", null_ok, len(null), "overall fit: J(Heston)",
                     "p_j_heston", "j_heston", 3)
        + _size_rows("Bates (fitted)", alt_ok, len(alt), "overall fit: J(Bates)",
                     "p_j_bates", "j_bates", 2)
    )
    out["null"] = pd.DataFrame(rows)
    out["null"].to_csv(out_dir / "validation_null_test.csv", index=False, float_format="%.6f")

    # --- power tests: the restriction being tested is FALSE in the simulated data
    rows = []
    for dgp, df, n_all, test, pcol, stat, dof, null_df in (
        ("Bates (fitted)", alt_ok, len(alt), "need jumps? ΔJ Heston→Bates",
         "p_chi2_jumps", "diff_j", 1, null_ok),
        ("Heston (no jumps)", null_ok, len(null), "need SV? ΔJ ConstantVol→Heston",
         "p_chi2_sv", "diff_j_sv", 2, cv_ok),
        ("Bates (fitted)", alt_ok, len(alt), "need SV? ΔJ ConstantVol→Heston",
         "p_chi2_sv", "diff_j_sv", 2, cv_ok),
        ("Bates (fitted)", alt_ok, len(alt), "need SV, given jumps? ΔJ Merton→Bates",
         "p_chi2_sv_jumps", "diff_j_sv_jumps", 2, mer_ok),
    ):
        n = len(df)
        if not n:
            continue
        for alpha in alphas:
            crit = float(null_df[stat].quantile(1 - alpha)) if len(null_df) else np.nan
            rows.append({
                "variant": variant, "spec": spec, "dgp": dgp, "test": test, "alpha": alpha,
                "n_runs": n,
                "selected_chi2_share": float((df[pcol] < alpha).mean()),
                "selected_simulated_cv_share": float((df[stat] > crit).mean())
                if np.isfinite(crit) else np.nan,
                "simulated_critical": crit,
                "chi2_critical": float(stats.chi2.ppf(1 - alpha, dof)),
                "stat_median": float(df[stat].median()),
                "n_errors": n_all - n,
            })
    out["power"] = pd.DataFrame(rows)
    out["power"].to_csv(out_dir / "validation_power_test.csv", index=False, float_format="%.6f")

    # --- recovery: do the Bates fits return the parameters that generated the data?
    rows = []
    for dgp, df, pairs in (
        ("Bates (fitted)", alt_ok, (("sigma_v", "bates_sigma_v"), ("sigma_J", "bates_sigma_J"),
                                    ("kappa", "bates_kappa"), ("stale_q", "bates_stale_q"))),
        ("Heston (no jumps)", null_ok, (("sigma_v", "heston_sigma_v"), ("kappa", "heston_kappa"),
                                        ("sigma_J", "bates_sigma_J"),
                                        ("stale_q", "heston_stale_q"))),
    ):
        for name, col in pairs:
            if not len(df) or col not in df:
                continue
            est = df[col].dropna()
            truth = float(df[f"dgp_{name}"].iloc[0])
            q = est.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            rows.append({
                "variant": variant, "spec": spec, "dgp": dgp, "parameter": name,
                "fitted_model": col.split("_")[0].capitalize(), "true_value": truth,
                "n_runs": len(est), "median": q[0.5], "mean": est.mean(), "sd": est.std(),
                "q05": q[0.05], "q25": q[0.25], "q75": q[0.75], "q95": q[0.95],
                "median_bias": q[0.5] - truth,
            })
    out["recovery"] = pd.DataFrame(rows)
    out["recovery"].to_csv(out_dir / "validation_recovery.csv", index=False, float_format="%.6f")

    for name, df in out.items():
        print(f"\n=== Validation: {name} ({variant}, {spec}) ===")
        print(df.drop(columns=["variant", "spec"]).to_string(index=False,
                                                              float_format="{:.4f}".format))
    print(f"\nSaved: {out_dir}/validation_{{null_test,power_test,recovery}}.csv")
    return out


# ---------------------------------------------------------------------------
# compare — old vs new
# ---------------------------------------------------------------------------

def run_compare(variant: str = "v2") -> dict[str, pd.DataFrame]:
    """Old-vs-new tables for the main ladder and the six-panel bootstrap."""
    v1p, v1e = config.DATA_DIR / "processed", config.DATA_DIR / "exports"
    out_dir = exports_dir(variant)
    out: dict[str, pd.DataFrame] = {}

    old = pd.read_csv(v1p / "smm_sampling_ladder_full.csv")
    new = pd.read_csv(ladder_csv(variant))
    cols = ["sigma_v", "se_sigma_v", "sigma_J", "se_sigma_J", "kappa", "se_kappa",
            "theta", "lambda", "j_stat", "j_dof", "j_pvalue"]
    m = old[["spec", "model"] + cols].merge(new[["spec", "model"] + cols + ["stale_q"]],
                                            on=["spec", "model"], suffixes=("_v1", "_v2"))
    out["ladder"] = m
    m.to_csv(out_dir / "old_vs_new_ladder.csv", index=False, float_format="%.6f")

    olds = pd.read_csv(v1p / "smm_sampling_sensitivity.csv")
    news = pd.read_csv(processed_dir(make_calibrator(variant)) / "smm_sampling_sensitivity.csv")
    keep = ["spec", "diffj_sv", "p_sv", "diffj_jumps", "p_jumps"]
    t = olds[keep].merge(news[keep], on="spec", suffixes=("_v1", "_v2"))
    boot_new = exports_dir(variant) / "bootstrap_diffj_summary.csv"
    if boot_new.exists():
        bo = pd.read_csv(v1e / "bootstrap_diffj_summary.csv")
        bn = pd.read_csv(boot_new)
        bcols = ["spec", "n_ok", "real_diff_j", "p_value", "sim_median", "sim_q95",
                 "chi2_rejects_share"]
        b = bo[bo["null_model"] == "Heston"][bcols].merge(
            bn[bn["null_model"] == "Heston"][bcols], on="spec", suffixes=("_v1", "_v2"))
        t = t.merge(b, on="spec", how="left")
    out["tests"] = t
    t.to_csv(out_dir / "old_vs_new_tests.csv", index=False, float_format="%.6f")
    for name, df in out.items():
        print(f"\n=== old vs new: {name} ===")
        print(df.to_string(index=False, float_format="{:.4f}".format))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["ladders", "sensitivities", "bootstrap", "validate",
                                     "compare"])
    ap.add_argument("--variant", default="v2", choices=list(VARIANTS))
    ap.add_argument("--specs", nargs="+", default=list(SPECS), choices=SPECS)
    ap.add_argument("--reps", type=int, help="override the per-spec replication count")
    ap.add_argument("--no-power", action="store_true", help="skip the Bates power runs")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, "1")     # one core per ladder

    if args.step == "ladders":
        run_ladders(args.variant, args.specs, args.workers)
    elif args.step == "sensitivities":
        run_sensitivities(args.variant)
    elif args.step == "bootstrap":
        run_bootstrap(args.variant, args.specs, args.reps, not args.no_power, args.workers)
    elif args.step == "validate":
        run_validate(args.variant)
    else:
        run_compare(args.variant)


if __name__ == "__main__":
    main()
