"""Parametric bootstrap for the difference-in-J jump test (§4.4 / §4.8).

The ladder's "need jumps?" verdict reads ΔJ = J(Heston) − J(Bates) against
χ²(1).  On panels simulated with no jumps at all that test still rejects in
every run, so the χ² p-values are not usable.  This script replaces them with
a simulated benchmark: how large does ΔJ get when the truth is Heston?

For each sampling spec:
  a. null model = the spec's Heston row in smm_sampling_ladder_full.csv
     (σ_v, κ, θ) with λ = 0 and σ_J = 0;
  b. simulate a fake panel with the real panel's contracts and increments per
     contract (_simulate_path, BURN_IN steps of burn-in per contract, fresh
     draws per run);
  c. run the main-run ladder on it, NestedLadder(...).run_selection with
     n_sim_multiplier=20, n_bootstrap=400, n_restarts=3 (rng_seed 42 and
     κ init 5 as in run_smm.py), so every replication is the exact procedure
     behind the real number;
  d. record test_jumps.diff_j;
and p = (#{simulated ΔJ ≥ real ΔJ} + 1) / (N + 1).  Real ΔJ is J(Heston) −
J(Bates) from the same CSV.  `--power` repeats a spec with the fitted Bates
row as the truth instead: if those runs rarely beat the no-jump runs, the
panel cannot detect jumps at its sample size.

A run whose ladder trips diff_j_test's containment guard (J_restricted <
J_full) is recorded with the error and the J values from its message, not
dropped.  The summary's p_value leaves those runs out; p_conservative
counts each one as an exceedance.

SPEED.  The pure-Python simulator costs about 5 min per ladder.  _install()
swaps a Numba copy into src.smm.bates_smm at runtime (src/ is not edited).
The copy takes √v through the C library's pow(), the same call Python's
`v ** 0.5` makes, so its paths are bit-for-bit identical to the original's;
check_simulator() proves that before any replication starts.  Replications
run in parallel, each seeded from (SEED, spec, null model, rep), so results
do not depend on worker count or completion order, and an interrupted run
resumes where it stopped.

    python -m analysis.bootstrap_diffj                      # everything
    python -m analysis.bootstrap_diffj --specs D/revision --reps 20
    python -m analysis.bootstrap_diffj --verify-real        # reproduce real J first
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import ctypes.util
import io
import os
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numba
import numpy as np
import pandas as pd
from scipy import stats

import config
from src.smm import bates_smm
from src.smm.bates_smm import BatesSMM
from src.smm.nested_ladder import NestedLadder

# Bound at import, before _install() can repoint the module attribute.
_python_simulate_path = bates_smm._simulate_path

PROCESSED = config.DATA_DIR / "processed"
EXPORTS = config.DATA_DIR / "exports"
LADDER_CSV = PROCESSED / "smm_sampling_ladder_full.csv"

SEED = 20261021
BURN_IN = 50
MAIN_RUN = {"n_sim_multiplier": 20, "n_bootstrap": 400, "n_restarts": 3}
KAPPA_INIT = 5.0

# Order is part of the seed: append new specs, never reorder.
SPECS = ("D/calendar", "2D/calendar", "3D/calendar", "4D/calendar",
         "D/revision", "D/calendar-matched")
NULL_MODELS = ("Heston", "Bates")
REPS = {"D/calendar": 100, "D/revision": 100}       # others: DEFAULT_REPS
DEFAULT_REPS = 50
POWER_SPEC, POWER_REPS = "D/revision", 50
CHI2_1_05 = float(stats.chi2.ppf(0.95, 1))         # the χ² test this replaces
MAX_POOL_RESTARTS = 20

_CPOW = ctypes.CDLL(ctypes.util.find_library("m")).pow
_CPOW.argtypes = [ctypes.c_double, ctypes.c_double]
_CPOW.restype = ctypes.c_double


# ---------------------------------------------------------------------------
# Numba simulator (line-for-line copy of bates_smm._simulate_path)
# ---------------------------------------------------------------------------

@numba.njit(cache=False)
def _path_kernel(sigma_v, sigma_J, dw_x, dw_v, u_jump, z_jump,
                 kappa, theta, lambda_, n_sim):
    v = theta
    dx_sim = np.empty(n_sim)
    for i in range(n_sim):
        v_pos = max(v, 0.0)
        sv = _CPOW(v_pos, 0.5)
        jump = sigma_J * z_jump[i] if u_jump[i] < lambda_ else 0.0
        dx_sim[i] = sv * dw_x[i] + jump
        v = v_pos + kappa * (theta - v_pos) + sigma_v * sv * dw_v[i]
        v = max(v, 0.0)
    return dx_sim


def _simulate_path_numba(sigma_v, sigma_J, randoms, kappa, theta, lambda_, rho, n_sim):
    """Drop-in for bates_smm._simulate_path (same signature, same output)."""
    return _path_kernel(float(sigma_v), float(sigma_J),
                        randoms["dw_x"], randoms["dw_v"],
                        randoms["u_jump"], randoms["z_jump"],
                        float(kappa), float(theta), float(lambda_), int(n_sim))


def _install() -> None:
    """Point the ladder at the Numba simulator for this process."""
    bates_smm._simulate_path = _simulate_path_numba


def check_simulator(n: int = 82_820, n_random: int = 40) -> int:
    """Raise unless the Numba copy reproduces the original path bit for bit."""
    randoms = BatesSMM()._draw_randoms(n, rho=0.0)
    ladder = pd.read_csv(LADDER_CSV)
    cases = [tuple(r) for r in ladder[["sigma_v", "sigma_J", "kappa", "theta", "lambda"]]
             .itertuples(index=False)]
    rng = np.random.default_rng(SEED)
    cases += list(zip(rng.uniform(0, 3, n_random), rng.uniform(0, 3, n_random),
                      rng.uniform(1e-4, 20, n_random), rng.uniform(1e-3, 0.3, n_random),
                      rng.uniform(0, 0.2, n_random), strict=True))
    for sv, sj, k, th, lam in cases:
        a = _python_simulate_path(sv, sj, randoms, k, th, lam, 0.0, n)
        b = _simulate_path_numba(sv, sj, randoms, k, th, lam, 0.0, n)
        if not np.array_equal(a, b, equal_nan=True):
            raise AssertionError(f"Numba path differs at {(sv, sj, k, th, lam)}: "
                                 f"max |diff| = {np.nanmax(np.abs(a - b)):.3e}")
    return len(cases)


# ---------------------------------------------------------------------------
# Panels and the fitted ladder
# ---------------------------------------------------------------------------

def load_panel(spec: str) -> pd.DataFrame:
    """The real panel behind one row of smm_sampling_ladder_full.csv."""
    freq, sampling = spec.split("/")
    if sampling == "calendar-matched":
        daily = load_panel("D/calendar")
        kept = load_panel("D/revision")["contract_id"].unique()
        panel = daily[daily["contract_id"].isin(kept)].reset_index(drop=True)
        if len(panel) != 3913:
            raise AssertionError(f"D/calendar-matched has {len(panel)} rows, expected 3,913")
        return panel
    name = ("smm_panel.parquet" if spec == "D/calendar"
            else f"smm_panel_{freq.lower()}_{sampling}.parquet")
    return pd.read_parquet(PROCESSED / name)


def ladder_row(spec: str, model: str) -> pd.Series:
    df = pd.read_csv(LADDER_CSV)
    row = df[(df["spec"] == spec) & (df["model"] == model)]
    if len(row) != 1:
        raise KeyError(f"{spec}/{model} not in {LADDER_CSV}")
    return row.iloc[0]


def real_diff_j(spec: str) -> float:
    return float(ladder_row(spec, "Heston")["j_stat"] - ladder_row(spec, "Bates")["j_stat"])


def null_params(spec: str, null_model: str) -> dict:
    """Data-generating parameters: Heston (no jumps) or the fitted Bates row."""
    r = ladder_row(spec, null_model)
    jumps = null_model == "Bates"
    return {"sigma_v": float(r["sigma_v"]), "kappa": float(r["kappa"]),
            "theta": float(r["theta"]),
            "sigma_J": float(r["sigma_J"]) if jumps else 0.0,
            "lambda_": float(r["lambda"]) if jumps else 0.0}


def simulate_panel(lengths: pd.Series, params: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Fake panel: one independent path per contract, BURN_IN steps discarded."""
    parts = []
    for cid, n in lengths.items():
        m = int(n) + BURN_IN
        randoms = {"dw_x": rng.standard_normal(m), "dw_v": rng.standard_normal(m),
                   "u_jump": rng.uniform(size=m), "z_jump": rng.standard_normal(m)}
        dx = _simulate_path_numba(params["sigma_v"], params["sigma_J"], randoms,
                                  params["kappa"], params["theta"], params["lambda_"],
                                  0.0, m)[BURN_IN:]
        parts.append(pd.DataFrame({"contract_id": cid, "t": np.arange(1, n + 1),
                                   "delta_X": dx}))
    return pd.concat(parts, ignore_index=True)


def _seed(spec: str, null_model: str, rep: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([SEED, SPECS.index(spec), NULL_MODELS.index(null_model), rep])


def _fit(panel: pd.DataFrame):
    ladder = NestedLadder(calibrator=BatesSMM(**MAIN_RUN), kappa_init=KAPPA_INIT)
    with contextlib.redirect_stdout(io.StringIO()):     # run_selection always prints
        return ladder.run_selection(panel, verbose=False, save_as=None, moment_table_as=None)


# ---------------------------------------------------------------------------
# One replication (runs in a worker)
# ---------------------------------------------------------------------------

_LENGTHS: dict[str, pd.Series] = {}
_GUARD = re.compile(r"J_restricted < J_full: (\w+) J=(\S+) vs (\w+) J=(\S+) \(ΔJ=(\S+)\)")


def _lengths(spec: str) -> pd.Series:
    if spec not in _LENGTHS:
        panel = load_panel(spec)
        _LENGTHS[spec] = panel.groupby("contract_id", sort=False)["delta_X"].count()
    return _LENGTHS[spec]


def run_rep(task: tuple[str, str, int]) -> dict:
    spec, null_model, rep = task
    params = null_params(spec, null_model)
    lengths = _lengths(spec)
    t0, c0 = time.time(), time.process_time()
    panel = simulate_panel(lengths, params, np.random.default_rng(_seed(spec, null_model, rep)))
    row = {"spec": spec, "null_model": null_model, "rep": rep,
           "seed": f"{SEED}-{SPECS.index(spec)}-{NULL_MODELS.index(null_model)}-{rep}",
           "n_contracts": len(lengths), "n_increments": int(lengths.sum()),
           "burn_in": BURN_IN, **{f"dgp_{k.rstrip('_')}": v for k, v in params.items()}}
    try:
        lr = _fit(panel)
        row.update({
            "theta_hat": lr.bates.theta, "lambda_hat": lr.bates.lambda_,
            "j_cv": lr.constant_vol.j_stat, "j_heston": lr.heston.j_stat,
            "j_merton": lr.merton.j_stat, "j_bates": lr.bates.j_stat,
            "diff_j_sv": lr.test_sv.diff_j, "diff_j": lr.test_jumps.diff_j,
            "heston_sigma_v": lr.heston.sigma_v, "heston_kappa": lr.heston.kappa,
            "bates_sigma_v": lr.bates.sigma_v, "bates_sigma_J": lr.bates.sigma_J,
            "bates_kappa": lr.bates.kappa, "error": "",
        })
    except AssertionError as e:          # diff_j_test's containment guard
        row["error"] = str(e).splitlines()[0]
        m = _GUARD.search(str(e))
        if m and (m.group(1), m.group(3)) == ("Heston", "Bates"):
            row.update({"j_heston": float(m.group(2)), "j_bates": float(m.group(4)),
                        "diff_j": float(m.group(5))})
    except Exception as e:               # anything else: keep the run, say why
        row["error"] = f"{type(e).__name__}: {e}"
    row["seconds"] = round(time.time() - t0, 1)          # wall clock, includes any sleep
    row["cpu_seconds"] = round(time.process_time() - c0, 1)
    return row


def _init_worker() -> None:
    _install()
    # np.corrcoef on a zero-variance trial path (e.g. a σ = 0 grid point);
    # the ladder already maps the resulting NaN to its penalty.
    warnings.filterwarnings("ignore", "invalid value encountered in divide", RuntimeWarning)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _out_path(spec: str, null_model: str):
    slug = spec.lower().replace("/", "_").replace("-", "_")
    suffix = "" if null_model == "Heston" else f"_{null_model.lower()}_power"
    return EXPORTS / f"bootstrap_diffj_{slug}{suffix}.csv"


def _load_done(spec: str, null_model: str) -> pd.DataFrame:
    path = _out_path(spec, null_model)
    return pd.read_csv(path, keep_default_na=False, na_values=[""]) if path.exists() \
        else pd.DataFrame()


def _write(spec: str, null_model: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).sort_values("rep").reset_index(drop=True)
    df["real_diff_j"] = real_diff_j(spec)
    df["exceeds_real"] = df["diff_j"] >= df["real_diff_j"]
    df.to_csv(_out_path(spec, null_model), index=False, float_format="%.6f")
    return df


def summarise(spec: str, null_model: str, df: pd.DataFrame) -> dict:
    ok = df[df["error"].fillna("") == ""]
    n_err = len(df) - len(ok)
    real = real_diff_j(spec)
    ge = int((ok["diff_j"] >= real).sum())
    q = ok["diff_j"].quantile([0.5, 0.9, 0.95, 0.99]) if len(ok) else pd.Series(dtype=float)
    return {
        "spec": spec, "null_model": null_model, "n_reps": len(df), "n_ok": len(ok),
        "n_errors": n_err, "real_diff_j": real, "n_ge_real": ge,
        "p_value": (ge + 1) / (len(ok) + 1),
        "p_conservative": (ge + n_err + 1) / (len(df) + 1),
        "sim_median": q.get(0.5, np.nan), "sim_q90": q.get(0.9, np.nan),
        "sim_q95": q.get(0.95, np.nan), "sim_q99": q.get(0.99, np.nan),
        "sim_max": ok["diff_j"].max() if len(ok) else np.nan,
        "chi2_rejects_share": float((ok["diff_j"] > CHI2_1_05).mean()) if len(ok) else np.nan,
    }


def _write_summary(jobs: list[tuple[str, str, int]]) -> pd.DataFrame:
    rows = []
    for spec, null_model, _ in jobs:
        df = _load_done(spec, null_model)
        if len(df):
            rows.append(summarise(spec, null_model, df))
    out = pd.DataFrame(rows)
    out.to_csv(EXPORTS / "bootstrap_diffj_summary.csv", index=False, float_format="%.6f")
    return out


def verify_real(specs: list[str], workers: int) -> pd.DataFrame:
    """Re-fit each real panel with the Numba simulator; J must match the CSV."""
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(min(workers, len(specs)), initializer=_init_worker) as pool:
        fits = pool.map(_verify_one, specs)
    df = pd.DataFrame(fits)
    print(df.to_string(index=False, float_format="{:.6f}".format))
    out = EXPORTS / "bootstrap_diffj_verify_real.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"Saved: {out}")
    return df


def _verify_one(spec: str) -> dict:
    lr = _fit(load_panel(spec))
    return {"spec": spec,
            "j_heston": lr.heston.j_stat, "csv_j_heston": ladder_row(spec, "Heston")["j_stat"],
            "j_bates": lr.bates.j_stat, "csv_j_bates": ladder_row(spec, "Bates")["j_stat"],
            "diff_j": lr.test_jumps.diff_j, "csv_diff_j": real_diff_j(spec)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--specs", nargs="+", default=list(SPECS), choices=SPECS)
    ap.add_argument("--reps", type=int, help="override the per-spec replication count")
    ap.add_argument("--no-power", action="store_true", help="skip the Bates power check")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--verify-real", action="store_true",
                    help="re-fit the real panels and compare J with the CSV, then stop")
    args = ap.parse_args(argv)

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, "1")     # one core per replication

    print(f"Numba simulator matches the original on {check_simulator()} parameter sets "
          "(bit-for-bit).")
    if args.verify_real:
        verify_real(args.specs, args.workers)
        return

    jobs = [(s, "Heston", args.reps or REPS.get(s, DEFAULT_REPS)) for s in args.specs]
    if not args.no_power and POWER_SPEC in args.specs:
        jobs.append((POWER_SPEC, "Bates", args.reps or POWER_REPS))

    rows: dict[tuple[str, str], list[dict]] = {}
    tasks = []
    for spec, null_model, n in jobs:
        done = _load_done(spec, null_model)
        done = done[done["rep"] < n] if len(done) else done
        rows[(spec, null_model)] = done.to_dict("records")
        have = set(done["rep"]) if len(done) else set()
        tasks += [(spec, null_model, r) for r in range(n) if r not in have]
        print(f"  {spec:<20} null={null_model:<7} reps={n:>4}  "
              f"done={len(have):>4}  real ΔJ={real_diff_j(spec):,.1f}")
    # Round-robin across specs so every spec fills in at the same pace.
    order = {job[:2]: i for i, job in enumerate(jobs)}
    tasks.sort(key=lambda t: (t[2], order[t[:2]]))
    print(f"{len(tasks)} replications to run on {args.workers} workers.")

    # ProcessPoolExecutor, not multiprocessing.Pool: when a worker dies (macOS
    # can kill them across a sleep/wake) Pool silently loses its task and
    # hangs at the end, whereas the executor raises and the lost reps are
    # resubmitted on a fresh pool.
    import multiprocessing as mp
    t0, i, pending = time.time(), 0, list(tasks)
    for attempt in range(MAX_POOL_RESTARTS + 1):
        try:
            with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                                     initializer=_init_worker) as ex:
                futures = {ex.submit(run_rep, t): t for t in pending}
                for fut in as_completed(futures):
                    row = fut.result()
                    pending.remove(futures[fut])
                    i += 1
                    key = (row["spec"], row["null_model"])
                    rows[key].append(row)
                    _write(*key, rows[key])
                    flag = f"  ERROR {row['error'][:60]}" if row.get("error") else ""
                    print(f"  [{i:>4}/{len(tasks)}] {key[0]:<20} {key[1]:<7} "
                          f"rep {row['rep']:>3}  ΔJ={row.get('diff_j', np.nan):>12,.1f}  "
                          f"cpu {row['cpu_seconds']:>6.1f}s  "
                          f"elapsed {(time.time() - t0) / 60:6.1f} min{flag}", flush=True)
                    if i % 10 == 0:
                        _write_summary(jobs)
            break
        except BrokenProcessPool:
            if attempt == MAX_POOL_RESTARTS:
                raise
            print(f"  A worker died; restarting the pool for the {len(pending)} "
                  "remaining replications.", flush=True)

    summary = _write_summary(jobs)
    print("\n=== Bootstrap ΔJ (jumps) — p = (#{sim ΔJ ≥ real} + 1) / (N + 1) ===")
    print(summary.to_string(index=False, float_format="{:,.4f}".format))
    print(f"\nSaved: {EXPORTS / 'bootstrap_diffj_summary.csv'}")


if __name__ == "__main__":
    main()
