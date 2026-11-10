"""The J-statistic and the standard errors must be on the χ² scale.

W is diag(1 / bootstrap variance of the full-sample moments), so g'Wg is
already O(1) when the model is true.  The v1 code multiplied by n_real once
more (J too large by n_real, SEs too small by √n_real).  These tests pin the
corrected scale and keep the legacy switch honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.smm.bates_smm import N_MOMENTS, BatesSMM, _j_test

THETA = 0.04
N_CONTRACTS, N_PER_CONTRACT = 8, 60
N_REAL = N_CONTRACTS * N_PER_CONTRACT


def _constant_vol_panel(seed: int) -> pd.DataFrame:
    """A small panel of iid N(0, θ) increments — the ConstantVol model, exactly."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "contract_id": np.repeat([f"C{i}" for i in range(N_CONTRACTS)], N_PER_CONTRACT),
        "t": np.tile(np.arange(1, N_PER_CONTRACT + 1), N_CONTRACTS),
        "delta_X": rng.normal(0.0, np.sqrt(THETA), N_REAL),
    })


def _j_at_truth(seed: int, j_scaling: str) -> float:
    """J evaluated at the TRUE parameters (nothing estimated → dof = 5)."""
    cal = BatesSMM(n_sim_multiplier=10, n_bootstrap=200, n_restarts=1,
                   rng_seed=seed, j_scaling=j_scaling,
                   moments="pooled", weights="iid")
    cache = cal.prepare(_constant_vol_panel(seed))
    cache["theta"] = THETA          # the truth, not the truncated-variance plug-in
    return cal.fit_constant_vol(cache, verbose=False).j_stat


def test_j_test_formula():
    g = np.array([0.3, -0.2, 0.1, 0.0, 0.5])
    W = np.diag([4.0, 9.0, 1.0, 2.0, 0.5])
    n_real, n_sim = 400, 8000
    quad = float(g @ W @ g)
    j, p = _j_test(g, W, n_real, n_sim, dof=3)
    assert j == pytest.approx(quad / (1.0 + n_real / n_sim))
    assert 0.0 < p < 1.0
    j_legacy, _ = _j_test(g, W, n_real, n_sim, dof=3, n_scaling=True)
    assert j_legacy == pytest.approx(n_real * j)


def test_j_is_chi2_scaled_at_the_true_parameters():
    """Across repeated draws at the truth, mean(J) ≈ dof — not ≈ n·dof."""
    draws = np.array([_j_at_truth(seed, "corrected") for seed in range(40)])
    # E[χ²(5)] = 5; SE of the mean over 40 draws ≈ 0.5.  The band is loose
    # because W is diagonal and bootstrapped, but it excludes anything like
    # the legacy scale by two orders of magnitude.
    assert 3.0 < draws.mean() < 8.0, draws.mean()
    # A χ²(5) variate exceeds its 95% point 5% of the time; allow up to 20%.
    assert (draws > 11.07).mean() <= 0.20


def test_legacy_switch_reproduces_the_n_inflated_statistic():
    seeds = range(5)
    new = np.array([_j_at_truth(s, "corrected") for s in seeds])
    old = np.array([_j_at_truth(s, "legacy_n") for s in seeds])
    np.testing.assert_allclose(old, N_REAL * new, rtol=1e-12)
    assert old.mean() > 100 * N_MOMENTS


def test_standard_errors_scale_by_sqrt_n():
    """SE(corrected) = SE(legacy) · √n_real — same Jacobian, same W."""
    panel = _constant_vol_panel(7)
    ses = {}
    for mode in ("corrected", "legacy_n"):
        cal = BatesSMM(n_sim_multiplier=10, n_bootstrap=200, n_restarts=1,
                       rng_seed=7, j_scaling=mode,
                       moments="pooled", weights="iid")
        cache = cal.prepare(panel)
        res = cal.fit_merton(cache, verbose=False)
        ses[mode] = cal.standard_errors(res, cache)[1]      # se_sigma_J
    assert np.isfinite(ses["corrected"]) and ses["corrected"] > 0
    assert ses["corrected"] == pytest.approx(ses["legacy_n"] * np.sqrt(N_REAL), rel=1e-9)


def test_corrected_runs_do_not_write_over_v1_outputs():
    assert BatesSMM.legacy().output_subdir == ""
    assert BatesSMM().output_subdir == "v2"
    assert BatesSMM(output_subdir="v2/scaling_only").output_subdir == "v2/scaling_only"
