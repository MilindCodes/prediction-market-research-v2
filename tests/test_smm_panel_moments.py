"""Within-contract moments, the panel simulator and the observation process.

These pin the properties the v2 estimator rests on:

* differences BETWEEN contracts cannot show up as kurtosis or volatility
  clustering in the pooled shape moments;
* the real and the simulated side go through one statistic function;
* the tick / clip / stale-quote observation process produces exact zeros the
  way the data does, and revision sampling collapses them;
* W comes from resampling whole contracts.

Everything here runs without Numba (panels are small).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from src.smm import panel_moments as pm
from src.smm.bates_smm import BatesSMM, compute_moments, pool_demeaned_increments

CLIP = (0.02, 0.98)


def _panel(increments: list[np.ndarray]) -> pd.DataFrame:
    return pd.concat([
        pd.DataFrame({"contract_id": f"C{i:02d}", "t": np.arange(1, len(d) + 1),
                      "delta_X": d})
        for i, d in enumerate(increments)
    ], ignore_index=True)


def _mixture(seed: int = 0, n_contracts: int = 40, n: int = 120) -> list[np.ndarray]:
    """iid Gaussian contracts whose SDs differ by a factor of ten."""
    rng = np.random.default_rng(seed)
    sds = np.geomspace(0.05, 0.5, n_contracts)
    return [rng.normal(0.0, sd, n) for sd in sds]


# ---------------------------------------------------------------------------
# Per-contract statistics
# ---------------------------------------------------------------------------

def test_contract_stats_match_numpy():
    rng = np.random.default_rng(1)
    d = rng.standard_t(4, 57) * 0.2
    out = np.zeros(pm.K_STATS)
    pm.contract_stats(d, out)
    dm = d - d.mean()
    assert out[pm.I_N] == 57
    assert out[pm.I_SS] == pytest.approx((dm ** 2).sum())
    assert out[pm.I_M4] == pytest.approx((dm ** 4).sum())
    assert out[pm.I_ACF1] == pytest.approx(np.corrcoef(dm[:-1] ** 2, dm[1:] ** 2)[0, 1])
    assert out[pm.I_ACF2] == pytest.approx(np.corrcoef(dm[:-2] ** 2, dm[2:] ** 2)[0, 1])
    assert out[pm.I_Q95] == pytest.approx(np.percentile(np.abs(dm), 95))


def test_all_zero_contract_has_no_shape_moments_but_does_not_break_pooling():
    rng = np.random.default_rng(2)
    incs = [rng.normal(0, 0.2, 60), np.zeros(41), rng.normal(0, 0.3, 80)]
    shape = pm.panel_shape(_panel(incs), CLIP)
    row = shape["stats"][1]
    assert row[pm.I_SS] == 0.0 and np.isnan(row[pm.I_ACF1])
    m = pm.pool_moments(shape["stats"])
    assert np.all(np.isfinite(m))
    # Dropping the dead contract leaves every shape moment unchanged ...
    m_live = pm.pool_moments(shape["stats"][[0, 2]])
    np.testing.assert_allclose(m[1:], m_live[1:], rtol=1e-12)
    # ... while the pooled variance still counts its (zero) increments.
    assert m[0] < m_live[0]


def test_pooled_kurtosis_is_kurtosis_of_contract_standardised_increments():
    """The two definitions the spec offers are the same number (ddof = 0)."""
    rng = np.random.default_rng(3)
    incs = [rng.standard_t(5, n) * sd for n, sd in ((41, 0.1), (93, 0.4), (180, 0.2))]
    kurt = pm.pool_moments(pm.panel_shape(_panel(incs), CLIP)["stats"])[1]
    z = np.concatenate([(d - d.mean()) / d.std(ddof=0) for d in incs])
    assert kurt == pytest.approx(sps.kurtosis(z, fisher=True), rel=1e-10)


def test_between_contract_variance_does_not_look_like_dynamics():
    """Gaussian iid contracts with different variances: v1 sees fat tails,
    the within-contract moments see none."""
    incs = _mixture()
    panel = _panel(incs)
    v1 = compute_moments(pool_demeaned_increments(panel))
    v2 = pm.pool_moments(pm.panel_shape(panel, CLIP)["stats"])
    assert v1[1] > 2.0                    # spurious excess kurtosis from mixing
    assert abs(v2[1]) < 0.25              # none within contracts
    assert v1[2] > 0.10                   # spurious ACF of squares (variance regimes)
    assert abs(v2[2]) < 0.05 and abs(v2[3]) < 0.05
    assert v2[4] == pytest.approx(1.96, abs=0.08)   # Gaussian 95th pct of |z|
    assert v2[0] == pytest.approx(v1[0])            # variance stays pooled


def test_raw_tail_moment_keeps_the_jensen_gap_standardised_does_not():
    """E[σ_c] < √E[σ_c²]: a raw per-contract percentile, read against the
    pooled variance, looks fat-tailed even though each contract is Gaussian."""
    panel = _panel(_mixture())
    st = pm.panel_shape(panel, CLIP)["stats"]
    var, raw = pm.pool_moments(st, q95="raw")[[0, 4]]
    assert raw / np.sqrt(var) < 0.85 * 1.96
    with pytest.raises(ValueError):
        pm.pool_moments(st, q95="nope")


# ---------------------------------------------------------------------------
# Contract bootstrap
# ---------------------------------------------------------------------------

def test_contract_bootstrap_weights():
    st = pm.panel_shape(_panel(_mixture(seed=5)), CLIP)["stats"]
    W, var = pm.contract_bootstrap_weights(st, 300, np.random.default_rng(0))
    assert W.shape == (5, 5) and np.all(np.diag(W) > 0)
    np.testing.assert_allclose(np.diag(W), 1.0 / var)
    W6, _ = pm.contract_bootstrap_weights(st, 300, np.random.default_rng(0),
                                          zero_share=True)
    assert W6.shape == (6, 6)


def test_contract_bootstrap_sees_serial_dependence_the_iid_bootstrap_destroys():
    """Persistent volatility: the ACF moment is far noisier than an iid
    resample of increments suggests."""
    rng = np.random.default_rng(11)
    incs = []
    for _ in range(30):
        vol = np.exp(np.cumsum(rng.normal(0, 0.25, 150)) * 0.5) * 0.1
        incs.append(rng.normal(0, 1, 150) * vol)
    panel = _panel(incs)
    iid = BatesSMM.legacy(n_bootstrap=200)
    clus = BatesSMM.legacy(n_bootstrap=200, weights="contract")
    var_iid = 1.0 / np.diag(iid.prepare(panel)["W"])
    var_clu = 1.0 / np.diag(clus.prepare(panel)["W"])
    assert var_clu[2] > 2.0 * var_iid[2]          # lag-1 ACF of squares


# ---------------------------------------------------------------------------
# Simulator and observation process
# ---------------------------------------------------------------------------

PARAMS = {"sigma_v": 0.15, "sigma_J": 0.8, "kappa": 0.3, "theta": 0.04, "lambda_": 0.03}
IDS = [f"S{i}" for i in range(6)]
N_STEPS = np.array([40, 55, 41, 90, 60, 45])
X0 = np.log(np.array([0.5, 0.07, 0.93, 0.3, 0.025, 0.6]) /
            (1 - np.array([0.5, 0.07, 0.93, 0.3, 0.025, 0.6])))


def _replay_randoms(seed: int, burn: int) -> dict:
    """The draws simulate_observed_panel makes, laid out for the stats kernel."""
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("dw_x", "dw_v", "u_jump", "z_jump", "u_stale")}
    for n in N_STEPS:
        m = int(n) + burn
        cols["dw_x"].append(rng.standard_normal(m))
        cols["dw_v"].append(rng.standard_normal(m))
        cols["u_jump"].append(rng.uniform(size=m))
        cols["z_jump"].append(rng.standard_normal(m))
        cols["u_stale"].append(rng.uniform(size=m))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out.update(kind="panel", n_steps=N_STEPS.astype(np.int64), n_rep=1, burn=burn)
    return out


@pytest.mark.parametrize("observation,collapse,stale_q", [
    ("none", False, 0.0), ("tick", False, 0.0), ("tick_stale", False, 0.3),
    ("tick_stale", True, 0.3),
])
def test_real_and_simulated_sides_compute_the_same_statistics(observation, collapse, stale_q):
    """A simulated panel written out as a DataFrame and read back like real
    data gives exactly the statistics the simulator computed internally."""
    burn = 20
    params = dict(PARAMS, stale_q=stale_q)
    panel = pm.simulate_observed_panel(params, IDS, N_STEPS, X0, np.random.default_rng(9),
                                       observation=observation, tick=0.001, clip=CLIP,
                                       collapse=collapse, burn=burn)
    internal, q_used = pm.simulate_panel_stats(
        PARAMS["sigma_v"], PARAMS["sigma_J"], PARAMS["kappa"], PARAMS["theta"],
        PARAMS["lambda_"], _replay_randoms(9, burn), X0, observation=observation,
        tick=0.001, clip=CLIP, collapse=collapse, stale_q=stale_q)
    internal = internal[0]
    assert q_used == stale_q
    shape = pm.panel_shape(panel, CLIP)
    cols = [pm.I_N, pm.I_SS, pm.I_M4, pm.I_ACF1, pm.I_ACF2, pm.I_Q95]
    np.testing.assert_allclose(shape["stats"][:, cols], internal[:, cols],
                               rtol=1e-9, atol=1e-12, equal_nan=True)
    if observation != "none":
        np.testing.assert_allclose(shape["x0"], X0, atol=0.03)   # start recovered to a tick
        if not collapse:
            # zero counts agree too, split into clip zeros and the rest
            for col in (pm.I_ZCLIP, pm.I_ZROUND, pm.I_STEPS):
                np.testing.assert_array_equal(shape["stats"][:, col], internal[:, col])


def test_tick_observation_puts_prices_on_the_grid_inside_the_clip():
    panel = pm.simulate_observed_panel(PARAMS, IDS, N_STEPS, X0, np.random.default_rng(4),
                                       observation="tick", tick=0.001, clip=CLIP)
    p = panel["p"].to_numpy()
    assert p.min() >= CLIP[0] - 1e-12 and p.max() <= CLIP[1] + 1e-12
    np.testing.assert_allclose(p / 0.001, np.round(p / 0.001), atol=1e-9)
    assert len(panel) == N_STEPS.sum()
    assert (panel["delta_X"] == 0.0).any()              # exact zeros exist
    # every zero increment is an unchanged price, and vice versa
    for _, g in panel.groupby("contract_id"):
        same = np.isclose(np.diff(g["p"].to_numpy()), 0.0)
        np.testing.assert_array_equal(same, g["delta_X"].to_numpy()[1:] == 0.0)


def test_stale_quotes_add_zeros_and_revision_sampling_removes_them():
    n_steps = np.full(8, 400)
    ids = [f"L{i}" for i in range(8)]
    x0 = np.zeros(8)
    quiet = {"sigma_v": 0.0, "sigma_J": 0.0, "kappa": 1.0, "theta": 0.01, "lambda_": 0.0}
    shares = []
    for q in (0.0, 0.25, 0.5):
        panel = pm.simulate_observed_panel(dict(quiet, stale_q=q), ids, n_steps, x0,
                                           np.random.default_rng(6), observation="tick_stale")
        shares.append(float((panel["delta_X"] == 0.0).mean()))
    assert shares[0] < 0.05                              # rounding alone: few zeros
    assert shares[1] == pytest.approx(0.25, abs=0.05)
    assert shares[2] == pytest.approx(0.50, abs=0.05)

    rev = pm.simulate_observed_panel(dict(quiet, stale_q=0.5), ids, n_steps, x0,
                                     np.random.default_rng(6), observation="tick_stale",
                                     collapse=True)
    assert not (rev["delta_X"] == 0.0).any()
    assert rev["dt_steps"].mean() == pytest.approx(2.0, abs=0.25)   # 1/(1−q)
    shape = pm.panel_shape(rev, CLIP)
    assert shape["revision"]
    assert np.all(shape["span"] <= 400) and np.all(shape["span"] >= shape["n"])
    assert pm.zero_shares(shape["stats"])["zero_share"] == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# The calibrator end to end
# ---------------------------------------------------------------------------

def _observed_panel(seed: int, stale_q: float = 0.25) -> pd.DataFrame:
    params = {"sigma_v": 0.0, "sigma_J": 0.0, "kappa": 1.0, "theta": 0.03,
              "lambda_": 0.0, "stale_q": stale_q}
    return pm.simulate_observed_panel(params, IDS, N_STEPS, X0, np.random.default_rng(seed),
                                      observation="tick_stale")


def test_v2_prepare_solves_the_plug_ins_through_the_observation_process():
    cal = BatesSMM(n_sim_multiplier=8, n_bootstrap=100, n_restarts=1)
    assert cal.observation == "tick_stale" and not cal.is_legacy
    assert cal.moment_labels()[-1] == "Zero share" and len(cal.moment_labels()) == 6
    cache = cal.prepare(_observed_panel(0))
    assert cache["m_real"].shape == (6,) and cache["W"].shape == (6, 6)
    cal_info = cache["theta_calibration"]
    # constant vol, pushed through clip + tick + stale quotes, reproduces the
    # observed variance and the observed share of unchanged prices
    assert cal_info["achieved"] == pytest.approx(cal_info["target"], rel=0.02)
    assert cal_info["zero_share_achieved"] == pytest.approx(
        cal_info["zero_share_target"], abs=0.005)
    assert 0.05 < cal_info["stale_q"] < 0.5          # truth is 0.25
    res = cal.fit_constant_vol(cache, verbose=False)
    # six moments, one spent on the concentrated-out stale-quote probability
    assert res.j_dof == 5 and np.isfinite(res.j_stat)
    assert res.stale_q == pytest.approx(cal_info["stale_q"])
    assert res.moments_sim[5] == pytest.approx(res.moments_real[5], abs=0.005)
    assert list(res.moment_labels) == cal.moment_labels()


def test_v2_fit_runs_and_keeps_the_nesting_guarantee():
    cal = BatesSMM(n_sim_multiplier=5, n_bootstrap=100, n_restarts=1)
    cache = cal.prepare(_observed_panel(1))
    cv = cal.fit_constant_vol(cache, verbose=False)
    mer = cal.fit_merton(cache, warm_starts=[np.array([0.0])], verbose=False)
    assert mer.objective_value <= cv.objective_value + 1e-9
    assert mer.j_dof == 4
    assert np.isfinite(cal.standard_errors(mer, cache)[1]) or mer.sigma_J == 0.0


def test_within_contract_j_is_chi2_scaled_at_the_truth():
    """Contract bootstrap W + replicate-panel simulation → mean(J) ≈ dof."""
    rng = np.random.default_rng(123)
    draws = []
    for seed in range(30):
        incs = [rng.normal(0.0, 0.2, 60) for _ in range(30)]
        cal = BatesSMM(n_sim_multiplier=10, n_bootstrap=200, rng_seed=seed,
                       observation="none")
        cache = cal.prepare(_panel(incs))
        cache["theta"] = 0.04
        draws.append(cal.fit_constant_vol(cache, verbose=False).j_stat)
    assert 3.0 < np.mean(draws) < 8.5, np.mean(draws)


def test_switch_validation_and_legacy_identity():
    assert BatesSMM.legacy().is_legacy
    assert not BatesSMM.legacy(j_scaling="corrected").is_legacy
    assert BatesSMM.legacy().moment_labels()[4] == "95th pct(ΔX)"
    with pytest.raises(ValueError):
        BatesSMM(moments="pooled", observation="tick")
    with pytest.raises(ValueError):
        BatesSMM(moments="within_contract", weights="iid")
    with pytest.raises(ValueError):
        BatesSMM(theta_mode="nope")
    cal = BatesSMM(tick=0.0005).replace(clip=(0.01, 0.99))
    assert cal.tick == 0.0005 and cal.clip == (0.01, 0.99)


def test_variance_targeting_carves_jumps_out_of_the_total():
    cal = BatesSMM(theta_mode="variance_target")
    assert cal.latent_theta(0.10, 0.02, 1.0) == pytest.approx(0.08)
    assert cal.latent_theta(0.10, 0.02, 5.0) > 0.0            # floored, never negative
    assert BatesSMM(theta_mode="truncated").latent_theta(0.10, 0.02, 1.0) == 0.10
    assert BatesSMM().theta_mode == "variance_target"
    assert BatesSMM.legacy().theta_mode == "truncated"
