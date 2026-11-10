"""
§4.3  Bates SMM Calibration

Model (log-odds space, Δt = 1):
  dX_t = √v_t · dW^X_t  +  J_t · dN_t
  dv_t = κ(θ − v_t)dt   + σ_v · √v_t · dW^v_t
  corr(dW^X, dW^v) = ρ    (baseline: ρ = 0)
  N_t ~ Poisson(λ),  J ~ N(0, σ_J²)

Parameter schedule
------------------
κ : FREE in the main selection run (shared spec across the ladder).  κ only
    enters the simulator through σ_v·√v, so it is identified in Heston and
    Bates and irrelevant (v ≡ θ) in ConstantVol and Merton.  §4.8 re-runs
    the ladder with κ fixed on the grid {1, 5, 10} as a robustness check.
ρ : fixed — 0 (baseline)
θ : fixed — Var(ΔX) / Δt  (with Δt=1: θ = sample variance of increments)
λ : fixed — fraction of |ΔX| > 3·SD (rough jump frequency)
σ_v : FREE — optimizer
σ_J : FREE — optimizer

Nesting guarantee
-----------------
Each model is warm-started at the optimum of the model it nests (Heston at
ConstantVol with σ_v=0; Bates at Heston with σ_J=0 and at Merton with
σ_v=0), so a richer model can never end at a worse objective than the
model it nests — worst case it stays at the warm start.  For this to hold
exactly, ConstantVol is scored by SIMULATION at (σ_v=0, σ_J=0) with the
shared CRN draws, not analytically.

Five moments (computed once on pooled, demeaned real increments)
  [0] Var(ΔX)
  [1] Excess kurtosis(ΔX)
  [2] lag-1 ACF(ΔX²)
  [3] lag-2 ACF(ΔX²)
  [4] 95th-percentile of ΔX

Weighting matrix W
  Diagonal, entries = 1/bootstrap-variance of each moment.
  Computed once, fixed throughout optimisation.

Common random numbers
  Pre-drawn once before optimisation; reused on every objective evaluation.
  Makes the objective deterministic in the parameters so Nelder-Mead converges.

Simulator: Euler–Maruyama with full-truncation (replace v with max(v,0)
           everywhere it appears, including the drift and the sqrt term).

Optimiser: Nelder-Mead with multiple restarts.

J-test: with 5 moments and 2 free params → 3 over-identifying restrictions.

Scale of J and of the standard errors (`j_scaling`)
----------------------------------------------------
W is diag(1 / bootstrap variance of the FULL-SAMPLE moments), so W already
grows like n.  The v1 code multiplied g'Wg by n_real again in the J-test and
divided (G'WG)⁻¹ by n_real again in the SEs, which inflated J by a factor of
n_real (≈4,141 on the daily panel) and shrank the SEs by √n_real.  Point
estimates never depended on this, and neither do bootstrap p-values for ΔJ
(real and simulated ΔJ carry the same factor).

  j_scaling="corrected" (default)  J = g'Wg / (1 + n_real/n_sim),  avar = (G'WG)⁻¹
  j_scaling="legacy_n"             the v1 formulas, kept so the numbers in the
                                   first draft (tag v1-paper-draft) can be
                                   reproduced and diffed against.

Within-contract moments, contract bootstrap, observation process (v2)
---------------------------------------------------------------------
See src/smm/panel_moments.py for the full rationale.  In short:

  moments="within_contract" (default)  the four shape moments are computed
      inside each contract and pooled with weights n_c (n_c − k for the ACF
      lags), on both the real and the simulated side, by one function.  The
      simulator produces n_sim_multiplier replicate PANELS shaped like the
      real one instead of one long path.
  moments="pooled"                     v1: one concatenated series, one path.

  weights="contract" (default)  W = diag(1/var) from resampling whole
      contracts.  weights="iid" is the v1 bootstrap of single increments
      (pooled moments only); it destroys the serial dependence the ACF
      moments measure and so understates their variance.

  observation="tick_stale" (default)  each simulated contract starts at the
      real contract's first price and is observed through the same [clip],
      a price tick and a stale-quote rule (with probability q the observed
      price does not update), so exact-zero increments arise as they do in
      the data.  The zero share is a SIXTH moment and q is concentrated
      out — solved, for every parameter vector, to reproduce the real zero
      share — which costs one degree of freedom in every model.  observation="tick" is the same without stale quotes
      (five moments) — tried first, but rounding alone yields ~2% zeros
      against 21% repeated prices in the daily panel.  observation="none"
      is v1 (latent increments observed directly).

  theta_mode="truncated"        θ plug-in targets the variance of increments
      within 3 SD, as in v1.  theta_mode="variance_target" targets the FULL
      variance and gives the diffusion θ − λσ_J², so that total variance is
      matched with or without jumps.  (With "truncated" the plug-in is only
      29% of the daily panel's variance and σ_J is the one parameter that
      can close the gap — see README, "What changed in v2".)  Under
      observation="tick" the plug-in is solved THROUGH the observation
      process: θ is the latent variance at which the constant-vol model
      reproduces the target.

Any non-legacy calibrator writes its outputs under data/processed/v2/ (see
`output_subdir`), never over the v1 CSVs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

import config
from src.smm import panel_moments as pm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOMENT_LABELS = [
    "Var(ΔX)",
    "Kurt(ΔX)",
    "ACF(ΔX², lag=1)",
    "ACF(ΔX², lag=2)",
    "95th pct(ΔX)",
]
N_MOMENTS = len(MOMENT_LABELS)
LARGE_PENALTY = 1e9
# Upper bound for free κ — twice the top of the {1,5,10} robustness grid;
# unbounded κ lets Nelder-Mead wander into Euler-unstable territory.
KAPPA_MAX = 20.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SMMResult:
    model: str           # "ConstantVol", "Heston", "Bates", "Merton"
    kappa: float
    theta: float
    rho: float
    lambda_: float
    # Free parameters (nan if not applicable to model)
    sigma_v: float
    sigma_J: float
    # Moments
    moments_real: np.ndarray
    moments_sim: np.ndarray
    moment_labels: list = field(default_factory=lambda: list(MOMENT_LABELS))
    # Fit quality
    objective_value: float = np.nan
    j_stat: float = np.nan
    j_dof: int = 0
    j_pvalue: float = np.nan
    # Inference
    se_sigma_v: float = np.nan
    se_sigma_J: float = np.nan
    se_kappa: float = np.nan
    # Stale-quote probability the simulator solved for at the optimum
    # (observation="tick_stale" only)
    stale_q: float = np.nan
    # Book-keeping
    kappa_free: bool = False
    n_real: int = 0
    n_sim: int = 0
    n_free: int = 2
    converged: bool = True


# ---------------------------------------------------------------------------
# Moment computation
# ---------------------------------------------------------------------------

def _acf_scalar(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag:
        return np.nan
    c = np.corrcoef(x[:-lag], x[lag:])
    return float(c[0, 1]) if c.shape == (2, 2) else np.nan


def compute_moments(dx: np.ndarray) -> np.ndarray:
    """Five SMM moments on a demeaned increment array."""
    n = len(dx)
    if n < 10:
        return np.full(N_MOMENTS, np.nan)
    dx2 = dx ** 2
    return np.array([
        float(np.var(dx, ddof=1)),
        float(stats.kurtosis(dx, fisher=True)),
        _acf_scalar(dx2, 1),
        _acf_scalar(dx2, 2),
        float(np.percentile(dx, 95)),
    ])


def pool_demeaned_increments(panel: pd.DataFrame) -> np.ndarray:
    """Concatenate within-contract demeaned ΔX across all contracts."""
    parts: list[np.ndarray] = []
    for _, grp in panel.groupby("contract_id", sort=False):
        dx = grp["delta_X"].dropna().values
        if len(dx) >= 2:
            parts.append(dx - dx.mean())
    return np.concatenate(parts) if parts else np.array([])


# ---------------------------------------------------------------------------
# Euler–Maruyama simulator (pure Python/NumPy, full truncation)
# ---------------------------------------------------------------------------

def _simulate_path(
    sigma_v: float,
    sigma_J: float,
    randoms: dict,
    kappa: float,
    theta: float,
    lambda_: float,
    rho: float,
    n_sim: int,
) -> np.ndarray:
    """Simulate one Bates path and return the ΔX array.

    The variance loop is sequential (v[i+1] depends on v[i]) so it cannot be
    fully vectorised.  If numba is installed, wrapping this function with
    @numba.jit(nopython=True) gives a ~100× speedup; for a research pipeline
    that runs once, the pure-Python version is acceptable.
    """
    dw_x   = randoms["dw_x"]
    dw_v   = randoms["dw_v"]   # already correlated: ρ·dw_x + √(1-ρ²)·dw_v_indep
    u_jump = randoms["u_jump"]
    z_jump = randoms["z_jump"]

    jump_threshold = lambda_   # Δt = 1 so P(jump in step) = λ·1

    v = theta          # initial variance
    dx_sim = np.empty(n_sim)

    for i in range(n_sim):
        v_pos = max(v, 0.0)
        sv    = v_pos ** 0.5

        jump = sigma_J * z_jump[i] if u_jump[i] < jump_threshold else 0.0
        dx_sim[i] = sv * dw_x[i] + jump

        # Full-truncation Euler for v (Δt = 1)
        v = v_pos + kappa * (theta - v_pos) + sigma_v * sv * dw_v[i]
        v = max(v, 0.0)

    return dx_sim


# ---------------------------------------------------------------------------
# Main calibrator
# ---------------------------------------------------------------------------

class BatesSMM:
    """Calibrate the Bates (or nested) model via SMM.

    Parameters
    ----------
    n_sim_multiplier : int
        n_sim = n_sim_multiplier × n_real.  50 in the spec; use 10 while
        debugging for speed, then increase to 50 for the final run.
    n_bootstrap : int
        Bootstrap resamples for the weighting matrix (spec: ~500).
    n_restarts : int
        Nelder-Mead restarts with different starting points.
    rng_seed : int
        Controls reproducibility of bootstrap and CRN draws.
    j_scaling : {"corrected", "legacy_n"}
        "corrected": J = g'Wg·1/(1+n_real/n_sim) and avar = (G'WG)⁻¹.
        "legacy_n": the v1 formulas with the extra factor of n_real (see the
        module docstring).  Parameter estimates are identical either way.
    moments : {"within_contract", "pooled"}
        Where the shape moments are computed; "pooled" is v1.
    weights : {"contract", "iid"} | None
        Bootstrap behind W.  None → "contract" (the only choice for
        within-contract moments); "iid" is v1 and needs moments="pooled".
    q95 : {"standardized", "raw"}
        Units of the within-contract tail moment (see panel_moments).
    observation : {"tick_stale", "tick", "none"}
        Observation process applied to simulated contracts; "none" is v1.
        Needs moments="within_contract".
    tick : float
        Price tick of the observation process (baseline 0.001; 0.0005 is
        the sensitivity — every raw price is a multiple of 0.0005).
    clip : (float, float) | None
        Price clip of the observation process.  None → config's
        [LOG_ODDS_CLIP_LO, LOG_ODDS_CLIP_HI], the clip the panel was built at.
    burn_in : int
        Variance burn-in steps per simulated contract (v starts at θ).
    theta_mode : {"truncated", "variance_target"}
        What the θ plug-in targets; "truncated" is v1.
    output_subdir : str | None
        Sub-directory of data/processed/ that result CSVs are written to.
        None → "" for an all-legacy calibrator (the v1 paths) and "v2" for
        anything else, so corrected runs never overwrite the v1 outputs.
    """

    J_SCALINGS = ("corrected", "legacy_n")
    MOMENTS = ("within_contract", "pooled")
    WEIGHTS = ("contract", "iid")
    Q95 = ("standardized", "raw")
    OBSERVATIONS = ("tick_stale", "tick", "none")
    THETA_MODES = ("truncated", "variance_target")

    def __init__(
        self,
        n_sim_multiplier: int = 20,
        n_bootstrap: int = 500,
        n_restarts: int = 3,
        rng_seed: int = 42,
        j_scaling: str = "corrected",
        moments: str = "within_contract",
        weights: "str | None" = None,
        q95: str = "standardized",
        observation: "str | None" = None,
        tick: float = 0.001,
        clip: "tuple[float, float] | None" = None,
        burn_in: int = pm.DEFAULT_BURN_IN,
        theta_mode: str = "truncated",
        output_subdir: "str | None" = None,
    ):
        weights = weights or "contract"
        observation = observation or ("tick_stale" if moments == "within_contract"
                                      else "none")
        for name, val, allowed in (
            ("j_scaling", j_scaling, self.J_SCALINGS),
            ("moments", moments, self.MOMENTS),
            ("weights", weights, self.WEIGHTS),
            ("q95", q95, self.Q95),
            ("observation", observation, self.OBSERVATIONS),
            ("theta_mode", theta_mode, self.THETA_MODES),
        ):
            if val not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {val!r}")
        if moments == "within_contract" and weights != "contract":
            raise ValueError("within-contract moments need weights='contract': an "
                             "iid bootstrap of increments has no contracts to keep")
        if moments == "pooled" and observation != "none":
            raise ValueError("an observation process needs moments='within_contract' "
                             "(the observation process is applied per contract)")
        self.n_sim_multiplier = n_sim_multiplier
        self.n_bootstrap = n_bootstrap
        self.n_restarts = n_restarts
        self.rng_seed = rng_seed
        self.j_scaling = j_scaling
        self.moments = moments
        self.weights = weights
        self.q95 = q95
        self.observation = observation
        self.tick = float(tick)
        self.clip = tuple(clip) if clip is not None else (
            config.LOG_ODDS_CLIP_LO, config.LOG_ODDS_CLIP_HI)
        self.burn_in = int(burn_in)
        self.theta_mode = theta_mode
        self._output_subdir = output_subdir

    _LEGACY = {"j_scaling": "legacy_n", "moments": "pooled", "weights": "iid",
               "observation": "none", "theta_mode": "truncated"}

    @classmethod
    def legacy(cls, **kwargs) -> "BatesSMM":
        """The v1 estimator exactly as tagged at v1-paper-draft.

        Individual fixes can be switched on from here, e.g.
        BatesSMM.legacy(j_scaling="corrected") is "Step 1 only".
        """
        for key, val in cls._LEGACY.items():
            kwargs.setdefault(key, val)
        return cls(**kwargs)

    def options(self) -> dict:
        """Constructor arguments that rebuild this calibrator."""
        return {
            "n_sim_multiplier": self.n_sim_multiplier, "n_bootstrap": self.n_bootstrap,
            "n_restarts": self.n_restarts, "rng_seed": self.rng_seed,
            "j_scaling": self.j_scaling, "moments": self.moments,
            "weights": self.weights, "q95": self.q95,
            "observation": self.observation, "tick": self.tick, "clip": self.clip,
            "burn_in": self.burn_in, "theta_mode": self.theta_mode,
            "output_subdir": self._output_subdir,
        }

    def replace(self, **changes) -> "BatesSMM":
        """A copy with some options changed (e.g. a different clip or tick)."""
        opts = self.options()
        opts.update(changes)
        return type(self)(**opts)

    @property
    def is_legacy(self) -> bool:
        """True when every switch is at its v1 setting."""
        return all(getattr(self, k) == v for k, v in self._LEGACY.items())

    @property
    def uses_zero_moment(self) -> bool:
        return self.observation == "tick_stale"

    def moment_labels(self) -> list:
        """Labels of the moments this calibrator matches (5 or 6)."""
        labels = list(MOMENT_LABELS)
        if self.moments == "within_contract":
            labels[4] = ("95th pct(|ΔX|/SD)" if self.q95 == "standardized"
                         else "95th pct(|ΔX|)")
        if self.uses_zero_moment:
            labels.append("Zero share")
        return labels

    def _pool(self, stats: np.ndarray) -> np.ndarray:
        return pm.pool_moments(stats, q95=self.q95, zero_share=self.uses_zero_moment)

    def latent_theta(self, theta: float, lambda_: float, sigma_J: float) -> float:
        """Diffusion variance level the simulator actually uses.

        theta_mode="truncated": θ itself.  "variance_target": θ − λσ_J², so
        the jump variance is carved out of the total instead of added to it
        (floored at a sliver of θ so the variance process stays defined).
        """
        if self.theta_mode == "variance_target":
            return max(theta - lambda_ * sigma_J * sigma_J, 1e-6 * theta)
        return theta

    @property
    def output_subdir(self) -> str:
        if self._output_subdir is not None:
            return self._output_subdir
        return "" if self.is_legacy else "v2"

    def _j(self, g: np.ndarray, W: np.ndarray, n_real: int, n_sim: int,
           dof: int) -> tuple[float, float]:
        return _j_test(g, W, n_real, n_sim, dof,
                       n_scaling=(self.j_scaling == "legacy_n"))

    # ------------------------------------------------------------------
    # Public API — called by BatesSMM.fit() and NestedLadder
    # ------------------------------------------------------------------

    def prepare(self, panel: pd.DataFrame) -> dict:
        """Pre-compute everything that is fixed across optimisation runs.

        Returns a 'cache' dict with:
          dx_real, m_real, W, theta, lambda_, n_real, n_sim, randoms
        (plus, for within-contract moments: shape, stats_real, zero_real,
        theta_target, theta_calibration).
        """
        if self.moments == "within_contract":
            return self._prepare_panel(panel)

        dx_real = pool_demeaned_increments(panel)
        n_real  = len(dx_real)
        n_sim   = self.n_sim_multiplier * n_real

        m_real = compute_moments(dx_real)
        if self.weights == "contract":
            W = self._contract_bootstrap_weights_pooled(panel)
        else:
            W = self._bootstrap_weights(dx_real)

        sd_real = float(np.std(dx_real, ddof=1))
        lambda_ = float(np.mean(np.abs(dx_real) > 3 * sd_real))
        theta   = self._theta_target(dx_real, sd_real)

        randoms = self._draw_randoms(n_sim, rho=0.0)

        return {
            "dx_real": dx_real,
            "m_real":  m_real,
            "W":       W,
            "theta":   theta,
            "lambda_": lambda_,
            "sd_real": sd_real,
            "n_real":  n_real,
            "n_sim":   n_sim,
            "randoms": randoms,
        }

    def _theta_target(self, dx_real: np.ndarray, sd_real: float) -> float:
        """Variance level the θ plug-in aims at."""
        if self.theta_mode == "variance_target":
            return float(np.var(dx_real, ddof=1))
        # v1: truncated variance (increments within 3 SD) so that jump
        # variance doesn't fold into θ and double-count once σ_J enters the
        # simulator.
        diffusion_mask = np.abs(dx_real) <= 3 * sd_real
        if diffusion_mask.sum() > 10:
            return float(np.var(dx_real[diffusion_mask], ddof=1))
        return float(np.var(dx_real, ddof=1))

    def _prepare_panel(self, panel: pd.DataFrame) -> dict:
        """prepare() for within-contract moments (v2)."""
        shape   = pm.panel_shape(panel, self.clip)
        dx_real = pool_demeaned_increments(panel)
        n_real  = len(dx_real)
        n_sim   = self.n_sim_multiplier * n_real   # S replicate panels

        stats_real = shape["stats"]
        m_real = self._pool(stats_real)
        W, _ = pm.contract_bootstrap_weights(
            stats_real, self.n_bootstrap, np.random.default_rng(self.rng_seed),
            q95=self.q95, zero_share=self.uses_zero_moment,
        )

        sd_real = float(np.std(dx_real, ddof=1))
        lambda_ = float(np.mean(np.abs(dx_real) > 3 * sd_real))
        target  = self._theta_target(dx_real, sd_real)

        # Revision-sampled panels are simulated on the calendar grid and
        # then collapsed, exactly as the panel builder does it.  Without an
        # observation process there are no zeros to collapse, so each
        # increment is one model step (the v1 convention).
        collapse = bool(shape["revision"] and self.observation != "none")
        n_steps = shape["span"] if collapse else shape["n"]
        randoms = pm.draw_panel_randoms(
            n_steps, self.n_sim_multiplier, self.burn_in,
            np.random.default_rng(self.rng_seed + 999), rho=0.0,
        )
        randoms["x0"] = shape["x0"]
        randoms["collapse"] = collapse
        zero_real = pm.zero_shares(stats_real)
        # Under "tick_stale" every simulation solves for the stale-quote
        # probability that reproduces this share.
        randoms["zero_target"] = (zero_real["zero_share"]
                                  if self.uses_zero_moment else None)

        calibration = {"target": target, "achieved": target, "bracketed": True}
        theta = target
        if self.observation != "none":
            theta, calibration = self._calibrate_theta(target, randoms)
            st, q = self._panel_stats(0.0, 0.0, 1.0, theta, 0.0, randoms)
            calibration["stale_q"] = q
            calibration["zero_share_target"] = zero_real["zero_share"]
            calibration["zero_share_achieved"] = pm.zero_shares(st)["zero_share"]

        return {
            "dx_real": dx_real,
            "m_real":  m_real,
            "W":       W,
            "theta":   theta,
            "lambda_": lambda_,
            "sd_real": sd_real,
            "n_real":  n_real,
            "n_sim":   n_sim,
            "randoms": randoms,
            "shape":   shape,
            "stats_real": stats_real,
            "zero_real":  zero_real,
            "theta_target": target,
            "theta_calibration": calibration,
        }

    def _panel_stats(self, sigma_v, sigma_J, kappa, theta_latent, lambda_,
                     randoms) -> tuple[np.ndarray, float]:
        """(S, C, K) per-contract statistics of the simulated panels, and the
        stale-quote probability the simulator solved for."""
        return pm.simulate_panel_stats(
            sigma_v, sigma_J, kappa, theta_latent, lambda_, randoms, randoms["x0"],
            observation=self.observation, tick=self.tick, clip=self.clip,
            collapse=randoms["collapse"], zero_target=randoms.get("zero_target"),
        )

    def _calibrate_theta(self, target: float, randoms: dict) -> tuple[float, dict]:
        """Latent θ at which constant vol, observed through the clip and the
        tick, reproduces the target variance.

        Observed variance is not latent variance: the clip removes moves
        beyond [lo, hi] and (under revision sampling) an increment spans
        several days.  Plugging the observed variance in as the latent θ
        would leave every model short on the variance moment by the same
        amount.  Solved once with the CRN draws; fixed across the ladder.
        """
        def observed_var(log_theta: float) -> float:
            st, _ = self._panel_stats(0.0, 0.0, 1.0, math.exp(log_theta), 0.0, randoms)
            return float(np.nanmean(pm.pool_moments(st, q95=self.q95)[:, 0]))

        grid = np.log(target) + np.linspace(np.log(1 / 30), np.log(300), 41)
        vals = np.array([observed_var(g) for g in grid])
        diff = vals - target
        theta, bracketed = float(np.exp(grid[np.nanargmin(np.abs(diff))])), False
        sign_change = np.flatnonzero(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)
        if len(sign_change):
            i = int(sign_change[0])        # lowest root: the least-clipped one
            try:
                theta = float(np.exp(optimize.brentq(
                    lambda g: observed_var(g) - target, grid[i], grid[i + 1],
                    xtol=1e-6, maxiter=60)))
                bracketed = True
            except ValueError:
                pass
        return theta, {"target": target, "bracketed": bracketed,
                       "achieved": observed_var(math.log(theta))}

    def simulated_zero_shares(self, result: "SMMResult", cache: dict) -> dict:
        """Zero-increment shares of the simulated panels at a fitted model —
        the calibration check for the observation process."""
        th = self.latent_theta(result.theta, result.lambda_, result.sigma_J)
        st, q = self._panel_stats(result.sigma_v, result.sigma_J, result.kappa, th,
                                  result.lambda_, cache["randoms"])
        out = pm.zero_shares(st)
        out["stale_q"] = q
        return out

    def _stale_q_at(self, sigma_v, sigma_J, kappa, theta, lambda_, randoms) -> float:
        if not (self.uses_zero_moment and randoms.get("kind") == "panel"):
            return np.nan
        return self._panel_stats(sigma_v, sigma_J, kappa,
                                 self.latent_theta(theta, lambda_, sigma_J),
                                 lambda_, randoms)[1]

    def _dof(self, n_moments: int, n_free: int) -> int:
        """Over-identifying restrictions.  The concentrated-out stale-quote
        probability is an estimated parameter too."""
        return n_moments - n_free - (1 if self.uses_zero_moment else 0)

    def fit_bates(
        self,
        cache: dict,
        kappa: float = 5.0,
        rho: float = 0.0,
        verbose: bool = True,
        free_kappa: bool = False,
        warm_starts: "list[np.ndarray] | None" = None,
    ) -> SMMResult:
        """Fit Bates (σ_v and σ_J free; κ free too when free_kappa)."""
        free = ("sigma_v", "sigma_J", "kappa") if free_kappa \
            else ("sigma_v", "sigma_J")
        return self._fit(
            cache, kappa=kappa, rho=rho,
            model="Bates", free=free,
            warm_starts=warm_starts, verbose=verbose,
        )

    def fit_heston(
        self,
        cache: dict,
        kappa: float = 5.0,
        rho: float = 0.0,
        verbose: bool = True,
        free_kappa: bool = False,
        warm_starts: "list[np.ndarray] | None" = None,
    ) -> SMMResult:
        """Fit Heston (σ_v free, σ_J = 0; κ free too when free_kappa)."""
        free = ("sigma_v", "kappa") if free_kappa else ("sigma_v",)
        return self._fit(
            cache, kappa=kappa, rho=rho,
            model="Heston", free=free,
            warm_starts=warm_starts, verbose=verbose,
        )

    def fit_merton(
        self,
        cache: dict,
        kappa: float = 5.0,
        rho: float = 0.0,
        verbose: bool = True,
        warm_starts: "list[np.ndarray] | None" = None,
    ) -> SMMResult:
        """Fit Merton jump-diffusion (σ_J free, σ_v = 0).

        κ is never free here: with σ_v = 0 the variance stays at θ and κ
        drops out of the model entirely.
        """
        return self._fit(
            cache, kappa=kappa, rho=rho,
            model="Merton", free=("sigma_J",),
            warm_starts=warm_starts, verbose=verbose,
        )

    def fit_constant_vol(
        self,
        cache: dict,
        kappa: float = 5.0,
        rho: float = 0.0,
        verbose: bool = True,
    ) -> SMMResult:
        """Evaluate constant-vol diffusion (no free params).

        Moments are SIMULATED at (σ_v=0, σ_J=0) with the shared CRN draws —
        not the analytical N(0, θ) values — so ConstantVol is exactly the
        σ_v=0 restriction of Heston under the same simulator.  Analytical
        moments would break the J(CV) ≥ J(Heston) nesting guarantee by the
        sampling noise of the CRN draws.  (κ is recorded but inert: σ_v=0
        pins v ≡ θ.)
        """
        m_real  = cache["m_real"]
        W       = cache["W"]
        theta   = cache["theta"]
        lambda_ = cache["lambda_"]
        n_real  = cache["n_real"]
        n_sim   = cache["n_sim"]
        randoms = cache["randoms"]

        m_sim = self._simulate_moments(
            0.0, 0.0, kappa, theta, lambda_, rho, n_sim, randoms
        )
        g = m_sim - m_real
        obj = float(g @ W @ g)
        n_mom = self._dof(len(m_real), 0)                      # no free params
        j_stat, j_p = self._j(g, W, n_real, n_sim, dof=n_mom)

        if verbose:
            print(f"  ConstantVol: obj={obj:.4f}, J={j_stat:.3f}"
                  f" (χ²({n_mom}), p={j_p:.3f})")

        return SMMResult(
            model="ConstantVol",
            kappa=kappa, theta=theta, rho=rho,
            lambda_=lambda_, sigma_v=0.0, sigma_J=0.0,
            moments_real=m_real, moments_sim=m_sim,
            moment_labels=self.moment_labels(),
            objective_value=obj,
            j_stat=j_stat, j_dof=n_mom, j_pvalue=j_p,
            n_real=n_real, n_sim=n_sim, n_free=0,
            stale_q=self._stale_q_at(0.0, 0.0, kappa, theta, lambda_, randoms),
        )

    def standard_errors(
        self,
        result: SMMResult,
        cache: dict,
        h_frac: "float | None" = None,
    ) -> tuple[float, float, float]:
        """Finite-difference Jacobian → asymptotic SEs for the free params.

        avar(θ̂) = (G'WG)⁻¹  [simplified: assuming W ≈ Ω⁻¹]

        W is the inverse variance of the full-sample moments, so (G'WG)⁻¹ is
        already the variance of θ̂ — no further division by n.  The v1 code
        divided by n_real again (SEs too small by √n_real); that formula is
        kept under j_scaling="legacy_n".

        κ is included in the Jacobian when the fit had κ free, so the σ
        SEs are not conditional on a fixed κ.

        Under observation="tick" the simulated moments are step functions
        of the parameters (a tiny parameter change moves no price across a
        tick), so the finite-difference step is widened from 0.1% to 2%.

        Returns (se_sigma_v, se_sigma_J, se_kappa); nan if not free.
        """
        if h_frac is None:
            h_frac = 2e-2 if self.observation != "none" else 1e-3
        W       = cache["W"]
        randoms = cache["randoms"]
        kappa   = result.kappa
        theta   = result.theta
        lambda_ = result.lambda_
        rho     = result.rho
        n_sim   = cache["n_sim"]

        sv  = result.sigma_v
        sJ  = result.sigma_J

        model = result.model
        if model == "Bates":
            free_names = ["sigma_v", "sigma_J"]
        elif model == "Heston":
            free_names = ["sigma_v"]
        elif model == "Merton":
            free_names = ["sigma_J"]
        else:
            return np.nan, np.nan, np.nan
        if result.kappa_free:
            free_names.append("kappa")
        base = {"sigma_v": sv, "sigma_J": sJ, "kappa": kappa}
        free_vals = np.array([base[n] for n in free_names])

        n_free = len(free_vals)
        G = np.zeros((len(result.moments_real), n_free))

        for j, val in enumerate(free_vals):
            h = max(abs(val) * h_frac, 1e-6)
            v_hi = free_vals.copy(); v_hi[j] += h
            v_lo = free_vals.copy(); v_lo[j] -= h

            m_hi = self._moments_at(
                v_hi, free_names, sv, sJ,
                kappa, theta, lambda_, rho, n_sim, randoms
            )
            m_lo = self._moments_at(
                v_lo, free_names, sv, sJ,
                kappa, theta, lambda_, rho, n_sim, randoms
            )
            G[:, j] = (m_hi - m_lo) / (2 * h)

        try:
            GWG = G.T @ W @ G
            avar = np.linalg.inv(GWG)
            if self.j_scaling == "legacy_n":
                avar = avar / cache["n_real"]
            ses = np.sqrt(np.abs(np.diag(avar)))
        except np.linalg.LinAlgError:
            ses = np.full(n_free, np.nan)

        se_sv = ses[free_names.index("sigma_v")] if "sigma_v" in free_names else np.nan
        se_sJ = ses[free_names.index("sigma_J")] if "sigma_J" in free_names else np.nan
        se_k  = ses[free_names.index("kappa")]   if "kappa" in free_names else np.nan
        return se_sv, se_sJ, se_k

    # ------------------------------------------------------------------
    # Internal: unified optimiser
    # ------------------------------------------------------------------

    def _fit(
        self,
        cache: dict,
        kappa: float,
        rho: float,
        model: str,
        free: tuple[str, ...],
        verbose: bool,
        warm_starts: "list[np.ndarray] | None" = None,
    ) -> SMMResult:
        m_real  = cache["m_real"]
        W       = cache["W"]
        theta   = cache["theta"]
        lambda_ = cache["lambda_"]
        sd_real = cache["sd_real"]
        n_real  = cache["n_real"]
        n_sim   = cache["n_sim"]
        randoms = cache["randoms"]

        n_free = len(free)
        j_dof  = self._dof(len(m_real), n_free)

        # Starting points: warm starts (the nested model's optimum) always
        # run first and are never dropped by the restart budget — they are
        # what guarantees the richer model can't end worse than the model
        # it nests.
        x0_pool = (
            [np.asarray(w, dtype=float) for w in (warm_starts or [])]
            + self._grid_starts(free, sd_real, kappa, theta, lambda_, rho,
                                n_sim, randoms, m_real, W)
            + self._starting_points(free, sd_real)[: self.n_restarts]
        )

        best_x   = x0_pool[0]
        best_val = np.inf
        best_sim_moments = compute_moments(np.zeros(10))

        for i, x0 in enumerate(x0_pool):
            try:
                res = optimize.minimize(
                    self._objective,
                    x0,
                    args=(free, kappa, theta, lambda_, rho, n_sim, randoms,
                          m_real, W),
                    method="Nelder-Mead",
                    options={
                        "maxiter": 10_000,
                        "xatol": 1e-7,
                        "fatol": 1e-9,
                        "adaptive": True,
                    },
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
                    if verbose:
                        print(f"  [{model}] restart {i+1}: "
                              + ", ".join(f"{n}={abs(v):.4f}"
                                         for n, v in zip(free, res.x))
                              + f"  obj={res.fun:.6f}")
            except Exception as e:
                if verbose:
                    print(f"  [{model}] restart {i+1} failed: {e}")

        # Unpack best parameters
        sv_hat = abs(float(best_x[free.index("sigma_v")])) \
            if "sigma_v" in free else 0.0
        sJ_hat = abs(float(best_x[free.index("sigma_J")])) \
            if "sigma_J" in free else 0.0
        kap_hat = abs(float(best_x[free.index("kappa")])) \
            if "kappa" in free else kappa

        m_sim = self._simulate_moments(
            sv_hat, sJ_hat, kap_hat, theta, lambda_, rho, n_sim, randoms
        )
        g = m_sim - m_real
        j_stat, j_p = self._j(g, W, n_real, n_sim, dof=j_dof)

        if verbose:
            kap_note = f", κ={kap_hat:.4f}" if "kappa" in free else ""
            print(f"  [{model}] → σ_v={sv_hat:.4f}, σ_J={sJ_hat:.4f}{kap_note}, "
                  f"J={j_stat:.3f} (χ²({j_dof}), p={j_p:.3f})")

        return SMMResult(
            model=model,
            kappa=kap_hat, theta=theta, rho=rho,
            lambda_=lambda_,
            sigma_v=sv_hat, sigma_J=sJ_hat,
            moments_real=m_real, moments_sim=m_sim,
            moment_labels=self.moment_labels(),
            objective_value=best_val,
            j_stat=j_stat, j_dof=j_dof, j_pvalue=j_p,
            kappa_free=("kappa" in free),
            n_real=n_real, n_sim=n_sim, n_free=n_free,
            stale_q=self._stale_q_at(sv_hat, sJ_hat, kap_hat, theta, lambda_, randoms),
        )

    def _objective(
        self,
        x: np.ndarray,
        free: tuple[str, ...],
        kappa: float,
        theta: float,
        lambda_: float,
        rho: float,
        n_sim: int,
        randoms: dict,
        m_real: np.ndarray,
        W: np.ndarray,
    ) -> float:
        # σ_v = 0 and σ_J = 0 are legitimate boundary values (they ARE the
        # nested restrictions), so only negatives are penalised.  κ must be
        # strictly positive and bounded — see KAPPA_MAX.
        for name, v in zip(free, x):
            if v < 0:
                return LARGE_PENALTY
            if name == "kappa" and (v <= 0 or v > KAPPA_MAX):
                return LARGE_PENALTY

        sv  = x[free.index("sigma_v")] if "sigma_v" in free else 0.0
        sJ  = x[free.index("sigma_J")] if "sigma_J" in free else 0.0
        kap = x[free.index("kappa")]   if "kappa" in free else kappa

        m_sim = self._simulate_moments(sv, sJ, kap, theta, lambda_, rho,
                                       n_sim, randoms)
        if np.any(np.isnan(m_sim)):
            return LARGE_PENALTY
        g = m_sim - m_real
        return float(g @ W @ g)

    def _simulate_moments(
        self,
        sigma_v: float,
        sigma_J: float,
        kappa: float,
        theta: float,
        lambda_: float,
        rho: float,
        n_sim: int,
        randoms: dict,
    ) -> np.ndarray:
        theta = self.latent_theta(theta, lambda_, sigma_J)
        if randoms.get("kind") == "panel":
            st, _ = self._panel_stats(sigma_v, sigma_J, kappa, theta, lambda_, randoms)
            return pm.mean_over_replicates(self._pool(st))
        dx = _simulate_path(
            sigma_v, sigma_J, randoms,
            kappa, theta, lambda_, rho, n_sim,
        )
        dx = dx - dx.mean()   # demean to match real increments
        return compute_moments(dx)

    def _moments_at(
        self,
        free_vals: np.ndarray,
        free_names: list[str],
        sv_base: float,
        sJ_base: float,
        kappa: float,
        theta: float,
        lambda_: float,
        rho: float,
        n_sim: int,
        randoms: dict,
    ) -> np.ndarray:
        sv = abs(float(free_vals[free_names.index("sigma_v")])) \
            if "sigma_v" in free_names else sv_base
        sJ = abs(float(free_vals[free_names.index("sigma_J")])) \
            if "sigma_J" in free_names else sJ_base
        kap = abs(float(free_vals[free_names.index("kappa")])) \
            if "kappa" in free_names else kappa
        return self._simulate_moments(sv, sJ, kap, theta, lambda_, rho,
                                      n_sim, randoms)

    # ------------------------------------------------------------------
    # Weighting matrix and random draws
    # ------------------------------------------------------------------

    def _bootstrap_weights(self, dx: np.ndarray) -> np.ndarray:
        """Diagonal W = diag(1/bootstrap-variance of each moment)."""
        rng = np.random.default_rng(self.rng_seed)
        n = len(dx)
        draws = np.zeros((self.n_bootstrap, N_MOMENTS))
        for b in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            dx_b = dx[idx] - dx[idx].mean()
            draws[b] = compute_moments(dx_b)
        variances = np.nanvar(draws, axis=0, ddof=1)
        variances = np.where(variances > 1e-30, variances, 1.0)
        return np.diag(1.0 / variances)

    def _contract_bootstrap_weights_pooled(self, panel: pd.DataFrame) -> np.ndarray:
        """W for POOLED moments from resampling whole contracts.

        Lets the W fix be switched on without the within-contract moments.
        """
        parts = [g["delta_X"].dropna().values
                 for _, g in panel.groupby("contract_id", sort=False)]
        parts = [d - d.mean() for d in parts if len(d) >= 2]
        rng = np.random.default_rng(self.rng_seed)
        draws = np.zeros((self.n_bootstrap, N_MOMENTS))
        for b in range(self.n_bootstrap):
            idx = rng.integers(0, len(parts), size=len(parts))
            draws[b] = compute_moments(np.concatenate([parts[i] for i in idx]))
        variances = np.nanvar(draws, axis=0, ddof=1)
        variances = np.where(variances > 1e-30, variances, 1.0)
        return np.diag(1.0 / variances)

    def _draw_randoms(self, n_sim: int, rho: float) -> dict:
        """Pre-draw all four CRN arrays; fix seed for reproducibility."""
        rng = np.random.default_rng(self.rng_seed + 999)
        dw_x       = rng.standard_normal(n_sim)
        dw_v_indep = rng.standard_normal(n_sim)
        u_jump     = rng.uniform(size=n_sim)
        z_jump     = rng.standard_normal(n_sim)
        dw_v = rho * dw_x + np.sqrt(max(1.0 - rho**2, 0.0)) * dw_v_indep
        return {
            "dw_x":   dw_x,
            "dw_v":   dw_v,
            "u_jump": u_jump,
            "z_jump": z_jump,
        }

    def _starting_points(
        self, free: tuple[str, ...], sd_real: float
    ) -> list[np.ndarray]:
        guesses = {
            "sigma_v": [0.3, 0.1, 0.6],
            "sigma_J": [2.0 * sd_real, sd_real, 3.0 * sd_real],
            "kappa":   [5.0, 1.0, 10.0],
        }
        return [
            np.array([guesses[name][i] for name in free])
            for i in range(3)
        ]

    # Coarse global grid used to seed Nelder-Mead.
    #
    # This objective has well-separated basins: the jump solution sits near
    # (sigma_v ~ 0.05, kappa ~ 0.001) while the pure-vol solution sits near
    # (sigma_v ~ 0.28, kappa ~ 0.27).  Nelder-Mead started from a handful of
    # hand-picked points can converge into the wrong basin and report it as
    # the optimum — which then feeds a difference-in-J test that reads the
    # optimizer's failure as a model-selection verdict.  Seeding from the
    # best few points of a coarse grid makes the search basin-agnostic.
    GRID_SIGMA_V = (0.0, 0.05, 0.1, 0.15, 0.2, 0.28, 0.4, 0.6, 0.9)
    GRID_SIGMA_J_MULT = (0.0, 0.3, 1.2, 2.5, 3.8, 4.7, 5.7)
    GRID_KAPPA = (0.001, 0.01, 0.1, 0.3, 1.0, 5.0)

    def _grid_starts(
        self,
        free: tuple[str, ...],
        sd_real: float,
        kappa: float,
        theta: float,
        lambda_: float,
        rho: float,
        n_sim: int,
        randoms: dict,
        m_real: np.ndarray,
        W: np.ndarray,
        n_keep: int = 3,
    ) -> list[np.ndarray]:
        """Evaluate a coarse grid and return the best `n_keep` points."""
        axes = []
        for name in free:
            if name == "sigma_v":
                axes.append(self.GRID_SIGMA_V)
            elif name == "sigma_J":
                axes.append(tuple(m * sd_real for m in self.GRID_SIGMA_J_MULT))
            else:
                axes.append(self.GRID_KAPPA)

        import itertools
        scored: list[tuple[float, np.ndarray]] = []
        for combo in itertools.product(*axes):
            x = np.array(combo, dtype=float)
            val = self._objective(x, free, kappa, theta, lambda_, rho,
                                  n_sim, randoms, m_real, W)
            scored.append((val, x))
        scored.sort(key=lambda t: t[0])
        return [x for _, x in scored[:n_keep]]


# ---------------------------------------------------------------------------
# Identification check (§4.3)
# ---------------------------------------------------------------------------

def identification_check(
    result: "SMMResult",
    cache: dict,
    h_frac: float = 0.20,
    calibrator: "BatesSMM | None" = None,
) -> "pd.DataFrame":
    """§4.3 identification check.

    Perturb σ_v and σ_J each by ±h_frac (default ±20%) and report how much
    each moment changes.  The spec requires that σ_v primarily moves the
    ACF-of-squares moments (vol clustering) and σ_J primarily moves the
    tail (95th pct) and kurtosis.  If both parameters move the same moments,
    the point estimates shouldn't be trusted.

    Returns a DataFrame with columns:
      moment, sigma_v_sensitivity, sigma_J_sensitivity, identified_by
    """
    sv    = result.sigma_v
    sJ    = result.sigma_J
    kappa = result.kappa
    theta = result.theta
    lam   = result.lambda_
    rho   = result.rho
    randoms = cache["randoms"]
    n_sim   = cache["n_sim"]

    # Reuse a BatesSMM instance to get _simulate_moments (avoids re-importing
    # _simulate_path which lives in the same module — the circular guard above
    # is no longer needed, but a method call is cleaner).
    # It must be configured like the calibrator that produced `cache`.
    _cal = calibrator or (BatesSMM() if randoms.get("kind") == "panel"
                          else BatesSMM.legacy())

    def _moments_at(sv_: float, sJ_: float) -> np.ndarray:
        return _cal._simulate_moments(sv_, sJ_, kappa, theta, lam, rho, n_sim, randoms)

    m_base = _moments_at(sv, sJ)

    rows = []
    for param_name, sv_p, sJ_p in [
        ("sigma_v", sv * (1 + h_frac), sJ),
        ("sigma_v", sv * (1 - h_frac), sJ),
        ("sigma_J", sv, sJ * (1 + h_frac)),
        ("sigma_J", sv, sJ * (1 - h_frac)),
    ]:
        m_pert = _moments_at(sv_p, sJ_p)
        for j, label in enumerate(result.moment_labels):
            base = m_base[j]
            if abs(base) > 1e-10:
                sens = (m_pert[j] - base) / abs(base)
            else:
                sens = m_pert[j] - base
            rows.append({"param": param_name, "moment": label,
                         "sensitivity": float(sens)})

    df = pd.DataFrame(rows)
    # Average absolute sensitivity per (param, moment)
    agg = (
        df.groupby(["param", "moment"])["sensitivity"]
          .apply(lambda x: float(np.mean(np.abs(x))))
          .reset_index(name="abs_sensitivity")
    )
    wide = agg.pivot(index="moment", columns="param", values="abs_sensitivity")
    wide.columns.name = None
    wide = wide.reindex(result.moment_labels)

    sv_col = "sigma_v" if "sigma_v" in wide.columns else None
    sJ_col = "sigma_J" if "sigma_J" in wide.columns else None

    if sv_col and sJ_col:
        wide["identified_by"] = wide.apply(
            lambda r: "sigma_v" if r[sv_col] > r[sJ_col] else "sigma_J",
            axis=1,
        )
    wide = wide.reset_index()

    print("\n=== §4.3 Identification Check ===")
    print(f"  Perturbation: ±{h_frac*100:.0f}% of optimal parameters")
    print(f"  σ_v = {sv:.4f},  σ_J = {sJ:.4f}")
    print()
    print(wide.to_string(index=False, float_format="{:.4f}".format))

    # Warn if identification is poor
    if sv_col and sJ_col:
        acf_moments = [m for m in result.moment_labels if "ACF" in m]
        tail_moments = [m for m in result.moment_labels if "95th" in m or "Kurt" in m]
        acf_rows  = wide[wide["moment"].isin(acf_moments)]
        tail_rows = wide[wide["moment"].isin(tail_moments)]

        sv_drives_acf  = (acf_rows[sv_col]  > acf_rows[sJ_col]).all()
        sJ_drives_tail = (tail_rows[sJ_col] > tail_rows[sv_col]).all()

        if sv_drives_acf and sJ_drives_tail:
            print("\n  Identification: PASS")
            print("    σ_v drives ACF-of-squares; σ_J drives tail + kurtosis")
        else:
            print("\n  Identification: WARNING")
            if not sv_drives_acf:
                print("    σ_v does NOT dominate ACF moments — vol-of-vol hard to pin down")
            if not sJ_drives_tail:
                print("    σ_J does NOT dominate tail + kurtosis — jump size hard to pin down")
            print("    Point estimates may not be trustworthy individually.")

    return wide


# ---------------------------------------------------------------------------
# J-test (standalone — also used by NestedLadder)
# ---------------------------------------------------------------------------

def _j_test(
    g: np.ndarray,
    W: np.ndarray,
    n_real: int,
    n_sim: int,
    dof: int,
    n_scaling: bool = False,
) -> tuple[float, float]:
    """Hansen over-identification J-statistic.

    J = g'Wg / (1 + n_real/n_sim).  W is the inverse variance of the
    full-sample moments, so g'Wg is already O(1) under the null; the only
    correction is for simulation noise in the simulated moments.

    n_scaling=True reproduces the v1 statistic, which multiplied by n_real a
    second time (J too large by a factor of n_real).
    """
    if dof <= 0:
        return 0.0, 1.0
    scale = 1.0 / (1.0 + n_real / n_sim)
    j_stat = scale * float(g @ W @ g)
    if n_scaling:
        j_stat *= n_real
    j_p    = float(1.0 - stats.chi2.cdf(j_stat, df=dof))
    return j_stat, j_p
