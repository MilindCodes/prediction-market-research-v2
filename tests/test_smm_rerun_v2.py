"""The v2 rerun must never write over v1 outputs, and its panels must be
buildable from the repository's own code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import config
from analysis import rerun_v2
from src.smm import panel_moments as pm
from src.smm.nested_ladder import SAMPLING_SPECS, processed_dir
from src.smm.panel import SAMPLING_SPEC_LABELS, load_sampling_panel

PROCESSED = config.DATA_DIR / "processed"
needs_panels = pytest.mark.skipif(
    not (PROCESSED / "smm_panel.parquet").exists()
    or not (PROCESSED / "smm_panel_d_revision.parquet").exists(),
    reason="cached SMM panels not present",
)


def test_every_variant_writes_below_v2_and_none_is_the_v1_estimator():
    seen = set()
    for name in rerun_v2.VARIANTS:
        cal = rerun_v2.make_calibrator(name)
        assert not cal.is_legacy, name
        sub = cal.output_subdir
        assert sub == "v2" or sub.startswith("v2/"), (name, sub)
        assert sub not in seen
        seen.add(sub)
        assert processed_dir(cal) != PROCESSED
        assert rerun_v2.exports_dir(name) != config.DATA_DIR / "exports"
    # the main estimator has all three fixes switched on
    main = rerun_v2.make_calibrator("v2")
    assert (main.j_scaling, main.moments, main.weights, main.observation) == (
        "corrected", "within_contract", "contract", "tick_stale")
    # "Step 1 only" differs from v1 in the scaling switch alone
    step1 = rerun_v2.make_calibrator("scaling_only")
    assert (step1.j_scaling, step1.moments, step1.weights, step1.observation) == (
        "corrected", "pooled", "iid", "none")


def test_sampling_spec_lists_agree():
    assert tuple(f"{f}/{s}" for f, s in SAMPLING_SPECS) == SAMPLING_SPEC_LABELS
    assert rerun_v2.SPECS == SAMPLING_SPEC_LABELS


@needs_panels
def test_calendar_matched_panel_is_built_by_repository_code():
    """Rishi's reproducibility gap: this panel used to exist only inside the
    bootstrap script, so its row of the sampling CSVs could not be rebuilt."""
    daily = load_sampling_panel("D/calendar")
    revision = load_sampling_panel("D/revision")
    matched = load_sampling_panel("D/calendar-matched")
    assert set(matched["contract_id"]) == set(revision["contract_id"])
    assert matched["contract_id"].nunique() == 34 and len(matched) == 3913
    assert len(matched) == daily["contract_id"].isin(revision["contract_id"]).sum()


@needs_panels
def test_daily_panel_zero_accounting_matches_the_validation_report():
    """1,515 of 4,141 increments are zero: 633 at the clip, 882 otherwise."""
    shape = pm.panel_shape(load_sampling_panel("D/calendar"), (0.02, 0.98))
    st = shape["stats"]
    assert int(st[:, pm.I_N].sum()) == 4141 and len(shape["ids"]) == 39
    assert int(st[:, pm.I_ZCLIP].sum()) == 633
    assert int(st[:, pm.I_ZROUND].sum()) == 882
    assert not shape["revision"]
    rev = pm.panel_shape(load_sampling_panel("D/revision"), (0.02, 0.98))
    assert rev["revision"] and np.all(rev["span"] >= rev["n"])


@needs_panels
def test_within_contract_acf_matches_the_validation_report():
    """Lag-1 ACF of ΔX² falls from 0.170 (pooled) to 0.134, lag-2 to ≈ 0."""
    st = pm.panel_shape(load_sampling_panel("D/calendar"), (0.02, 0.98))["stats"]
    m = pm.pool_moments(st)
    assert m[2] == pytest.approx(0.134, abs=0.002)
    assert abs(m[3]) < 0.02


def test_drop_opening_variant_removes_one_increment_per_contract():
    if not (PROCESSED / "smm_panel.parquet").exists():
        pytest.skip("cached SMM panel not present")
    full = rerun_v2._quiet_panel("D/calendar")
    cut = rerun_v2._quiet_panel("D/calendar", "drop_opening")
    assert len(full) - len(cut) == full["contract_id"].nunique()


def test_repo_paths_exist():
    assert Path(rerun_v2.__file__).name == "rerun_v2.py"


def test_null_models_keep_their_seed_positions_and_merton_has_jumps_only():
    """NULL_MODELS indexes the seed, so the first three must never move."""
    assert rerun_v2.NULL_MODELS[:3] == ("Heston", "Bates", "ConstantVol")
    assert set(rerun_v2._SUFFIX) == set(rerun_v2.NULL_MODELS)
    assert len(set(rerun_v2._SUFFIX.values())) == len(rerun_v2._SUFFIX)
    if not rerun_v2.ladder_csv("v2").exists():
        pytest.skip("v2 ladder results not present")
    merton = rerun_v2.dgp_params("v2", "D/calendar", "Merton")
    assert merton["sigma_v"] == 0.0 and merton["sigma_J"] > 0 and merton["lambda_"] > 0
    heston = rerun_v2.dgp_params("v2", "D/calendar", "Heston")
    assert heston["sigma_J"] == 0.0 and heston["lambda_"] == 0.0 and heston["sigma_v"] > 0
