"""Sanity tests for the ATT pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nyc_cp.analysis.effects import (
    build_long_df,
    compute_effects,
    summarize_by_unit,
    summarize_over_time,
    summarize_overall,
)


def _make_inputs(offset: float = 5.0, n_days: int = 30, n_units: int = 3):
    dates = pd.date_range("2025-01-05", periods=n_days, freq="D")
    cols = list("ABCDEFG"[:n_units])
    rng = np.random.default_rng(0)
    actual = pd.DataFrame(rng.poisson(100, size=(n_days, n_units)), index=dates, columns=cols)
    mu = actual - offset                    # cf says actual is "offset" above counterfactual
    lower = mu - 10
    upper = mu + 10
    return actual, mu, lower, upper


def test_build_long_has_expected_columns():
    actual, mu, lo, hi = _make_inputs()
    long = build_long_df(actual, mu, lo, hi, id_col="route_id")
    assert set(long.columns) >= {"date", "route_id", "actual", "cf_mean", "cf_lower", "cf_upper"}
    assert len(long) == actual.size


def test_compute_effects_recovers_offset():
    actual, mu, lo, hi = _make_inputs(offset=7.0)
    long = build_long_df(actual, mu, lo, hi, id_col="route_id")
    eff = compute_effects(long, id_col="route_id")
    assert eff["tau"].mean() == pytest.approx(7.0)


def test_signif_when_pi_excludes_zero():
    actual, mu, _, _ = _make_inputs(offset=20.0)
    lower = mu - 1   # tight PI; effect of 20 is well outside
    upper = mu + 1
    long = build_long_df(actual, mu, lower, upper, id_col="route_id")
    eff = compute_effects(long, id_col="route_id")
    assert eff["signif"].all()
    assert (eff["direction"] == "positive").all()


def test_unit_summary_one_row_per_unit():
    actual, mu, lo, hi = _make_inputs(offset=5.0, n_units=4)
    long = build_long_df(actual, mu, lo, hi, id_col="route_id")
    eff = compute_effects(long, id_col="route_id")
    unit = summarize_by_unit(eff, id_col="route_id")
    assert len(unit) == 4
    np.testing.assert_allclose(unit["avg_daily"].to_numpy(), 5.0, rtol=1e-9)


def test_daily_and_overall_consistency():
    actual, mu, lo, hi = _make_inputs(offset=5.0, n_units=3, n_days=30)
    long = build_long_df(actual, mu, lo, hi, id_col="route_id")
    eff = compute_effects(long, id_col="route_id")
    daily = summarize_over_time(eff)
    overall = summarize_overall(summarize_by_unit(eff, id_col="route_id"), id_col="route_id")
    # total ATT = sum of daily mean tau × n_units
    assert overall["total_att"].iloc[0] == pytest.approx(daily["mean_tau"].sum() * 3)
