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


# ---- PI propagation: variance cumulates, not bounds --------------------------


def _constant_pi_inputs(sigma: float = 5.0, offset: float = 0.0, n_days: int = 100, n_units: int = 4):
    """Build inputs where every (unit, day) PI has the same width 2*z*sigma."""
    from scipy.stats import norm

    z = norm.ppf((1 + 0.9) / 2)
    dates = pd.date_range("2025-01-05", periods=n_days, freq="D")
    cols = [f"u{i}" for i in range(n_units)]
    rng = np.random.default_rng(0)
    mu = pd.DataFrame(rng.uniform(80, 120, size=(n_days, n_units)), index=dates, columns=cols)
    actual = mu + offset
    lower = mu - z * sigma
    upper = mu + z * sigma
    return actual, mu, lower, upper, sigma, z


def test_per_unit_cum_pi_grows_as_sqrt_t():
    """cum_se at day T should equal sigma * sqrt(T) (independent-day Gaussian)."""
    actual, mu, lo, hi, sigma, _ = _constant_pi_inputs(sigma=5.0, n_days=50, n_units=3)
    long = build_long_df(actual, mu, lo, hi, id_col="route_id")
    eff = compute_effects(long, id_col="route_id")
    last = eff.groupby("route_id").tail(1)
    expected = sigma * np.sqrt(50)
    np.testing.assert_allclose(last["cum_se"].to_numpy(), expected, rtol=1e-9)
    # Sanity: must NOT equal sigma * T (the old buggy "cumsum-of-bounds" width).
    assert not np.allclose(last["cum_se"].to_numpy(), sigma * 50)


def test_cross_unit_mean_pi_shrinks_as_one_over_sqrt_n():
    """daily se_mean_tau = sigma / sqrt(N) when all units have the same sigma."""
    actual, mu, lo, hi, sigma, _ = _constant_pi_inputs(sigma=5.0, n_days=10, n_units=9)
    long = build_long_df(actual, mu, lo, hi, id_col="route_id")
    eff = compute_effects(long, id_col="route_id")
    daily = summarize_over_time(eff)
    np.testing.assert_allclose(daily["se_mean_tau"].to_numpy(), sigma / np.sqrt(9), rtol=1e-9)


def test_att_se_per_unit():
    """unit-level att_se = sigma * sqrt(T) when all per-day sigmas are equal."""
    actual, mu, lo, hi, sigma, _ = _constant_pi_inputs(sigma=5.0, n_days=40, n_units=3)
    long = build_long_df(actual, mu, lo, hi, id_col="route_id")
    eff = compute_effects(long, id_col="route_id")
    unit = summarize_by_unit(eff, id_col="route_id")
    np.testing.assert_allclose(unit["att_se"].to_numpy(), sigma * np.sqrt(40), rtol=1e-9)


def test_signif_share_excess_is_share_minus_alpha():
    """signif_share_excess = signif_share - (1 - coverage_level)."""
    actual, mu, lo, hi = _make_inputs(offset=0.0, n_days=30, n_units=3)
    long = build_long_df(actual, mu, lo, hi, id_col="route_id")
    eff = compute_effects(long, id_col="route_id", coverage_level=0.9)
    unit = summarize_by_unit(eff, id_col="route_id", coverage_level=0.9)
    np.testing.assert_allclose(
        (unit["signif_share"] - 0.10).to_numpy(),
        unit["signif_share_excess"].to_numpy(),
        atol=1e-12,
    )


def test_overall_has_total_att_ci():
    """summarize_overall now exposes Gaussian CI on total_att."""
    actual, mu, lo, hi = _make_inputs(offset=5.0, n_units=4, n_days=30)
    long = build_long_df(actual, mu, lo, hi, id_col="route_id")
    eff = compute_effects(long, id_col="route_id")
    overall = summarize_overall(summarize_by_unit(eff, id_col="route_id"), id_col="route_id")
    for col in ("total_att_se", "total_att_lo", "total_att_hi", "total_att_signif", "avg_signif_share_excess"):
        assert col in overall.columns, f"missing {col}"
    # CI must contain the point estimate.
    row = overall.iloc[0]
    assert row["total_att_lo"] <= row["total_att"] <= row["total_att_hi"]
