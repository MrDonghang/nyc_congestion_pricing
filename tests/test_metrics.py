"""Sanity tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nyc_cp.evaluation.metrics import evaluate_forecasts, evaluate_per_series


def _build_inputs(n_days: int = 10, n_series: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-05", periods=n_days, freq="D")
    cols = [f"s{i}" for i in range(n_series)]
    actual = pd.DataFrame(rng.poisson(100, size=(n_days, n_series)), index=dates, columns=cols)
    mu = actual + 5
    lower = mu - 10
    upper = mu + 10
    return actual, mu, lower, upper


def test_perfect_forecast_has_zero_error():
    actual, *_ = _build_inputs()
    mu = actual.copy()
    lower = actual - 1
    upper = actual + 1
    per = evaluate_per_series(actual, mu, lower, upper, coverage_level=0.9)
    assert (per["RMSE"] == 0).all()
    assert (per["MAE"] == 0).all()
    assert (per["R2"] >= 0.99).all()


def test_constant_offset_recovers_mae():
    actual, mu, lower, upper = _build_inputs()
    # mu = actual + 5, so MAE should be exactly 5 for every series.
    per = evaluate_per_series(actual, mu, lower, upper, coverage_level=0.9)
    assert per["MAE"].max() == pytest.approx(5.0)
    assert per["MAE"].min() == pytest.approx(5.0)


def test_coverage_full_when_pi_dominates():
    actual, mu, _, _ = _build_inputs()
    lower = mu - 1000
    upper = mu + 1000
    per = evaluate_per_series(actual, mu, lower, upper, coverage_level=0.9)
    assert (per[col := per.columns[per.columns.str.startswith("Coverage")][0]] == 1.0).all()


def test_evaluate_forecasts_returns_means():
    actual, mu, lower, upper = _build_inputs()
    per = evaluate_per_series(actual, mu, lower, upper, coverage_level=0.9)
    avg = evaluate_forecasts(actual, mu, lower, upper, coverage_level=0.9)
    for c in per.columns:
        assert avg[c] == pytest.approx(per[c].mean())
