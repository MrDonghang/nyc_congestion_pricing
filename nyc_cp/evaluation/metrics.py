"""Forecast evaluation metrics.

All metrics operate on aligned (date × series) DataFrames. ``evaluate_per_series``
returns one row per series; ``evaluate_forecasts`` returns the column-wise mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-8

METRIC_NAMES = ("RMSE", "MAE", "MAPE", "WMAPE", "SMAPE", "R2", "Coverage")


def _align(*dfs: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    common_cols = dfs[0].columns
    for d in dfs[1:]:
        common_cols = common_cols.intersection(d.columns)
    common_idx = dfs[0].index
    for d in dfs[1:]:
        common_idx = common_idx.intersection(d.index)
    return tuple(d.loc[common_idx, common_cols] for d in dfs)


def evaluate_per_series(
    actual: pd.DataFrame,
    mu: pd.DataFrame,
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    coverage_level: float = 0.9,
) -> pd.DataFrame:
    """One row per series; columns are the metrics in METRIC_NAMES."""
    actual, mu, lower, upper = _align(actual, mu, lower, upper)

    rows = []
    for col in actual.columns:
        y = actual[col].to_numpy(dtype=float)
        p = mu[col].to_numpy(dtype=float)
        lo = lower[col].to_numpy(dtype=float)
        hi = upper[col].to_numpy(dtype=float)

        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        mae = float(np.mean(np.abs(y - p)))
        mape = float(np.mean(np.abs((y - p) / (y + EPS))))
        wmape = float(np.sum(np.abs(y - p)) / (np.sum(np.abs(y)) + EPS))
        smape = float(np.mean(np.abs(y - p) / ((np.abs(y) + np.abs(p)) / 2 + EPS)))
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + EPS)
        coverage = float(np.mean((y >= lo) & (y <= hi)))

        rows.append(
            dict(RMSE=rmse, MAE=mae, MAPE=mape, WMAPE=wmape, SMAPE=smape, R2=r2, Coverage=coverage)
        )

    out = pd.DataFrame(rows, index=actual.columns)
    out.index.name = "series"
    out = out.rename(columns={"Coverage": f"Coverage[{coverage_level}]"})
    return out


def evaluate_forecasts(
    actual: pd.DataFrame,
    mu: pd.DataFrame,
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    coverage_level: float = 0.9,
) -> pd.Series:
    """Column-wise mean of ``evaluate_per_series``."""
    df = evaluate_per_series(actual, mu, lower, upper, coverage_level=coverage_level)
    return df.mean(numeric_only=True)
