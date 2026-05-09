"""Quantile-regression calibration of forecaster outputs.

Use the validation window as a calibration set: learn the conditional quantiles
of forecaster residuals (``actual - mu``) given calendar + level features, then
add those residual quantiles to test forecasts. This corrects systematic bias
(e.g. weekly-cycle dampening) in the point forecast and recalibrates the
prediction interval simultaneously.

Identifying assumption: the conditional-quantile structure of residuals is
stable from val to test, i.e. the policy does not interact with the calibration
features. If the policy itself shifts (say) the weekly cycle, this method will
absorb part of the real policy effect.

Features
--------
DOW (6 dummies, Monday omitted), month (11 dummies, Jan omitted), federal
holiday indicator, log per-unit historical level, the forecaster's own
prediction ``mu_pred``, and DOW × log_level interactions (so weekly amplitude
is allowed to scale with unit size).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.linear_model import QuantileRegressor

log = logging.getLogger(__name__)

DEFAULT_QUANTILES = (0.05, 0.5, 0.95)
ID_DROP = {"date", "unit_id"}


def _melt_panel(df: pd.DataFrame, value_name: str, id_col: str = "unit_id") -> pd.DataFrame:
    df = df.copy()
    df.index.name = "date" if df.index.name is None else df.index.name
    long = df.reset_index().melt(id_vars=df.index.name, var_name=id_col, value_name=value_name)
    long = long.rename(columns={df.index.name: "date"})
    long["date"] = pd.to_datetime(long["date"])
    long[id_col] = long[id_col].astype(str)
    return long


def _us_federal_holidays(start: pd.Timestamp, end: pd.Timestamp) -> set[pd.Timestamp]:
    cal = USFederalHolidayCalendar()
    return set(pd.DatetimeIndex(cal.holidays(start=start, end=end)).normalize())


def build_features(
    actual_for_levels: pd.DataFrame,
    mu: pd.DataFrame,
    id_col: str = "unit_id",
) -> pd.DataFrame:
    """Build a long-format feature dataframe aligned with ``mu``.

    ``actual_for_levels`` is a history panel used only to compute per-unit
    log mean ridership. Pass *pre-policy* history (e.g. the val train history)
    so the level feature isn't contaminated by the policy.
    """
    long = _melt_panel(mu, value_name="mu_pred", id_col=id_col)

    # Per-unit log mean level
    levels = actual_for_levels.mean(axis=0)
    levels = np.log(levels.clip(lower=1.0))
    levels.index = levels.index.astype(str)
    levels.index.name = id_col
    long = long.merge(levels.rename("log_level").reset_index(), on=id_col, how="left")

    # DOW dummies (drop Monday as reference). Fixed category set so the
    # column schema is identical across val/test even if some DOWs are absent.
    dow_cats = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    long["_dow"] = pd.Categorical(long["date"].dt.day_name(), categories=dow_cats)
    dow = pd.get_dummies(long["_dow"], prefix="dow", dtype=float)
    dow = dow.drop(columns=[c for c in ["dow_Monday"] if c in dow.columns])
    long = pd.concat([long, dow], axis=1)

    # Month dummies (drop January as reference). Fixed 1..12 category set.
    long["_month"] = pd.Categorical(long["date"].dt.month, categories=list(range(1, 13)))
    months = pd.get_dummies(long["_month"], prefix="month", dtype=float)
    months = months.drop(columns=[c for c in ["month_1"] if c in months.columns])
    long = pd.concat([long, months], axis=1)

    # Federal holiday flag
    if len(long):
        hset = _us_federal_holidays(long["date"].min(), long["date"].max())
        long["is_holiday"] = long["date"].isin(hset).astype(float)
    else:
        long["is_holiday"] = 0.0

    # DOW × log_level interaction (lets weekly amplitude scale with unit size)
    for c in [c for c in long.columns if c.startswith("dow_")]:
        long[f"{c}_x_loglvl"] = long[c] * long["log_level"]

    return long.drop(columns=["_dow", "_month"])


@dataclass
class QuantileCalibration:
    feature_cols: list[str]
    models: dict[float, QuantileRegressor]
    median_q: float

    def predict_deltas(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        Xv = X[self.feature_cols].astype(float).to_numpy()
        return {q: m.predict(Xv) for q, m in self.models.items()}


def fit_calibration(
    val_features: pd.DataFrame,
    val_residuals: pd.Series,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    alpha: float = 1e-4,
) -> QuantileCalibration:
    """Fit one ``sklearn.QuantileRegressor`` per requested quantile."""
    feature_cols = [c for c in val_features.columns if c not in ID_DROP]
    X = val_features[feature_cols].astype(float).to_numpy()
    y = val_residuals.astype(float).to_numpy()
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    log.info("QR calibration: %d obs × %d features", len(y), X.shape[1])
    models: dict[float, QuantileRegressor] = {}
    for q in quantiles:
        m = QuantileRegressor(quantile=q, alpha=alpha, solver="highs", fit_intercept=True)
        m.fit(X, y)
        models[q] = m

    qs_sorted = sorted(quantiles)
    median_q = qs_sorted[len(qs_sorted) // 2]
    return QuantileCalibration(feature_cols=feature_cols, models=models, median_q=median_q)


def apply_calibration(
    cal: QuantileCalibration,
    test_features: pd.DataFrame,
    mu_test: pd.DataFrame,
    quantile_lo: float,
    quantile_hi: float,
    id_col: str = "unit_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the fitted calibration to test forecasts.

    Returns calibrated wide-format ``(mu_cal, lower_cal, upper_cal)`` aligned
    with ``mu_test``'s index/columns. Quantiles are sorted across q-levels to
    prevent crossings.
    """
    deltas = cal.predict_deltas(test_features)
    qs = sorted(deltas.keys())
    arr = np.stack([deltas[q] for q in qs], axis=0)
    arr = np.sort(arr, axis=0)
    deltas_sorted = {q: arr[i] for i, q in enumerate(qs)}

    out = test_features[["date", id_col, "mu_pred"]].copy()
    out["delta_lo"] = deltas_sorted[quantile_lo]
    out["delta_med"] = deltas_sorted[cal.median_q]
    out["delta_hi"] = deltas_sorted[quantile_hi]
    out["mu_cal"] = out["mu_pred"] + out["delta_med"]
    out["lower_cal"] = out["mu_pred"] + out["delta_lo"]
    out["upper_cal"] = out["mu_pred"] + out["delta_hi"]

    def _pivot(col: str) -> pd.DataFrame:
        wide = out.pivot(index="date", columns=id_col, values=col)
        return wide.reindex(mu_test.index).reindex(columns=mu_test.columns)

    return _pivot("mu_cal"), _pivot("lower_cal"), _pivot("upper_cal")


def residuals_long(actual: pd.DataFrame, mu: pd.DataFrame, id_col: str = "unit_id") -> pd.Series:
    """Compute long-format residuals ``actual - mu`` aligned to ``mu``."""
    common = actual.columns.intersection(mu.columns)
    a = actual[common].reindex(mu.index)
    m = mu[common]
    r = a - m
    long = _melt_panel(r, value_name="residual", id_col=id_col)
    return long.set_index(["date", id_col])["residual"]
