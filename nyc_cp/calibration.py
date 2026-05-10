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


# --------------------------------------------------------------------------
# Per-unit calibration (one QuantileRegressor set per unit, fall back to
# pooled when a unit has too few val rows).
# --------------------------------------------------------------------------

# Features that vary across units but not within a unit — useless when fitting
# per-unit (constant column ⇒ contributes only to the intercept).
PER_UNIT_DROP = {"log_level"}


def _per_unit_feature_cols(all_feature_cols: list[str]) -> list[str]:
    return [
        c for c in all_feature_cols
        if c not in ID_DROP
        and c not in PER_UNIT_DROP
        and not c.endswith("_x_loglvl")  # DOW × log_level interactions are also constant per unit
    ]


@dataclass
class PerUnitQuantileCalibration:
    """One ``QuantileCalibration`` per unit; ``fallback`` used for missing units."""

    feature_cols: list[str]
    per_unit: dict[str, QuantileCalibration]
    fallback: QuantileCalibration   # pooled cal, used when a unit has too few obs
    fallback_feature_cols: list[str]
    median_q: float


def fit_per_unit_calibration(
    val_features: pd.DataFrame,
    val_residuals: pd.Series,
    fallback: QuantileCalibration,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    alpha: float = 1e-4,
    min_obs: int = 60,
    id_col: str = "unit_id",
) -> PerUnitQuantileCalibration:
    """Fit a separate ``QuantileRegressor`` set per ``unit_id``.

    ``val_features`` is the long-format feature panel (same as for pooled).
    ``val_residuals`` is a Series indexed by ``(date, unit_id)``. Units with
    fewer than ``min_obs`` rows are skipped — at apply time we fall back to
    the pooled ``fallback`` calibration for those.
    """
    feature_cols = _per_unit_feature_cols(list(val_features.columns))

    # Align features ↔ residuals on (date, unit_id)
    feats = val_features.set_index(["date", id_col])
    common_idx = feats.index.intersection(val_residuals.index)
    feats = feats.loc[common_idx]
    res = val_residuals.loc[common_idx]

    units = feats.index.get_level_values(id_col).unique()
    log.info("Per-unit QR calibration over %d units (min_obs=%d, %d features)",
             len(units), min_obs, len(feature_cols))

    qs_sorted = sorted(quantiles)
    median_q = qs_sorted[len(qs_sorted) // 2]
    per_unit: dict[str, QuantileCalibration] = {}
    n_skip = 0

    for unit in units:
        Xu = feats.xs(unit, level=id_col)[feature_cols].astype(float).to_numpy()
        yu = res.xs(unit, level=id_col).astype(float).to_numpy()
        mask = np.isfinite(Xu).all(axis=1) & np.isfinite(yu)
        Xu, yu = Xu[mask], yu[mask]
        if len(yu) < min_obs:
            n_skip += 1
            continue

        try:
            models = {}
            for q in quantiles:
                m = QuantileRegressor(quantile=q, alpha=alpha, solver="highs", fit_intercept=True)
                m.fit(Xu, yu)
                models[q] = m
            per_unit[str(unit)] = QuantileCalibration(
                feature_cols=feature_cols, models=models, median_q=median_q,
            )
        except Exception as e:  # pragma: no cover — solver may fail on degenerate units
            log.warning("per-unit QR failed for %s: %s — falling back to pooled", unit, e)
            n_skip += 1

    log.info("Per-unit fitted: %d  fallback: %d / %d", len(per_unit), n_skip, len(units))
    return PerUnitQuantileCalibration(
        feature_cols=feature_cols,
        per_unit=per_unit,
        fallback=fallback,
        fallback_feature_cols=fallback.feature_cols,
        median_q=median_q,
    )


def predict_per_unit_deltas(
    cal: PerUnitQuantileCalibration,
    features: pd.DataFrame,
    id_col: str = "unit_id",
) -> dict[float, np.ndarray]:
    """Predict per-quantile residual deltas for ``features``, falling back to
    pooled ``cal.fallback`` for units not in ``cal.per_unit``.

    Returns ``{q: array of length N}`` aligned to ``features`` rows. Output is
    NOT row-sorted across quantiles — the caller should ``np.sort`` to enforce
    non-crossing if needed.
    """
    n = len(features)
    units = features[id_col].astype(str).to_numpy()
    Xu_per = features[cal.feature_cols].astype(float).to_numpy()
    Xu_fb = features[cal.fallback_feature_cols].astype(float).to_numpy()
    qs = sorted(cal.fallback.models.keys())
    out = {q: np.full(n, np.nan) for q in qs}
    n_fb = 0
    for unit in np.unique(units):
        mask = units == unit
        if unit in cal.per_unit:
            cu = cal.per_unit[unit]
            for q in qs:
                out[q][mask] = cu.models[q].predict(Xu_per[mask])
        else:
            n_fb += 1
            for q in qs:
                out[q][mask] = cal.fallback.models[q].predict(Xu_fb[mask])
    log.info("Per-unit predict: %d units, %d fallback to pooled", len(np.unique(units)), n_fb)
    return out


def apply_per_unit_calibration(
    cal: PerUnitQuantileCalibration,
    test_features: pd.DataFrame,
    mu_test: pd.DataFrame,
    quantile_lo: float,
    quantile_hi: float,
    id_col: str = "unit_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply per-unit calibration; fall back to ``cal.fallback`` for missing units."""
    out = test_features[["date", id_col, "mu_pred"]].copy()

    deltas = predict_per_unit_deltas(cal, test_features, id_col=id_col)
    qs = sorted(deltas.keys())
    arr = np.sort(np.stack([deltas[q] for q in qs], axis=0), axis=0)
    delta_lo = arr[qs.index(quantile_lo)]
    delta_med = arr[qs.index(cal.median_q)]
    delta_hi = arr[qs.index(quantile_hi)]

    out["mu_cal"] = out["mu_pred"] + delta_med
    out["lower_cal"] = out["mu_pred"] + delta_lo
    out["upper_cal"] = out["mu_pred"] + delta_hi

    def _pivot(col: str) -> pd.DataFrame:
        wide = out.pivot(index="date", columns=id_col, values=col)
        return wide.reindex(mu_test.index).reindex(columns=mu_test.columns)

    return _pivot("mu_cal"), _pivot("lower_cal"), _pivot("upper_cal")


# --------------------------------------------------------------------------
# Per-unit intercept + pooled QR (cheap intermediate between pooled and
# fully per-unit). Per unit: 1 free param (mean residual). Pooled QR: same
# design as global qrcal, fit on de-biased residuals so it learns PI shape
# only.
# --------------------------------------------------------------------------


@dataclass
class InterceptPlusPooledCalibration:
    intercepts: dict[str, float]
    pooled: QuantileCalibration            # fit on residuals AFTER subtracting per-unit intercept
    fallback_intercept: float = 0.0


def fit_intercept_plus_pooled_calibration(
    val_features: pd.DataFrame,
    val_residuals: pd.Series,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    alpha: float = 1e-4,
    id_col: str = "unit_id",
) -> InterceptPlusPooledCalibration:
    """Step 1: per-unit mean residual = level shift correction.
    Step 2: pooled QR on de-biased residuals = PI shape.

    ``val_residuals`` is a Series indexed by ``(date, unit_id)``.
    ``val_features`` is the long-format feature panel (same as for pooled).
    """
    feats = val_features.set_index(["date", id_col])
    common_idx = feats.index.intersection(val_residuals.index)
    feats = feats.loc[common_idx]
    res = val_residuals.loc[common_idx]

    intercepts = res.dropna().groupby(level=id_col).mean()
    intercepts.index = intercepts.index.astype(str)
    intercepts_dict = intercepts.to_dict()

    res_debiased = res - res.index.get_level_values(id_col).astype(str).map(intercepts_dict)
    pooled = fit_calibration(feats.reset_index(), res_debiased, quantiles=quantiles, alpha=alpha)

    log.info("Intercept+pooled cal: %d unit intercepts (mean=%+.1f, std=%.1f), pooled QR on %d obs",
             len(intercepts_dict), float(intercepts.mean()), float(intercepts.std()), len(res_debiased))
    return InterceptPlusPooledCalibration(intercepts=intercepts_dict, pooled=pooled)


def predict_intercept_plus_pooled_deltas(
    cal: InterceptPlusPooledCalibration,
    features: pd.DataFrame,
    id_col: str = "unit_id",
) -> dict[float, np.ndarray]:
    """Predict per-quantile deltas = per-unit intercept + pooled QR delta.

    Output ``{q: array length N}`` matches the API of
    ``QuantileCalibration.predict_deltas`` so the k-fold loop can use it
    interchangeably. NOT row-sorted across quantiles — caller sorts.
    """
    units = features[id_col].astype(str).to_numpy()
    intercepts_arr = np.array([cal.intercepts.get(u, cal.fallback_intercept) for u in units])
    pooled_deltas = cal.pooled.predict_deltas(features)
    return {q: intercepts_arr + pooled_deltas[q] for q in pooled_deltas}


def apply_intercept_plus_pooled_calibration(
    cal: InterceptPlusPooledCalibration,
    test_features: pd.DataFrame,
    mu_test: pd.DataFrame,
    quantile_lo: float,
    quantile_hi: float,
    id_col: str = "unit_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = test_features[["date", id_col, "mu_pred"]].copy()

    deltas = predict_intercept_plus_pooled_deltas(cal, test_features, id_col=id_col)
    qs = sorted(deltas.keys())
    arr = np.sort(np.stack([deltas[q] for q in qs], axis=0), axis=0)
    delta_lo = arr[qs.index(quantile_lo)]
    delta_med = arr[qs.index(cal.pooled.median_q)]
    delta_hi = arr[qs.index(quantile_hi)]

    out["mu_cal"] = out["mu_pred"] + delta_med
    out["lower_cal"] = out["mu_pred"] + delta_lo
    out["upper_cal"] = out["mu_pred"] + delta_hi

    def _pivot(col: str) -> pd.DataFrame:
        wide = out.pivot(index="date", columns=id_col, values=col)
        return wide.reindex(mu_test.index).reindex(columns=mu_test.columns)

    return _pivot("mu_cal"), _pivot("lower_cal"), _pivot("upper_cal")
