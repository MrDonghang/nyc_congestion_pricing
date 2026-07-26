"""Counterfactual treatment-effect computation.

Given an *actual* series and a *counterfactual* forecast (mean + lower / upper
prediction interval), compute per-unit per-day effects, daily / cumulative
aggregates, and per-unit summaries — all parameterised by ``id_col`` so the
same code works for bus routes, subway stations, and census tracts.

Sign convention
---------------
``tau = actual - cf_mean``. When the policy *increases* the outcome (e.g.
post-pricing transit demand goes up), tau is positive.

A unit-day is "significant" when the actual lies outside the counterfactual PI:
``signif = (actual > cf_upper) | (actual < cf_lower)``.

Uncertainty propagation
-----------------------
The forecaster gives us a per-day, per-unit Gaussian PI ``[cf_lower, cf_upper]``
at coverage ``c``. We treat that PI as encoding ``cf ~ N(mu, sigma^2)`` with
``sigma = (cf_upper - cf_lower) / (2 * z(c))`` (symmetric Gaussian assumption —
correct for ARIMA / PCN analytic; an approximation for sample-based PIs).
Cumulative effects are then propagated via the **variance**, not by summing
the PI bounds directly:

* per-unit cumulative effect:    ``cum_tau ± z * sqrt(sum_t sigma_t^2)``
* cross-unit mean (one day):     ``mean_tau ± z * sqrt(sum_u sigma_ut^2 / N^2)``
* cross-unit cumulative mean:    ``cum_mean_tau ± z * sqrt(sum_t var(mean_tau_t))``

This avoids the ``O(T)`` over-coverage that comes from cumsumming per-day PI
bounds (which would give ``O(T)`` width instead of the correct ``O(sqrt(T))``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from nyc_cp.models.base import ForecastResult


# --------------------------------------------------------------- helpers ---


def _z(coverage_level: float) -> float:
    """Two-sided z value for a Gaussian PI of given coverage."""
    return float(norm.ppf((1 + coverage_level) / 2))


def _recover_sigma(lower: pd.Series, upper: pd.Series, coverage_level: float) -> pd.Series:
    """Recover Gaussian sigma from a symmetric PI: ``sigma = (upper - lower) / (2z)``.

    Numerical noise can make ``upper < lower`` by epsilon; clamp to non-negative.
    """
    z = _z(coverage_level)
    return ((upper - lower) / (2 * z)).clip(lower=0)


def load_forecast_triplet(output_dir: Path, prefix: str) -> ForecastResult:
    """Convenience wrapper around :meth:`ForecastResult.load`."""
    return ForecastResult.load(Path(output_dir), prefix)


# ------------------------------------------------------------------ I/O ---


def _melt(df: pd.DataFrame, value_name: str, id_col: str, columns: list | None = None) -> pd.DataFrame:
    """Wide → long.

    Note: previous versions clipped *all* numeric columns at 0 here, including
    ``cf_lower``. That artificially compressed the PI from below and biased the
    upper bound of the effect. We now keep the bounds untouched; if the user
    wants non-negative ``mu`` it should happen at the forecaster (e.g.
    truncated-Gaussian head).
    """
    df = df.copy()
    df.index.name = "date" if df.index.name is None else df.index.name
    df = df.reset_index()
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])

    if columns is not None:
        cols = [str(c) for c in columns if str(c) in df.columns]
        df = df[[date_col, *cols]]

    long = df.melt(id_vars=[date_col], var_name=id_col, value_name=value_name)
    long = long.rename(columns={date_col: "date"})
    long[id_col] = long[id_col].astype(str)
    return long


def build_long_df(
    actual: pd.DataFrame,
    mu: pd.DataFrame,
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    id_col: str = "unit_id",
    columns: list | None = None,
) -> pd.DataFrame:
    """Combine actual + counterfactual into one long-format DataFrame.

    Output columns: ``[id_col, date, actual, cf_mean, cf_lower, cf_upper]``.
    """
    a = _melt(actual, "actual", id_col=id_col, columns=columns)
    m = _melt(mu, "cf_mean", id_col=id_col, columns=columns)
    lo = _melt(lower, "cf_lower", id_col=id_col, columns=columns)
    hi = _melt(upper, "cf_upper", id_col=id_col, columns=columns)
    key = [id_col, "date"]
    return (
        a.merge(m, on=key, how="inner")
        .merge(lo, on=key, how="inner")
        .merge(hi, on=key, how="inner")
        .sort_values(key)
        .reset_index(drop=True)
    )


# ------------------------------------------------------------------ core ---


def compute_effects(df_long: pd.DataFrame, id_col: str = "unit_id", coverage_level: float = 0.9) -> pd.DataFrame:
    """Add per-day and per-unit-cumulative effect columns.

    New columns:
      Per-day:
        * ``tau``                 — point estimate (``actual - cf_mean``)
        * ``eff_lo`` / ``eff_hi`` — per-day effect PI (``actual - cf_upper`` / ``- cf_lower``)
        * ``signif``              — bool: per-day PI excludes zero
        * ``direction``           — "positive" / "negative" / "ns"
        * ``cf_sigma``            — recovered Gaussian sigma of the per-day forecast

      Per-unit cumulative (Gaussian variance propagation, NOT bound-cumsum):
        * ``cum_effect``                    — running ``tau`` per unit
        * ``cum_var`` / ``cum_se``          — running variance / SE of cum_effect
        * ``cum_effect_lo`` / ``cum_effect_hi`` — Gaussian PI on cum_effect
        * ``cum_signif``                    — bool: cumulative PI excludes zero
        * ``cum_relative_effect``           — ``cum_effect / cum_cf``
        * ``cum_rel_lo`` / ``cum_rel_hi``   — relative PI bounds
    """
    z = _z(coverage_level)
    df = df_long.copy()
    df = df.sort_values([id_col, "date"]).reset_index(drop=True)

    # Per-day point estimate and PI for the effect.
    df["tau"] = df["actual"] - df["cf_mean"]
    df["eff_lo"] = df["actual"] - df["cf_upper"]
    df["eff_hi"] = df["actual"] - df["cf_lower"]
    df["signif"] = (df["eff_lo"] > 0) | (df["eff_hi"] < 0)
    df["direction"] = np.where(df["tau"] > 0, "positive", "negative")
    df.loc[~df["signif"], "direction"] = "ns"

    # Recover per-day Gaussian sigma of the forecast.
    df["cf_sigma"] = _recover_sigma(df["cf_lower"], df["cf_upper"], coverage_level)

    # Per-unit cumulative tau and variance-based PI.
    df["cum_effect"] = df.groupby(id_col)["tau"].cumsum()
    df["cum_var"] = df.groupby(id_col)["cf_sigma"].transform(lambda s: (s ** 2).cumsum())
    df["cum_se"] = np.sqrt(df["cum_var"])
    df["cum_effect_lo"] = df["cum_effect"] - z * df["cum_se"]
    df["cum_effect_hi"] = df["cum_effect"] + z * df["cum_se"]
    df["cum_signif"] = (df["cum_effect_lo"] > 0) | (df["cum_effect_hi"] < 0)

    df["cum_cf"] = df.groupby(id_col)["cf_mean"].cumsum()
    safe_cf = df["cum_cf"].where(df["cum_cf"] > 0)
    df["cum_relative_effect"] = df["cum_effect"] / safe_cf
    df["cum_rel_lo"] = df["cum_effect_lo"] / safe_cf
    df["cum_rel_hi"] = df["cum_effect_hi"] / safe_cf

    return df


# -------------------------------------------------------------- summaries ---


def summarize_by_unit(df_eff: pd.DataFrame, id_col: str = "unit_id", coverage_level: float = 0.9) -> pd.DataFrame:
    """One row per unit: total ATT (with Gaussian SE / CI), signif share + excess."""
    z = _z(coverage_level)

    # Aggregate per unit: total ATT, mean daily, count, etc.
    summary = df_eff.groupby(id_col, as_index=False).agg(
        n_days=("date", "count"),
        att=("tau", "sum"),
        avg_daily=("tau", "mean"),
        signif_days=("signif", "sum"),
        att_var=("cf_sigma", lambda s: float((s ** 2).sum())),
    )
    summary["att_se"] = np.sqrt(summary["att_var"])
    summary["att_lo"] = summary["att"] - z * summary["att_se"]
    summary["att_hi"] = summary["att"] + z * summary["att_se"]
    summary["att_signif"] = (summary["att_lo"] > 0) | (summary["att_hi"] < 0)

    summary["signif_share"] = summary["signif_days"] / summary["n_days"]
    summary["signif_share_excess"] = summary["signif_share"] - (1 - coverage_level)

    # Pull cumulative (last-day) values from compute_effects' output.
    last = (
        df_eff.sort_values([id_col, "date"]).groupby(id_col).tail(1)[
            [id_col, "cum_effect", "cum_relative_effect", "cum_effect_lo", "cum_effect_hi", "cum_signif"]
        ]
        .rename(columns={
            "cum_effect_lo": "cum_effect_ci_lo",
            "cum_effect_hi": "cum_effect_ci_hi",
            "cum_signif": "cum_signif_at_end",
        })
    )
    cum_cf = df_eff.groupby(id_col, as_index=False).agg(cum_cf=("cf_mean", "sum"))
    return summary.merge(cum_cf, on=id_col).merge(last, on=id_col)


def summarize_over_time(df_eff: pd.DataFrame, coverage_level: float = 0.9) -> pd.DataFrame:
    """One row per date: cross-unit mean tau with proper variance-based PI.

    The per-day cross-unit mean variance is ``(1/N^2) * sum_u sigma_ut^2``, not
    the mean of per-unit PI bounds. Cumulative variance is the cumsum of those
    daily variances (assumes day-to-day independence).
    """
    z = _z(coverage_level)

    grouped = df_eff.groupby("date")
    n_per_day = grouped.size().rename("n_units")
    mean_tau = grouped["tau"].mean().rename("mean_tau")
    sum_var = grouped["cf_sigma"].apply(lambda s: float((s ** 2).sum())).rename("sum_var")
    mean_cf = grouped["cf_mean"].mean().rename("mean_cf")

    daily = (
        pd.concat([n_per_day, mean_tau, sum_var, mean_cf], axis=1)
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Variance of the cross-unit mean tau on day t (independence across units).
    daily["var_mean_tau"] = daily["sum_var"] / (daily["n_units"] ** 2)
    daily["se_mean_tau"] = np.sqrt(daily["var_mean_tau"])
    daily["mean_eff_lo"] = daily["mean_tau"] - z * daily["se_mean_tau"]
    daily["mean_eff_hi"] = daily["mean_tau"] + z * daily["se_mean_tau"]

    # Cumulative aggregates (variance cumsum, not bound cumsum).
    daily["cum_tau"] = daily["mean_tau"].cumsum()
    cum_var = daily["var_mean_tau"].cumsum()
    cum_se = np.sqrt(cum_var)
    daily["cum_eff_lo"] = daily["cum_tau"] - z * cum_se
    daily["cum_eff_hi"] = daily["cum_tau"] + z * cum_se

    cum_cf = daily["mean_cf"].cumsum()
    safe_cum_cf = cum_cf.where(cum_cf > 0)
    daily["cum_rel_effect"] = daily["cum_tau"] / safe_cum_cf
    daily["cum_rel_lo"] = daily["cum_eff_lo"] / safe_cum_cf
    daily["cum_rel_hi"] = daily["cum_eff_hi"] / safe_cum_cf

    daily["signif_daily"] = (daily["mean_eff_lo"] > 0) | (daily["mean_eff_hi"] < 0)
    daily["signif_cum"] = (daily["cum_eff_lo"] > 0) | (daily["cum_eff_hi"] < 0)

    # Drop intermediate columns to keep the CSV small.
    return daily.drop(columns=["sum_var"])


def summarize_overall(unit_summary: pd.DataFrame, id_col: str = "unit_id", coverage_level: float = 0.9) -> pd.DataFrame:
    """One-row global summary, including a Gaussian SE on total ATT.

    The total ATT variance assumes independence across units; with shared
    city-wide shocks it understates uncertainty. Treat as a lower bound on SE.
    """
    z = _z(coverage_level)
    n_units = int(unit_summary[id_col].nunique())
    total_att = float(unit_summary["att"].sum())
    total_att_var = float(unit_summary["att_var"].sum())
    total_att_se = float(np.sqrt(total_att_var))
    total_att_lo = total_att - z * total_att_se
    total_att_hi = total_att + z * total_att_se

    total_cum = float(unit_summary["cum_effect"].sum())
    total_cum_cf = float(unit_summary["cum_cf"].sum())

    return pd.DataFrame(
        {
            "n_units": [n_units],
            "coverage_level": [coverage_level],
            "total_att": [total_att],
            "total_att_se": [total_att_se],
            "total_att_lo": [total_att_lo],
            "total_att_hi": [total_att_hi],
            "total_att_signif": [bool((total_att_lo > 0) or (total_att_hi < 0))],
            "avg_daily_all": [float(unit_summary["avg_daily"].mean())],
            "total_signif_days": [int(unit_summary["signif_days"].sum())],
            "avg_signif_share": [float(unit_summary["signif_share"].mean())],
            "avg_signif_share_excess": [float(unit_summary["signif_share_excess"].mean())],
            "total_cum_effect": [total_cum],
            "total_cum_cf": [total_cum_cf],
            "total_cum_relative_effect": [np.nan if total_cum_cf == 0 else total_cum / total_cum_cf],
        }
    )
