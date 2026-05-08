"""Counterfactual treatment-effect computation.

Given an *actual* series and a *counterfactual* forecast (mean + lower / upper
prediction interval), compute per-unit per-day effects, daily / cumulative
aggregates, and per-unit summaries — all parameterised by ``id_col`` so the
same code works for bus routes, subway stations, and citibike tracts.

Sign convention
---------------
``tau = actual - cf_mean``. When the policy *increases* the outcome (e.g.
post-pricing transit demand goes up), tau is positive.

A unit-day is "significant" when the actual lies outside the counterfactual PI:
``signif = (actual > cf_upper) | (actual < cf_lower)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nyc_cp.models.base import ForecastResult


# ---------------------------------------------------------------------- I/O ---


def load_forecast_triplet(output_dir: Path, prefix: str) -> ForecastResult:
    """Convenience wrapper around :meth:`ForecastResult.load`."""
    return ForecastResult.load(Path(output_dir), prefix)


def _melt(df: pd.DataFrame, value_name: str, id_col: str, columns: list | None = None) -> pd.DataFrame:
    """Wide → long, with negatives clipped to zero (forecasts can dip below 0)."""
    df = df.copy()
    df.index.name = "date" if df.index.name is None else df.index.name
    df = df.reset_index()
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])

    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].clip(lower=0)

    if columns is not None:
        cols = [str(c) for c in columns if str(c) in df.columns]
        df = df[[date_col, *cols]]

    long = df.melt(id_vars=[date_col], var_name=id_col, value_name=value_name)
    long = long.rename(columns={date_col: "date"})
    long[id_col] = long[id_col].astype(str)
    return long


# --------------------------------------------------------------------- core ---


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


def compute_effects(df_long: pd.DataFrame, id_col: str = "unit_id") -> pd.DataFrame:
    """Add per-day effect columns to a long-format DataFrame.

    New columns:
      * ``tau``                  — point estimate (``actual - cf_mean``)
      * ``eff_lo`` / ``eff_hi``  — lower / upper effect bounds (``actual - cf_upper`` and ``actual - cf_lower``)
      * ``signif``               — bool: PI excludes zero
      * ``direction``            — "positive" / "negative" / "ns"
      * ``cum_effect``           — running ``tau`` per unit
      * ``cum_relative_effect``  — running ``tau / cf_mean`` per unit
    """
    df = df_long.copy()
    df["tau"] = df["actual"] - df["cf_mean"]
    df["eff_lo"] = df["actual"] - df["cf_upper"]
    df["eff_hi"] = df["actual"] - df["cf_lower"]
    df["signif"] = (df["eff_lo"] > 0) | (df["eff_hi"] < 0)

    df["direction"] = np.where(df["tau"] > 0, "positive", "negative")
    df.loc[~df["signif"], "direction"] = "ns"

    df["cum_effect"] = df.groupby(id_col)["tau"].cumsum()
    df["cum_relative_effect"] = (
        df.groupby(id_col)["tau"].cumsum() / df.groupby(id_col)["cf_mean"].cumsum()
    )
    return df


# -------------------------------------------------------------- summaries ---


def summarize_by_unit(df_eff: pd.DataFrame, id_col: str = "unit_id") -> pd.DataFrame:
    """One row per unit: total ATT, average daily effect, signif share, cumulative."""
    last_day = df_eff.groupby(id_col).tail(1)[[id_col, "cum_effect", "cum_relative_effect"]]
    cum_cf = df_eff.groupby(id_col, as_index=False).agg(cum_cf=("cf_mean", "sum"))
    summary = df_eff.groupby(id_col, as_index=False).agg(
        n_days=("date", "count"),
        att=("tau", "sum"),
        avg_daily=("tau", "mean"),
        signif_days=("signif", "sum"),
    )
    summary["signif_share"] = summary["signif_days"] / summary["n_days"]
    return summary.merge(cum_cf, on=id_col).merge(last_day, on=id_col)


def summarize_over_time(df_eff: pd.DataFrame) -> pd.DataFrame:
    """One row per date: cross-unit means + cumulative effect with PI."""
    daily = df_eff.groupby("date", as_index=False).agg(
        mean_tau=("tau", "mean"),
        mean_eff_lo=("eff_lo", "mean"),
        mean_eff_hi=("eff_hi", "mean"),
        mean_cf=("cf_mean", "mean"),
    )
    daily["cum_tau"] = daily["mean_tau"].cumsum()
    daily["cum_eff_lo"] = daily["mean_eff_lo"].cumsum()
    daily["cum_eff_hi"] = daily["mean_eff_hi"].cumsum()
    cum_cf = daily["mean_cf"].cumsum()
    daily["cum_rel_effect"] = daily["cum_tau"] / cum_cf
    daily["cum_rel_lo"] = daily["cum_eff_lo"] / cum_cf
    daily["cum_rel_hi"] = daily["cum_eff_hi"] / cum_cf
    daily["signif_daily"] = (daily["mean_eff_lo"] > 0) | (daily["mean_eff_hi"] < 0)
    daily["signif_cum"] = (daily["cum_eff_lo"] > 0) | (daily["cum_eff_hi"] < 0)
    return daily


def summarize_overall(unit_summary: pd.DataFrame, id_col: str = "unit_id") -> pd.DataFrame:
    """One-row global summary (sum/mean across units)."""
    total_cum = float(unit_summary["cum_effect"].sum())
    total_cum_cf = float(unit_summary["cum_cf"].sum())
    return pd.DataFrame(
        {
            "n_units": [unit_summary[id_col].nunique()],
            "total_att": [unit_summary["att"].sum()],
            "avg_daily_all": [unit_summary["avg_daily"].mean()],
            "total_signif_days": [unit_summary["signif_days"].sum()],
            "avg_signif_share": [unit_summary["signif_share"].mean()],
            "total_cum_effect": [total_cum],
            "total_cum_cf": [total_cum_cf],
            "total_cum_relative_effect": [np.nan if total_cum_cf == 0 else total_cum / total_cum_cf],
        }
    )
