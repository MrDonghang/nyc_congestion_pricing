"""Spatial joins, CRZ classification, and figure helpers.

* :func:`classify_crz` — classify bus routes / census tracts / subway stations
  as inside / partially-inside / outside the Congestion Relief Zone polygon.
* :func:`map_units_to_tracts` — bring per-unit ATT (e.g. per bus route) into
  census-tract space via spatial join + averaging.
* :func:`plot_choropleth`, :func:`plot_significance_calendar`,
  :func:`plot_effects_over_time` — three figures used throughout the report.

All inputs are projected to ``EPSG:2263`` (NY State Plane, feet) for
area-correct geometry.
"""

from __future__ import annotations

import calendar
import math
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_CRS = "EPSG:2263"


# --------------------------------------------------------------------- CRZ ---


def classify_crz(gdf, crz_polygon, kind: Literal["routes", "tracts", "stations"], tract_share_threshold: float = 0.5):
    """Tag each row with its CRZ relationship.

    * ``routes``   — three classes via ``within`` / ``intersects``
    * ``tracts``   — binary by intersection-area share
    * ``stations`` — binary by ``within``
    """
    import geopandas as gpd  # noqa: F401

    gdf = gdf.copy()
    crz_geom = crz_polygon.geometry.unary_union if hasattr(crz_polygon, "geometry") else crz_polygon

    if kind == "routes":
        within = gdf.geometry.within(crz_geom)
        inter = gdf.geometry.intersects(crz_geom)
        gdf["crz_class"] = "fully_outside"
        gdf.loc[inter & ~within, "crz_class"] = "partially_inside"
        gdf.loc[within, "crz_class"] = "fully_inside"
    elif kind == "tracts":
        share = gdf.geometry.intersection(crz_geom).area / gdf.geometry.area
        gdf["crz_share"] = share
        gdf["crz_class"] = np.where(share > tract_share_threshold, "Inside CRZ", "Outside CRZ")
    elif kind == "stations":
        gdf["crz_class"] = np.where(gdf.geometry.within(crz_geom), "Inside CRZ", "Outside CRZ")
    else:
        raise ValueError(f"Unknown kind: {kind!r}")
    return gdf


# ---------------------------------------------------------- spatial joins ---


def map_units_to_tracts(
    units_gdf,
    tracts_gdf,
    effects_df: pd.DataFrame,
    join_col_units: str,
    join_col_effects: str,
    tract_id_col: str = "GEOID",
    metric_cols: Iterable[str] = ("att", "avg_daily", "signif_days", "signif_share", "cum_effect", "cum_cf"),
):
    """Attach unit-level ATT to a unit-with-geometry GDF, then average per tract.

    Parameters
    ----------
    units_gdf : GeoDataFrame
        E.g. bus stops or subway stations, with one geometry per unit.
    tracts_gdf : GeoDataFrame
        Census-tract polygons.
    effects_df : DataFrame
        Output of :func:`summarize_by_unit` — must have ``join_col_effects``.
    join_col_units : str
        Column in ``units_gdf`` that matches ``join_col_effects`` in ``effects_df``.

    Returns a GDF with tract-level mean of every metric in ``metric_cols`` plus
    ``cum_relative_effect = sum(cum_effect) / sum(cum_cf)``.
    """
    import geopandas as gpd

    units_gdf = units_gdf.to_crs(TARGET_CRS)
    tracts_gdf = tracts_gdf.to_crs(TARGET_CRS)

    units = units_gdf.merge(
        effects_df, left_on=join_col_units, right_on=join_col_effects, how="left"
    )

    joined = gpd.sjoin(
        units,
        tracts_gdf[[tract_id_col, "geometry"]],
        how="left",
        predicate="within",
    )
    tract_col = f"{tract_id_col}_right" if f"{tract_id_col}_right" in joined.columns else tract_id_col

    agg = joined.groupby(tract_col).agg({c: "mean" for c in metric_cols}).reset_index()
    if "cum_effect" in metric_cols and "cum_cf" in metric_cols:
        agg["cum_relative_effect"] = agg["cum_effect"] / agg["cum_cf"].replace(0, np.nan)

    out = tracts_gdf.merge(agg, left_on=tract_id_col, right_on=tract_col, how="left")
    return out


# ----------------------------------------------------------- visualisation ---


def plot_choropleth(
    gdf,
    column: str,
    crz_polygon=None,
    cmap: str = "Reds",
    legend_label: str | None = None,
    quantile_clip: tuple[float, float] = (0.02, 0.98),
    ax=None,
    figsize=(10, 10),
    dpi: int = 300,
):
    """Choropleth of ``column`` clipped to per-quantile vmin/vmax."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    gdf = gdf.to_crs(TARGET_CRS)
    if column not in gdf.columns:
        raise KeyError(f"Column {column!r} not in GDF.")

    missing = gdf[gdf[column].isna()]
    valid = gdf[~gdf[column].isna()]
    vals = valid[column].astype(float)
    vmin, vmax = np.nanpercentile(vals, [quantile_clip[0] * 100, quantile_clip[1] * 100])

    if not missing.empty:
        missing.plot(ax=ax, color="lightgrey", linewidth=0.4)
    valid.plot(ax=ax, column=column, cmap=cmap, linewidth=0.4, legend=True, vmin=vmin, vmax=vmax)

    if crz_polygon is not None:
        crz_polygon.to_crs(TARGET_CRS).boundary.plot(ax=ax, color="red", linewidth=2)

    ax.set_axis_off()
    if legend_label:
        cbar = fig.axes[-1]
        cbar.set_ylabel(legend_label, rotation=270, labelpad=18)
    return fig, ax


def plot_significance_calendar(
    df_daily: pd.DataFrame,
    year: int | None = None,
    start_month: int = 1,
    end_month: int = 12,
    week_starts_monday: bool = True,
    figsize=(14, 8),
    date_col: str = "date",
    signif_col: str = "signif_daily",
    tau_col: str = "mean_tau",
):
    """Calendar grid showing significant +ve / −ve / not-significant days."""
    d = df_daily[[date_col, signif_col, tau_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    if year is None:
        year = int(d[date_col].dt.year.mode().iloc[0])
    d = d[d[date_col].dt.year == year].copy()
    d["month"], d["day"] = d[date_col].dt.month, d[date_col].dt.day

    status: dict[tuple[int, int], int] = {}
    for m, day, sig, tau in zip(d["month"], d["day"], d[signif_col], d[tau_col]):
        if pd.isna(sig) or pd.isna(tau):
            continue
        status[(int(m), int(day))] = (1 if float(tau) > 0 else -1) if bool(sig) else 0

    cal = calendar.Calendar(firstweekday=0 if week_starts_monday else 6)
    day_labels = (
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if week_starts_monday
        else ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    )
    months = list(range(start_month, end_month + 1))
    n = len(months)
    nrows, ncols = (2, 2) if n == 4 else (math.ceil(n / 3), 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=300)
    axes = np.array(axes).reshape(-1)
    for ax in axes[n:]:
        ax.axis("off")

    palette = {1: ("black", "white"), -1: ("dimgray", "white"), 0: ("lightgray", "black")}

    for i, m in enumerate(months):
        ax = axes[i]
        ax.set_title(f"{calendar.month_name[m]} {year}", fontsize=14, pad=10)
        weeks = cal.monthdayscalendar(year, m)
        ax.set_xlim(0, 7)
        ax.set_ylim(0, len(weeks) + 1)
        ax.invert_yaxis()
        ax.axis("off")
        for j, lab in enumerate(day_labels):
            ax.text(j + 0.5, 0.5, lab, ha="center", va="center", fontsize=10)
        for w_idx, week in enumerate(weeks):
            for d_idx, day in enumerate(week):
                if day == 0:
                    ax.add_patch(patches.Rectangle((d_idx, w_idx + 1), 1, 1, facecolor="white", edgecolor="lightgray", linewidth=1))
                    continue
                if (m, day) in status:
                    fc, tc = palette[status[(m, day)]]
                else:
                    fc, tc = "whitesmoke", "black"
                ax.add_patch(patches.Rectangle((d_idx, w_idx + 1), 1, 1, facecolor=fc, edgecolor="white", linewidth=1.5))
                ax.text(d_idx + 0.5, w_idx + 1.5, str(day), ha="center", va="center", fontsize=10, color=tc)
        legend_handles = [
            patches.Patch(facecolor="black", label="Positive significant"),
            patches.Patch(facecolor="dimgray", label="Negative significant"),
            patches.Patch(facecolor="lightgray", label="Not significant"),
            patches.Patch(facecolor="whitesmoke", label="Missing"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8)

    fig.tight_layout()
    return fig, axes


def plot_effects_over_time(
    df_daily: pd.DataFrame,
    mode: Literal["daily_att", "cum_att", "cum_rel"] = "daily_att",
    title_prefix: str = "",
    ci_color: str = "lightgray",
    line_color: str = "black",
    figsize=(10, 5),
):
    """Line plot of daily / cumulative ATT (with PI shading)."""
    pickers = {
        "daily_att": ("mean_tau", "mean_eff_lo", "mean_eff_hi", "Daily ATT"),
        "cum_att": ("cum_tau", "cum_eff_lo", "cum_eff_hi", "Cumulative ATT"),
        "cum_rel": ("cum_rel_effect", "cum_rel_lo", "cum_rel_hi", "Cumulative relative ATT"),
    }
    if mode not in pickers:
        raise ValueError(f"mode must be one of {list(pickers)}")
    y, lo, hi, ylabel = pickers[mode]

    d = df_daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.fill_between(d["date"], d[lo], d[hi], color=ci_color, alpha=0.6, label="90% PI")
    ax.plot(d["date"], d[y], color=line_color, linewidth=2, label=ylabel)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix} {ylabel}".strip())
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax
