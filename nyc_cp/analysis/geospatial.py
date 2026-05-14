"""Spatial joins, CRZ classification, and figure helpers.

* :func:`classify_crz` — classify bus routes / census tracts / subway stations
  as inside / partially-inside / outside the Congestion Relief Zone polygon.
* :func:`map_units_to_tracts` — bring per-unit ATT (e.g. per bus route) into
  census-tract space via spatial join + averaging.
* :func:`plot_choropleth`, :func:`plot_unit_effects`,
  :func:`plot_significance_calendar`, :func:`plot_effects_over_time` —
  figures used throughout the report. ``plot_choropleth`` aggregates to
  tract polygons, ``plot_unit_effects`` colours raw units (bus polylines /
  subway points) by per-unit ATT — useful when tract aggregation hides
  inter-unit variation.

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


def _add_basemap(ax, extent_gdf=None) -> None:
    """Underlay a CartoDB Positron tile basemap. No-op if ``contextily`` missing.

    Called **before** data layers are plotted (so data sits on top). Pass the
    GDF whose bounds the map should cover via ``extent_gdf`` — required because
    ax limits are still defaults before any artist is drawn. Also makes the
    axes patch transparent so the basemap isn't hidden by mpl's default white
    axes background.
    """
    try:
        import contextily as cx
    except ImportError:
        return
    if extent_gdf is not None and len(extent_gdf):
        xmin, ymin, xmax, ymax = extent_gdf.to_crs(TARGET_CRS).total_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_facecolor("none")
    cx.add_basemap(
        ax,
        crs=TARGET_CRS,
        source=cx.providers.CartoDB.Positron,
        attribution=False,
    )


# --------------------------------------------------------------------- CRZ ---


def classify_crz(gdf, crz_polygon, kind: Literal["routes", "tracts", "stations"], tract_share_threshold: float = 0.5):
    """Tag each row with its CRZ relationship.

    * ``routes``   — three classes (``fully_inside`` / ``partially_inside`` /
                     ``fully_outside``) — keeps the partial-crossing routes as
                     a distinct group because their behaviour (small/negative
                     ATT) differs materially from fully-inside (~+11%) and
                     fully-outside (~+8%) routes.
    * ``tracts``   — binary by intersection-area share.
    * ``stations`` — binary by ``within``.
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
    puma_polygon=None,
    cmap: str = "Reds",
    legend_label: str | None = None,
    quantile_clip: tuple[float, float] = (0.02, 0.98),
    diverging: bool | None = None,
    center: float = 0.0,
    basemap: bool = False,
    ax=None,
    figsize=(10, 10),
    dpi: int = 300,
):
    """Choropleth of ``column`` clipped to per-quantile vmin/vmax.

    For diverging quantities (signed ATT), pass ``diverging=True`` so the
    colormap is normalised with ``TwoSlopeNorm(vcenter=center)`` — that way
    ``center`` (default 0) is always the white midpoint regardless of how
    skewed the value distribution is. Without this, a heavy-negative tail
    pushes the white midpoint deep into the negative range and most slightly-
    negative tracts appear in the *positive* (red) half of the colour bar.
    Auto-detected when ``diverging is None`` and any value crosses ``center``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    gdf = gdf.to_crs(TARGET_CRS)
    if column not in gdf.columns:
        raise KeyError(f"Column {column!r} not in GDF.")

    missing = gdf[gdf[column].isna()]
    valid = gdf[~gdf[column].isna()]
    if valid.empty:
        # All NaN — usually a broken upstream join. Render only the missing
        # polygons so the user can see the geometry without crashing.
        if not missing.empty:
            missing.plot(ax=ax, color="white", edgecolor="dimgrey", linewidth=0.25)
        if crz_polygon is not None:
            crz_polygon.to_crs(TARGET_CRS).boundary.plot(ax=ax, color="red", linewidth=2)
        ax.set_axis_off()
        ax.set_title(f"{column} — all values missing", fontsize=11, color="red")
        return fig, ax
    vals = valid[column].astype(float)
    pct = np.nanpercentile(vals, [quantile_clip[0] * 100, quantile_clip[1] * 100])
    vmin, vmax = float(pct[0]), float(pct[1])

    if diverging is None:
        diverging = bool((vals.min() < center) and (vals.max() > center))

    plot_kwargs = dict(
        column=column, cmap=cmap,
        edgecolor="dimgrey", linewidth=0.25,
        legend=True, legend_kwds={"shrink": 0.5, "aspect": 30},
    )
    if diverging:
        from matplotlib.colors import TwoSlopeNorm

        # Symmetric extent around ``center`` so red/blue intensity is comparable.
        half = max(abs(vmin - center), abs(vmax - center))
        if half == 0:
            half = 1.0  # degenerate; avoid TwoSlopeNorm raising on equal bounds
        plot_kwargs["norm"] = TwoSlopeNorm(vcenter=center, vmin=center - half, vmax=center + half)
    else:
        plot_kwargs["vmin"], plot_kwargs["vmax"] = vmin, vmax

    # Basemap drawn FIRST so data layers sit on top.
    if basemap:
        _add_basemap(ax, extent_gdf=gdf)

    if not missing.empty:
        missing_face = "none" if basemap else "white"
        missing.plot(ax=ax, facecolor=missing_face, edgecolor="dimgrey", linewidth=0.25)
    valid.plot(ax=ax, **plot_kwargs)

    if puma_polygon is not None:
        puma_polygon.to_crs(TARGET_CRS).plot(
            ax=ax, facecolor="none", edgecolor="black", linewidth=0.9
        )

    if crz_polygon is not None:
        crz_polygon.to_crs(TARGET_CRS).boundary.plot(ax=ax, color="red", linewidth=2)

    ax.set_axis_off()
    if legend_label:
        cbar = fig.axes[-1]
        cbar.set_ylabel(legend_label, rotation=270, labelpad=18)
    return fig, ax


def plot_unit_effects(
    units_gdf,
    effects_df: pd.DataFrame,
    join_col: str,
    value_col: str = "att",
    crz_polygon=None,
    puma_polygon=None,
    base_layer=None,
    cmap: str = "RdBu_r",
    legend_label: str | None = None,
    quantile_clip: tuple[float, float] = (0.02, 0.98),
    diverging: bool | None = None,
    center: float = 0.0,
    point_size: float = 30.0,
    point_size_by_abs: bool = False,
    line_width: float = 2.0,
    basemap: bool = False,
    ax=None,
    figsize=(11, 11),
    dpi: int = 300,
):
    """Map per-unit ATT directly onto unit geometries (bus polylines / subway points).

    Avoids the tract-aggregation step in :func:`plot_choropleth`, so each
    individual route or station is visible. Use ``base_layer`` to pass a
    grey background (e.g. census tract polygons or borough boundaries).

    Parameters
    ----------
    units_gdf : GeoDataFrame
        Geometry per unit (rows may repeat per direction; merge handles it).
    effects_df : DataFrame
        Per-unit metrics, e.g. the ``*_unit.csv`` written by ``compute_effects``.
    join_col : str
        Column name shared by both frames (``route_id`` or ``station_id``).
    value_col : str
        Column in ``effects_df`` to colour by (default ``"att"``).
    diverging : bool or None
        ``True`` → use ``TwoSlopeNorm(vcenter=center)``. Auto-detect when None.
    point_size_by_abs : bool
        Subway-only convenience: scale Point markers by ``|value|`` so big
        effects look big. Has no effect on line geometries.
    """
    from matplotlib.colors import TwoSlopeNorm

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    if value_col not in effects_df.columns:
        raise KeyError(f"Column {value_col!r} not in effects_df.")
    if join_col not in units_gdf.columns:
        raise KeyError(f"Column {join_col!r} not in units_gdf.")

    # Coerce join keys to a common dtype (handles int vs str id mismatches).
    units = units_gdf.copy()
    eff = effects_df.copy()
    units[join_col] = units[join_col].astype(str)
    eff[join_col] = eff[join_col].astype(str)
    merged = units.merge(eff[[join_col, value_col]], on=join_col, how="left")

    merged = merged.to_crs(TARGET_CRS)

    # Basemap drawn FIRST (before any data) so tract/route/point layers sit on top.
    if basemap:
        _add_basemap(ax, extent_gdf=base_layer if base_layer is not None else merged)

    if base_layer is not None:
        # When basemap is on, only draw tract outlines so the tiles show through.
        base_face = "none" if basemap else "white"
        base_layer.to_crs(TARGET_CRS).plot(
            ax=ax, facecolor=base_face, edgecolor="dimgrey", linewidth=0.25
        )

    valid = merged[~merged[value_col].isna()]
    if valid.empty:
        if crz_polygon is not None:
            crz_polygon.to_crs(TARGET_CRS).boundary.plot(ax=ax, color="red", linewidth=2)
        ax.set_axis_off()
        ax.set_title(f"{value_col} — all values missing", fontsize=11, color="red")
        return fig, ax
    vals = valid[value_col].astype(float)
    pct = np.nanpercentile(vals, [quantile_clip[0] * 100, quantile_clip[1] * 100])
    vmin, vmax = float(pct[0]), float(pct[1])

    if diverging is None:
        diverging = bool((vals.min() < center) and (vals.max() > center))
    if diverging:
        half = max(abs(vmin - center), abs(vmax - center)) or 1.0
        norm = TwoSlopeNorm(vcenter=center, vmin=center - half, vmax=center + half)
        plot_kwargs = {"norm": norm}
    else:
        plot_kwargs = {"vmin": vmin, "vmax": vmax}

    legend_kwds = {"shrink": 0.5, "aspect": 30}
    geom_kind = valid.geometry.geom_type.iloc[0]
    if geom_kind in ("Point", "MultiPoint"):
        if point_size_by_abs:
            # Quadratic scaling on |val| so the size ratio between largest and
            # smallest dots is visually dramatic (~50× rather than ~7×).
            mag = vals.abs()
            mag_max = mag.max() or 1.0
            sizes = (point_size * 0.1) + ((mag / mag_max) ** 1.5) * point_size * 5.0
        else:
            sizes = point_size
        valid.plot(ax=ax, column=value_col, cmap=cmap, markersize=sizes,
                   edgecolor="none", linewidth=0,
                   legend=True, legend_kwds=legend_kwds, **plot_kwargs)
    elif geom_kind in ("LineString", "MultiLineString"):
        valid.plot(ax=ax, column=value_col, cmap=cmap, linewidth=line_width,
                   legend=True, legend_kwds=legend_kwds, **plot_kwargs)
    else:
        valid.plot(ax=ax, column=value_col, cmap=cmap,
                   edgecolor="dimgrey", linewidth=0.25,
                   legend=True, legend_kwds=legend_kwds, **plot_kwargs)

    if puma_polygon is not None:
        puma_polygon.to_crs(TARGET_CRS).plot(
            ax=ax, facecolor="none", edgecolor="black", linewidth=0.9
        )

    if crz_polygon is not None:
        crz_polygon.to_crs(TARGET_CRS).boundary.plot(ax=ax, color="red", linewidth=2)

    ax.set_axis_off()
    if legend_label:
        cbar = fig.axes[-1]
        cbar.set_ylabel(legend_label, rotation=270, labelpad=18)
    return fig, ax


def plot_unit_effects_by_significance(
    units_gdf,
    effects_df: pd.DataFrame,
    join_col: str,
    crz_polygon=None,
    base_layer=None,
    att_col: str = "att",
    signif_col: str = "att_signif",
    point_size: float = 35.0,
    point_size_by_abs: bool = True,
    line_width: float = 2.5,
    ax=None,
    figsize=(11, 11),
    dpi: int = 300,
    title: str | None = None,
):
    """Categorical map: each unit colored by SIGN × SIGNIFICANCE of its ATT.

    Unlike :func:`plot_unit_effects` (which fades color smoothly across all
    units regardless of confidence), this distinguishes:

    * **red**  — significantly positive (att > 0 AND ``signif_col`` True)
    * **blue** — significantly negative (att < 0 AND ``signif_col`` True)
    * **grey** — not significant (CI on ATT covers zero)
    * **white** — missing (no ATT could be computed)

    "Significant" uses the per-unit ATT 90% CI (``att_signif`` from
    ``compute_effects``), which is the proper test for *whether the unit's
    effect is detectably non-zero across the test horizon* — strictly
    stronger than just thresholding on the sign of the point estimate.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    import geopandas as gpd

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    units = units_gdf.copy()
    eff = effects_df.copy()
    units[join_col] = units[join_col].astype(str)
    eff[join_col] = eff[join_col].astype(str)
    keep_cols = [join_col, att_col]
    if signif_col in eff.columns:
        keep_cols.append(signif_col)
    merged = units.merge(eff[keep_cols], on=join_col, how="left").to_crs(TARGET_CRS)

    # Categorize
    sig = merged[signif_col].fillna(False).astype(bool) if signif_col in merged.columns else pd.Series(False, index=merged.index)
    pos = (merged[att_col] > 0) & sig
    neg = (merged[att_col] < 0) & sig
    nonsig = merged[att_col].notna() & ~sig
    missing = merged[att_col].isna()

    if base_layer is not None:
        base_layer.to_crs(TARGET_CRS).plot(ax=ax, color="whitesmoke",
                                           edgecolor="lightgray", linewidth=0.3)

    geom_kind = merged.geometry.geom_type.iloc[0]

    def _sized(s):
        if not point_size_by_abs:
            return point_size
        mag = s[att_col].abs()
        m_max = mag.max() or 1.0
        return (point_size * 0.3) + (mag / m_max) * point_size * 2.0

    if geom_kind in ("Point", "MultiPoint"):
        if missing.any():
            merged[missing].plot(ax=ax, color="white", markersize=point_size * 0.4,
                                 edgecolor="lightgray", linewidth=0.3)
        if nonsig.any():
            merged[nonsig].plot(ax=ax, color="lightgrey", markersize=point_size * 0.5,
                                edgecolor="dimgray", linewidth=0.3, alpha=0.7)
        if neg.any():
            merged[neg].plot(ax=ax, color="steelblue", markersize=_sized(merged[neg]),
                             edgecolor="navy", linewidth=0.4, alpha=0.85)
        if pos.any():
            merged[pos].plot(ax=ax, color="firebrick", markersize=_sized(merged[pos]),
                             edgecolor="darkred", linewidth=0.4, alpha=0.85)
    elif geom_kind in ("LineString", "MultiLineString"):
        if missing.any():
            merged[missing].plot(ax=ax, color="white", linewidth=0.6)
        if nonsig.any():
            merged[nonsig].plot(ax=ax, color="lightgrey", linewidth=line_width * 0.5, alpha=0.7)
        if neg.any():
            merged[neg].plot(ax=ax, color="steelblue", linewidth=line_width, alpha=0.85)
        if pos.any():
            merged[pos].plot(ax=ax, color="firebrick", linewidth=line_width, alpha=0.85)
    else:
        # Polygons
        cats = pd.Series("missing", index=merged.index)
        cats[nonsig] = "not significant"; cats[neg] = "signif negative"; cats[pos] = "signif positive"
        merged["_cat"] = cats
        cmap = {"missing": "white", "not significant": "lightgrey",
                "signif negative": "steelblue", "signif positive": "firebrick"}
        for cat, color in cmap.items():
            sub = merged[merged._cat == cat]
            if not sub.empty:
                sub.plot(ax=ax, color=color, edgecolor="white", linewidth=0.1, alpha=0.85)

    if crz_polygon is not None:
        crz_polygon.to_crs(TARGET_CRS).boundary.plot(ax=ax, color="black", linewidth=2)

    ax.set_axis_off()

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="firebrick",
               markeredgecolor="darkred", markersize=10,
               label=f"Signif positive  (n={int(pos.sum())})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markeredgecolor="navy", markersize=10,
               label=f"Signif negative  (n={int(neg.sum())})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgrey",
               markeredgecolor="dimgray", markersize=8,
               label=f"Not significant  (n={int(nonsig.sum())})"),
    ]
    if missing.any():
        legend_elems.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                                   markeredgecolor="lightgray", markersize=6,
                                   label=f"Missing  (n={int(missing.sum())})"))
    ax.legend(handles=legend_elems, loc="upper left", frameon=True, fontsize=10)

    if title:
        ax.set_title(title, fontsize=12)
    return fig, ax


def summarize_effects_by_crz(
    units_gdf,
    effects_df: pd.DataFrame,
    crz_polygon,
    kind: Literal["routes", "tracts", "stations"],
    join_col: str,
    att_col: str = "att",
    cum_effect_col: str = "cum_effect",
    cum_cf_col: str = "cum_cf",
    signif_col: str = "att_signif",
    cum_lo_col: str = "cum_effect_ci_lo",
    cum_hi_col: str = "cum_effect_ci_hi",
    coverage_level: float = 0.9,
) -> pd.DataFrame:
    """One row per CRZ class: count of units, count signif +/-, mean ATT,
    pooled cumulative ATT, and pooled relative effect = Σ cum_effect / Σ cum_cf.

    Use this to check whether the policy effect is concentrated in CRZ-inside
    units (the policy target) and whether the spatial story is compositional
    (e.g. inside +X%, outside -Y%, average ≈ 0)."""
    units = units_gdf.copy()
    eff = effects_df.copy()
    units[join_col] = units[join_col].astype(str)
    eff[join_col] = eff[join_col].astype(str)
    crz_in_target = crz_polygon.to_crs(TARGET_CRS) if hasattr(crz_polygon, "to_crs") else crz_polygon
    units = classify_crz(units.to_crs(TARGET_CRS), crz_in_target, kind=kind)
    cols = [join_col, att_col]
    for c in (cum_effect_col, cum_cf_col, signif_col, cum_lo_col, cum_hi_col):
        if c in eff.columns: cols.append(c)
    merged = units.merge(eff[cols], on=join_col, how="left")

    from scipy.stats import norm as _norm
    z = float(_norm.ppf(0.5 + coverage_level / 2))

    rows = []
    for crz_class, grp in merged.groupby("crz_class"):
        valid = grp[grp[att_col].notna()]
        sig = valid[signif_col].fillna(False).astype(bool) if signif_col in valid.columns else pd.Series(False, index=valid.index)
        pos = (valid[att_col] > 0) & sig
        neg = (valid[att_col] < 0) & sig
        row = {
            "crz_class": crz_class,
            "n_units": len(valid),
            "n_signif_positive": int(pos.sum()),
            "n_signif_negative": int(neg.sum()),
            "n_not_signif": int(valid[att_col].notna().sum() - pos.sum() - neg.sum()),
            "mean_att": float(valid[att_col].mean()) if len(valid) else float("nan"),
            "median_att": float(valid[att_col].median()) if len(valid) else float("nan"),
        }
        if cum_effect_col in valid.columns and cum_cf_col in valid.columns:
            cum_eff_sum = float(valid[cum_effect_col].sum())
            cum_cf_sum = float(valid[cum_cf_col].sum())
            row["total_cum_effect"] = cum_eff_sum
            row["total_cum_cf"] = cum_cf_sum
            row["pooled_relative_effect"] = cum_eff_sum / cum_cf_sum if cum_cf_sum else float("nan")
            if cum_lo_col in valid.columns and cum_hi_col in valid.columns:
                unit_se = (valid[cum_hi_col] - valid[cum_lo_col]) / (2 * z)
                class_se = float(np.sqrt(float((unit_se ** 2).sum())))
                row["total_cum_se"] = class_se
                row["total_cum_lo"] = cum_eff_sum - z * class_se
                row["total_cum_hi"] = cum_eff_sum + z * class_se
                if cum_cf_sum:
                    row["pooled_rel_lo"] = (cum_eff_sum - z * class_se) / cum_cf_sum
                    row["pooled_rel_hi"] = (cum_eff_sum + z * class_se) / cum_cf_sum
                else:
                    row["pooled_rel_lo"] = float("nan")
                    row["pooled_rel_hi"] = float("nan")
                row["pooled_rel_signif"] = bool(
                    (row["pooled_rel_lo"] > 0) or (row["pooled_rel_hi"] < 0)
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("crz_class").reset_index(drop=True)


def plot_trends_by_crz(
    long_df: pd.DataFrame,
    units_gdf,
    crz_polygon,
    kind: Literal["routes", "tracts", "stations"],
    join_col: str,
    title_prefix: str = "",
    figsize=(13, 9),
    dpi: int = 150,
    line_colors: dict[str, str] | None = None,
):
    """Two-row figure: daily ATT (top) and cumulative ATT% (bottom), with one
    line per CRZ class. Replaces the single-line ``plot_effects_over_time``
    when you want to see Inside-vs-Outside dynamics.

    Parameters
    ----------
    long_df : per-(date, unit) ATT panel from ``compute_effects`` _long.csv.
              Must have columns: date, <join_col>, tau, cf_mean.
    units_gdf : geometry per unit, indexed/keyed by ``join_col``.
    crz_polygon : GeoSeries / GeoDataFrame with the CRZ polygon.
    kind : "routes" (3-class) | "stations" (2-class) | "tracts" (2-class).
    """
    import geopandas as gpd

    # Classify each unit into CRZ class
    crz_in_target = crz_polygon.to_crs(TARGET_CRS) if hasattr(crz_polygon, "to_crs") else crz_polygon
    units_classified = classify_crz(units_gdf.to_crs(TARGET_CRS), crz_in_target, kind=kind)
    unit_to_class = dict(zip(units_classified[join_col].astype(str), units_classified["crz_class"]))

    long = long_df.copy()
    long[join_col] = long[join_col].astype(str)
    long["crz_class"] = long[join_col].map(unit_to_class)
    long = long.dropna(subset=["crz_class"])
    long["date"] = pd.to_datetime(long["date"])

    if line_colors is None:
        if kind == "routes":
            line_colors = {"fully_inside": "C3", "partially_inside": "C1", "fully_outside": "C0"}
        else:
            line_colors = {"Inside CRZ": "C3", "Outside CRZ": "C0"}

    # Daily aggregate per (date, class): mean tau across units in that class
    daily = (long.groupby(["date", "crz_class"], as_index=False)
                  .agg(daily_mean_tau=("tau", "mean"),
                       daily_se_tau=("tau", lambda s: float(s.std(ddof=1) / max(len(s), 1) ** 0.5)),
                       n_units=("tau", "count")))

    # Cumulative pooled relative effect per (date, class):
    #   sum_units(cum_effect_to_date) / sum_units(cum_cf_to_date)
    long_sorted = long.sort_values(["crz_class", join_col, "date"])
    long_sorted["cum_eff_unit"] = long_sorted.groupby([join_col])["tau"].cumsum()
    long_sorted["cum_cf_unit"]  = long_sorted.groupby([join_col])["cf_mean"].cumsum()
    pooled = (long_sorted.groupby(["date", "crz_class"], as_index=False)
                          .agg(cum_eff_pool=("cum_eff_unit", "sum"),
                               cum_cf_pool=("cum_cf_unit", "sum")))
    pooled["cum_rel_pct"] = 100.0 * pooled["cum_eff_pool"] / pooled["cum_cf_pool"].replace(0, np.nan)

    fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)

    # Top: daily mean tau per class with ±1 SE shading
    ax = axes[0]
    for cls in sorted(daily["crz_class"].unique()):
        sub = daily[daily.crz_class == cls].sort_values("date")
        col = line_colors.get(cls, "k")
        ax.fill_between(sub["date"],
                        sub["daily_mean_tau"] - sub["daily_se_tau"],
                        sub["daily_mean_tau"] + sub["daily_se_tau"],
                        color=col, alpha=0.18)
        ax.plot(sub["date"], sub["daily_mean_tau"], color=col, lw=1.6,
                label=f"{cls}  (n={int(sub['n_units'].iloc[0])} units)")
    ax.axhline(0, color="red", ls="--", lw=0.8)
    ax.set_ylabel("Daily ATT per unit\n(mean across class)")
    ax.set_title(f"{title_prefix} Daily ATT by CRZ class".strip())
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # Bottom: cumulative pooled relative effect (%)
    ax = axes[1]
    for cls in sorted(pooled["crz_class"].unique()):
        sub = pooled[pooled.crz_class == cls].sort_values("date")
        col = line_colors.get(cls, "k")
        ax.plot(sub["date"], sub["cum_rel_pct"], color=col, lw=1.8,
                label=f"{cls}  (final {sub['cum_rel_pct'].iloc[-1]:+.2f}%)")
    ax.axhline(0, color="red", ls="--", lw=0.8)
    ax.set_ylabel("Cumulative pooled\nrelative ATT (%)")
    ax.set_xlabel("Date")
    ax.set_title(f"{title_prefix} Cumulative relative ATT by CRZ class".strip())
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, axes


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
