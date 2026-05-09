"""Batch renderer for timesfm_qrcal × test figures across (mode, direction)
combos. Same layout as ``_render_tft_figs.py``. Writes PNGs to
``outputs/figures/<mode>_timesfm_qrcal_test{_dir}/``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from nyc_cp.analysis.geospatial import (
    plot_choropleth,
    plot_effects_over_time,
    plot_significance_calendar,
    plot_unit_effects,
)
from nyc_cp.config import load_paths, output_dir

MODEL = "timesfm_qrcal"
WINDOW = "test"
COMBOS = [
    ("bus", "all"),
    ("subway", "O"),
    ("subway", "D"),
    ("citibike", "O"),
    ("citibike", "D"),
]


def render(mode: str, direction: str) -> None:
    paths = load_paths()
    out_dir = output_dir(mode, MODEL, direction=direction, paths=paths)
    suffix = f"_{direction}" if direction != "all" else ""
    prefix = f"{mode}_{MODEL}_{WINDOW}{suffix}"

    daily_csv = out_dir / "effects" / f"{prefix}_daily.csv"
    tract_geojson = out_dir / "causal" / "tract_effects.geojson"
    if not daily_csv.exists() or not tract_geojson.exists():
        print(f"SKIP {mode}/{direction}: missing {daily_csv} or {tract_geojson}")
        return

    daily = pd.read_csv(daily_csv, parse_dates=["date"])
    tract_eff = gpd.read_file(tract_geojson)
    crz = gpd.read_file(Path(paths["geo_root"]) / "nyc_cp_boundary_poly_json.geojson")

    fig_dir = Path("outputs") / "figures" / f"{mode}_{MODEL}_{WINDOW}{suffix}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{mode}/{direction}] tracts={len(tract_eff)} days={len(daily)} → {fig_dir}")

    fig, ax = plot_choropleth(
        tract_eff, column="avg_daily", crz_polygon=crz,
        cmap="RdBu_r", legend_label="Mean daily ATT (rides/day)", figsize=(11, 11),
    )
    ax.set_title(f"{mode}/{direction}/{WINDOW} — mean daily ATT by tract  ({MODEL})", fontsize=13)
    fig.savefig(fig_dir / "choropleth_att.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if "cum_relative_effect" in tract_eff.columns:
        fig, ax = plot_choropleth(
            tract_eff, column="cum_relative_effect", crz_polygon=crz,
            cmap="RdBu_r", legend_label="Cumulative ATT / counterfactual", figsize=(11, 11),
        )
        ax.set_title(f"{mode}/{direction}/{WINDOW} — relative cumulative ATT  ({MODEL})", fontsize=13)
        fig.savefig(fig_dir / "choropleth_relative.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if mode in ("bus", "subway"):
        geo_root = Path(paths["geo_root"])
        unit_csv = out_dir / "effects" / f"{prefix}_unit.csv"
        unit_df = pd.read_csv(unit_csv)
        if mode == "bus":
            units_gdf = gpd.read_file(geo_root / "bus_routes" / "bus_routes_nyc_dec2019.shp")
            join_col, line_width, point_size, size_by_abs = "route_id", 1.8, 30.0, False
        else:
            units_gdf = gpd.read_file(geo_root / "MTA_Subway_Stations_20251029.geojson")
            units_gdf["station_id"] = units_gdf["station_id"].astype(str)
            unit_df["station_id"] = unit_df["station_id"].astype(str)
            join_col, line_width, point_size, size_by_abs = "station_id", 2.0, 35.0, True

        fig, ax = plot_unit_effects(
            units_gdf, unit_df, join_col=join_col, value_col="avg_daily",
            crz_polygon=crz, base_layer=tract_eff,
            cmap="RdBu_r", legend_label=f"Per-{join_col} mean daily ATT (rides/day)",
            line_width=line_width, point_size=point_size, point_size_by_abs=size_by_abs,
            figsize=(11, 11),
        )
        title = f"{mode}/{direction}/{WINDOW} — per-{'route' if mode=='bus' else 'station'} ATT  ({MODEL})"
        if mode == "subway" and size_by_abs:
            title += "  (size ∝ |ATT|)"
        ax.set_title(title, fontsize=13)
        fig.savefig(fig_dir / "per_unit_att.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    year = int(daily["date"].dt.year.mode().iloc[0])
    months = sorted(daily["date"].dt.month.unique().tolist())
    fig, _ = plot_significance_calendar(
        daily, year=year, start_month=int(months[0]), end_month=int(months[-1]),
    )
    fig.suptitle(f"{mode}/{direction}/{WINDOW} — significance calendar  ({MODEL})", y=1.02, fontsize=13)
    fig.savefig(fig_dir / "significance_calendar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plot_effects_over_time(daily, mode="daily_att", title_prefix=f"{mode.title()} ({MODEL})")
    fig.savefig(fig_dir / "daily_att.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plot_effects_over_time(daily, mode="cum_rel", title_prefix=f"{mode.title()} ({MODEL})")
    fig.savefig(fig_dir / "cum_rel_att.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for mode, direction in COMBOS:
        render(mode, direction)
    print("Done.")
