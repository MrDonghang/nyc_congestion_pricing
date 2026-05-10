"""Geospatial + causal analysis for one (mode, model, window).

Steps
-----
1. Load the per-unit ATT summary written by ``compute_effects.py``.
2. Spatially join unit effects to census tracts (bus stops / subway stations
   / Citibike already-tract-keyed).
3. Build ACS-derived demographic features per tract.
4. Run VIF filter + OLS / ML_Lag / ML_Error spatial regressions and tree-based
   ML (RF / XGB / LightGBM) on the tract-level effect.

Outputs
-------
Under ``<output_root>/<mode>/<model>/causal/``:
  * ``tract_effects.geojson``       — per-tract effect + demographics
  * ``spatial_regression.txt``      — OLS / ML_Lag / ML_Error summaries
  * ``ml_models.csv``               — tree-based feature-importance comparison

Examples
--------
    python -m scripts.geospatial_analysis --mode bus --model pcn --window test
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from nyc_cp.analysis import demographics
from nyc_cp.analysis.causal import run_spatial_regression, run_tree_models, stepwise_vif_filter
from nyc_cp.analysis.geospatial import (
    map_units_to_tracts,
    plot_choropleth,
    plot_effects_over_time,
    plot_trends_by_crz,
    plot_unit_effects,
    plot_unit_effects_by_significance,
    summarize_effects_by_crz,
)
from nyc_cp.config import load_paths, normalize_window_name, output_dir
from nyc_cp.utils import setup_logging

log = logging.getLogger("geospatial_analysis")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["bus", "subway", "citibike", "replica"])
    p.add_argument("--model", required=True, choices=["arima", "prophet", "deepar", "pcn", "chronos", "timesfm", "nhits", "tft", "bsts", "chronos_qrcal", "timesfm_qrcal", "chronos_qrcal_perunit", "timesfm_qrcal_perunit", "chronos_qrcal_intercept", "timesfm_qrcal_intercept", "chronos_qrcal_oos", "timesfm_qrcal_oos", "chronos_qrcal_intercept_insample", "timesfm_qrcal_intercept_insample"])
    p.add_argument("--window", required=True, choices=["val", "validation", "test"], help="Forecast window (val and validation are aliases).")
    p.add_argument("--direction", choices=["all", "O", "D"], default="all")
    p.add_argument("--y-col", default="att", help="Tract-level outcome column for regressions.")
    p.add_argument("--vif-threshold", type=float, default=20.0)
    p.add_argument(
        "--skip-regression",
        action="store_true",
        help="Stop after writing tract_effects.geojson; skip spatial OLS/ML and tree models.",
    )
    return p.parse_args()


def _load_geo(geo_root: Path):
    import geopandas as gpd

    # The tract shapefile ships with lowercase ``geoid``; downstream code
    # (map_units_to_tracts, the citibike merge) defaults to ``GEOID``.
    # Normalize once at load so every consumer sees the same name.
    tracts = gpd.read_file(geo_root / "NYC_Census_Tracts_2020" / "NYC_Census_Tracts_2020.shp")
    if "GEOID" not in tracts.columns:
        tracts = tracts.rename(columns={c: "GEOID" for c in tracts.columns if c.lower() == "geoid"})

    return {
        "tracts": tracts,
        "puma": gpd.read_file(geo_root / "NYC_PUMA" / "NYC_Public_Use_Microdata_Areas_PUMAs_2010.shp"),
        "crz": gpd.read_file(geo_root / "nyc_cp_boundary_poly_json.geojson"),
        "demographics": geo_root / "nyc_census_tracts.geojson",
        "bus_routes": geo_root / "bus_routes" / "bus_routes_nyc_dec2019.shp",
        "bus_stops": geo_root / "NYC_bus_stop" / "Stops_by_Route.shp",
        "subway_stations": geo_root / "MTA_Subway_Stations_20251029.geojson",
    }


def _tract_effects(args, paths, geo) -> "geopandas.GeoDataFrame":
    """Bring per-unit ATT into census-tract space."""
    import geopandas as gpd

    out_dir = output_dir(args.mode, args.model, direction=args.direction, paths=paths)
    prefix = f"{args.mode}_{args.model}_{args.window}" + (f"_{args.direction}" if args.direction != "all" else "")
    unit_csv = out_dir / "effects" / f"{prefix}_unit.csv"
    if not unit_csv.exists():
        raise SystemExit(f"Unit-effect file not found: {unit_csv}. Run scripts/compute_effects.py first.")
    unit = pd.read_csv(unit_csv)

    metric_cols = [c for c in ("att", "avg_daily", "signif_days", "signif_share", "cum_effect", "cum_cf") if c in unit.columns]

    if args.mode == "bus":
        bus_stops = gpd.read_file(geo["bus_stops"])
        unit = unit.rename(columns={unit.columns[0]: "route_id"}) if unit.columns[0] != "route_id" else unit
        n_stops = bus_stops.groupby("route_id").size().reset_index(name="n_stops")
        unit = unit.merge(n_stops, on="route_id", how="left")
        for c in metric_cols:
            unit[c] = unit[c] / unit["n_stops"].replace(0, 1)
        return map_units_to_tracts(
            bus_stops, geo["tracts"], unit, join_col_units="route_id", join_col_effects="route_id", metric_cols=metric_cols
        )

    if args.mode == "subway":
        stations = gpd.read_file(geo["subway_stations"])
        unit = unit.rename(columns={unit.columns[0]: "station_id"}) if unit.columns[0] != "station_id" else unit
        # The MTA stations geojson stores station_id as int; the unit-effects csv
        # stores it as str. Coerce both to str so the merge inside
        # map_units_to_tracts succeeds.
        stations["station_id"] = stations["station_id"].astype(str)
        unit["station_id"] = unit["station_id"].astype(str)
        return map_units_to_tracts(
            stations, geo["tracts"], unit, join_col_units="station_id", join_col_effects="station_id", metric_cols=metric_cols
        )

    if args.mode == "replica":
        # Replica units are tracts keyed by full 11-digit FIPS GEOID strings
        # (e.g. "36061000100"). Direct merge with the NYC tract shapefile's
        # GEOID — no index pkl needed. Tracts in the replica panel that lie
        # outside NYC (e.g. Suffolk County 36103xx) won't match and become
        # NaN ATT in the output GDF — correct behaviour.
        unit = unit.rename(columns={unit.columns[0]: "tract_id"}) if unit.columns[0] != "tract_id" else unit
        unit["tract_id"] = unit["tract_id"].astype(str)
        tracts = geo["tracts"].copy()
        tracts["GEOID"] = tracts["GEOID"].astype(str)
        return tracts.merge(unit, left_on="GEOID", right_on="tract_id", how="left")

    if args.mode == "citibike":
        # citibike units are tract-indexed by *integer position*, not FIPS GEOID.
        # The processing step (nyc_cp.data.citibike) builds an index over 6-digit
        # ct2020 codes (which are not unique across NYC's 5 boroughs — e.g. 000100
        # exists in Manhattan, Brooklyn, and Bronx), so 1530 indices cover ~2325
        # tracts. Recover the mapping from the pkl saved at processing time and
        # merge on ``ct2020``: a single citibike tract_id thus paints every
        # borough's tract that shares the ct2020 with the same ATT value, which
        # is the honest rendering given the lossy upstream index.
        import pickle
        unit = unit.rename(columns={unit.columns[0]: "tract_id"}) if unit.columns[0] != "tract_id" else unit
        pkl_path = Path(paths["data_root"]) / "citibike" / "census" / "censustract_idx_mapping.pkl"
        if not pkl_path.exists():
            raise SystemExit(
                f"citibike index mapping pkl not found at {pkl_path}. "
                "Re-run nyc_cp.data.citibike.process to regenerate."
            )
        with open(pkl_path, "rb") as f:
            idx_map = pickle.load(f)            # ct2020 (str) -> idx (int)
        inv = {v: k for k, v in idx_map.items()}  # idx -> ct2020
        unit["ct2020"] = unit["tract_id"].astype(int).map(inv)
        unit = unit.dropna(subset=["ct2020"])
        tracts = geo["tracts"].copy()
        tracts["ct2020"] = tracts["ct2020"].astype(str)
        return tracts.merge(unit, on="ct2020", how="left")

    raise SystemExit(f"Unsupported mode for spatial mapping: {args.mode}")


def main() -> None:
    args = parse_args()
    args.window = normalize_window_name(args.window)
    paths = load_paths()
    setup_logging(f"geospatial_{args.mode}_{args.model}_{args.window}", log_root=paths["log_root"])

    geo_root = Path(paths["geo_root"])
    geo = _load_geo(geo_root)

    log.info("Mapping unit effects → tracts")
    tract_eff = _tract_effects(args, paths, geo)

    log.info("Building ACS demographics")
    geoid_col = "geoid" if "geoid" in tract_eff.columns else "GEOID"
    tracts_with_dem = demographics.build(tract_eff, geo["demographics"], year=2023, geoid_col=geoid_col)

    out_dir = output_dir(args.mode, args.model, direction=args.direction, paths=paths) / "causal"
    out_dir.mkdir(parents=True, exist_ok=True)
    tracts_with_dem.to_file(out_dir / "tract_effects.geojson", driver="GeoJSON")
    log.info("Wrote %s", out_dir / "tract_effects.geojson")

    # ----------------------------------------------------------------------
    # Significance-categorical map + CRZ-grouped summary
    # ----------------------------------------------------------------------
    log.info("Building significance map + CRZ summary")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import geopandas as gpd

    eff_dir = output_dir(args.mode, args.model, direction=args.direction, paths=paths) / "effects"
    prefix = f"{args.mode}_{args.model}_{args.window}" + (f"_{args.direction}" if args.direction != "all" else "")
    unit_csv = eff_dir / f"{prefix}_unit.csv"
    unit = pd.read_csv(unit_csv)
    id_col = unit.columns[0]
    unit = unit.rename(columns={id_col: id_col})  # no-op, keeps name explicit

    # Pick the right unit-geometry GDF + CRZ kind per mode
    if args.mode == "bus":
        units_gdf = gpd.read_file(geo["bus_routes"])
        join_col = "route_id"
        kind = "routes"
    elif args.mode == "subway":
        units_gdf = gpd.read_file(geo["subway_stations"])
        units_gdf["station_id"] = units_gdf["station_id"].astype(str)
        join_col = "station_id"
        kind = "stations"
    elif args.mode == "replica":
        # Replica unit is the tract itself (GEOID-keyed). Use the NYC tract
        # polygons directly so that CRZ classification is by polygon-share.
        units_gdf = geo["tracts"].copy()
        units_gdf["GEOID"] = units_gdf["GEOID"].astype(str)
        units_gdf = units_gdf.rename(columns={"GEOID": "tract_id"})
        join_col = "tract_id"
        kind = "tracts"
    else:  # citibike — units are tract polygons; reuse the tract_eff result
        # tract_eff already has merged effect columns (att, signif, etc.); the
        # downstream plotting helpers do their own merge against `unit`, so
        # we'd get att_x / att_y suffixes. Keep only geometry + join key here.
        keep = ["tract_id", "geometry"] if "tract_id" in tract_eff.columns else [id_col, "geometry"]
        units_gdf = tract_eff[keep].copy()
        join_col = keep[0]
        kind = "tracts"

    crz_polygon = geo["crz"]

    # Categorical significance map
    fig, ax = plt.subplots(figsize=(11, 11), dpi=200)
    plot_unit_effects_by_significance(
        units_gdf=units_gdf,
        effects_df=unit.rename(columns={id_col: join_col}) if id_col != join_col else unit,
        join_col=join_col,
        crz_polygon=crz_polygon,
        ax=ax,
        title=f"{args.mode} {args.direction} — {args.model}: per-unit ATT significance",
    )
    fig.savefig(out_dir / "significance_map.png", bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_dir / "significance_map.png")

    # CRZ-grouped summary table
    crz_summary = summarize_effects_by_crz(
        units_gdf=units_gdf,
        effects_df=unit.rename(columns={id_col: join_col}) if id_col != join_col else unit,
        crz_polygon=crz_polygon,
        kind=kind,
        join_col=join_col,
    )
    crz_summary.to_csv(out_dir / "crz_summary.csv", index=False)
    log.info("Wrote %s\n%s", out_dir / "crz_summary.csv", crz_summary.to_string(index=False))

    # ----------------------------------------------------------------------
    # 4 additional standard plots:
    #   (a) daily ATT trend (mean_tau over time, with PI band)
    #   (b) cumulative ATT trend (cum_tau over time)
    #   (c) per-unit effect on raw geometry (continuous color)
    #   (d) tract-level choropleth (aggregated to census tracts)
    # ----------------------------------------------------------------------
    log.info("Building daily/cumulative trend + per-unit/tract maps")

    # (a) + (b) — trends BY CRZ CLASS from _long.csv (per-(unit, date) panel)
    # This replaces the panel-aggregate single-line plot with one line per
    # CRZ class so the spatial heterogeneity is visible in the time domain.
    long_csv = eff_dir / f"{prefix}_long.csv"
    if long_csv.exists():
        try:
            long_df = pd.read_csv(long_csv)
            # Make the join column name match what the unit_gdf uses
            if id_col != join_col and id_col in long_df.columns:
                long_df = long_df.rename(columns={id_col: join_col})
            fig, _ = plot_trends_by_crz(
                long_df=long_df,
                units_gdf=units_gdf,
                crz_polygon=crz_polygon,
                kind=kind,
                join_col=join_col,
                title_prefix=f"{args.mode} {args.direction} — {args.model}",
            )
            fig.savefig(out_dir / "trends_by_crz.png", bbox_inches="tight")
            plt.close(fig)
            log.info("Wrote %s", out_dir / "trends_by_crz.png")
        except Exception as e:
            log.warning("Failed trends_by_crz.png: %s", e)
        # Also keep the panel-aggregate version for completeness
        daily_csv = eff_dir / f"{prefix}_daily.csv"
        if daily_csv.exists():
            df_daily = pd.read_csv(daily_csv)
            for kind_, fname in [("daily_att", "daily_att.png"), ("cum_att", "cumulative_att.png")]:
                try:
                    fig, _ = plot_effects_over_time(df_daily, mode=kind_,
                                                    title_prefix=f"{args.mode} {args.direction} — {args.model}")
                    fig.savefig(out_dir / fname, bbox_inches="tight")
                    plt.close(fig)
                    log.info("Wrote %s", out_dir / fname)
                except Exception as e:
                    log.warning("Failed %s: %s", fname, e)
    else:
        log.warning("long.csv not found at %s; skipping trend plots", long_csv)

    # (c) — per-unit effect on raw geometry (continuous diverging color)
    try:
        fig, _ = plot_unit_effects(
            units_gdf=units_gdf,
            effects_df=unit.rename(columns={id_col: join_col}) if id_col != join_col else unit,
            join_col=join_col,
            value_col="att",
            crz_polygon=crz_polygon,
            point_size_by_abs=(kind == "stations"),
            figsize=(11, 11), dpi=150,
        )
        fig.savefig(out_dir / "unit_effects_map.png", bbox_inches="tight")
        plt.close(fig)
        log.info("Wrote %s", out_dir / "unit_effects_map.png")
    except Exception as e:
        log.warning("Failed unit_effects_map.png: %s", e)

    # (d) — tract-level choropleth (aggregate); skip for replica/citibike where
    # units already are tracts (the unit map IS the tract map).
    if args.mode in ("bus", "subway"):
        try:
            fig, _ = plot_choropleth(
                tracts_with_dem,
                column="att",
                crz_polygon=crz_polygon,
                cmap="RdBu_r",
                legend_label="mean per-tract ATT",
                figsize=(10, 10), dpi=150,
            )
            fig.savefig(out_dir / "tract_choropleth.png", bbox_inches="tight")
            plt.close(fig)
            log.info("Wrote %s", out_dir / "tract_choropleth.png")
        except Exception as e:
            log.warning("Failed tract_choropleth.png: %s", e)

    if args.skip_regression:
        log.info("--skip-regression set; stopping before OLS/ML and tree models.")
        return

    # Use the standard variable groups, then VIF-filter.
    x_cols = sum([demographics.GROUPS[g] for g in ["demographics", "race_ethnicity", "economics", "travel", "education", "housing"]], [])
    x_cols = [c for c in x_cols if c in tracts_with_dem.columns]
    if args.y_col not in tracts_with_dem.columns:
        raise SystemExit(f"y-column {args.y_col!r} not found. Available: {sorted(tracts_with_dem.columns)}")

    df = tracts_with_dem.dropna(subset=[args.y_col, *x_cols]).reset_index(drop=True)
    log.info("Regression sample size: %d", len(df))
    kept, dropped = stepwise_vif_filter(df[x_cols], threshold=args.vif_threshold)
    log.info("VIF kept %d / dropped %d", len(kept), len(dropped))

    spatial = run_spatial_regression(df, args.y_col, kept)
    with (out_dir / "spatial_regression.txt").open("w") as f:
        f.write("=== OLS ===\n")
        f.write(spatial.ols.summary)
        f.write("\n\n=== ML_Lag ===\n")
        f.write(spatial.ml_lag.summary)
        f.write("\n\n=== ML_Error ===\n")
        f.write(spatial.ml_error.summary)
    log.info("Wrote %s", out_dir / "spatial_regression.txt")

    ml = run_tree_models(df, args.y_col, kept)
    ml.to_csv(out_dir / "ml_models.csv")
    log.info("Wrote %s", out_dir / "ml_models.csv")
    print("\nML model summary:")
    print(ml[["r2_train", "r2_test", "rmse_test"]].to_string())


if __name__ == "__main__":
    main()
