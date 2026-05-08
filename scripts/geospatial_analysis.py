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
from nyc_cp.analysis.geospatial import map_units_to_tracts
from nyc_cp.config import load_paths, normalize_window_name, output_dir
from nyc_cp.utils import setup_logging

log = logging.getLogger("geospatial_analysis")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["bus", "subway", "citibike"])
    p.add_argument("--model", required=True, choices=["arima", "prophet", "deepar", "pcn", "chronos", "timesfm", "nhits", "tft", "bsts"])
    p.add_argument("--window", required=True, choices=["val", "validation", "test"], help="Forecast window (val and validation are aliases).")
    p.add_argument("--direction", choices=["all", "O", "D"], default="all")
    p.add_argument("--y-col", default="att_mean", help="Tract-level outcome column for regressions.")
    p.add_argument("--vif-threshold", type=float, default=20.0)
    return p.parse_args()


def _load_geo(geo_root: Path):
    import geopandas as gpd

    return {
        "tracts": gpd.read_file(geo_root / "NYC_Census_Tracts_2020" / "NYC_Census_Tracts_2020.shp"),
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
        return map_units_to_tracts(
            stations, geo["tracts"], unit, join_col_units="station_id", join_col_effects="station_id", metric_cols=metric_cols
        )

    if args.mode == "citibike":
        # citibike units are already tract-indexed; just merge directly to tract polygons.
        unit = unit.rename(columns={unit.columns[0]: "tract_id"}) if unit.columns[0] != "tract_id" else unit
        tracts = geo["tracts"].copy()
        if "GEOID" not in tracts.columns:
            tracts = tracts.rename(columns={c: "GEOID" for c in tracts.columns if c.lower() == "geoid"})
        unit["tract_id"] = unit["tract_id"].astype(str)
        return tracts.merge(unit.rename(columns={c: f"{c}_mean" for c in metric_cols}), left_on="GEOID", right_on="tract_id", how="left")

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
