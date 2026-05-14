"""Publication spatial figures: per-unit ATT and tract-aggregated ATT
for the 5 (mode, direction) pairs used in the paper —
bus (all), subway O, subway D, replica O, replica D.

Reads the artefacts produced by ``scripts.compute_effects`` and
``scripts.geospatial_analysis``:
  * ``effects/<prefix>_unit.csv``        — per-unit ATT (route/station/tract)
  * ``causal/tract_effects.geojson``     — per-tract effect + ACS demographics

Produces, for each ``(mode, direction)`` in ``MODE_CONFIGS``:
  1. **Per-unit map** — raw unit geometry coloured by mean daily ATT.
     Bus uses route polylines, subway uses station points (size ∝ |ATT|),
     replica uses tract polygons (its units *are* tracts).
  2. **Tract choropleth** — same metric aggregated to census tracts.

Open as a notebook in VS Code / Jupyter; cells delimited by ``# %%``.
"""

# %% Imports + publication rcParams
# Auto-reload so edits to nyc_cp.* are picked up without restarting the kernel.
try:
    get_ipython().run_line_magic("load_ext", "autoreload")     # type: ignore[name-defined]
    get_ipython().run_line_magic("autoreload", "2")            # type: ignore[name-defined]
except (NameError, ImportError):
    pass

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from nyc_cp.analysis.geospatial import plot_choropleth, plot_unit_effects
from nyc_cp.config import REPO_ROOT, load_paths, output_dir
from nyc_cp.analysis.geospatial import _add_basemap

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,  # editable text in vector PDF (journal requirement)
    "ps.fonttype": 42,
})

# %% Config — edit which (mode, direction) combinations and which model to plot
MODE_CONFIGS = [
    {"mode": "bus",     "direction": "all"},
    {"mode": "subway",  "direction": "O"},
    {"mode": "subway",  "direction": "D"},
    {"mode": "replica", "direction": "O"},
    {"mode": "replica", "direction": "D"},
]
MODEL = "timesfm_qrcal_intercept"   # HQC-Chronos (paper headline)
WINDOW = "test"
VALUE_COL = "avg_daily"             # mean daily ATT (rides/day). "att" is sum-over-window.

SAVE_FIGS = True
FIG_DIR = REPO_ROOT / "outputs" / "figures" / "paper" / "spatial"

# %% Load shared geometry (paths resolved against repo root, so cwd-independent)
paths = load_paths()

_geo_rel = Path(paths["geo_root"])
GEO_ROOT = _geo_rel if _geo_rel.is_absolute() else REPO_ROOT / _geo_rel

tracts = gpd.read_file(GEO_ROOT / "NYC_Census_Tracts_2020" / "NYC_Census_Tracts_2020.shp")
if "GEOID" not in tracts.columns:
    tracts = tracts.rename(columns={c: "GEOID" for c in tracts.columns if c.lower() == "geoid"})
tracts["GEOID"] = tracts["GEOID"].astype(str)

crz = gpd.read_file(GEO_ROOT / "nyc_cp_boundary_poly_json.geojson")
puma = gpd.read_file(GEO_ROOT / "NYC_PUMA" / "NYC_Public_Use_Microdata_Areas_PUMAs_2010.shp")
bus_routes = gpd.read_file(GEO_ROOT / "bus_routes" / "bus_routes_nyc_dec2019.shp")
subway_stations = gpd.read_file(GEO_ROOT / "MTA_Subway_Stations_20251029.geojson")
subway_stations["station_id"] = subway_stations["station_id"].astype(str)

print(f"Geo loaded: {len(tracts)} tracts, {len(puma)} PUMAs, {len(bus_routes)} routes, {len(subway_stations)} stations")

# %% Helper — return (units_gdf, unit_df, join_col) for any (mode, direction)
def load_unit_geo(mode: str, direction: str):
    """Mirror of `_tract_effects` logic in scripts/geospatial_analysis.py,
    but returns the per-unit pieces needed for `plot_unit_effects` (not the
    tract-aggregated frame).
    """
    out = output_dir(mode, MODEL, direction=direction, paths=paths)
    prefix = f"{mode}_{MODEL}_{WINDOW}" + (f"_{direction}" if direction != "all" else "")
    unit_csv = out / "effects" / f"{prefix}_unit.csv"
    if not unit_csv.exists():
        raise FileNotFoundError(f"missing {unit_csv} — run scripts.compute_effects first")
    unit_df = pd.read_csv(unit_csv)

    if mode == "bus":
        unit_df = unit_df.rename(columns={unit_df.columns[0]: "route_id"})
        return bus_routes, unit_df, "route_id"

    if mode == "subway":
        unit_df = unit_df.rename(columns={unit_df.columns[0]: "station_id"})
        unit_df["station_id"] = unit_df["station_id"].astype(str)
        return subway_stations, unit_df, "station_id"

    if mode == "replica":
        # Replica unit_id is the full 11-digit FIPS GEOID. Join directly on tracts.GEOID.
        unit_df = unit_df.rename(columns={unit_df.columns[0]: "tract_id"})
        unit_df["tract_id"] = unit_df["tract_id"].astype(str)
        units_gdf = tracts.rename(columns={"GEOID": "tract_id"}).copy()
        return units_gdf, unit_df, "tract_id"

    raise ValueError(f"unknown mode: {mode}")


def _label(mode: str, direction: str) -> str:
    return mode if direction == "all" else f"{mode}_{direction}"


def _save(fig, name: str):
    if not SAVE_FIGS:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=200)
    print(f"  saved → {FIG_DIR / name}.{{pdf,png}}")


# %% Loop 1 — Per-unit spatial distribution (5 figures)
for cfg in MODE_CONFIGS:
    mode, direction = cfg["mode"], cfg["direction"]
    print(f"=== per-unit: {mode}/{direction} ===")
    units_gdf, unit_df, join_col = load_unit_geo(mode, direction)

    if mode == "bus":
        style = dict(line_width=1.8, point_size=30.0, point_size_by_abs=False)
    elif mode == "subway":
        style = dict(line_width=2.0, point_size=35.0, point_size_by_abs=True)
    else:
        style = {}

    fig, ax = plot_unit_effects(
        units_gdf, unit_df,
        join_col=join_col,
        value_col=VALUE_COL,
        crz_polygon=crz,
        puma_polygon=puma,
        base_layer=tracts,
        cmap="RdBu_r",
        legend_label="Mean daily ATT (rides/day)",
        basemap=True,
        figsize=(8, 8),
        dpi=200,
        **style,
    )
    _save(fig, f"perunit_{_label(mode, direction)}")
    plt.show()

# %% Loop 2 — Tract-aggregated choropleth (5 figures)
for cfg in MODE_CONFIGS:
    mode, direction = cfg["mode"], cfg["direction"]
    print(f"=== tract aggregate: {mode}/{direction} ===")
    out = output_dir(mode, MODEL, direction=direction, paths=paths)
    tract_geojson = out / "causal" / "tract_effects.geojson"
    if not tract_geojson.exists():
        print(f"  SKIP — missing {tract_geojson} (run scripts.geospatial_analysis first)")
        continue
    tract_eff = gpd.read_file(tract_geojson)

    fig, ax = plot_choropleth(
        tract_eff,
        column=VALUE_COL,
        crz_polygon=crz,
        puma_polygon=puma,
        cmap="RdBu_r",
        legend_label="Mean daily ATT (rides/day)",
        basemap=True,
        figsize=(8, 8),
        dpi=200,
    )
    _save(fig, f"tract_{_label(mode, direction)}")
    plt.show()

print("Done." + (f" Figures under {FIG_DIR}" if SAVE_FIGS else ""))

# %%