"""Geographic + temporal visualisation of one (mode, model, window) ATT run.

Reads the artefacts produced by ``scripts.compute_effects`` and
``scripts.geospatial_analysis``:
  * ``effects/<prefix>_daily.csv``         — daily mean / cumulative ATT
  * ``causal/tract_effects.geojson``       — per-tract effect + ACS demographics

Produces three figures:
  1. **Choropleth** of tract-level mean ATT (with CRZ boundary overlay).
  2. **Choropleth** of cumulative relative ATT (% of counterfactual).
  3. **Calendar** of significant ± vs non-significant days.
  4. **Time series** of daily ATT and cumulative relative ATT, with PI shading.

Open as a notebook in VS Code / Jupyter; cells delimited by ``# %%``.
"""

# %% Imports + config
from pathlib import Path

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

# ---- Edit to point at the run you want to visualise.
MODE = "bus"             # "bus" | "subway" | "citibike"
DIRECTION = "all"        # "all" | "O" | "D"
WINDOW = "test"          # "val" | "test"
MODEL = "chronos"        # any model name with saved effects
SAVE_FIGS = False        # True → write PNGs under outputs/figures/<mode>_<model>_<window>/

# %% Load artefacts
paths = load_paths()
out_dir = output_dir(MODE, MODEL, direction=DIRECTION, paths=paths)
prefix = f"{MODE}_{MODEL}_{WINDOW}" + (f"_{DIRECTION}" if DIRECTION != "all" else "")

daily_csv = out_dir / "effects" / f"{prefix}_daily.csv"
tract_geojson = out_dir / "causal" / "tract_effects.geojson"
if not daily_csv.exists():
    raise SystemExit(f"Missing {daily_csv} — run scripts.compute_effects first.")
if not tract_geojson.exists():
    raise SystemExit(f"Missing {tract_geojson} — run scripts.geospatial_analysis first.")

daily = pd.read_csv(daily_csv, parse_dates=["date"])
# plot_significance_calendar expects ``signif_daily`` (already present) and
# ``mean_tau``; rename so its tau_col default works without repeated edits.
daily_for_calendar = daily.rename(columns={"signif_daily": "signif_daily"}).copy()

tract_eff = gpd.read_file(tract_geojson)
crz = gpd.read_file(Path(paths["geo_root"]) / "nyc_cp_boundary_poly_json.geojson")
print(f"Loaded {MODE}/{MODEL}/{WINDOW}: {len(tract_eff)} tracts, {len(daily)} days")

if SAVE_FIGS:
    fig_dir = Path("outputs") / "figures" / f"{MODE}_{MODEL}_{WINDOW}{('_' + DIRECTION) if DIRECTION != 'all' else ''}"
    fig_dir.mkdir(parents=True, exist_ok=True)
else:
    fig_dir = None

# %% Figure 1 — Choropleth: mean daily ATT per tract
# NOTE on column choice: ``att`` in unit.csv / tract_effects.geojson is the
# *sum* of tau over the test window (≈116 days), not a daily rate. The true
# mean-daily ATT lives in ``avg_daily``. We use avg_daily here so the colour
# scale matches ridership-per-day in the ground truth.
fig, ax = plot_choropleth(
    tract_eff, column="avg_daily", crz_polygon=crz,
    cmap="RdBu_r", legend_label="Mean daily ATT (rides/day)",
    figsize=(11, 11),
)
ax.set_title(f"{MODE}/{DIRECTION}/{WINDOW} — mean daily ATT by tract  ({MODEL})", fontsize=13)
plt.show()
if fig_dir:
    fig.savefig(fig_dir / "choropleth_att.png", dpi=200, bbox_inches="tight")

# %% Figure 2 — Choropleth: cumulative relative effect (% of counterfactual)
if "cum_relative_effect" in tract_eff.columns:
    fig, ax = plot_choropleth(
        tract_eff, column="cum_relative_effect", crz_polygon=crz,
        cmap="RdBu_r", legend_label="Cumulative ATT / counterfactual",
        figsize=(11, 11),
    )
    ax.set_title(f"{MODE}/{DIRECTION}/{WINDOW} — relative cumulative ATT  ({MODEL})", fontsize=13)
    plt.show()
    if fig_dir:
        fig.savefig(fig_dir / "choropleth_relative.png", dpi=200, bbox_inches="tight")

# %% Figure 2b — Per-unit effects (bus polylines / subway points). Skip for citibike.
if MODE in ("bus", "subway"):
    geo_root = Path(paths["geo_root"])
    unit_csv = out_dir / "effects" / f"{prefix}_unit.csv"
    unit_df = pd.read_csv(unit_csv)

    if MODE == "bus":
        units_gdf = gpd.read_file(geo_root / "bus_routes" / "bus_routes_nyc_dec2019.shp")
        join_col = "route_id"
        line_width = 1.8
        point_size = 30.0
        size_by_abs = False
    else:  # subway
        units_gdf = gpd.read_file(geo_root / "MTA_Subway_Stations_20251029.geojson")
        join_col = "station_id"
        line_width = 2.0
        point_size = 35.0
        size_by_abs = True  # bigger dot = bigger |ATT|

    # Use ``avg_daily`` (mean daily ATT) — ``att`` is sum-over-window so its
    # scale would be ~116× the daily ridership we want to compare against.
    fig, ax = plot_unit_effects(
        units_gdf, unit_df, join_col=join_col, value_col="avg_daily",
        crz_polygon=crz, base_layer=tract_eff,
        cmap="RdBu_r", legend_label=f"Per-{join_col} mean daily ATT (rides/day)",
        line_width=line_width, point_size=point_size, point_size_by_abs=size_by_abs,
        figsize=(11, 11),
    )
    title = f"{MODE}/{DIRECTION}/{WINDOW} — per-{'route' if MODE=='bus' else 'station'} ATT  ({MODEL})"
    if MODE == "subway" and size_by_abs:
        title += "  (size ∝ |ATT|)"
    ax.set_title(title, fontsize=13)
    plt.show()
    if fig_dir:
        fig.savefig(fig_dir / "per_unit_att.png", dpi=200, bbox_inches="tight")

# %% Figure 3 — Calendar: significance per day
year = int(daily["date"].dt.year.mode().iloc[0])
months = sorted(daily["date"].dt.month.unique().tolist())
fig, _ = plot_significance_calendar(
    daily_for_calendar, year=year,
    start_month=int(months[0]), end_month=int(months[-1]),
)
fig.suptitle(f"{MODE}/{DIRECTION}/{WINDOW} — significance calendar  ({MODEL})", y=1.02, fontsize=13)
plt.show()
if fig_dir:
    fig.savefig(fig_dir / "significance_calendar.png", dpi=200, bbox_inches="tight")

# %% Figure 4 — Time series of daily ATT (with PI shading)
fig, ax = plot_effects_over_time(daily, mode="daily_att", title_prefix=f"{MODE.title()} ({MODEL})")
plt.show()
if fig_dir:
    fig.savefig(fig_dir / "daily_att.png", dpi=200, bbox_inches="tight")

# %% Figure 5 — Cumulative relative ATT
fig, ax = plot_effects_over_time(daily, mode="cum_rel", title_prefix=f"{MODE.title()} ({MODEL})")
plt.show()
if fig_dir:
    fig.savefig(fig_dir / "cum_rel_att.png", dpi=200, bbox_inches="tight")

print("Done." + (f" Figures saved to {fig_dir}" if fig_dir else ""))
