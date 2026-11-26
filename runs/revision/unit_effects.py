"""Station-level (subway) and route-level (bus) effect tables for a given model; same definitions as station_level.py / route_level.py
(those two also draw the TimesFM-only diagnostic figures). Subway: the system-level seasonal calibration is passed down as one
scaling factor per origin. Bus: raw route forecast (the calibration factor is applied by the figure / analysis scripts).
Writes uq/station_effects_subway_<model>.csv, uq/route_effects_bus_<model>.csv
Usage: python runs/revision/unit_effects.py chronos_bolt
"""
import glob, sys
import geopandas as gpd, pandas as pd
from nyc_cp.config import REPO_ROOT
model = sys.argv[1]; U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; POLICY, B0 = pd.Timestamp("2025-01-05"), pd.Timestamp("2024-08-29")
rd = lambda pat: pd.concat([pd.read_csv(f, parse_dates=["origin"], dtype={"unit": str}) for f in glob.glob(str(U / pat))])
def table(d, col, vol, extra):
    real = d[d.origin == POLICY].set_index("unit"); nul = d[(d.origin >= "2024-01-01") & (d.origin + pd.Timedelta(days=115) < B0)]; g = nul.groupby("unit").s.agg(placebo_mean="mean", placebo_sd="std", placebo_max="max", n="size")
    T = pd.DataFrame({"effect_pct": real.s, vol: real[col] / 116, extra: (real.actual - real[col]) / 116}).join(g); mx = nul.pivot_table(index="unit", columns="origin", values="s")
    T["z"] = (T.effect_pct - T.placebo_mean) / T.placebo_sd; T["n_ge"] = (mx.reindex(T.index).ge(T.effect_pct, axis=0)).sum(axis=1).astype(int); return T
qd = pd.read_csv(U / "qr_native_daily.csv", parse_dates=["origin"]); qd = qd[(qd.model == model) & (qd.panel == "subway")]; k = qd.groupby("origin").apply(lambda g: g.cc.sum() / g.cf.sum())      # QR (median) calibration factor per origin
d = rd(f"denseunitq_subway_ns22_{model}*.csv"); d = d[(d.h == 116) & d.origin.isin(k.index)]; d["cfc"] = d.cf * d.origin.map(k); d["s"] = 100 * (d.actual / d.cfc - 1)
geo = pd.read_csv(U / "unit_geo_subway_ns.csv", dtype={"unit": str}).set_index("unit"); geo = geo[~geo.yoy_closed]
T = table(d, "cfc", "entries_per_day_cf", "extra_entries_per_day").join(geo[["stop_name", "borough", "in_crz", "dist_km"]], how="inner"); T.index.name = "station_index"
T = T[["effect_pct", "entries_per_day_cf", "extra_entries_per_day", "placebo_mean", "placebo_sd", "placebo_max", "n", "stop_name", "borough", "in_crz", "dist_km", "z", "n_ge"]]; T.sort_values("effect_pct", ascending=False).to_csv(U / f"station_effects_subway_{model}.csv")
print(f"{model} subway: stations {len(T)}, volume-weighted effect {100 * T.extra_entries_per_day.sum() / T.entries_per_day_cf.sum():+.2f}%, median {T.effect_pct.median():+.1f}, above all own placebos {100 * (T.n_ge == 0).mean():.0f}%")
d = rd(f"denseunitq_bus_ns_{model}*.csv"); d = d[d.h == 116]; d["s"] = 100 * (d.actual / d.cf - 1); geo = pd.read_csv(U / "unit_geo_bus_ns.csv", dtype={"unit": str}).set_index("unit")
names = gpd.read_file(REPO_ROOT / "geo_data" / "bus_routes" / "bus_routes_nyc_dec2019.shp")[["route_id", "route_long"]].drop_duplicates("route_id").set_index("route_id").route_long
R = table(d, "cf", "riders_per_day_cf", "extra_riders_per_day").join(geo[["prefix", "express", "share_in_crz", "touches_crz"]]); R["effect_minus_placebo_mean"] = R.effect_pct - R.placebo_mean
R["name"] = [names.get(u, names.get(u.replace("+", "-SBS"), "")) for u in R.index]; R["bronx_local"] = R.prefix == "BX"; R.index.name = "route"
R = R[["effect_pct", "riders_per_day_cf", "extra_riders_per_day", "placebo_mean", "placebo_sd", "placebo_max", "n", "prefix", "express", "share_in_crz", "touches_crz", "z", "n_ge", "effect_minus_placebo_mean", "name", "bronx_local"]]
R.sort_values("effect_pct", ascending=False).to_csv(U / f"route_effects_bus_{model}.csv"); nb = R[~R.bronx_local]
print(f"{model} bus: routes {len(R)}, non-Bronx raw volume-weighted effect {100 * nb.extra_riders_per_day.sum() / nb.riders_per_day_cf.sum():+.2f}%, median {nb.effect_pct.median():+.1f}")
