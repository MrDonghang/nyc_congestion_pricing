"""Weekly rolling-origin replay for Replica trips INTO the Congestion Relief Zone, by travel mode.
Source: data_processed/replica/mode/<mode>.csv (weekly OD-pair counts, typical weekday; destinations in Manhattan).
Pairs with a destination tract inside the zone are kept and summed by mode × origin county; each such series is forecast
17 weeks ahead from every weekly origin (raw foundation model), then summed to the mode.
Output: uq/densewk_modes_<model>.csv (origin, h, date, mode, county, actual, cf); uq/replica_mode_panel.csv
Usage: CUDA_VISIBLE_DEVICES=8 python runs/revision/dense_weekly_modes.py --model timesfm
"""
import argparse, logging
import geopandas as gpd, numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT, load_model, load_paths
from nyc_cp.models import build_forecaster
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--min-context", type=int, default=52); a = ap.parse_args()
logging.basicConfig(level=logging.WARNING); H, POLICY = 17, pd.Timestamp("2025-01-04"); U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; D = load_paths()["data_root"] + "/replica/mode/"
MODES = ["private_auto", "auto_passenger", "on_demand_auto", "public_transit", "walking", "biking", "other"]
CNAME = {"36005": "Bronx", "36047": "Brooklyn", "36061": "Manhattan", "36081": "Queens", "36085": "Staten Island"}
tr = gpd.read_file(REPO_ROOT / "geo_data" / "NYC_Census_Tracts_2020" / "NYC_Census_Tracts_2020.shp").to_crs(2263); idc = [c for c in tr.columns if c.lower() == "geoid"][0]
crz = gpd.read_file(REPO_ROOT / "geo_data" / "nyc_cp_boundary_poly_json.geojson").to_crs(2263).geometry.unary_union; inz = set(tr[tr.geometry.intersection(crz).area / tr.geometry.area > 0.5][idc].astype(str))
cols = {}
for m in MODES:
    x = pd.read_csv(D + m + ".csv", index_col=0, parse_dates=True); o = x.columns.str.split("_").str[0]; d = x.columns.str.split("_").str[1]
    keep = d.isin(inz) & ~o.isin(inz); x = x.loc[:, keep]; cty = pd.Index(o[keep].str[:5]).map(lambda c: CNAME.get(c, "Other"))
    g = x.T.groupby(cty.to_numpy()).sum().T
    for c in g.columns:
        if g[c].mean() >= 50: cols[f"{m}|{c}"] = g[c]
panel = pd.DataFrame(cols).astype(float); panel.to_csv(U / "replica_mode_panel.csv"); print("panel", panel.shape, panel.index.min().date(), panel.index.max().date(), flush=True)
print((panel.loc["2024"].mean().groupby(lambda s: s.split("|")[0]).sum() / 1e3).round(1).to_dict(), flush=True)
cfg = load_model(a.model); cfg["freq"] = "W-SAT"; fc = build_forecaster(cfg)
origins = [w for w in panel.index[a.min_context:] if w <= POLICY - pd.Timedelta(weeks=2)] + [POLICY]; rows = []
for i, O in enumerate(origins):
    fc.fit(panel.loc[panel.index < O], train_end=O - pd.Timedelta(weeks=1), prediction_length=H, actual=panel)
    mu = fc.predict(O, O + pd.Timedelta(weeks=H - 1), freq="W-SAT").mu
    last = panel.index.max() if O == POLICY else POLICY - pd.Timedelta(weeks=1); idx = mu.index[(mu.index <= last) & mu.index.isin(panel.index)]
    for c in panel.columns:
        m, cty = c.split("|"); rows.append(pd.DataFrame({"origin": O, "h": range(1, len(idx) + 1), "date": idx, "mode": m, "county": cty, "actual": panel.loc[idx, c].to_numpy(), "cf": mu.loc[idx, c].to_numpy()}))
    if i % 25 == 0 or O == POLICY: print(i + 1, len(origins), O.date(), flush=True)
pd.concat(rows).to_csv(U / f"densewk_modes_{a.model}.csv", index=False); print("wrote")
