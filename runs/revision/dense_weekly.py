"""Weekly rolling-origin replay for Replica (all trips by origin tract), raw foundation model.
One pseudo-origin per week; horizon 17 weeks (the real test window 2025-01-04 … 2025-04-26); pre-policy weeks only for pseudo-origins.
Stores weekly system sums and sums inside / outside the Congestion Relief Zone per origin.
Output: uq/densewk_replica_<model>.csv (origin, h, date, grp, actual, cf); uq/denseunitwk_replica_<model>.csv (origin, unit, 17-week sums)
Usage: CUDA_VISIBLE_DEVICES=8 python runs/revision/dense_weekly.py --model timesfm [--direction O]
"""
import argparse, logging
import geopandas as gpd, numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT, load_mode, load_model, load_paths
from nyc_cp.data import load_actual
from nyc_cp.models import build_forecaster
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--direction", default="O"); ap.add_argument("--min-context", type=int, default=52); a = ap.parse_args()
logging.basicConfig(level=logging.WARNING); H, POLICY = 17, pd.Timestamp("2025-01-04"); U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"
actual = load_actual("replica", direction=a.direction, mode_cfg=load_mode("replica"), paths=load_paths()); actual.columns = actual.columns.astype(str)
tr = gpd.read_file(REPO_ROOT / "geo_data" / "NYC_Census_Tracts_2020" / "NYC_Census_Tracts_2020.shp").to_crs(2263); idc = [c for c in tr.columns if c.lower() == "geoid"][0]
crz = gpd.read_file(REPO_ROOT / "geo_data" / "nyc_cp_boundary_poly_json.geojson").to_crs(2263).geometry.unary_union
tr["in_crz"] = tr.geometry.intersection(crz).area / tr.geometry.area > 0.5; grp = tr.set_index(tr[idc].astype(str)).in_crz.reindex(actual.columns).fillna(False).map({True: "inside", False: "outside"})
print("tracts:", actual.shape[1], "inside CRZ:", int((grp == "inside").sum()), flush=True)
cfg = load_model(a.model); cfg["freq"] = "W-SAT"; fc = build_forecaster(cfg)
weeks = actual.index; origins = [w for w in weeks[a.min_context:] if w <= POLICY - pd.Timedelta(weeks=2)] + [POLICY]; rows = []; urows = []
for i, D in enumerate(origins):
    fc.fit(actual.loc[actual.index < D], train_end=D - pd.Timedelta(weeks=1), prediction_length=H, actual=actual)
    mu = fc.predict(D, D + pd.Timedelta(weeks=H - 1), freq="W-SAT").mu; mu.columns = mu.columns.astype(str)
    last = actual.index.max() if D == POLICY else POLICY - pd.Timedelta(weeks=1); idx = mu.index[(mu.index <= last) & mu.index.isin(actual.index)]
    A, M = actual.loc[idx], mu.loc[idx, actual.columns]
    for hh in (8, H):   # tract-level cumulative sums at 8 and 17 weeks
        if len(idx) >= hh: urows.append(pd.DataFrame({"origin": D, "h": hh, "unit": actual.columns, "actual": A.iloc[:hh].sum().to_numpy(), "cf": M.iloc[:hh].sum().to_numpy()}))
    for g in ("inside", "outside", "all"):
        c = actual.columns if g == "all" else actual.columns[(grp == g).to_numpy()]
        rows.append(pd.DataFrame({"origin": D, "h": range(1, len(idx) + 1), "date": idx, "grp": g, "actual": A[c].sum(axis=1).to_numpy(), "cf": M[c].sum(axis=1).to_numpy()}))
    if i % 20 == 0 or D == POLICY: print(i + 1, len(origins), D.date(), f"cum={100 * (A.sum().sum() / M.sum().sum() - 1):+.2f}% (h={len(idx)})", flush=True)
sfx = "" if a.direction == "O" else f"_{a.direction}"      # origin-based files keep their names; destination-based get _D
pd.concat(rows).to_csv(U / f"densewk_replica{sfx}_{a.model}.csv", index=False); pd.concat(urows).to_csv(U / f"denseunitwk_replica{sfx}_{a.model}.csv", index=False); print("wrote")
