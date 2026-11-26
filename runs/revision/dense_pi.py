"""Dense rolling-origin replay that also keeps the models' own predictive spread (user request 2026-11-26).
Same origins and forecasts as dense_origins.py. Per origin, day and route group it stores system sums of
  actual, cf (point forecast), med (model median), and the model spread aggregated over units two ways:
  sig_sum = sum of unit sigmas (units perfectly dependent), sig_rss = root of summed squared sigmas (units independent),
where sigma is the unit's (q90 - q10) / (2 * 1.2816) from the model's native 10 % / 90 % quantiles.
Output: uq/densepi_<mode>_<model>_<shard>.csv    Usage: CUDA_VISIBLE_DEVICES=0 python runs/revision/dense_pi.py --mode subway_ns22 --direction O --model timesfm [--groups uq/bus_groups.csv] [--shard 0/2]
"""
import argparse, logging
import numpy as np, pandas as pd
from scipy.stats import norm
from nyc_cp.config import REPO_ROOT, load_mode, load_model, load_paths
from nyc_cp.data import load_actual
from nyc_cp.models import build_forecaster
ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--direction", default="all"); ap.add_argument("--model", required=True); ap.add_argument("--groups", default=None); ap.add_argument("--step", type=int, default=3)
ap.add_argument("--min-context", type=int, default=180); ap.add_argument("--shard", default="0/1"); a = ap.parse_args(); logging.basicConfig(level=logging.WARNING)
H, POLICY = 116, pd.Timestamp("2025-01-05"); paths = load_paths(); actual = load_actual(a.mode, direction=a.direction, mode_cfg=load_mode(a.mode), paths=paths); actual.columns = actual.columns.astype(str)
cfg = load_model(a.model); cfg.setdefault("freq", "D"); fc = build_forecaster(cfg); zc = norm.ppf(1 - (1 - float(cfg.get("coverage_level", 0.9))) / 2)
origins = list(pd.date_range(actual.index.min() + pd.Timedelta(days=a.min_context), POLICY - pd.Timedelta(days=14), freq=f"{a.step}D")) + [POLICY]; k, n = map(int, a.shard.split("/")); origins = [d for i, d in enumerate(origins) if i % n == k]
out = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq" / f"densepi_{a.mode}_{a.model}_{k}.csv"; gmap = pd.read_csv(a.groups, dtype={"unit": str}).set_index("unit").grp if a.groups else None; rows = []
for i, D in enumerate(origins):
    fc.fit(actual.loc[actual.index < D], train_end=D - pd.Timedelta(days=1), prediction_length=H, actual=actual); r = fc.predict(D, D + pd.Timedelta(days=H - 1), freq="D")
    mu, lo, hi = (x.set_axis(x.columns.astype(str), axis=1) for x in (r.mu, r.lower, r.upper)); last = actual.index.max() if D == POLICY else POLICY - pd.Timedelta(days=1)
    idx = mu.index[(mu.index <= last) & mu.index.isin(actual.index)]; cols = actual.columns.intersection(mu.columns); g = (gmap.reindex(cols).fillna("other") if gmap is not None else pd.Series("all", index=cols)).to_numpy()
    A, M, Q, S = actual.loc[idx, cols], mu.loc[idx, cols], (lo.loc[idx, cols] + hi.loc[idx, cols]) / 2, (hi.loc[idx, cols] - lo.loc[idx, cols]) / (2 * zc)
    for grp in np.unique(g):
        c = cols[g == grp]; rows.append(pd.DataFrame({"origin": D, "h": range(1, len(idx) + 1), "date": idx, "grp": grp, "actual": A[c].sum(axis=1).to_numpy(), "cf": M[c].sum(axis=1).to_numpy(), "med": Q[c].sum(axis=1).to_numpy(),
                                                       "sig_sum": S[c].sum(axis=1).to_numpy(), "sig_rss": np.sqrt((S[c] ** 2).sum(axis=1)).to_numpy()}))
    if i % 20 == 0 or D == POLICY: pd.concat(rows).to_csv(out, index=False); print(f"{i + 1}/{len(origins)} {D.date()}", flush=True)
pd.concat(rows).to_csv(out, index=False); print("wrote", out)
