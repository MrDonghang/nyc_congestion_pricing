"""Dense rolling-origin replay (raw foundation model, no calibration): one pseudo-origin every --step days.

For each origin D: history = data ≤ D−1, forecast D … D+H−1 (H = 116), evaluated only on pre-policy days (≤ 2025-01-04),
so origins after 2024-09-11 have a truncated horizon. The real origin 2025-01-05 is appended (full horizon).
Stores system-level daily sums (actual, counterfactual) per origin → cumulative effect at any horizon h.
Output: outputs/_summary/revision/uq/dense_<mode>_<model>[_shard].csv   (origin, h, date, actual, cf, n_units)
Usage: CUDA_VISIBLE_DEVICES=8 python runs/revision/dense_origins.py --mode subway_ns --direction O --model timesfm [--step 3] [--shard 0/2]
"""
from __future__ import annotations
import argparse, logging
import pandas as pd
from nyc_cp.config import REPO_ROOT, load_mode, load_model, load_paths
from nyc_cp.data import load_actual
from nyc_cp.models import build_forecaster
ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--direction", default="all"); ap.add_argument("--model", required=True)
ap.add_argument("--groups", default=None, help="csv with columns unit,grp: also store daily sums per group"); ap.add_argument("--step", type=int, default=3); ap.add_argument("--min-context", type=int, default=180); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
logging.basicConfig(level=logging.WARNING)
H, POLICY = 116, pd.Timestamp("2025-01-05")
paths = load_paths(); actual = load_actual(a.mode, direction=a.direction, mode_cfg=load_mode(a.mode), paths=paths); actual.columns = actual.columns.astype(str)
cfg = load_model(a.model); cfg.setdefault("freq", "D"); fc = build_forecaster(cfg)
first = actual.index.min() + pd.Timedelta(days=a.min_context)
origins = list(pd.date_range(first, POLICY - pd.Timedelta(days=14), freq=f"{a.step}D")) + [POLICY]
k, n = map(int, a.shard.split("/")); origins = [d for i, d in enumerate(origins) if i % n == k]
out = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq" / f"dense_{a.mode}_{a.model}{'' if n == 1 else f'_{k}'}.csv"
rows, urows, grows = [], [], []
gmap = pd.read_csv(a.groups, dtype={"unit": str}).set_index("unit").grp if a.groups else None
uout = out.with_name(out.name.replace("dense_", "denseunit_"))
for i, D in enumerate(origins):
    hist = actual.loc[actual.index < D]
    fc.fit(hist, train_end=D - pd.Timedelta(days=1), prediction_length=H, actual=actual)
    mu = fc.predict(D, D + pd.Timedelta(days=H - 1), freq="D").mu; mu.columns = mu.columns.astype(str)
    last = actual.index.max() if D == POLICY else POLICY - pd.Timedelta(days=1)
    idx = mu.index[(mu.index <= last) & mu.index.isin(actual.index)]; cols = actual.columns.intersection(mu.columns)
    A, M = actual.loc[idx, cols].sum(axis=1), mu.loc[idx, cols].sum(axis=1)
    rows.append(pd.DataFrame({"origin": D, "h": range(1, len(idx) + 1), "date": idx, "actual": A.to_numpy(), "cf": M.to_numpy(), "n_units": len(cols)}))
    if gmap is not None:
        g = gmap.reindex(cols); Ag = actual.loc[idx, cols].T.groupby(g.to_numpy()).sum().T; Mg = mu.loc[idx, cols].T.groupby(g.to_numpy()).sum().T
        for k in Ag.columns: grows.append(pd.DataFrame({"origin": D, "h": range(1, len(idx) + 1), "date": idx, "grp": k, "actual": Ag[k].to_numpy(), "cf": Mg[k].to_numpy()}))
    Au, Mu = actual.loc[idx, cols].cumsum(), mu.loc[idx, cols].cumsum()
    for h in (28, 56, 84, 116):
        if len(idx) >= h: urows.append(pd.DataFrame({"origin": D, "h": h, "unit": cols, "actual": Au.iloc[h - 1].to_numpy(), "cf": Mu.iloc[h - 1].to_numpy()}))
    if i % 10 == 0 or D == POLICY:
        pd.concat(rows).to_csv(out, index=False); print(f"{i + 1}/{len(origins)} {D.date()} cum={100 * (A.sum() / M.sum() - 1):+.2f}% (h={len(idx)})", flush=True)
pd.concat(rows).to_csv(out, index=False); pd.concat(urows).to_csv(uout, index=False)
if grows: pd.concat(grows).to_csv(out.with_name(out.name.replace("dense_", "densegrp_")), index=False)
print("wrote", out)
