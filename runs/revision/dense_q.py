"""Dense rolling-origin replay keeping the models' NATIVE 10 % / 50 % / 90 % quantiles (no normal approximation), summed over units.
Same origins and point forecasts as dense_origins.py / dense_pi.py.  Output: uq/denseq_<mode>_<model>_<shard>.csv (origin, h, date, grp, actual, cf, q10, q50, q90)
Usage: CUDA_VISIBLE_DEVICES=0 python runs/revision/dense_q.py --mode subway_ns22 --direction O --model timesfm [--groups uq/bus_groups.csv] [--shard 0/2]
"""
import argparse, logging
import numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT, load_mode, load_model, load_paths
from nyc_cp.data import load_actual
from nyc_cp.models import build_forecaster
ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--direction", default="all"); ap.add_argument("--model", required=True); ap.add_argument("--groups", default=None); ap.add_argument("--step", type=int, default=3)
ap.add_argument("--min-context", type=int, default=180); ap.add_argument("--shard", default="0/1"); ap.add_argument("--clean-holidays", action="store_true", help="model input: replace holiday days by the mean of the same weekday in the 4 preceding non-holiday weeks")
ap.add_argument("--context-days", type=int, default=0, help="fixed context length (0 = all history)"); ap.add_argument("--tag", default="")
ap.add_argument("--units", action="store_true", help="also write cumulative unit totals at 28/56/84/116 days -> denseunitq_<mode>_<model>_<shard>.csv"); ap.add_argument("--eval-only", action="store_true", help="policy date and the 21 pseudo-policy dates only"); a = ap.parse_args(); logging.basicConfig(level=logging.WARNING)
H, POLICY = 116, pd.Timestamp("2025-01-05"); actual = load_actual(a.mode, direction=a.direction, mode_cfg=load_mode(a.mode), paths=load_paths()); actual.columns = actual.columns.astype(str)
hist_all = actual
if a.clean_holidays:      # causal: every replacement uses earlier days only
    from pandas.tseries.holiday import USFederalHolidayCalendar
    ix = actual.index; fed = USFederalHolidayCalendar().holidays("2019-01-01", "2026-12-31"); tg = fed[(fed.month == 11) & (fed.day >= 22) & (fed.dayofweek == 3)] + pd.Timedelta(days=1)
    m = np.asarray(ix.isin(fed) | ix.isin(tg) | ((ix.month == 12) & (ix.day >= 24)) | ((ix.month == 1) & (ix.day == 1))); An = actual.mask(pd.Series(m, index=ix), axis=0)
    fill = pd.concat([An.shift(7 * j) for j in range(1, 5)]).groupby(level=0).mean(); hist_all = actual.where(~pd.Series(m, index=ix), fill.reindex(ix), axis=0).fillna(actual); print("holiday days replaced in the model input:", int(m.sum()), flush=True)
cfg = load_model(a.model); cfg.setdefault("freq", "D"); fc = build_forecaster(cfg); cap = {}
if cfg["type"] == "timesfm":
    f0 = fc_forecast = None
def hook():
    if cfg["type"] == "timesfm":
        f0 = fc._tfm.forecast
        def f1(*x, **k): out = f0(*x, **k); cap["q"] = np.asarray(out[1]); return out
        fc._tfm.forecast = f1; return lambda: setattr(fc._tfm, "forecast", f0)
    p0 = fc._pipeline.predict_df
    def p1(*x, **k): k["quantile_levels"] = [0.1, 0.5, 0.9]; cap["df"] = p0(*x, **k); return cap["df"]
    fc._pipeline.predict_df = p1; return lambda: setattr(fc._pipeline, "predict_df", p0)
origins = list(pd.date_range(actual.index.min() + pd.Timedelta(days=a.min_context), POLICY - pd.Timedelta(days=14), freq=f"{a.step}D")) + [POLICY]; origins = [d for i, d in enumerate(origins) if d == POLICY or (i % 2 == 0 and d >= pd.Timestamp("2024-01-01") and d + pd.Timedelta(days=H - 1) < pd.Timestamp("2024-08-29"))] if a.eval_only else origins; urows = []
k, n = map(int, a.shard.split("/")); origins = [d for i, d in enumerate(origins) if i % n == k]
out = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq" / f"denseq{a.tag}_{a.mode}_{a.model}_{k}.csv"; gmap = pd.read_csv(a.groups, dtype={"unit": str}).set_index("unit").grp if a.groups else None; rows = []
for i, D in enumerate(origins):
    hist = hist_all.loc[hist_all.index < D]; hist = hist.iloc[-a.context_days:] if a.context_days else hist; fc.fit(hist, train_end=D - pd.Timedelta(days=1), prediction_length=H, actual=actual); undo = hook(); r = fc.predict(D, D + pd.Timedelta(days=H - 1), freq="D"); undo(); mu = r.mu.set_axis(r.mu.columns.astype(str), axis=1)
    if cfg["type"] == "timesfm": Q = {q: pd.DataFrame(cap["q"][:, :len(mu), j].T, index=mu.index, columns=mu.columns) for q, j in (("q10", 1), ("q50", 5), ("q90", 9))}
    else:
        df = cap["df"]; Q = {}
        for q, c in (("q10", "0.1"), ("q50", "0.5"), ("q90", "0.9")): t = df.pivot(index="timestamp", columns="item_id", values=c); t.index = pd.to_datetime(t.index); t.columns = t.columns.astype(str); Q[q] = t.reindex(mu.index).reindex(columns=mu.columns)
    last = actual.index.max() if D == POLICY else POLICY - pd.Timedelta(days=1); idx = mu.index[(mu.index <= last) & mu.index.isin(actual.index)]; cols = actual.columns.intersection(mu.columns)
    g = (gmap.reindex(cols).fillna("other") if gmap is not None else pd.Series("all", index=cols)).to_numpy()
    for grp in np.unique(g):
        c = cols[g == grp]; rows.append(pd.DataFrame({"origin": D, "h": range(1, len(idx) + 1), "date": idx, "grp": grp, "actual": actual.loc[idx, c].sum(axis=1).to_numpy(), "cf": mu.loc[idx, c].sum(axis=1).to_numpy(), **{q: Q[q].loc[idx, c].sum(axis=1).to_numpy() for q in Q}}))
    if a.units:
        Au, Mu = actual.loc[idx, cols].cumsum(), mu.loc[idx, cols].cumsum()
        for h in (28, 56, 84, 116):
            if len(idx) >= h: urows.append(pd.DataFrame({"origin": D, "h": h, "unit": cols, "actual": Au.iloc[h - 1].to_numpy(), "cf": Mu.iloc[h - 1].to_numpy()}))
    if i % 20 == 0 or D == POLICY: pd.concat(rows).to_csv(out, index=False); print(f"{i + 1}/{len(origins)} {D.date()}", flush=True)
pd.concat(rows).to_csv(out, index=False); print("wrote", out)
if a.units: pd.concat(urows).to_csv(out.with_name(f"denseunitq_{a.mode}_{a.model}_{k}.csv"), index=False)
