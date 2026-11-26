"""Quantile-regression calibration on the models' NATIVE quantiles (denseq_*.csv; no normal approximation). System level, both models.
Per origin D (policy date + 21 pseudo-policy dates), fitted on forecasts with target date before D; design = holiday, day of week, lead, target month:
  tau 0.50 of log(y / point forecast)  -> calibrated counterfactual (identical to qr_final.py)
  tau 0.05 / 0.95 of log(y / q10), log(y / q90) -> calibrated 90 % band;   tau 0.10 / 0.90 of the same -> calibrated 80 % band
Writes uq/qr_native_daily.csv, uq/qr_native_summary.csv
"""
import glob, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.linear_model import QuantileRegressor
from nyc_cp.config import REPO_ROOT
warnings.filterwarnings("ignore"); U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; POLICY, B0, H = pd.Timestamp("2025-01-05"), pd.Timestamp("2024-08-29"), 116; hol = set(USFederalHolidayCalendar().holidays("2021-12-01", "2026-12-31"))
def load(panel, model):
    g = pd.concat([pd.read_csv(f, parse_dates=["origin", "date"]) for f in glob.glob(str(U / f"denseq_{'subway_ns22' if panel == 'subway' else 'bus_ns'}_{model}_*.csv"))]); g = g[g.grp != "bronx"]
    d = g.groupby(["origin", "h", "date"], as_index=False)[["actual", "cf", "q10", "q50", "q90"]].sum(); d = d[d.h <= H].sort_values(["origin", "h"]).reset_index(drop=True); d["straddle"] = (d.origin < B0) & (d.date >= B0); return d
def X3(d): return np.hstack([d.date.isin(hol).to_numpy(float)[:, None], np.eye(7)[d.date.dt.dayofweek.to_numpy()][:, 1:], (d.h.to_numpy() / 116.0)[:, None], np.eye(12)[d.date.dt.month.to_numpy() - 1][:, 1:]])
def one(panel, model, d, o):
    te = d[d.origin == o].copy(); tr = d[(d.date < o) & ~d.straddle]; Xtr, Xte = X3(tr), X3(te)
    for col, base, q in (("cc", "cf", 0.5), ("lo90", "q10", 0.05), ("hi90", "q90", 0.95), ("lo80", "q10", 0.10), ("hi80", "q90", 0.90)):
        te[col] = te[base].to_numpy() * np.exp(QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(Xtr, np.log(tr.actual.to_numpy() / tr[base].to_numpy())).predict(Xte))
    te["panel"], te["model"] = panel, model; return te.drop(columns="straddle")
if __name__ == "__main__":
    tasks = []
    for panel in ("subway", "bus"):
        for model in ("timesfm", "chronos_bolt"):
            d = load(panel, model); origins = pd.DatetimeIndex(sorted(d.origin.unique())); tasks += [(panel, model, d, o) for i, o in enumerate(origins) if o == POLICY or (i % 2 == 0 and o >= pd.Timestamp("2024-01-01") and o + pd.Timedelta(days=H - 1) < B0)]
    D = pd.concat(Parallel(n_jobs=44)(delayed(one)(*t) for t in tasks)); D.to_csv(U / "qr_native_daily.csv", index=False); out = []
    for (panel, model), g in D.groupby(["panel", "model"], sort=False):
        pl, re = g[g.origin != POLICY], g[g.origin == POLICY]; r = dict(panel=panel, model=model, est116=100 * (re.actual.sum() / re.cc.sum() - 1), raw80_cover=float(((pl.actual >= pl.q10) & (pl.actual <= pl.q90)).mean()))
        for L in ("90", "80"):
            lo, hi = f"lo{L}", f"hi{L}"; r[f"cover{L}"] = float(((pl.actual >= pl[lo]) & (pl.actual <= pl[hi])).mean()); r[f"below{L}"] = float((pl.actual < pl[lo]).mean()); r[f"above{L}"] = float((pl.actual > pl[hi]).mean())
            r[f"hw{L}_dn"] = 100 * float(((pl.cc - pl[lo]) / pl.cc).median()); r[f"hw{L}_up"] = 100 * float(((pl[hi] - pl.cc) / pl.cc).median()); r[f"cross{L}"] = float(((g[lo] > g.cc) | (g[hi] < g.cc)).mean())
            r[f"policy_above{L}"] = float((re.actual > re[hi]).mean()); r[f"policy_below{L}"] = float((re.actual < re[lo]).mean())
            pc = pl.groupby("origin").apply(lambda t: ((t.actual >= t[lo]) & (t.actual <= t[hi])).mean()); r[f"cover{L}_min_origin"], r[f"cover{L}_max_origin"] = float(pc.min()), float(pc.max())
        out.append(r)
    R = pd.DataFrame(out); R.to_csv(U / "qr_native_summary.csv", index=False); pd.set_option("display.width", 250); print(R.round(3).T.to_string())
    old = pd.read_csv(U / "qr_final_daily.csv", parse_dates=["origin", "date"]).merge(D, on=["panel", "model", "origin", "h"], suffixes=("_old", "")); print("max rel diff: point forecast", float((old.cf / old.cf_old - 1).abs().max()), " calibrated cf", float((old.cc / old.cc_old - 1).abs().max()), " rows", len(old), "of", len(D))
