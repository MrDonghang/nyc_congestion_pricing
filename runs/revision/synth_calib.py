"""Synthetic comparison of calibration methods under the placebo framework (truth known: no policy effect in the series).
Stage 1 (GPU):  python runs/revision/synth_calib.py replay <model>     -> uq/synth_calib/cf_<model>_<dgp>_<start>.npz
Stage 2 (CPU):  python runs/revision/synth_calib.py eval               -> uq/synth_calib_series.csv, uq/synth_calib_summary.csv
DGPs (daily, <start> .. 2025-04-30, 20 series each):
  simple  shared growth 3 %/yr, weekly pattern, annual cycle, holidays; AR(1) noise (sd 3 %, rho 0.5) differs by series
  varied  growth, seasonal amplitude / phase, weekly pattern, noise sd and persistence drawn per series
  rw      simple + random-walk log level (daily sd 0.2 %)
Framework as in the paper: 116-day window, policy date 2025-01-05, 21 pseudo-policy dates in Jan-May 2024 (6-day grid),
calibration fitted on forecast errors with target date before the origin only. Effects are injected after the fact
(counterfactuals never see post-origin data), so one replay serves every effect size.
"""
import sys, logging, warnings
import numpy as np, pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.linear_model import HuberRegressor
from nyc_cp.config import REPO_ROOT
warnings.filterwarnings("ignore"); logging.basicConfig(level=logging.WARNING)
NS, H = 20, 116; POLICY, B0, END = pd.Timestamp("2025-01-05"), pd.Timestamp("2024-08-29"), "2025-04-30"
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; C = U / "synth_calib"; C.mkdir(exist_ok=True)
HOL = set(USFederalHolidayCalendar().holidays("2018-12-01", "2026-12-31")); DGPS, STARTS, TAUS = ("simple", "varied", "rw"), ("2022-01-01", "2020-01-01"), (2.7, 5.0)


def generate(dgp, start):
    idx = pd.date_range(start, END); n = len(idx); rng = np.random.default_rng({"simple": 7, "varied": 11, "rw": 13}[dgp]); t = np.arange(n) / 365.25; doy = idx.dayofyear.to_numpy()
    hd = np.where(idx.isin(HOL), 0.65, 1.0) * np.where(((idx.month == 12) & (idx.day >= 24)) | ((idx.month == 1) & (idx.day == 1)), 0.8, 1.0); dow0 = np.array([1.00, 1.04, 1.05, 1.04, 0.98, 0.62, 0.50]); Y = {}
    for s in range(NS):
        if dgp == "varied":
            g, a1, p1, a2, sd, rho = rng.uniform(0, 0.06), rng.uniform(0.02, 0.08), rng.uniform(0, 365), rng.uniform(0, 0.03), rng.uniform(0.02, 0.05), rng.uniform(0.3, 0.8); dw = dow0 * np.exp(rng.normal(0, 0.05, 7))
        else: g, a1, p1, a2, sd, rho, dw = 0.03, 0.04, 100.0, 0.02, 0.03, 0.5, dow0
        seas = a1 * np.sin(2 * np.pi * (doy - p1) / 365.25) + a2 * np.cos(4 * np.pi * doy / 365.25); e = np.empty(n); e[0] = rng.normal(0, sd)
        for i in range(1, n): e[i] = rho * e[i - 1] + rng.normal(0, sd * np.sqrt(1 - rho ** 2))
        lvl = np.cumsum(rng.normal(0, 0.002, n)) if dgp == "rw" else 0.0
        Y[f"s{s}"] = 1e6 * np.exp(g * t + seas + e + lvl) * dw[idx.dayofweek] * hd
    return pd.DataFrame(Y, index=idx)


def replay(model):
    from nyc_cp.config import load_model
    from nyc_cp.models import build_forecaster
    cfg = load_model(model); cfg.setdefault("freq", "D"); fc = build_forecaster(cfg)
    for dgp in DGPS:
        for start in STARTS:
            Y = generate(dgp, start); idx = Y.index
            origins = pd.DatetimeIndex(list(pd.date_range(idx[0] + pd.Timedelta(days=180), POLICY - pd.Timedelta(days=14), freq="3D")) + [POLICY]); cf = np.full((len(origins), H, NS), np.nan)
            for i, D in enumerate(origins):
                fc.fit(Y.loc[Y.index < D], train_end=D - pd.Timedelta(days=1), prediction_length=H, actual=Y); mu = fc.predict(D, D + pd.Timedelta(days=H - 1), freq="D").mu[Y.columns]; mu = mu.loc[mu.index.isin(idx)]; cf[i, :len(mu)] = mu.to_numpy()
            np.savez_compressed(C / f"cf_{model}_{dgp}_{start[:4]}.npz", cf=cf, origins=origins.values, Y=Y.to_numpy(), idx=idx.values); print(model, dgp, start, cf.shape, flush=True)


def four(d, k=2): return np.column_stack([f(2 * np.pi * (j + 1) * d / 365.25) for j in range(k) for f in (np.sin, np.cos)])


def one_series(cf, origins, y, idx, ev):
    """-> list of (method, {origin: window-total error in %}) for one series"""
    oi, hi = np.nonzero(~np.isnan(cf)); org = origins[oi]; date = org + pd.to_timedelta(hi, unit="D"); keep = (date < POLICY) | (org == POLICY); oi, hi, org, date = oi[keep], hi[keep], org[keep], date[keep]
    c = cf[oi, hi]; a = pd.Series(y, index=idx).reindex(date).to_numpy(); ly = np.log(a / c); h = (hi + 1.0) / 116.0; odoy = org.dayofyear.to_numpy(); tdoy = date.dayofyear.to_numpy(); last = org + pd.Timedelta(days=H - 1)
    base = np.hstack([date.isin(HOL).astype(float)[:, None], np.eye(7)[date.dayofweek][:, 1:], h[:, None]])
    F = {"dowhol": base, "V3": np.hstack([base, np.eye(12)[date.month - 1][:, 1:]]), "V3_fourier": np.hstack([base, four(tdoy)]), "V3_origin": np.hstack([base, np.eye(12)[date.month - 1][:, 1:], four(odoy, 1)]),
         "origin_season": np.hstack([base, four(odoy), four(odoy, 1) * h[:, None]])}
    out = {m: {} for m in ("raw", "level", "recent", "analog", "analog_shrunk", "V3_analog", *F)}
    hub = lambda X, tr, te: HuberRegressor(alpha=1e-3, max_iter=500).fit(X[tr], ly[tr]).predict(X[te])
    tot = lambda m, k=0.0: np.log(a[m].sum() / (c[m] * np.exp(k)).sum())
    for i in ev:
        o = origins[i]; te = oi == i; tr = date < o; done = last < o; d = np.abs(odoy - o.dayofyear); an = done & (np.minimum(d, 365 - d) <= 21) & (org < o - pd.Timedelta(days=200)); k = {"raw": np.zeros(te.sum())}
        k["level"] = np.full(te.sum(), tot(done)) if done.any() else k["raw"]; rec = done & (org >= o - pd.Timedelta(days=H + 180)); k["recent"] = np.full(te.sum(), tot(rec)) if rec.any() else k["level"]
        nw = len(np.unique(oi[an])); k["analog"] = np.full(te.sum(), tot(an)) if nw else k["level"]
        k["analog_shrunk"] = k["level"] + (nw / (nw + 14.0)) * (k["analog"] - k["level"])           # shrink the same-season correction towards the all-season level (14 windows = one season-year)
        for m, X in F.items(): k[m] = hub(X, tr, te)
        if nw:                                                                                        # V3, then the V3 residual of same-season windows of earlier years (V3 refitted on data before those windows end is approximated by the current fit)
            kv = HuberRegressor(alpha=1e-3, max_iter=500).fit(F["V3"][tr], ly[tr]); k["V3_analog"] = k["V3"] + tot(an, kv.predict(F["V3"][an]))
        else: k["V3_analog"] = k["V3"]
        for m in out: out[m][o] = 100 * (a[te].sum() / (c[te] * np.exp(k[m])).sum() - 1)
    return out


def evaluate():
    from joblib import Parallel, delayed
    rows = []
    for f in sorted(C.glob("cf_*.npz")):
        model, dgp, start = f.stem[3:].rsplit("_", 2); z = np.load(f); origins, idx = pd.DatetimeIndex(z["origins"]), pd.DatetimeIndex(z["idx"])
        ev = [i for i, o in enumerate(origins) if o == POLICY or (i % 2 == 0 and o >= pd.Timestamp("2024-01-01") and o + pd.Timedelta(days=H - 1) < B0)]
        res = Parallel(n_jobs=NS)(delayed(one_series)(z["cf"][:, :, j], origins, z["Y"][:, j], idx, ev) for j in range(NS))
        for j, out in enumerate(res):
            for m, sc in out.items():
                nul = np.array([v for o, v in sc.items() if o != POLICY]); e0 = sc[POLICY]; cen = nul - nul.mean(); q95, q05 = np.quantile(cen, 0.95), np.quantile(cen, 0.05)
                r = dict(model=model, dgp=dgp, start=start, method=m, series=j, est0=e0, null_mean=nul.mean(), null_sd=nul.std(ddof=1), null_max=nul.max(), n_null=len(nul), fp_max=e0 > nul.max(), fp_ci=(e0 - q95 > 0) | (e0 - q05 < 0))
                for tau in TAUS:
                    et = 100 * ((1 + e0 / 100) * (1 + tau / 100) - 1); r[f"est{tau}"] = et; r[f"det{tau}"] = et > nul.max(); r[f"cov{tau}"] = (et - q95 <= tau) & (et - q05 >= tau)
                rows.append(r)
        print(f.stem, flush=True)
    R = pd.DataFrame(rows); R.to_csv(U / "synth_calib_series.csv", index=False)
    S = R.groupby(["model", "dgp", "start", "method"]).agg(bias=("est0", "mean"), rmse=("est0", lambda x: np.sqrt((x ** 2).mean())), null_mean=("null_mean", "mean"), null_sd=("null_sd", "mean"), fp_max=("fp_max", "mean"), fp_ci=("fp_ci", "mean"),
                                                             det27=("det2.7", "mean"), det5=("det5.0", "mean"), cov27=("cov2.7", "mean")).reset_index(); S["gap"] = S.bias - S.null_mean
    S.to_csv(U / "synth_calib_summary.csv", index=False); return S


if __name__ == "__main__":
    if sys.argv[1] == "replay": replay(sys.argv[2])
    else:
        S = evaluate(); pd.set_option("display.width", 250)
        print(S.groupby(["model", "method"])[["bias", "rmse", "null_mean", "null_sd", "gap", "fp_max", "fp_ci", "det27", "det5", "cov27"]].mean().round(2).to_string())
