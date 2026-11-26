"""Synthetic check of the main specification (Chronos-Bolt, median quantile-regression calibration with holiday, day of week, lead and target month; placebo inference).
Uses the Bolt replays cached by synth_calib.py (3 data-generating processes x 2 history lengths x 20 series; no policy in any series). Effects are injected after the fact.
Per series: estimate at the policy date, 21 pseudo-policy dates in 2024, 'detected' = estimate above all pseudo-policy dates, 90 % interval from the mean-centred pseudo-policy quantiles.
Writes uq/synth_bolt_qr_series.csv, uq/synth_bolt_qr_summary.csv
"""
import warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.linear_model import QuantileRegressor
from nyc_cp.config import REPO_ROOT
warnings.filterwarnings("ignore"); U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; C = U / "synth_calib"; POLICY, B0, H = pd.Timestamp("2025-01-05"), pd.Timestamp("2024-08-29"), 116
HOL = set(USFederalHolidayCalendar().holidays("2018-12-01", "2026-12-31")); TAUS = (0.0, 1.0, 2.6, 3.5, 5.0)
def one(cf, origins, y, idx, ev):
    oi, hi = np.nonzero(~np.isnan(cf)); org = origins[oi]; date = org + pd.to_timedelta(hi, unit="D"); keep = (date < POLICY) | (org == POLICY); oi, hi, date = oi[keep], hi[keep], date[keep]
    c = cf[oi, hi]; a = pd.Series(y, index=idx).reindex(date).to_numpy(); ly = np.log(a / c); X = np.hstack([date.isin(HOL).astype(float)[:, None], np.eye(7)[date.dayofweek][:, 1:], ((hi + 1.0) / 116.0)[:, None], np.eye(12)[date.month - 1][:, 1:]]); out = {}
    for i in ev:
        o = origins[i]; te = oi == i; tr = date < o; k = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(X[tr], ly[tr]).predict(X[te]); out[o] = (100 * (a[te].sum() / c[te].sum() - 1), 100 * (a[te].sum() / (c[te] * np.exp(k)).sum() - 1))
    return out
rows = []
for f in sorted(C.glob("cf_chronos_bolt_*.npz")):
    _, _, dgp, start = f.stem[3:].rsplit("_", 2)[0], *f.stem[3:].rsplit("_", 2); z = np.load(f); origins, idx = pd.DatetimeIndex(z["origins"]), pd.DatetimeIndex(z["idx"])
    ev = [i for i, o in enumerate(origins) if o == POLICY or (i % 2 == 0 and o >= pd.Timestamp("2024-01-01") and o + pd.Timedelta(days=H - 1) < B0)]
    res = Parallel(n_jobs=20)(delayed(one)(z["cf"][:, :, j], origins, z["Y"][:, j], idx, ev) for j in range(z["Y"].shape[1]))
    for j, out in enumerate(res):
        for vi, v in enumerate(("no calibration", "main specification")):
            e0 = out[POLICY][vi]; nul = np.array([x[vi] for o, x in out.items() if o != POLICY]); cen = nul - nul.mean(); q95, q05 = np.quantile(cen, 0.95), np.quantile(cen, 0.05)
            for tau in TAUS:
                et = 100 * ((1 + e0 / 100) * (1 + tau / 100) - 1)
                rows.append(dict(dgp=dgp, start=start, series=j, variant=v, tau=tau, estimate=et, placebo_mean=nul.mean(), placebo_sd=nul.std(ddof=1), placebo_max=nul.max(), detected=et > nul.max(), ci_lo=et - q95, ci_hi=et - q05, covered=(et - q95 <= tau) & (et - q05 >= tau), ci_excludes_0=et - q95 > 0))
    print(f.stem, flush=True)
R = pd.DataFrame(rows); R.to_csv(U / "synth_bolt_qr_series.csv", index=False)
def summ(g): return pd.Series(dict(n=len(g), mean_estimate=g.estimate.mean(), bias=(g.estimate - g.tau).mean(), sd=g.estimate.std(), placebo_mean=g.placebo_mean.mean(), placebo_sd=g.placebo_sd.mean(), detected=g.detected.mean(), ci_excludes_0=g.ci_excludes_0.mean(), coverage=g.covered.mean()))
S = pd.concat([R.groupby(["variant", "dgp", "start", "tau"]).apply(summ).reset_index(), R.groupby(["variant", "tau"]).apply(summ).reset_index().assign(dgp="all", start="all")]); S.to_csv(U / "synth_bolt_qr_summary.csv", index=False)
pd.set_option("display.width", 220); print(S[S.dgp == "all"].drop(columns=["dgp", "start"]).round(2).to_string(index=False)); print(S[(S.dgp != "all") & S.tau.isin([0.0, 2.6]) & (S.variant == "main specification")].round(2).to_string(index=False))
