"""Original experiment with quantile-regression calibration and the models' own predictive quantiles (user request 2026-11-26).
Per origin D (policy date and the 21 pseudo-policy dates of 2024), system level, both models:
  model output    point forecast c, and the model's 5 % / 95 % quantiles (median -/+ 1.645 sigma, sigma summed over units; densepi_*.csv)
  calibration     quantile regression on forecasts with target date before D, design = holiday, day of week, lead, target month (seasonal profile):
                  tau = 0.5 of log(y / c) -> calibrated counterfactual;  tau = 0.05 of log(y / q05) and tau = 0.95 of log(y / q95) -> calibrated daily 90 % band
  effect          daily y / cf - 1;  cumulative sum(y) / sum(cf) - 1 at 28 / 56 / 84 / 116 days
  inference       rank of the policy estimate among the pseudo-policy dates; 90 % interval from the mean-centred placebo quantiles
Writes uq/qr_final_daily.csv, uq/qr_final_scores.csv, uq/qr_final_summary.csv
"""
import glob, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.linear_model import QuantileRegressor
from nyc_cp.config import REPO_ROOT
warnings.filterwarnings("ignore"); U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; POLICY, B0, H, HS, Z = pd.Timestamp("2025-01-05"), pd.Timestamp("2024-08-29"), 116, (28, 56, 84, 116), 1.645
hol = set(USFederalHolidayCalendar().holidays("2021-12-01", "2026-12-31"))
def load(panel, model):
    g = pd.concat([pd.read_csv(f, parse_dates=["origin", "date"]) for f in glob.glob(str(U / f"densepi_{'subway_ns22' if panel == 'subway' else 'bus_ns'}_{model}_*.csv"))]); g = g[g.grp != "bronx"]
    d = g.groupby(["origin", "h", "date"], as_index=False)[["actual", "cf", "med", "sig_sum"]].sum(); d = d[d.h <= H].sort_values(["origin", "h"]).reset_index(drop=True)
    d["q05"] = np.maximum(d.med - Z * d.sig_sum, 0.05 * d.med); d["q95"] = d.med + Z * d.sig_sum; d["straddle"] = (d.origin < B0) & (d.date >= B0); return d
def X3(d): return np.hstack([d.date.isin(hol).to_numpy(float)[:, None], np.eye(7)[d.date.dt.dayofweek.to_numpy()][:, 1:], (d.h.to_numpy() / 116.0)[:, None], np.eye(12)[d.date.dt.month.to_numpy() - 1][:, 1:]])
def one(panel, model, d, o):
    te = d[d.origin == o].copy(); tr = d[(d.date < o) & ~d.straddle]; Xtr, Xte = X3(tr), X3(te)
    for col, base, q in (("cc", "cf", 0.5), ("lo", "q05", 0.05), ("hi", "q95", 0.95)):
        te[col] = te[base].to_numpy() * np.exp(QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(Xtr, np.log(tr.actual.to_numpy() / tr[base].to_numpy())).predict(Xte))
    te["panel"], te["model"] = panel, model; return te[["panel", "model", "origin", "h", "date", "actual", "cf", "q05", "q95", "cc", "lo", "hi"]]
if __name__ == "__main__":
    tasks = []
    for panel in ("subway", "bus"):
        for model in ("timesfm", "chronos_bolt"):
            d = load(panel, model); origins = pd.DatetimeIndex(sorted(d.origin.unique()))
            tasks += [(panel, model, d, o) for i, o in enumerate(origins) if o == POLICY or (i % 2 == 0 and o >= pd.Timestamp("2024-01-01") and o + pd.Timedelta(days=H - 1) < B0)]
    D = pd.concat(Parallel(n_jobs=44)(delayed(one)(*t) for t in tasks)); D.to_csv(U / "qr_final_daily.csv", index=False); sc, out = [], []
    for (panel, model, o), g in D.groupby(["panel", "model", "origin"], sort=False):
        g = g.sort_values("h"); a, c = g.actual.to_numpy(), g.cc.to_numpy(); r = dict(panel=panel, model=model, origin=o, raw116=100 * (a.sum() / g.cf.sum() - 1), cover_cal=float(((a >= g.lo) & (a <= g.hi)).mean()), cover_raw=float(((a >= g.q05) & (a <= g.q95)).mean()),
                                                                                   above=float((a > g.hi).mean()), below=float((a < g.lo).mean()), halfwidth=100 * float(((g.hi - g.lo) / 2 / g.cc).mean()), halfwidth_raw=100 * float(((g.q95 - g.q05) / 2 / g.cf).mean()))
        for h in HS: r[f"e{h}"] = 100 * (a[:h].sum() / c[:h].sum() - 1)
        sc.append(r)
    S = pd.DataFrame(sc); S.to_csv(U / "qr_final_scores.csv", index=False)
    for (panel, model), g in S.groupby(["panel", "model"], sort=False):
        real = g[g.origin == POLICY].iloc[0]; nul = g[g.origin != POLICY]
        for h in HS:
            x = nul[f"e{h}"]; e = real[f"e{h}"]; cen = x - x.mean()
            out.append(dict(panel=panel, model=model, h=h, estimate=e, ci_lo=e - np.quantile(cen, 0.95), ci_hi=e - np.quantile(cen, 0.05), placebo_mean=x.mean(), placebo_sd=x.std(ddof=1), n_placebo=len(x), n_ge=int((x >= e).sum()), z=(e - x.mean()) / x.std(ddof=1),
                            raw_estimate=real.raw116 if h == 116 else np.nan, band_cover_placebo=nul.cover_cal.mean(), model_band_cover_placebo=nul.cover_raw.mean(), band_halfwidth=nul.halfwidth.mean(), model_band_halfwidth=nul.halfwidth_raw.mean(),
                            policy_days_above=real.above, policy_days_below=real.below, placebo_days_above=nul.above.mean(), placebo_days_below=nul.below.mean()))
    R = pd.DataFrame(out); R.to_csv(U / "qr_final_summary.csv", index=False); pd.set_option("display.width", 260)
    print(R[["panel", "model", "h", "estimate", "ci_lo", "ci_hi", "placebo_mean", "placebo_sd", "n_ge", "z"]].round(2).to_string(index=False))
    print(R[R.h == 116][["panel", "model", "raw_estimate", "band_cover_placebo", "model_band_cover_placebo", "band_halfwidth", "model_band_halfwidth", "policy_days_above", "policy_days_below", "placebo_days_above", "placebo_days_below"]].round(2).to_string(index=False))
