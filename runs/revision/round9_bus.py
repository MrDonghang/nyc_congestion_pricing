"""Round 9 — repairing the bus null. Per route group (Bronx local / serves CRZ / other / non-Bronx / all), raw and with the
holiday + day-of-week pooled calibration (V1, past-only). Null sets: clean pre-break by half-year of origin (to see the
drift), clean 2024, and post-break origins (issued after 8 Sep 2024, truncated horizons). Real policy last.
Writes uq/round9_bus.csv, uq/round9_scores.csv
"""
import glob, warnings
import numpy as np, pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.linear_model import HuberRegressor
from nyc_cp.config import REPO_ROOT
warnings.filterwarnings("ignore")
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; POLICY = pd.Timestamp("2025-01-05"); B0, B1 = pd.Timestamp("2024-08-29"), pd.Timestamp("2024-09-08"); HS = (28, 56, 84, 116)
hol = set(USFederalHolidayCalendar().holidays("2021-12-01", "2026-12-31"))
def X(d): dw = np.eye(7)[d.date.dt.dayofweek.to_numpy()]; return np.hstack([d.date.isin(hol).to_numpy(float)[:, None], dw[:, 1:] - 1 / 7])
def X2(d):   # V2: holiday + day of week + lead (intercept fitted)
    dw = np.eye(7)[d.date.dt.dayofweek.to_numpy()]; return np.hstack([d.date.isin(hol).to_numpy(float)[:, None], dw[:, 1:], (d.h.to_numpy() / 116.0)[:, None]])
def X3(d):   # V3: holiday + day of week + lead + target-month profile (intercept fitted)
    dw = np.eye(7)[d.date.dt.dayofweek.to_numpy()]; mo = np.eye(12)[d.date.dt.month.to_numpy() - 1]
    return np.hstack([d.date.isin(hol).to_numpy(float)[:, None], dw[:, 1:], (d.h.to_numpy() / 116.0)[:, None], mo[:, 1:]])
scores = []
for model in ("timesfm", "chronos_bolt"):
    g = pd.concat([pd.read_csv(f, parse_dates=["origin", "date"]) for f in glob.glob(str(U / f"densegrp_bus_ns_{model}*.csv"))])
    parts = {k: v for k, v in g.groupby("grp")}
    comb = lambda ks: pd.concat([parts[k] for k in ks]).groupby(["origin", "h", "date"], as_index=False)[["actual", "cf"]].sum()
    series = {"Bronx local": parts["bronx"], "serves CRZ": parts["crz"], "other": parts["other"], "non-Bronx": comb(["crz", "other"]), "all routes": comb(["bronx", "crz", "other"])}
    for name, d in series.items():
        d = d.sort_values(["origin", "h"]).reset_index(drop=True); d["y"] = np.log(d.actual / d.cf); d["straddle"] = (d.origin < B0) & (d.date >= B0)
        origins = pd.DatetimeIndex(sorted(d.origin.unique())); first_eval = origins.min() + pd.Timedelta(days=365)
        for i, o in enumerate(origins):
            if not (o == POLICY or i % 2 == 0): continue
            te = d[d.origin == o]; corr = {"raw": np.zeros(len(te))}
            tr = d[(d.date < o) & ~d.straddle]
            if o >= first_eval and len(tr) > 2000:
                corr["V1"] = HuberRegressor(fit_intercept=False, alpha=1e-3, max_iter=500).fit(X(tr), tr.y.to_numpy()).predict(X(te))
                corr["V2"] = HuberRegressor(fit_intercept=True, alpha=1e-3, max_iter=500).fit(X2(tr), tr.y.to_numpy()).predict(X2(te))
                if tr.date.dt.month.nunique() == 12: corr["V3"] = HuberRegressor(fit_intercept=True, alpha=1e-3, max_iter=500).fit(X3(tr), tr.y.to_numpy()).predict(X3(te))
            for v, c in corr.items():
                cfc = te.cf.to_numpy() * np.exp(c)
                for H in HS:
                    if te.h.max() >= H: scores.append(dict(model=model, group=name, variant=v, h=H, origin=o, score=100 * (te.actual.to_numpy()[:H].sum() / cfc[:H].sum() - 1)))
S = pd.DataFrame(scores); S.to_csv(U / "round9_scores.csv", index=False); rows = []
for (model, group, v, H), gdf in S.groupby(["model", "group", "variant", "h"]):
    s = gdf.set_index("origin").score; real = s.get(POLICY, np.nan); nul = s.drop(index=POLICY, errors="ignore"); nul = nul[nul.index >= "2023-07-01"]
    pre = nul[nul.index + pd.Timedelta(days=H - 1) < B0]; post = nul[nul.index > B1]
    sets = {"pre-break 2023H2": pre[pre.index < "2024-01-01"], "pre-break 2024": pre[pre.index >= "2024-01-01"], "pre-break all": pre, "post-break": post, "pre+post": pd.concat([pre, post])}
    for lab, sub in sets.items():
        if len(sub) < 4: continue
        rows.append(dict(model=model, group=group, variant=v, h=H, null=lab, n=len(sub), mean=sub.mean(), sd=sub.std(), rms=np.sqrt((sub ** 2).mean()), min=sub.min(), max=sub.max(), real=real,
                         z=(real - sub.mean()) / sub.std(), n_ge=int((sub >= real).sum()), p_two=(1 + (sub.abs() >= abs(real)).sum()) / (len(sub) + 1)))
R = pd.DataFrame(rows); R.to_csv(U / "round9_bus.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
print("=== NULL ONLY, h=116 (post-break has no 116-day windows) ==="); print(R[(R.h == 116)][["model", "group", "variant", "null", "n", "mean", "sd", "rms", "min", "max"]].round(2).to_string(index=False))
print("\n=== NULL ONLY, post-break origins, h=28/56/84 ==="); print(R[(R.null == "post-break")][["model", "group", "variant", "h", "n", "mean", "sd", "rms", "min", "max"]].round(2).to_string(index=False))
print("\n=== REAL ==="); print(R[R.null.isin(["pre-break 2024", "post-break", "pre+post"])][["model", "group", "variant", "h", "null", "n", "real", "mean", "sd", "max", "z", "n_ge", "p_two"]].round(2).to_string(index=False))
