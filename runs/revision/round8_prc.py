"""Round 8 — pooled rolling-origin calibration (PRC), system level.

Idea: learn the *structured, recurring* part of the forecast error (holiday, day of week, season, lead time) from ALL past
rolling-origin forecasts whose target dates are already observed, with a robust (Huber) fit, instead of from one
year-displaced window. One-off level shocks average out across origins instead of being baked into the correction.
For evaluation origin D0 the training rows are (origin, target date) pairs with target date < D0, excluding rows whose
forecast straddles the documented Sept-2024 break (issued before 29 Aug 2024, target on/after it).
Response: log(actual / cf). Variants:
  V0 raw
  V1 holiday + day-of-week                      (structure only; mean of the correction over a normal week forced to 0)
  V2 V1 + intercept + lead                      (adds the average level bias of the model at that lead)
  V3 V2 + target-month effects                  (adds a seasonal profile)
Scores: cumulative effect at h; evaluated on clean no-policy origins (weekly grid), then on the real policy.
Writes uq/round8_prc.csv, uq/round8_scores.csv, uq/round8_coefs.csv
"""
import glob, sys, warnings
import numpy as np, pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.linear_model import HuberRegressor
from nyc_cp.config import REPO_ROOT
warnings.filterwarnings("ignore")
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; POLICY = pd.Timestamp("2025-01-05"); B0, B1 = pd.Timestamp("2024-08-29"), pd.Timestamp("2024-09-08"); HS = (28, 56, 84, 116)
hol = set(USFederalHolidayCalendar().holidays("2021-12-01", "2026-12-31"))
def load(mode, model):
    d = pd.concat([pd.read_csv(f, parse_dates=["origin", "date"]) for f in glob.glob(str(U / f"dense_{mode}_{model}*.csv"))]).sort_values(["origin", "h"]).reset_index(drop=True)
    d["y"] = np.log(d.actual / d.cf); d["hol"] = d.date.isin(hol).astype(float); d["dow"] = d.date.dt.dayofweek; d["mon"] = d.date.dt.month; d["lead"] = d.h / 116.0
    d["straddle"] = (d.origin < B0) & (d.date >= B0)
    return d
def design(d, variant):
    X = [d[["hol"]].to_numpy()]
    dw = np.eye(7)[d.dow.to_numpy()]; X.append(dw[:, 1:] - 1 / 7 if variant == "V1" else dw[:, 1:])   # V1: centred so a normal week averages to ~0
    if variant in ("V2", "V3"): X.append(d[["lead"]].to_numpy())
    if variant == "V3": X.append(np.eye(12)[d.mon.to_numpy() - 1][:, 1:])
    return np.hstack(X)
rows, scores, coefs = [], [], []
for mode in sys.argv[1:] or ["subway_ns22", "bus_ns"]:
    for model in ("timesfm", "chronos_bolt"):
        d = load(mode, model); origins = pd.DatetimeIndex(sorted(d.origin.unique()))
        first_eval = origins.min() + pd.Timedelta(days=365)                           # need a year of past forecasts to learn from
        evals = [o for i, o in enumerate(origins) if o >= first_eval and (o == POLICY or i % 2 == 0)]
        for o in evals:
            te = d[d.origin == o]; tr = d[(d.date < o) & ~d.straddle]
            if len(tr) < 2000: continue
            out = {"V0": np.zeros(len(te))}
            for v in ("V1", "V2", "V3"):
                if v == "V3" and tr.mon.nunique() < 12: out[v] = np.full(len(te), np.nan); continue
                m = HuberRegressor(fit_intercept=(v != "V1"), alpha=1e-3, max_iter=500).fit(design(tr, v), tr.y.to_numpy())
                out[v] = m.predict(design(te, v))
                if o == POLICY: coefs.append(dict(panel=mode, model=model, variant=v, n_train=len(tr), intercept=float(m.intercept_) if v != "V1" else 0.0, holiday=m.coef_[0],
                                                  **{f"dow{k}": m.coef_[k] for k in range(1, 7)}, lead=m.coef_[7] if v != "V1" else np.nan))
            for v, corr in out.items():
                cfc = te.cf.to_numpy() * np.exp(corr)
                for H in HS:
                    if te.h.max() < H or np.isnan(cfc[:H]).any(): continue
                    scores.append(dict(panel=mode, model=model, variant=v, h=H, origin=o, score=100 * (te.actual.to_numpy()[:H].sum() / cfc[:H].sum() - 1)))
S = pd.DataFrame(scores); S.to_csv(U / "round8_scores.csv", index=False); pd.DataFrame(coefs).to_csv(U / "round8_coefs.csv", index=False)
for (panel, model, v, H), g in S.groupby(["panel", "model", "variant", "h"]):
    s = g.set_index("origin").score; real = s.get(POLICY, np.nan); nul = s.drop(index=POLICY, errors="ignore")
    clean = nul[~((nul.index <= B1) & (nul.index + pd.Timedelta(days=H - 1) >= B0))]
    for lab, sub in {"clean": clean, "all": nul}.items():
        if len(sub) < 5: continue
        rows.append(dict(panel=panel, model=model, variant=v, h=H, null=lab, n=len(sub), first=str(sub.index.min().date()), mean=sub.mean(), sd=sub.std(), rms=np.sqrt((sub ** 2).mean()), min=sub.min(), max=sub.max(),
                         real=real, z=(real - sub.mean()) / sub.std(), n_ge=int((sub >= real).sum()), p_two=(1 + (sub.abs() >= abs(real)).sum()) / (len(sub) + 1)))
R = pd.DataFrame(rows); R.to_csv(U / "round8_prc.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print("=== NULL ONLY (clean origins): RMS by horizon ==="); print(R[R.null == "clean"].pivot_table(index=["panel", "model", "variant"], columns="h", values="rms").round(2).to_string())
print("\n=== NULL ONLY (clean), h=116 ==="); print(R[(R.null == "clean") & (R.h == 116)][["panel", "model", "variant", "n", "first", "mean", "sd", "rms", "min", "max"]].round(2).to_string(index=False))
print("\n=== coefficients learned for the real-policy origin (log points ×100) ==="); C = pd.DataFrame(coefs); print((C.set_index(["panel", "model", "variant"]).drop(columns="n_train") * 100).round(1).to_string())
print("\n=== REAL (clean null) ==="); print(R[R.null == "clean"][["panel", "model", "variant", "h", "real", "mean", "sd", "max", "z", "n_ge", "n", "p_two"]].round(2).to_string(index=False))
