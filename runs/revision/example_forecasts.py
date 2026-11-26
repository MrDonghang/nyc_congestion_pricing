"""Example units for Fig. 1a,b: observed vs forecast with a CALIBRATED model interval, in a no-policy rehearsal window (same season one
year earlier) and in the policy window. All series are 7-day means.
Forecast: foundation-model point forecast x system-level quantile-regression (median) calibration factor of that origin (qr_native.py).
Interval: the model's NATIVE 10 % / 90 % quantiles (no normal approximation), each side rescaled by split-conformal calibration on windows that
  end before the origin: lower score = (forecast - observed) / (forecast - q10), upper score = (observed - forecast) / (q90 - forecast) (7-day means),
  pooled over the unit's calibration forecasts issued 116-240 days earlier; band = [forecast - q95(lower score) x (forecast - q10),
  forecast + q95(upper score) x (q90 - forecast)], i.e. a 90 % band that keeps the model's asymmetry. No pseudo-policy (placebo) information is used.
Step 1 (screen): every station >= 8,000 entries/day and every non-Bronx route >= 4,000 riders/day -> uq/example_screen_<model>.csv
Step 2: the chosen units (EX below) -> uq/example_forecasts_<model>.csv
Usage: CUDA_VISIBLE_DEVICES=9 python runs/revision/example_forecasts.py timesfm
"""
import sys, os, logging
import numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT, load_mode, load_model, load_paths
from nyc_cp.data import load_actual
from nyc_cp.models import build_forecaster
model = sys.argv[1]; logging.basicConfig(level=logging.WARNING); U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; H, POLICY = 116, pd.Timestamp("2025-01-05"); paths = load_paths()
EX = {"subway": ("220", "Northern Blvd station"), "bus": ("B11", "B11 bus route")}      # 2026-11-27: user asked for a clearer station; screen widened to >= 2,500 entries/day (EX_MIN_SUBWAY), Northern Blvd chosen (Rockaway Blvd skipped: A-train Rockaway closure from Jan 2025). Before, 145 St (119). 2026-11-26: Chronos-Bolt is the main model; Houston St (267) no longer passes the screen under Bolt, 145 St does (Bolt and TimesFM)      # overwritten after screening, see ITERATION_LOG
if len(sys.argv) > 3: EX = {"subway": (sys.argv[2], ""), "bus": (sys.argv[3], "")}
qd = pd.read_csv(U / "qr_native_daily.csv", parse_dates=["origin"]); qd = qd[qd.model == model]      # system-level quantile-regression calibration (qr_native.py)
K = {p: qd[qd.panel == p].groupby("origin").apply(lambda g: g.cc.sum() / g.cf.sum()) for p in ("subway", "bus")}      # calibrated cf / raw cf, per origin
cfg = load_model(model); cfg.setdefault("freq", "D"); fc = build_forecaster(cfg); r7 = lambda x: x.rolling(7, center=True).mean(); OUT, SCR = [], []
for kind, mode, direction in (("subway", "subway_ns22", "O"), ("bus", "bus_ns", "all")):
    a = load_actual(mode, direction=direction, mode_cfg=load_mode(mode), paths=paths).astype(float); a.columns = a.columns.astype(str); k = K[kind]
    vol = a.loc["2024-09-15":"2024-12-15"].mean(); keep = vol[vol >= (float(os.environ.get("EX_MIN_SUBWAY", 8000)) if kind == "subway" else 4000)].index
    if kind == "subway": g = pd.read_csv(U / "unit_geo_subway_ns.csv", dtype={"unit": str}); keep = keep.intersection(g[~g.yoy_closed].unit)
    else: keep = keep[~keep.str.match(r"^BX[0-9]")]
    a = a[keep]; val = k.index[abs(k.index - pd.Timestamp("2024-01-05")).argmin()]
    def run(D):      # -> observed, calibrated forecast, and the model's NATIVE lower / upper spread (forecast - q10, q90 - forecast), all per unit
        fc.fit(a.loc[a.index < D], train_end=D - pd.Timedelta(days=1), prediction_length=H, actual=a); cap = {}
        if cfg["type"] == "timesfm":
            f0 = fc._tfm.forecast
            def f1(*x, **kw): out = f0(*x, **kw); cap["q"] = np.asarray(out[1]); return out
            fc._tfm.forecast = f1; r = fc.predict(D, D + pd.Timedelta(days=H - 1), freq="D"); fc._tfm.forecast = f0
            Q = {q: pd.DataFrame(cap["q"][:, :len(r.mu), jq].T, index=r.mu.index, columns=r.mu.columns) for q, jq in (("q10", 1), ("q50", 5), ("q90", 9))}
            assert np.allclose(Q["q50"].to_numpy(), ((r.lower + r.upper) / 2).to_numpy(), rtol=1e-4, atol=1e-2), "quantile / unit order mismatch"
        else:
            p0 = fc._pipeline.predict_df
            def p1(*x, **kw): kw["quantile_levels"] = [0.1, 0.5, 0.9]; cap["df"] = p0(*x, **kw); return cap["df"]
            fc._pipeline.predict_df = p1; r = fc.predict(D, D + pd.Timedelta(days=H - 1), freq="D"); fc._pipeline.predict_df = p0; Q = {}
            for q, c in (("q10", "0.1"), ("q90", "0.9")): t = cap["df"].pivot(index="timestamp", columns="item_id", values=c); t.index = pd.to_datetime(t.index); t.columns = t.columns.astype(str); Q[q] = t.reindex(r.mu.index).reindex(columns=r.mu.columns.astype(str)).set_axis(r.mu.columns, axis=1)
        idx = r.mu.index[r.mu.index.isin(a.index)]; kD = k.iloc[abs(k.index - D).argmin()]      # calibration factor of the nearest pseudo-policy grid date (factors exist from Jan 2024; earlier calibration windows use the first one)
        M = r.mu.loc[idx, a.columns] * kD; return a.loc[idx], M, (M - Q["q10"].loc[idx, a.columns] * kD).clip(lower=0.01 * M), (Q["q90"].loc[idx, a.columns] * kD - M).clip(lower=0.01 * M)
    res = {}
    for window, D in (("rehearsal", val), ("policy", POLICY)):
        cal = list(pd.date_range(D - pd.Timedelta(days=242), D - pd.Timedelta(days=H), freq="18D")); S = []
        E = []
        SU = []
        for o in cal: A, M, DN, UP = run(o); S.append((r7(M) - r7(A)) / r7(DN)); SU.append((r7(A) - r7(M)) / r7(UP)); E.append(100 * (A.sum() / M.sum() - 1))
        ch = pd.concat(E, axis=1).abs().quantile(0.9, axis=1)      # calibrated 90 % half-width of the 116-day total: q90 of |window-total error| over the calibration windows
        ql, qu = pd.concat(S).quantile(0.95).clip(lower=0.05), pd.concat(SU).quantile(0.95).clip(lower=0.05); q = (ql + qu) / 2      # each side scaled so that 5 % of the calibration days fall outside it
        A, M, DN, UP = run(D); obs, cf = r7(A), r7(M); lo, hi = cf - r7(DN) * ql, cf + r7(UP) * qu; half = (hi - lo) / 2; ok = cf.notna()
        res[window] = dict(D=D, A=A, M=M, lo=lo, hi=hi, q=q, cumhalf=ch, eff=100 * (A.sum() / M.sum() - 1), above=100 * ((obs > hi) & ok).sum() / ok.sum(), below=100 * ((obs < lo) & ok).sum() / ok.sum(), width=100 * (half / cf).mean(), ncal=len(cal))
        print(f"{model} {kind} {window}: origin {D.date()}, calibration origins {len(cal)}, median conformal factor {q.median():.2f}, median band half-width {res[window]['width'].median():.1f}% of forecast", flush=True)
    T = pd.DataFrame({f"{w}_{c}": res[w][c] for w in res for c in ("eff", "cumhalf", "above", "below", "width")}); T["vol"] = vol[keep]; T["mode"] = kind; T.index.name = "unit"; SCR.append(T.reset_index())
    u = EX[kind][0]
    for w in res:
        r = res[w]; D = r["D"]; ctx = load_actual(mode, direction=direction, mode_cfg=load_mode(mode), paths=paths).astype(float); ctx.columns = ctx.columns.astype(str); ctx = ctx.loc[(ctx.index >= D - pd.Timedelta(days=38)) & (ctx.index < D), u]
        OUT.append(pd.DataFrame({"mode": kind, "unit": u, "name": EX[kind][1], "window": w, "origin": D, "date": ctx.index, "actual": ctx.to_numpy()}))
        OUT.append(pd.DataFrame({"mode": kind, "unit": u, "name": EX[kind][1], "window": w, "origin": D, "date": r["A"].index, "actual": r["A"][u].to_numpy(), "cf": r["M"][u].to_numpy(), "lower7": r["lo"][u].to_numpy(), "upper7": r["hi"][u].to_numpy(),
                                 "effect_pct": r["eff"][u], "effect_halfwidth": r["cumhalf"][u], "days_above_pct": r["above"][u], "days_below_pct": r["below"][u], "conformal_factor": r["q"][u]}))
pd.concat(SCR).to_csv(U / f"example_screen_{model}.csv", index=False); pd.concat(OUT).to_csv(U / f"example_forecasts_{model}.csv", index=False); print("wrote")
