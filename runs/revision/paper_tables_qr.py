"""Tables behind Fig. 1 (panels c-f) under the quantile-regression specification (qr_final.py): same schema as the paper_* tables of the Huber version.
  paper_daily_qr_<model>.csv         daily effect; 90 % interval from the model's native 10 % / 90 % quantile lines, calibrated by quantile regression at tau = 0.05 / 0.95 (effect = y / q - 1);
                                     for the 7-day mean the band keeps that shape and is rescaled so that it covers 90 % of the 7-day-mean errors on the pseudo-policy dates (factor kappa)
  paper_cumulative_qr_<model>.csv    cumulative additional riders; 90 % interval from the mean-centred pseudo-policy paths
  paper_placebo_paths_qr_<model>.csv cumulative effect paths, policy date and pseudo-policy dates
"""
import numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; POLICY, H = pd.Timestamp("2025-01-05"), 116; D = pd.read_csv(U / "qr_native_daily.csv", parse_dates=["origin", "date"]).rename(columns={"lo90": "lo", "hi90": "hi"})   # native model quantiles (qr_native.py)
LAB = {"subway": ("Subway entries", "Subway"), "bus": ("Bus (excl. Bronx local routes)", "Bus")}; MLAB = {"timesfm": "TimesFM-2.0", "chronos_bolt": "Chronos-Bolt"}; r7 = lambda x: pd.Series(np.asarray(x)).rolling(7, center=True, min_periods=4).mean().to_numpy()
for model in ("timesfm", "chronos_bolt"):
    daily, cum, paths = [], [], []
    for panel in ("subway", "bus"):
        W = {o: g.sort_values("h") for o, g in D[(D.panel == panel) & (D.model == model)].groupby("origin")}; real = W.pop(POLICY)
        def parts(g): e = 100 * (g.actual / g.cc - 1).to_numpy(); return e, e - 100 * (g.actual / g.hi - 1).to_numpy(), 100 * (g.actual / g.lo - 1).to_numpy() - e      # effect, lower and upper half-widths
        ratio = []
        for g in W.values(): e, dl, du = parts(g); m = r7(e); ratio.append(np.where(m < 0, -m / r7(du), m / r7(dl)))                                                     # no-policy 7-day-mean error relative to the 7-day-mean half-width on its side
        kappa = float(np.nanquantile(np.concatenate(ratio), 0.9)); e, dl, du = parts(real)
        daily.append(pd.DataFrame({"series": LAB[panel][0], "model": MLAB[model], "date": real.date.to_numpy(), "h": real.h.to_numpy(), "actual": real.actual.to_numpy(), "cfc": real.cc.to_numpy(), "eff": e, "ci_lo": e - dl, "ci_hi": e + du,
                                   "eff_7d": r7(e), "ci7_lo": r7(e) - kappa * r7(dl), "ci7_hi": r7(e) + kappa * r7(du), "kappa": kappa}))
        cA, cC = real.actual.cumsum().to_numpy(), real.cc.cumsum().to_numpy(); rel = cA / cC - 1; P = np.vstack([g.actual.cumsum().to_numpy() / g.cc.cumsum().to_numpy() - 1 for g in W.values()]); cen = P - P.mean(axis=0); q05, q95 = np.percentile(cen, 5, axis=0), np.percentile(cen, 95, axis=0)
        cum.append(pd.DataFrame({"series": LAB[panel][0], "model": MLAB[model], "date": real.date.to_numpy(), "h": real.h.to_numpy(), "cum_actual": cA, "cum_counterfactual": cC, "cum_effect": cA - cC, "ci_lo": (rel - q95) * cC, "ci_hi": (rel - q05) * cC, "cum_rel_pct": 100 * rel}))
        paths.append(pd.DataFrame({"series": LAB[panel][1], "day": np.arange(1, H + 1), "real": 100 * rel, **{f"placebo_{o:%Y-%m-%d}": 100 * P[i] for i, o in enumerate(W)}}))
        print(f"{model:13s} {panel:7s} kappa {kappa:.2f}  7-day band half-width {np.nanmean(kappa * (r7(dl) + r7(du)) / 2):.1f} pp  days with 7-day band above 0: {100 * np.nanmean(r7(e) - kappa * r7(dl) > 0):.0f}%  riders {(cA - cC)[-1] / 1e6:+.2f} M [{((rel - q95) * cC)[-1] / 1e6:+.2f}, {((rel - q05) * cC)[-1] / 1e6:+.2f}]")
    pd.concat(daily).to_csv(U / f"paper_daily_qr_{model}.csv", index=False); pd.concat(cum).to_csv(U / f"paper_cumulative_qr_{model}.csv", index=False); pd.concat(paths).to_csv(U / f"paper_placebo_paths_qr_{model}.csv", index=False)
