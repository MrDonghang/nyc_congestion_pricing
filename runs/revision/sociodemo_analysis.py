"""Policy effect vs socio-demographics (ACS 2023), with the same in-time placebo logic as the headline estimates.
Units: subway stations (800 m catchment), bus routes excl. Bronx local (400 m corridor), Replica origin tracts.
For each variable, units are split into five groups holding equal shares of baseline volume. Group effect = sum(actual) / sum(cf) - 1
over the group, MINUS the system-wide effect of the same origin (so a common forecast bias or calibration factor cancels).
Gradient = top group minus bottom group. Standardised slope = volume-weighted regression of log(actual / cf) on the z-scored
variable, alone and with distance to the zone as a control. Every statistic is recomputed for the 2024 pseudo-policy origins;
the real value is ranked against them and the 90 % interval uses their centred dispersion.
Writes uq/sociodemo_{quintiles,gradients}.csv
Usage: python runs/revision/sociodemo_analysis.py [--model timesfm]
"""
import argparse, glob
import numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="timesfm"); a = ap.parse_args()
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"
VARS = [("income_k", "Median household income"), ("pct_bachelor_plus", "Bachelor's degree or higher"), ("pct_wfh", "Work from home"), ("pct_car_commute", "Commute by car"), ("pct_transit_commute", "Commute by transit"),
        ("pop_density_k", "Population density"), ("pct_white", "White residents"), ("pct_black", "Black residents"), ("pct_hispanic", "Hispanic residents"), ("pct_asian", "Asian residents"), ("age_median", "Median age"), ("dist_km", "Distance to the zone")]
def load(mode, model):
    if mode == "subway":
        d = pd.concat([pd.read_csv(f, parse_dates=["origin"], dtype={"unit": str}) for f in glob.glob(str(U / f"denseunit_subway_ns22_{model}*.csv"))]); d = d[d.h == 116]
        g = pd.read_csv(U / "unit_geo_subway_ns.csv", dtype={"unit": str}); d = d[d.unit.isin(g[~g.yoy_closed].unit)]; pol, end = pd.Timestamp("2025-01-05"), lambda o: o + pd.Timedelta(days=115) < pd.Timestamp("2024-08-29")
    elif mode == "bus":
        d = pd.concat([pd.read_csv(f, parse_dates=["origin"], dtype={"unit": str}) for f in glob.glob(str(U / f"denseunit_bus_ns_{model}*.csv"))]); d = d[(d.h == 116) & ~d.unit.str.match(r"^BX[0-9]")]
        pol, end = pd.Timestamp("2025-01-05"), lambda o: o + pd.Timedelta(days=115) < pd.Timestamp("2024-08-29")
    else:
        d = pd.read_csv(U / f"denseunitwk_replica_{model}.csv", parse_dates=["origin"], dtype={"unit": str}); d = d[d.h == (8 if mode == "replica8" else 17)]
        if mode == "replica_out": d = d[d.unit.isin(pd.read_csv(U / "sociodemo_replica.csv", dtype={"unit": str}).query("dist_km > 0").unit)]      # tracts outside the zone only
        pol, end = pd.Timestamp("2025-01-04"), lambda o: o + pd.Timedelta(weeks=16) < pd.Timestamp("2024-09-07")
    S = pd.read_csv(U / f"sociodemo_{ {'replica8': 'replica', 'replica_out': 'replica'}.get(mode, mode)}.csv", dtype={"unit": str}).set_index("unit"); d = d[d.unit.isin(S.index) & (d.cf > 0)]
    origins = pd.DatetimeIndex(sorted(d.origin.unique())); nul = [o for o in origins if o >= pd.Timestamp("2024-01-01") and end(o)]
    post = [o for o in origins if pd.Timestamp("2024-09-28") <= o < pol] if mode == "replica8" else []
    # wide no-policy set: every origin with >= 18 months of history whose window contains no policy and no known break
    H = pd.Timedelta(days=115) if mode in ("subway", "bus") else pd.Timedelta(weeks=7 if mode == "replica8" else 16); stop = {"subway": pol, "bus": pd.Timestamp("2024-08-29")}.get(mode, pd.Timestamp("2024-09-07"))
    wide = [o for o in origins if o >= pd.Timestamp("2023-07-08") and o + H < stop]
    return d, S, pol, nul, post, wide
def wq(x, w, n=5):
    """volume-weighted quantile groups 0..n-1"""
    o = np.argsort(x.to_numpy(), kind="stable"); c = np.cumsum(w.to_numpy()[o]) / w.sum(); g = np.empty(len(x), int); g[o] = np.minimum((c * n - 1e-9).astype(int), n - 1); return pd.Series(g, index=x.index)
def wls(y, X, w):
    Xd = np.column_stack([np.ones(len(y)), X]); sw = np.sqrt(w); return np.linalg.lstsq(Xd * sw[:, None], y * sw, rcond=None)[0][1]
Q, Gr = [], []
for mode, mlab in (("subway", "Subway stations"), ("bus", "Bus routes"), ("replica", "Replica origin tracts"), ("replica_out", "Replica origin tracts outside the zone"), ("replica8", "Replica, first 8 weeks")):
    try: d, S, pol, nul, post, wide = load(mode, a.model)
    except FileNotFoundError as e: print("skip", mode, e); continue
    base = d[d.origin == pol].set_index("unit").cf; base = base[base > base.sum() * 2e-5]      # drop negligible units
    A = d.pivot(index="unit", columns="origin", values="actual").reindex(base.index); C = d.pivot(index="unit", columns="origin", values="cf").reindex(base.index); use = sorted(set([pol] + nul + post + wide))
    print(f"\n=== {mlab} ({a.model}): units {len(base)}, placebo origins {len(nul)}, wide no-policy set {len(wide)}" + (f", post-step origins {len(post)}" if post else "") + f", system effect {100 * (A[pol].sum() / C[pol].sum() - 1):+.2f}%")
    Y = np.log((A[use] / C[use]).clip(0.5, 2.0)); sysd = 100 * (A[use].sum() / C[use].sum() - 1); zd = (S.dist_km.reindex(base.index) - np.average(S.dist_km.reindex(base.index), weights=base)) ; zd = zd / zd.std()
    for v, lab in VARS:
        x = S[v].reindex(base.index).astype(float); ok = x.notna() & Y[pol].notna(); x = x[ok]; w = base[ok]; g = wq(x, w); z = (x - np.average(x, weights=w)); z = z / np.sqrt(np.average(z ** 2, weights=w))
        E = pd.DataFrame({q: 100 * (A.loc[g.index[g == q], use].sum() / C.loc[g.index[g == q], use].sum() - 1) - sysd for q in range(5)})          # origin × group, deviation from system
        for q in range(5):
            c = E.loc[nul, q] - E.loc[nul, q].mean(); Q.append(dict(mode=mlab, variable=lab, group=q + 1, lo=x[g == q].min(), hi=x[g == q].max(), units=int((g == q).sum()), dev=E.loc[pol, q], placebo_mean=E.loc[nul, q].mean(), placebo_sd=E.loc[nul, q].std(),
                                                                    ci_lo=E.loc[pol, q] - c.quantile(0.95), ci_hi=E.loc[pol, q] - c.quantile(0.05)))
        stats = {"gradient_pp": E[4] - E[0]}
        yy = Y.loc[x.index].fillna(0); stats["slope_pct_per_sd"] = pd.Series({o: 100 * wls(yy[o].to_numpy(), z.to_numpy()[:, None], w.to_numpy()) for o in use})
        yr = np.log((A.loc[x.index, use] / C.loc[x.index, use]).clip(np.exp(-0.2), np.exp(0.2))).fillna(0); stats["slope_robust"] = pd.Series({o: 100 * wls(yr[o].to_numpy(), z.to_numpy()[:, None], w.to_numpy()) for o in use})   # unit effects clipped at about ±20 %
        if v != "dist_km": stats["slope_ctrl_distance"] = pd.Series({o: 100 * wls(yy[o].to_numpy(), np.column_stack([z.to_numpy(), zd[x.index].to_numpy()]), w.to_numpy()) for o in use})
        for k, s in stats.items():
            n0 = s[nul]; c = n0 - n0.mean(); r = dict(mode=mlab, variable=lab, stat=k, real=s[pol], placebo_mean=n0.mean(), placebo_sd=n0.std(), placebo_min=n0.min(), placebo_max=n0.max(), n=len(n0), n_abs_ge=int((n0.abs() >= abs(s[pol])).sum()),
                                              n_centred_abs_ge=int((c.abs() >= abs(s[pol] - n0.mean())).sum()), ci_lo=s[pol] - c.quantile(0.95), ci_hi=s[pol] - c.quantile(0.05),
                                              wide_n=len(wide), wide_mean=s[wide].mean(), wide_sd=s[wide].std(), wide_min=s[wide].min(), wide_max=s[wide].max(), wide_abs_ge=int((s[wide].abs() >= abs(s[pol])).sum()))
            if post: r.update(post_mean=s[post].mean(), post_min=s[post].min(), post_max=s[post].max(), n_post=len(post))
            Gr.append(r)
Q, Gr = pd.DataFrame(Q), pd.DataFrame(Gr); Q.to_csv(U / f"sociodemo_quintiles_{a.model}.csv", index=False); Gr.to_csv(U / f"sociodemo_gradients_{a.model}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
for m, g in Gr.groupby("mode", sort=False):
    print("\n" + m)
    for k in ("gradient_pp", "slope_pct_per_sd", "slope_robust", "slope_ctrl_distance"):
        t = g[g.stat == k].drop(columns=["mode", "stat", "placebo_min", "placebo_max", "n_abs_ge"]); print(f"-- {k}"); print(t.round(2).to_string(index=False))
for m, g in Q.groupby("mode", sort=False):
    print("\nGroup deviations from the system effect (pp), " + m); print(g.pivot(index="variable", columns="group", values="dev").reindex([l for _, l in VARS]).round(2).to_string())
