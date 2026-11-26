"""Shared loader for the socio-demographic figures: per-origin standardised slopes of the unit-level effect on each variable.
Definitions identical to sociodemo_analysis.py (statistic `slope_pct_per_sd`)."""
import glob
import numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"
VARS = [("income_k", "Median household income"), ("pct_bachelor_plus", "Bachelor's degree or higher"), ("pct_wfh", "Work from home"), ("pct_car_commute", "Commute by car"), ("pct_transit_commute", "Commute by transit"),
        ("pop_density_k", "Population density"), ("pct_white", "White residents"), ("pct_black", "Black residents"), ("pct_hispanic", "Hispanic residents"), ("pct_asian", "Asian residents"), ("age_median", "Median age"), ("dist_km", "Distance to the zone")]
def wls(y, x, w): X = np.column_stack([np.ones(len(y)), x]); sw = np.sqrt(w); return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0][1]
def load(mode, model):
    if mode in ("subway", "bus"):
        d = pd.concat([pd.read_csv(f, parse_dates=["origin"], dtype={"unit": str}) for f in glob.glob(str(U / f"denseunit_{'subway_ns22' if mode == 'subway' else 'bus_ns'}_{model}*.csv"))]); d = d[d.h == 116]
        if mode == "subway": g = pd.read_csv(U / "unit_geo_subway_ns.csv", dtype={"unit": str}); d = d[d.unit.isin(g[~g.yoy_closed].unit)]
        else: d = d[~d.unit.str.match(r"^BX[0-9]")]
        pol, H, stop_n, stop_w = pd.Timestamp("2025-01-05"), pd.Timedelta(days=115), pd.Timestamp("2024-08-29"), (pd.Timestamp("2025-01-05") if mode == "subway" else pd.Timestamp("2024-08-29"))
    else:
        d = pd.read_csv(U / f"denseunitwk_replica_{model}.csv", parse_dates=["origin"], dtype={"unit": str}); d = d[d.h == 17]; pol, H, stop_n, stop_w = pd.Timestamp("2025-01-04"), pd.Timedelta(weeks=16), pd.Timestamp("2024-09-07"), pd.Timestamp("2024-09-07")
    S = pd.read_csv(U / f"sociodemo_{mode}.csv", dtype={"unit": str}).set_index("unit"); d = d[d.unit.isin(S.index) & (d.cf > 0)]; origins = pd.DatetimeIndex(sorted(d.origin.unique()))
    nul = [o for o in origins if o >= pd.Timestamp("2024-01-01") and o + H < stop_n]; wide = [o for o in origins if o >= pd.Timestamp("2023-07-08") and o + H < stop_w]
    return d, S, pol, nul, wide
def slopes(mode, model):
    d, S, pol, nul, wide = load(mode, model); base = d[d.origin == pol].set_index("unit").cf; base = base[base > base.sum() * 2e-5]; use = sorted(set([pol] + nul + wide))
    A = d.pivot(index="unit", columns="origin", values="actual").reindex(base.index)[use]; C = d.pivot(index="unit", columns="origin", values="cf").reindex(base.index)[use]; Y = np.log((A / C).clip(0.5, 2.0)); out = {}
    for v, _ in VARS:
        x = S[v].reindex(base.index).astype(float); ok = x.notna() & Y[pol].notna(); x = x[ok]; w = base[ok]; z = x - np.average(x, weights=w); z = z / np.sqrt(np.average(z ** 2, weights=w)); yy = Y.loc[x.index].fillna(0)
        out[v] = pd.Series({o: 100 * wls(yy[o].to_numpy(), z.to_numpy(), w.to_numpy()) for o in use})
    return dict(slope=out, pol=pol, nul=nul, wide=wide, A=A, C=C, S=S, base=base)
