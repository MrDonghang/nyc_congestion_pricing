"""Route-level bus effects (TimesFM, student fares excluded, 116 days). Each route is compared with its own 2024
pseudo-policy scores. Route-level numbers use the uncalibrated forecast (the seasonal calibration is fitted at system level);
subtracting the route's own placebo mean plays the role of the bias correction.
Writes uq/route_effects_bus_timesfm.csv and figures route_top_nonbronx / route_top_bronx.
"""
import glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import geopandas as gpd, numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT, load_mode, load_paths
from nyc_cp.data import load_actual
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; FIG = REPO_ROOT / "outputs" / "figures" / "paper" / "robustness"; paths = load_paths()
POLICY, B0 = pd.Timestamp("2025-01-05"), pd.Timestamp("2024-08-29"); C_MAIN, C_CF, C_PREV, INK, MUTED = "#2a78d6", "#1f1f1f", "#b9b7ae", "#1f1f1f", "#5f5e5a"
d = pd.concat([pd.read_csv(f, parse_dates=["origin"], dtype={"unit": str}) for f in glob.glob(str(U / "denseunit_bus_ns_timesfm*.csv"))]); d = d[d.h == 116]
d["s"] = 100 * (d.actual / d.cf - 1)
real = d[d.origin == POLICY].set_index("unit"); nul = d[(d.origin >= "2024-01-01") & (d.origin + pd.Timedelta(days=115) < B0)]
g = nul.groupby("unit").s.agg(placebo_mean="mean", placebo_sd="std", placebo_max="max", n="size")
geo = pd.read_csv(U / "unit_geo_bus_ns.csv", dtype={"unit": str}).set_index("unit")
names = gpd.read_file(REPO_ROOT / "geo_data" / "bus_routes" / "bus_routes_nyc_dec2019.shp")[["route_id", "route_long"]].drop_duplicates("route_id").set_index("route_id").route_long
T = pd.DataFrame({"effect_pct": real.s, "riders_per_day_cf": real.cf / 116, "extra_riders_per_day": (real.actual - real.cf) / 116}).join(g).join(geo[["prefix", "express", "share_in_crz", "touches_crz"]])
T["z"] = (T.effect_pct - T.placebo_mean) / T.placebo_sd; T["n_ge"] = [int((nul[nul.unit == u].s >= T.effect_pct[u]).sum()) for u in T.index]
T["effect_minus_placebo_mean"] = T.effect_pct - T.placebo_mean; T["name"] = [names.get(u, names.get(u.replace("+", "-SBS"), "")) for u in T.index]
T["bronx_local"] = T.prefix == "BX"; T.index.name = "route"; T.sort_values("effect_pct", ascending=False).to_csv(U / "route_effects_bus_timesfm.csv")
big = T[T.riders_per_day_cf >= 2000]
pd.set_option("display.width", 230); cols = ["name", "riders_per_day_cf", "effect_pct", "extra_riders_per_day", "placebo_mean", "placebo_sd", "placebo_max", "z", "n_ge", "n", "share_in_crz"]
print("routes:", len(T), "| ≥2000 riders/day:", len(big), "| Bronx local:", int(big.bronx_local.sum()))
print("\nTOP 12, all routes ≥ 2,000 riders/day, by relative effect"); print(big.sort_values("effect_pct", ascending=False).head(12)[cols].round(1).to_string())
nb = big[~big.bronx_local]
print("\nTOP 12 excluding Bronx local, by relative effect"); print(nb.sort_values("effect_pct", ascending=False).head(12)[cols].round(1).to_string())
print("\nTOP 8 excluding Bronx local, by extra riders per day"); print(nb.sort_values("extra_riders_per_day", ascending=False).head(8)[cols].round(1).to_string())
print("\nnon-Bronx distribution: share with effect>0 %.2f, share above all own placebos %.2f, median effect %.1f, median z %.1f" % ((nb.effect_pct > 0).mean(), (nb.n_ge == 0).mean(), nb.effect_pct.median(), nb.z.median()))
# ---- per-route time-series figures
act = load_actual("bus_ns", direction="all", mode_cfg=load_mode("bus_ns"), paths=paths); act.columns = act.columns.astype(str)
mu = pd.read_csv(f"{paths['output_root']}/bus_ns/timesfm/bus_ns_timesfm_test_mu.csv", index_col=0, parse_dates=True); mu.columns = mu.columns.astype(str)
plt.rcParams.update({"font.size": 8.5, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
def panel(routes, fname):
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 11), sharex=True)
    for ax, r in zip(axes.ravel(), routes):
        a = act[r].rolling(7, center=True).mean() / 1e3; prev = a.copy(); prev.index = prev.index + pd.Timedelta(weeks=52); cf = mu[r].rolling(7, center=True, min_periods=4).mean() / 1e3
        ax.plot(prev.loc["2024-09-15":"2025-04-30"].index, prev.loc["2024-09-15":"2025-04-30"], color=C_PREV, lw=1.3, label="same weeks one year earlier")
        ax.plot(a.loc["2024-09-15":"2025-04-30"].index, a.loc["2024-09-15":"2025-04-30"], color=C_MAIN, lw=1.9, label="observed (7-day mean)")
        ax.plot(cf.index, cf, color=C_CF, lw=1.3, ls="--", label="forecast without the policy")
        ax.axvline(POLICY, color=INK, lw=0.9, ls=":"); ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True); ax.set_ylim(bottom=0)
        t = T.loc[r]; ax.set_title(f"{r}  {t['name']}"[:62], loc="left", fontweight="bold", fontsize=9, color=INK)
        ax.text(0.01, 0.04, f"{t.effect_pct:+.0f}%  ({t.extra_riders_per_day:+,.0f} riders/day)   own placebos: mean {t.placebo_mean:+.0f}%, max {t.placebo_max:+.0f}%   z = {t.z:.1f}", transform=ax.transAxes, fontsize=7.8, color=MUTED)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(10, 12, 2, 4))); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")); ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]: ax.set_ylabel("thousand riders / day", color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend([h[1], h[2], h[0]], [l[1], l[2], l[0]], loc="lower center", ncol=3, frameon=False, fontsize=8.5); fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=1.4)
    for e in ("png", "pdf"): fig.savefig(FIG / f"{fname}.{e}", dpi=180, bbox_inches="tight")
    plt.close(fig)
panel(nb.sort_values("effect_pct", ascending=False).head(8).index.tolist(), "route_top_nonbronx")
panel(big[big.bronx_local].sort_values("effect_pct", ascending=False).head(8).index.tolist(), "route_top_bronx")
