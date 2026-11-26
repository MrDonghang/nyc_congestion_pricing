"""Station-level subway effects (TimesFM, entries, student fares excluded, history from 2022, 116 days).
The system-level seasonal calibration is passed down to stations as one scaling factor per origin
(k_o = calibrated / raw system counterfactual), so station effects add up to the headline estimate.
Each station is compared with its own 2024 pseudo-policy scores (same scaling). Stations closed in the policy period dropped.
Writes uq/station_effects_subway_timesfm.csv; figures station_top.{png,pdf}, station_effect_map.{png,pdf}
"""
import glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import geopandas as gpd, numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT, load_mode, load_paths
from nyc_cp.data import load_actual
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; FIG = REPO_ROOT / "outputs" / "figures" / "paper" / "robustness"; G = REPO_ROOT / "geo_data"; paths = load_paths()
POLICY, B0 = pd.Timestamp("2025-01-05"), pd.Timestamp("2024-08-29"); C_MAIN, C_PREV, INK, MUTED = "#2a78d6", "#b9b7ae", "#1f1f1f", "#5f5e5a"
s8 = pd.read_csv(U / "round8_scores.csv", parse_dates=["origin"]); s8 = s8[(s8.panel == "subway_ns22") & (s8.model == "timesfm") & (s8.h == 116)]
k = ((1 + s8[s8.variant == "V0"].set_index("origin").score / 100) / (1 + s8[s8.variant == "V3"].set_index("origin").score / 100)).dropna()      # calibrated cf / raw cf, per origin
d = pd.concat([pd.read_csv(f, parse_dates=["origin"], dtype={"unit": str}) for f in glob.glob(str(U / "denseunit_subway_ns22_timesfm*.csv"))]); d = d[(d.h == 116) & d.origin.isin(k.index)]
d["cfc"] = d.cf * d.origin.map(k); d["s"] = 100 * (d.actual / d.cfc - 1)
geo = pd.read_csv(U / "unit_geo_subway_ns.csv", dtype={"unit": str}).set_index("unit"); geo = geo[~geo.yoy_closed]
real = d[d.origin == POLICY].set_index("unit"); nul = d[(d.origin >= "2024-01-01") & (d.origin + pd.Timedelta(days=115) < B0)]
g = nul.groupby("unit").s.agg(placebo_mean="mean", placebo_sd="std", placebo_max="max", n="size")
T = pd.DataFrame({"effect_pct": real.s, "entries_per_day_cf": real.cfc / 116, "extra_entries_per_day": (real.actual - real.cfc) / 116}).join(g).join(geo[["stop_name", "borough", "in_crz", "dist_km"]], how="inner")
T["z"] = (T.effect_pct - T.placebo_mean) / T.placebo_sd; T["n_ge"] = [int((nul[nul.unit == u].s >= T.effect_pct[u]).sum()) for u in T.index]; T.index.name = "station_index"
T.sort_values("effect_pct", ascending=False).to_csv(U / "station_effects_subway_timesfm.csv")
tot = 100 * (real.loc[T.index].actual.sum() / real.loc[T.index].cfc.sum() - 1)
pd.set_option("display.width", 230); cols = ["stop_name", "borough", "dist_km", "entries_per_day_cf", "effect_pct", "extra_entries_per_day", "placebo_mean", "placebo_sd", "placebo_max", "z", "n_ge", "n"]
big = T[T.entries_per_day_cf >= 3000]
print(f"stations {len(T)} (closed ones dropped) | ≥3000 entries/day: {len(big)} | system effect over these stations {tot:+.2f}% | placebo origins per station {int(T.n.median())}")
print("\nTOP 12 by relative effect (≥3000 entries/day)"); print(big.sort_values("effect_pct", ascending=False).head(12)[cols].round(1).to_string())
print("\nTOP 10 by extra entries per day"); print(T.sort_values("extra_entries_per_day", ascending=False).head(10)[cols].round(1).to_string())
print("\nBOTTOM 8 by relative effect (≥3000)"); print(big.sort_values("effect_pct").head(8)[cols].round(1).to_string())
T["band"] = pd.cut(T.dist_km, [-1, 0, 2, 5, 10, 100], labels=["inside CRZ", "0–2 km", "2–5 km", "5–10 km", "> 10 km"])
print("\nby distance to the zone:"); print(T.groupby("band", observed=True).apply(lambda x: pd.Series({"stations": len(x), "share_of_entries": x.entries_per_day_cf.sum() / T.entries_per_day_cf.sum(), "effect_pct_volume_weighted": 100 * x.extra_entries_per_day.sum() / x.entries_per_day_cf.sum(), "median_effect": x.effect_pct.median(), "share_positive": (x.effect_pct > 0).mean(), "share_above_all_placebos": (x.n_ge == 0).mean()})).round(2).to_string())
print("by borough:"); print(T.groupby("borough").apply(lambda x: pd.Series({"stations": len(x), "effect_pct_volume_weighted": 100 * x.extra_entries_per_day.sum() / x.entries_per_day_cf.sum(), "median_effect": x.effect_pct.median(), "share_above_all_placebos": (x.n_ge == 0).mean()})).round(2).to_string())
print("all: share positive %.2f, share above all own placebos %.2f, median %.1f" % ((T.effect_pct > 0).mean(), (T.n_ge == 0).mean(), T.effect_pct.median()))
# ---- per-station time series (top 8 by relative effect among larger stations)
act = load_actual("subway_ns22", direction="O", mode_cfg=load_mode("subway_ns22"), paths=paths); act.columns = act.columns.astype(str)
mu = pd.read_csv(f"{paths['output_root']}/subway_ns22/timesfm/O/subway_ns22_timesfm_test_O_mu.csv", index_col=0, parse_dates=True); mu.columns = mu.columns.astype(str); mu = mu * k[POLICY]
plt.rcParams.update({"font.size": 8.5, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(4, 2, figsize=(12.5, 11), sharex=True)
for ax, r in zip(axes.ravel(), big.sort_values("effect_pct", ascending=False).head(8).index):
    a = act[r].rolling(7, center=True).mean() / 1e3; prev = a.copy(); prev.index = prev.index + pd.Timedelta(weeks=52); cf = mu[r].rolling(7, center=True, min_periods=4).mean() / 1e3
    ax.plot(prev.loc["2024-09-15":"2025-04-30"].index, prev.loc["2024-09-15":"2025-04-30"], color=C_PREV, lw=1.3, label="same weeks one year earlier")
    ax.plot(a.loc["2024-09-15":"2025-04-30"].index, a.loc["2024-09-15":"2025-04-30"], color=C_MAIN, lw=1.9, label="observed (7-day mean)"); ax.plot(cf.index, cf, color=INK, lw=1.3, ls="--", label="counterfactual without the policy")
    ax.axvline(POLICY, color=INK, lw=0.9, ls=":"); ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True); ax.set_ylim(bottom=0); t = T.loc[r]
    ax.set_title(f"{t.stop_name}  ({ {'M': 'Manhattan', 'Bk': 'Brooklyn', 'Q': 'Queens', 'Bx': 'Bronx', 'SI': 'Staten Island'}.get(t.borough, t.borough)}, {'inside the zone' if t.in_crz else f'{t.dist_km:.0f} km from the zone'})"[:70], loc="left", fontweight="bold", fontsize=9, color=INK)
    ax.text(0.01, 0.04, f"{t.effect_pct:+.0f}%  ({t.extra_entries_per_day:+,.0f} entries/day)   own placebos: mean {t.placebo_mean:+.0f}%, max {t.placebo_max:+.0f}%   z = {t.z:.1f}", transform=ax.transAxes, fontsize=7.8, color=MUTED)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(10, 12, 2, 4))); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")); ax.tick_params(labelbottom=True)
for ax in axes[:, 0]: ax.set_ylabel("thousand entries / day", color=MUTED)
h, l = axes[0, 0].get_legend_handles_labels(); fig.legend([h[1], h[2], h[0]], [l[1], l[2], l[0]], loc="lower center", ncol=3, frameon=False, fontsize=8.5); fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=1.4)
for e in ("png", "pdf"): fig.savefig(FIG / f"station_top.{e}", dpi=180, bbox_inches="tight")
plt.close(fig)
# ---- map
st = gpd.read_file(G / "MTA_Subway_Stations_20251029.geojson").to_crs(2263); st["complex_id"] = st.complex_id.astype(int); cx = st.dissolve("complex_id"); cx["geometry"] = cx.geometry.centroid
mp = pd.read_csv(paths["data_root"] + "/subway/patterns/station_mapping.csv"); mp["unit"] = mp.station_index.astype(str)
P = gpd.GeoDataFrame(T.join(mp.set_index("unit").station_id).merge(cx[["geometry"]], left_on="station_id", right_index=True), geometry="geometry", crs=2263)
crz = gpd.read_file(G / "nyc_cp_boundary_poly_json.geojson").to_crs(2263); tr = gpd.read_file(G / "NYC_Census_Tracts_2020" / "NYC_Census_Tracts_2020.shp").to_crs(2263)
boro = tr.dissolve([c for c in tr.columns if c.lower() in ("boroname", "boro_name", "borocode", "boroct2020")][0])
cmap = LinearSegmentedColormap.from_list("div", ["#a8501c", "#e3a878", "#dcdad2", "#86b6ef", "#0d366b"]); norm = TwoSlopeNorm(vmin=-12, vcenter=0, vmax=12)
plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(14, 8.2)); size = 6 + 70 * np.sqrt(P.entries_per_day_cf / P.entries_per_day_cf.max())
for ax in axes: boro.plot(ax=ax, color="#f4f3ee", edgecolor="#c9c7bd", linewidth=0.7); ax.set_axis_off(); ax.set_xlim(970000, 1068000); ax.set_ylim(145000, 272000)
o = P.effect_pct.abs().argsort()
axes[0].scatter(P.geometry.x.iloc[o], P.geometry.y.iloc[o], s=size.iloc[o], c=P.effect_pct.iloc[o], cmap=cmap, norm=norm, edgecolor="white", linewidth=0.5); crz.boundary.plot(ax=axes[0], color=INK, linewidth=1.6)
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes[0], orientation="horizontal", fraction=0.035, pad=0.01, shrink=0.6, extend="both"); cb.set_label("effect on entries over 116 days (%)", color=MUTED); cb.outline.set_visible(False)
axes[0].set_title("Estimated effect by station (circle area ∝ entries)", loc="left", fontweight="bold", fontsize=11, color=INK)
up = (P.n_ge == 0) & (P.effect_pct > 0)
axes[1].scatter(P.geometry.x[~up], P.geometry.y[~up], s=size[~up], color="#c9c7bd", edgecolor="white", linewidth=0.5); axes[1].scatter(P.geometry.x[up], P.geometry.y[up], s=size[up], color=C_MAIN, edgecolor="white", linewidth=0.5)
crz.boundary.plot(ax=axes[1], color=INK, linewidth=1.6); axes[1].set_title("Stations above every one of their own no-policy placebos", loc="left", fontweight="bold", fontsize=11, color=INK)
axes[1].legend(handles=[plt.Line2D([], [], marker="o", ls="none", color=C_MAIN, ms=7, label=f"effect above all own placebos ({int(up.sum())} stations)"), plt.Line2D([], [], marker="o", ls="none", color="#c9c7bd", ms=7, label=f"other stations ({int((~up).sum())})"), plt.Line2D([], [], color=INK, lw=1.6, label="Congestion Relief Zone")], loc="lower right", frameon=False, fontsize=8.5)
fig.tight_layout(w_pad=0.5)
for e in ("png", "pdf"): fig.savefig(FIG / f"station_effect_map.{e}", dpi=190, bbox_inches="tight")
print("mapped stations:", len(P))
