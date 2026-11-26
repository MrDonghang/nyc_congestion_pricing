"""Where did people travel INTO the Congestion Relief Zone mainly by car before the policy? (Replica, typical weekday)
Source: data_processed/replica/mode/<mode>.csv, OD pairs with destination tract inside the zone and origin tract outside it
(origins are NYC tracts only; the same 9,289 pairs are present for every car and transit mode).
Baseline window: 2024-01-06 .. 2024-08-31 (2024, before the mid-September Replica vintage step). Checked against 2023 and Sep-Dec 2024.
Private car = private_auto (driver) + auto_passenger. Taxi / ride-hail (on_demand_auto) is reported separately.
Writes uq/replica_carshare_{tract,cdta,borough,corr}.csv and figure fig_replica_carshare.{pdf,svg,tiff,png}
"""
import warnings
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd, numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT, load_paths
warnings.filterwarnings("ignore")
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; FIG = REPO_ROOT / "outputs" / "figures" / "paper" / "main"; MM = 1 / 25.4; D = load_paths()["data_root"] + "/replica/mode/"
MODES = ["private_auto", "auto_passenger", "on_demand_auto", "public_transit", "walking", "biking", "other"]
WIN = {"base": ("2024-01-06", "2024-08-31"), "y2023": ("2023-01-07", "2023-12-30"), "post_step": ("2024-09-28", "2024-12-28")}
tr = gpd.read_file(REPO_ROOT / "geo_data" / "NYC_Census_Tracts_2020" / "NYC_Census_Tracts_2020.shp").to_crs(2263); tr["geoid"] = tr.geoid.astype(str)
crz = gpd.read_file(REPO_ROOT / "geo_data" / "nyc_cp_boundary_poly_json.geojson").to_crs(2263).geometry.union_all(); inz = set(tr[tr.geometry.intersection(crz).area / tr.geometry.area > 0.5].geoid)
T = {}
for m in MODES:
    x = pd.read_csv(D + m + ".csv", index_col=0, parse_dates=True); o = x.columns.str.split("_").str[0]; d = x.columns.str.split("_").str[1]; k = d.isin(inz) & ~o.isin(inz)
    for w, (a, b) in WIN.items(): T[(w, m)] = x.loc[a:b, k].mean().groupby(o[k].to_numpy()).sum()
T = pd.DataFrame(T).fillna(0.0)
def shares(w):
    t = T[w].copy(); t["car"] = t.private_auto + t.auto_passenger; t["total"] = t[MODES].sum(axis=1); return t
B = shares("base"); B = B.join(tr.set_index("geoid")[["boroname", "cdta2020", "cdtaname", "ntaname"]]); S = pd.read_csv(U / "sociodemo_replica.csv", dtype={"unit": str}).set_index("unit"); B = B.join(S, rsuffix="_acs")
for c, lab in (("car", "car_share"), ("private_auto", "driver_share"), ("on_demand_auto", "taxi_share"), ("public_transit", "transit_share")): B[lab] = 100 * B[c] / B.total
for w in ("y2023", "post_step"): t = shares(w); B[f"car_share_{w}"] = 100 * t.car / t.total
B.index.name = "tract"; B.to_csv(U / "replica_carshare_tract.csv")
def agg(key):
    g = B.groupby(key)[MODES + ["car", "total"]].sum(); g["tracts"] = B.groupby(key).size(); g["car_share"] = 100 * g.car / g.total; g["driver_share"] = 100 * g.private_auto / g.total; g["taxi_share"] = 100 * g.on_demand_auto / g.total
    g["transit_share"] = 100 * g.public_transit / g.total; g["share_of_all_car_trips"] = 100 * g.car / g.car.sum(); g["dist_km"] = B.groupby(key).apply(lambda q: np.average(q.dist_km.fillna(q.dist_km.median()), weights=q.total)); return g
boro = agg("boroname"); cd = agg(["boroname", "cdtaname"]).reset_index(); cd = cd[cd.total >= 500].sort_values("car_share", ascending=False); boro.to_csv(U / "replica_carshare_borough.csv"); cd.to_csv(U / "replica_carshare_cdta.csv", index=False)
tot = B[MODES + ["car", "total"]].sum(); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(f"origin tracts {len(B)}, trips into zone per weekday {tot.total:,.0f}; car {100 * tot.car / tot.total:.1f}% (driver {100 * tot.private_auto / tot.total:.1f}, passenger {100 * tot.auto_passenger / tot.total:.1f}), taxi {100 * tot.on_demand_auto / tot.total:.1f}%, transit {100 * tot.public_transit / tot.total:.1f}%, walk {100 * tot.walking / tot.total:.1f}%, bike {100 * tot.biking / tot.total:.1f}%")
print("\nBy borough"); print(boro[["tracts", "total", "car", "car_share", "driver_share", "taxi_share", "transit_share", "share_of_all_car_trips", "dist_km"]].round(1).to_string())
print("\nCommunity districts, highest car share"); print(cd.head(20)[["cdtaname", "tracts", "total", "car", "car_share", "driver_share", "transit_share", "share_of_all_car_trips", "dist_km"]].round(1).to_string(index=False))
print("\nCommunity districts, lowest car share"); print(cd.tail(8)[["cdtaname", "tracts", "total", "car_share", "transit_share", "dist_km"]].round(1).to_string(index=False))
print("\nCommunity districts sending most car trips"); print(cd.sort_values("car", ascending=False).head(12)[["cdtaname", "total", "car", "car_share", "share_of_all_car_trips"]].round(1).to_string(index=False))
big = B[B.total >= 100]; print(f"\ntracts with >=100 trips: {len(big)}; car share > 50%: {(big.car_share > 50).sum()} tracts, {100 * big[big.car_share > 50].total.sum() / big.total.sum():.1f}% of trips, {100 * big[big.car_share > 50].car.sum() / big.car.sum():.1f}% of car trips")
print("car share quantiles (tracts >=100 trips):", big.car_share.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(1).to_dict())
print("stability: corr with 2023", round(big.car_share.corr(big.car_share_y2023), 3), " with Sep-Dec 2024", round(big.car_share.corr(big.car_share_post_step), 3), "; citywide car share 2023 / post-step:", *[round(100 * shares(w).car.sum() / shares(w).total.sum(), 1) for w in ("y2023", "post_step")])
bins = pd.cut(big.dist_km, [-0.1, 3, 6, 10, 15, 60]); print("\nBy distance to the zone (km)"); print(big.groupby(bins).apply(lambda q: pd.Series(dict(tracts=len(q), trips=q.total.sum(), car_share=100 * q.car.sum() / q.total.sum(), transit_share=100 * q.public_transit.sum() / q.total.sum(), share_of_car_trips=100 * q.car.sum() / big.car.sum()))).round(1).to_string())
V = [("income_k", "Median household income"), ("pct_bachelor_plus", "Bachelor's degree or higher"), ("pct_wfh", "Work from home"), ("pct_car_commute", "Commute by car (ACS)"), ("pct_transit_commute", "Commute by transit (ACS)"), ("pop_density_k", "Population density"),
     ("pct_white", "White residents"), ("pct_black", "Black residents"), ("pct_hispanic", "Hispanic residents"), ("pct_asian", "Asian residents"), ("age_median", "Median age"), ("dist_km", "Distance to the zone")]
bc = big.dropna(subset=["dist_km"]); rk = lambda q: q.fillna(q.median()).rank().to_numpy(); res = lambda q: q - np.polyval(np.polyfit(rk(bc.dist_km), q, 1), rk(bc.dist_km))
C = pd.DataFrame([dict(variable=l, spearman=bc.car_share.corr(bc[v], method="spearman"), partial_given_distance=np.corrcoef(res(rk(bc.car_share)), res(rk(bc[v])))[0, 1] if v != "dist_km" else np.nan) for v, l in V])
C.to_csv(U / "replica_carshare_corr.csv", index=False); print("\nSpearman correlation of tract car share with neighbourhood variables"); print(C.round(2).to_string(index=False))
# ---------------- figure
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2, "ytick.major.size": 2.2, "legend.frameon": False, "axes.spines.right": False, "axes.spines.top": False, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6})
INK, GREY, NOD = "#272727", "#767676", "#EFEFEF"; cmap = mpl.colors.LinearSegmentedColormap.from_list("seq", ["#F1ECE4", "#D9B98C", "#B0703A", "#6E2F14", "#2E0F06"])
G = tr.merge(B[["car_share", "car", "total"]], left_on="geoid", right_index=True, how="left"); G.loc[G.total < 30, ["car_share", "car"]] = np.nan; G["car_km2"] = G.car / (G.geometry.area / 10.7639e6)
boros = tr.dissolve("boroname"); zone = gpd.GeoSeries([crz], crs=2263)
fig = plt.figure(figsize=(183 * MM, 92 * MM)); ax1 = fig.add_axes([0.0, 0.04, 0.33, 0.88]); ax2 = fig.add_axes([0.33, 0.04, 0.33, 0.88]); ax3 = fig.add_axes([0.83, 0.13, 0.15, 0.74])
for ax, col, norm, lab, ticks in ((ax1, "car_share", mpl.colors.Normalize(0, 70), "Trips into the zone made by private car (%)", [0, 35, 70]), (ax2, "car_km2", mpl.colors.LogNorm(10, 3000), "Car trips into the zone per km² of origin tract", [10, 100, 1000])):
    tr.plot(ax=ax, color=NOD, linewidth=0); G.dropna(subset=[col]).plot(ax=ax, column=col, cmap=cmap, norm=norm, linewidth=0); boros.boundary.plot(ax=ax, color="white", linewidth=0.5); zone.boundary.plot(ax=ax, color=INK, linewidth=0.7)
    ax.set_axis_off(); ax.set_aspect("equal"); cax = ax.inset_axes([0.05, 0.90, 0.34, 0.022]); cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=ticks); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4)
    if col == "car_km2": cb.ax.set_xticklabels(["10", "100", "1,000"]); cb.ax.minorticks_off()
    ax.text(0.05, 0.945, lab, transform=ax.transAxes, fontsize=6, color=INK, va="bottom", ha="left")
ax1.legend(handles=[mpl.patches.Patch(facecolor="none", edgecolor=INK, linewidth=0.7, label="Congestion Relief Zone"), mpl.patches.Patch(facecolor=NOD, edgecolor="none", label="No data or fewer than 30 trips")], loc="lower right", fontsize=5.5, bbox_to_anchor=(1.0, 0.02), handlelength=1.2)
top = cd.head(15).iloc[::-1]; name = top.cdtaname.str.replace(r"^[A-Z]{2}\d+\s+", "", regex=True).str.replace(r"\s*\(.*\)$", "", regex=True).str.split("-").map(lambda q: "–".join(q[:2]) if len("–".join(q[:2])) <= 24 else q[0]) + ", " + top.boroname.map({"Staten Island": "SI", "Brooklyn": "Bk", "Queens": "Qn", "Bronx": "Bx", "Manhattan": "Mn"})
ax3.barh(range(len(top)), top.car_share, color="#B0703A", height=0.62); ax3.set_yticks(range(len(top))); ax3.set_yticklabels(name, fontsize=5.5); ax3.tick_params(axis="y", length=0); ax3.spines["left"].set_visible(False)
for i, v in enumerate(top.car_share): ax3.text(v + 1, i, f"{v:.0f}", va="center", fontsize=5.5, color=INK)
ax3.axvline(100 * tot.car / tot.total, color=INK, lw=0.6, ls=(0, (3, 2)), clip_on=False); ax3.text(100 * tot.car / tot.total, len(top) - 0.3, "All origins", fontsize=5.5, color=GREY, ha="center", va="bottom"); ax3.set_xlim(0, 60); ax3.set_xticks([0, 30, 60]); ax3.set_ylim(-0.6, len(top) - 0.35); ax3.set_xlabel("Private car share (%)")
for x0, s in ((0.012, "a"), (0.342, "b"), (0.672, "c")): fig.text(x0, 0.945, s, fontsize=8, fontweight="bold", va="bottom")
for e, kw in (("pdf", {"dpi": 600}), ("svg", {"dpi": 600}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}), ("png", {"dpi": 300})): fig.savefig(FIG / f"fig_replica_carshare.{e}", **kw)
print("wrote", FIG / "fig_replica_carshare.png")
