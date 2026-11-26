"""Figure 4 (who responded, and where people drove before the policy) - Nature-style, 183 mm, Arial, editable text.
Row 1  a bus routes, b subway stations, c origin tracts (Replica): standardised coefficients of the spatial-lag regression of the unit effect on
       neighbourhood characteristics (ACS 2023), 95 % interval; red / blue = significant positive / negative, grey = not significant.
Row 2  d share of trips into the Congestion Relief Zone made by private car before the policy, by origin tract (Replica, Jan-Aug 2024);
       e the 15 community districts with the highest car share.
Usage: python runs/revision/fig4_drivers.py [timesfm|chronos_bolt]  -> main/fig4_drivers[_chronos_bolt].{pdf,svg,tiff,png}
"""
import glob, os, sys
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import contextily as cx, geopandas as gpd, numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT, load_paths
for f in glob.glob(os.path.expanduser("~/.fonts/arial*.ttf")): fm.fontManager.addfont(f)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "chronos_bolt"; SUF = "" if MODEL == "chronos_bolt" else "_" + MODEL; DSUF = "" if MODEL == "timesfm" else "_" + MODEL      # main model: Chronos-Bolt (2026-11-26); data tables keep their model-named suffix
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; FIG = REPO_ROOT / "outputs" / "figures" / "paper" / "main"; G = REPO_ROOT / "geo_data"; paths = load_paths(); MM = 1 / 25.4
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2, "ytick.major.size": 2.2, "legend.frameon": False, "axes.spines.right": False, "axes.spines.top": False,
                     "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.labelpad": 2.5})
COL = {"bus": "#D4722A", "subway": "#16788A", "replica": "#5B4B8A"}; DARK = {"bus": "#A8551A", "subway": "#0F5563", "replica": "#3F3366"}
RED, BLUE, INK, GREY, LIGHT, BROWN = "#B64342", "#0F4D92", "#1C1C1C", "#6E6E6E", "#B8B8B8", "#B0703A"; mn = lambda s: s.replace("-", "−")
CAR = mpl.colors.LinearSegmentedColormap.from_list("car", ["#F6EFE4", "#E3C08F", "#C48646", "#8E4A1C", "#4A1F0B"])
# ------------------------------------------------------------------ regression results
coef = pd.read_csv(U / f"sociodemo_spatial_coef{DSUF}.csv"); coef = coef[coef.model == "Spatial lag"]; fit = pd.read_csv(U / f"sociodemo_spatial_fit{DSUF}.csv").set_index("mode")
VARS = ["Median household income", "Bachelor's degree or higher", "Work from home", "Commute by car", "Commute by transit", "Population density", "White residents", "Black residents", "Hispanic residents", "Asian residents", "Median age", "Distance to the zone"]
REG = [("bus", "Bus routes"), ("subway", "Subway stations"), ("replica", "All trips by origin tract (Replica)")]; RNAME = {"bus": "Bus routes", "subway": "Subway stations", "replica": "Origin tracts, all trips (Replica)"}
# ------------------------------------------------------------------ pre-policy car share
cs = pd.read_csv(U / "replica_carshare_tract.csv", dtype={"tract": str}).set_index("tract"); cd = pd.read_csv(U / "replica_carshare_cdta.csv"); ALLSHARE = 100 * cs.car.sum() / cs.total.sum()
TR = gpd.read_file(G / "NYC_Census_Tracts_2020" / "NYC_Census_Tracts_2020.shp"); TR["geoid"] = TR.geoid.astype(str); T3 = TR.to_crs(3857); BORO = T3.dissolve("boroname"); PUMA = gpd.read_file(G / "NYC_PUMA" / "NYC_Public_Use_Microdata_Areas_PUMAs_2010.shp").to_crs(3857)
crz = gpd.read_file(G / "nyc_cp_boundary_poly_json.geojson").to_crs(3857); M = T3.merge(cs[["car_share", "car", "total"]], left_on="geoid", right_index=True); M = M[M.total >= 30]
# ------------------------------------------------------------------ basemap
xmin, ymin, xmax, ymax = T3.total_bounds; pad = 0.015 * (xmax - xmin); xmin -= pad; xmax += pad; ymin -= pad; ymax += pad
cx.set_cache_dir(str(REPO_ROOT / "outputs" / "_cache_tiles_osm")); cx.tile.USER_AGENT = "transit-figure-renderer/1.0 (one-off static research figure)"
img, ext = cx.bounds2img(xmin, ymin, xmax, ymax, ll=False, source=cx.providers.OpenStreetMap.Mapnik, zoom=12)      # credited in the caption
base = np.clip(255 * 0.66 + 0.34 * (img[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])), 0, 255).astype(np.uint8)
# ------------------------------------------------------------------ layout (mm)
W = 183.0; LAB, R, G1 = 37.0, 3.0, 6.0; rw = (W - LAB - R - 2 * G1) / 3; TOP, RH, GAP, BOT = 10.0, 58.0, 27.0, 4.0; mw = 72.0; mh = mw * (ymax - ymin) / (xmax - xmin); Hh = TOP + RH + GAP + mh + BOT
fig = plt.figure(figsize=(W * MM, Hh * MM)); y1 = Hh - TOP - RH; y2 = BOT
def add(x, y, w, h): return fig.add_axes([x / W, y / Hh, w / W, h / Hh])
def plabel(x, ytop, s): fig.text(x / W, (ytop + 1.4) / Hh, s, fontsize=8, fontweight="bold", va="bottom", ha="left")
# ---- a-c: coefficients
yy = {v: -i for i, v in enumerate(VARS)}; yrho = -len(VARS) - 0.5
for j, (kind, mode) in enumerate(REG):
    ax = add(LAB + j * (rw + G1), y1, rw, RH); c = coef[coef["mode"] == mode].set_index("label")
    for v, y in list(yy.items()) + [("rho (spatial lag of effect)", yrho)]:
        if v not in c.index: ax.text(0.03, y, "dropped (collinear)", fontsize=5.5, color=GREY, va="center", ha="left"); continue
        r = c.loc[v]; sig = r.p < 0.05; col = (RED if r.coef > 0 else BLUE) if sig else LIGHT
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=1.0, solid_capstyle="butt", zorder=2); ax.scatter([r.coef], [y], marker="D", s=14, facecolor=col if sig else "white", edgecolor=col, linewidth=0.8, zorder=3)
    ax.axvline(0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1); ax.set_xlim(-0.8, 0.8); ax.set_xticks([-0.5, 0, 0.5]); ax.set_xticklabels(["−0.5", "0", "0.5"]); ax.set_ylim(yrho - 0.6, 0.6); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_yticks(list(yy.values()) + [yrho]); ax.set_yticklabels(VARS + ["Effect in neighbouring units"] if j == 0 else []); f_ = fit.loc[mode]
    ax.text(0.0, 1.075, RNAME[kind], transform=ax.transAxes, fontsize=7, fontweight="bold", color=DARK[kind], va="bottom", ha="left"); ax.text(0.0, 1.02, f"n = {int(f_.n):,}, pseudo-R² = {f_.lag_pseudo_r2:.2f}", transform=ax.transAxes, fontsize=5.8, color=GREY, va="bottom", ha="left")
    plabel(LAB + j * (rw + G1) - (34.0 if j == 0 else 4.5), y1 + RH + 3.2, "abc"[j])
    if j == 1: ax.set_xlabel("Standardised coefficient, spatial-lag model (s.d. of effect per s.d. of the variable)")
fig.legend(handles=[Line2D([], [], marker="D", color=RED, lw=1.0, ms=3.5, label="Positive, P < 0.05"), Line2D([], [], marker="D", color=BLUE, lw=1.0, ms=3.5, label="Negative, P < 0.05"), Line2D([], [], marker="D", color=LIGHT, markerfacecolor="white", lw=1.0, ms=3.5, label="Not significant (95% interval)")],
           loc="upper center", bbox_to_anchor=((LAB + (W - LAB - R) / 2) / W, (y1 - 9.5) / Hh), ncol=3, fontsize=6.5, handlelength=2.0, columnspacing=2.2, borderpad=0)
# ---- d: car share map
ax = add(4.0, y2, mw, mh); ax.imshow(base, extent=ext, cmap="gray", vmin=0, vmax=255, interpolation="bilinear", zorder=0, rasterized=True); nrm = mpl.colors.Normalize(10, 60)
M.plot(ax=ax, column="car_share", cmap=CAR, norm=nrm, linewidth=0, zorder=2, rasterized=True); PUMA.boundary.plot(ax=ax, color="#5E5B56", linewidth=0.3, zorder=4); BORO.boundary.plot(ax=ax, color="#33312E", linewidth=0.55, zorder=5); crz.boundary.plot(ax=ax, color=INK, linewidth=1.1, zorder=6)
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color(GREY)
fq = 1 / np.cos(np.deg2rad(40.7)); x0, y0 = xmax - 0.05 * (xmax - xmin) - 10000 * fq, ymin + 0.04 * (ymax - ymin); ax.plot([x0, x0 + 10000 * fq], [y0, y0], color=INK, lw=1.1, solid_capstyle="butt", zorder=8); ax.text(x0 + 5000 * fq, y0 + 0.012 * (ymax - ymin), "10 km", ha="center", va="bottom", fontsize=6, zorder=8)
ax.add_patch(mpl.patches.Rectangle((0.015, 0.70), 0.47, 0.285, transform=ax.transAxes, facecolor="white", edgecolor="none", alpha=0.92, zorder=7))
cax = ax.inset_axes([0.045, 0.875, 0.30, 0.022]); cax.set_zorder(9); cb = fig.colorbar(mpl.cm.ScalarMappable(norm=nrm, cmap=CAR), cax=cax, orientation="horizontal", ticks=[10, 35, 60], extend="both"); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1.5)
ax.text(0.04, 0.925, "Trips into the zone by private car (%)", transform=ax.transAxes, fontsize=6, color=INK, va="bottom", ha="left", zorder=9)
ax.legend([Line2D([], [], color=INK, lw=1.1), Line2D([], [], color="#33312E", lw=0.55), Line2D([], [], color="#5E5B56", lw=0.35)], ["Congestion Relief Zone", "Borough", "Public Use Microdata Area"], loc="upper left", bbox_to_anchor=(0.02, 0.83), fontsize=5.5, handletextpad=0.6, labelspacing=0.35, borderpad=0.2, frameon=False).set_zorder(9)
plabel(4.0, y2 + mh, "d"); fig.text(9.5 / W, (y2 + mh + 1.6) / Hh, "Driving into the zone before the policy", fontsize=7, fontweight="bold", color=INK, va="bottom", ha="left")
# ---- e: top community districts
top = cd.sort_values("car_share", ascending=False).head(15).iloc[::-1]; nm_full = None; nm = top.cdtaname.str.replace(r"^[A-Z]{2}\d+\s+", "", regex=True).str.replace(r"\s*\(.*\)$", "", regex=True).str.split("-").map(lambda q: "–".join(q[:2])) + ", " + top.boroname.map({"Staten Island": "Staten Island", "Brooklyn": "Brooklyn", "Queens": "Queens", "Bronx": "Bronx", "Manhattan": "Manhattan"})
ax = add(124.0, y2 + 8.5, 48.0, mh - 8.5); ax.barh(range(len(top)), top.car_share, color=BROWN, height=0.66, linewidth=0); ax.set_yticks(range(len(top))); ax.set_yticklabels(nm, fontsize=6); ax.tick_params(axis="y", length=0, pad=2); ax.spines["left"].set_visible(False)
for i, v in enumerate(top.car_share): ax.text(v + 1.2, i, f"{v:.0f}", va="center", fontsize=5.5, color=INK)
ax.axvline(ALLSHARE, color=INK, lw=0.6, ls=(0, (3, 2))); ax.text(ALLSHARE + 1.0, len(top) - 0.3, f"All origins, {ALLSHARE:.0f}%", fontsize=5.5, color=GREY, va="bottom", ha="left"); ax.set_xlim(0, 62); ax.set_xticks([0, 30, 60]); ax.set_ylim(-0.6, len(top) + 0.5); ax.set_xlabel("Private car share (%)"); ax.tick_params(axis="x", pad=2)
plabel(84.0, y2 + mh, "e"); fig.text(89.5 / W, (y2 + mh + 1.6) / Hh, "Highest car share, by community district", fontsize=7, fontweight="bold", color=INK, va="bottom", ha="left")
for e_, kw in (("pdf", {"dpi": 600}), ("svg", {"dpi": 600}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}), ("png", {"dpi": 300})): fig.savefig(FIG / f"fig4_drivers{SUF}.{e_}", **kw)
print(f"figure {W:.0f} x {Hh:.0f} mm ->", FIG / f"fig4_drivers{SUF}.png")
