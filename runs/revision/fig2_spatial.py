"""Figure 2 (where the gains are) - Nature-style, 183 mm, Arial, editable text.
  a  bus routes coloured by the 116-day effect (>= 2,000 riders/day)        b  subway stations coloured by the effect (circle area = entries per day)
  c  bus effect for routes mostly inside / crossing / outside the zone       d  subway effect inside vs outside the Congestion Relief Zone and by distance band
  e  station effect against distance from the zone, with the volume-weighted running mean
Unit effects carry the system-level seasonal calibration as one factor per origin, so groups add up to the headline estimates.
Group intervals (90 %): group estimate minus the 95th / 5th percentile of the centred group error over the 2024 pseudo-policy dates.
Red = increase, blue = decrease on the maps (greyscale OpenStreetMap basemap, credited in the caption; borough, PUMA and census-tract boundaries); orange = bus, teal = subway elsewhere. Bronx local bus routes are not analysed.
Usage: python runs/revision/fig2_spatial.py [timesfm|chronos_bolt]  -> main/fig2_spatial[_chronos_bolt].{pdf,svg,tiff,png}, uq/paper_strata_fig2[_chronos_bolt].csv
"""
import glob, os, sys
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import contextily as cx, geopandas as gpd, numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT, load_paths
for f in glob.glob(os.path.expanduser("~/.fonts/arial*.ttf")): fm.fontManager.addfont(f)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "chronos_bolt"; SUF = "" if MODEL == "chronos_bolt" else "_" + MODEL      # main model: Chronos-Bolt; TimesFM = robustness (2026-11-26)
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; FIG = REPO_ROOT / "outputs" / "figures" / "paper" / "main"; G = REPO_ROOT / "geo_data"; paths = load_paths(); MM = 1 / 25.4
POLICY, B0, H = pd.Timestamp("2025-01-05"), pd.Timestamp("2024-08-29"), 116
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2, "ytick.major.size": 2.2, "legend.frameon": False, "axes.spines.right": False, "axes.spines.top": False,
                     "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.labelpad": 2.5})
COL = {"subway": "#16788A", "bus": "#D4722A"}; DARK = {"subway": "#0F5563", "bus": "#A8551A"}; TINT = {"subway": "#BFE0E4", "bus": "#F6D4B4"}
RED, BLUE, INK, GREY, LIGHT = "#B64342", "#0F4D92", "#1C1C1C", "#6E6E6E", "#C9C9C9"
CMAP = LinearSegmentedColormap.from_list("bwr_nat", ["#0F4D92", "#7FA6CF", "#F2F2F2", "#E9A6A1", "#B64342"]); NORM = TwoSlopeNorm(vmin=-12, vcenter=0, vmax=12); mn = lambda s: s.replace("-", "−")      # softer scale, same as Fig. 3 (user, 2026-11-25)
def is_null(o): return (o >= pd.Timestamp("2024-01-01")) & (o + pd.Timedelta(days=H - 1) < B0)
# ------------------------------------------------------------------ data
qd = pd.read_csv(U / "qr_native_daily.csv", parse_dates=["origin"]); qd = qd[qd.model == MODEL]      # quantile-regression (median) calibration, seasonal design (qr_native.py)
K = {p: qd[qd.panel == p].groupby("origin").apply(lambda g: g.cc.sum() / g.cf.sum()) for p in ("subway", "bus")}; HEAD = {p: 100 * (qd[(qd.panel == p) & (qd.origin == POLICY)].actual.sum() / qd[(qd.panel == p) & (qd.origin == POLICY)].cc.sum() - 1) for p in ("subway", "bus")}
def units(kind):
    pat = f"denseunitq_subway_ns22_{MODEL}*.csv" if kind == "subway" else f"denseunitq_bus_ns_{MODEL}*.csv"
    d = pd.concat([pd.read_csv(f, parse_dates=["origin"], dtype={"unit": str}) for f in glob.glob(str(U / pat))]); d = d[(d.h == H) & d.origin.isin(K[kind].index)].copy(); d["cfc"] = d.cf * d.origin.map(K[kind]); return d
du, db = units("subway"), units("bus")
gs = pd.read_csv(U / "unit_geo_subway_ns.csv", dtype={"unit": str}); gs = gs[~gs.yoy_closed].copy(); gs["band"] = pd.cut(gs.dist_km, [-1, 0, 2, 5, 10, 100], labels=["Inside the zone", "0–2 km", "2–5 km", "5–10 km", "> 10 km"]).astype(str); gs["io"] = np.where(gs.dist_km <= 0, "Inside the zone", "Outside the zone")
gb = pd.read_csv(U / "unit_geo_bus_ns.csv", dtype={"unit": str}); gb = gb[gb.prefix != "BX"].copy(); gb["grp"] = np.select([gb.share_in_crz.fillna(0) > 0.5, gb.share_in_crz.fillna(0) > 0], ["Mostly inside the zone", "Crossing the boundary"], "Entirely outside")
def strata(d, g, col, order):
    x = d.merge(g[["unit", col]], on="unit"); a = x.groupby(["origin", col])[["actual", "cfc"]].sum(); s = (100 * (a.actual / a.cfc - 1)).unstack(col); nul = s[is_null(s.index)]; c = nul - nul.mean(); r = a.xs(POLICY)
    return pd.DataFrame({"est": s.loc[POLICY], "lo": s.loc[POLICY] - c.quantile(0.95), "hi": s.loc[POLICY] - c.quantile(0.05), "units": g.groupby(col).size(), "riders_per_day": r.cfc / H, "extra_per_day": (r.actual - r.cfc) / H,
                         "placebo_mean": nul.mean(), "n_placebo_at_or_above": (nul.ge(s.loc[POLICY], axis=1)).sum(), "n_placebo": len(nul)}).loc[order]
ST = {"subway_io": strata(du, gs, "io", ["Inside the zone", "Outside the zone"]), "subway_band": strata(du, gs, "band", ["0–2 km", "2–5 km", "5–10 km", "> 10 km"]), "bus": strata(db, gb, "grp", ["Mostly inside the zone", "Crossing the boundary", "Entirely outside"])}
for k in ST: ST[k]["share_of_riders"] = 100 * ST[k].riders_per_day / (ST["subway_io"].riders_per_day.sum() if k.startswith("subway") else ST["bus"].riders_per_day.sum())
ru, rb = du[du.origin == POLICY].copy(), db[db.origin == POLICY].copy(); ru["eff"] = 100 * (ru.actual / ru.cfc - 1); ru["vol"] = ru.cfc / H; rb["eff"] = 100 * (rb.actual / rb.cfc - 1); rb["vol"] = rb.cfc / H
mp = pd.read_csv(paths["data_root"] + "/subway/patterns/station_mapping.csv"); mp["unit"] = mp.station_index.astype(str)
pts = gpd.read_file(G / "MTA_Subway_Stations_20251029.geojson").to_crs(3857); pts["complex_id"] = pts.complex_id.astype(int); cxs = pts.dissolve("complex_id"); cxs["geometry"] = cxs.geometry.centroid
S = gpd.GeoDataFrame(ru.merge(gs[["unit", "dist_km"]], on="unit").merge(mp, on="unit").merge(cxs[["geometry"]], left_on="station_id", right_index=True), geometry="geometry", crs=3857)
br = gpd.read_file(G / "bus_routes" / "bus_routes_nyc_dec2019.shp").to_crs(3857).dissolve("route_id").reset_index()
rb = rb.merge(gb[["unit"]], on="unit"); rb = rb[rb.vol >= 2000]; rb["key"] = rb.unit.where(rb.unit.isin(br.route_id), rb.unit.str.replace("+", "-SBS", regex=False)); B = br.merge(rb, left_on="route_id", right_on="key")
bxl = br[br.route_id.str.upper().str.match(r"^BX[0-9]")]; crz = gpd.read_file(G / "nyc_cp_boundary_poly_json.geojson").to_crs(3857)
xmin, ymin, xmax, ymax = gpd.read_file(G / "NYC_Census_Tracts_2020" / "NYC_Census_Tracts_2020.shp").to_crs(3857).total_bounds
pad = 0.015 * (xmax - xmin); xmin -= pad; xmax += pad; ymin -= pad; ymax += pad      # whole city, small margin
# greyscale-lightened OpenStreetMap basemap (credit goes in the caption, not on the figure) + census-tract, PUMA and borough boundaries
cx.set_cache_dir(str(REPO_ROOT / "outputs" / "_cache_tiles_osm")); cx.tile.USER_AGENT = "transit-figure-renderer/1.0 (one-off static research figure)"
img, ext = cx.bounds2img(xmin, ymin, xmax, ymax, ll=False, source=cx.providers.OpenStreetMap.Mapnik, zoom=12)
base = np.clip(255 * 0.66 + 0.34 * (img[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])), 0, 255).astype(np.uint8)
TR = gpd.read_file(G / "NYC_Census_Tracts_2020" / "NYC_Census_Tracts_2020.shp").to_crs(3857); BORO = TR.dissolve("boroname"); PUMA = gpd.read_file(G / "NYC_PUMA" / "NYC_Public_Use_Microdata_Areas_PUMAs_2010.shp").to_crs(3857)
# ------------------------------------------------------------------ layout (mm)
W = 183.0; ML, MR, MG = 4.0, 4.0, 7.0; mw = (W - ML - MR - MG) / 2; asp = (ymax - ymin) / (xmax - xmin); mh = mw * asp
TOP, CB, ROW2, BOT = 6.0, 19.0, 40.0, 11.0; Hh = TOP + mh + CB + ROW2 + BOT; fig = plt.figure(figsize=(W * MM, Hh * MM)); y_map = Hh - TOP - mh; y2 = BOT
def add(x, y, w, h): return fig.add_axes([x / W, y / Hh, w / W, h / Hh])
def plabel(x, ytop, s): fig.text(x / W, (ytop + 1.4) / Hh, s, fontsize=8, fontweight="bold", va="bottom", ha="left")
axa, axb = add(ML, y_map, mw, mh), add(ML + mw + MG, y_map, mw, mh); fq = 1 / np.cos(np.deg2rad(40.7))
for ax, s, k in ((axa, "a", "bus"), (axb, "b", "subway")):
    ax.imshow(base, extent=ext, cmap="gray", vmin=0, vmax=255, interpolation="bilinear", zorder=0, rasterized=True)
    TR.boundary.plot(ax=ax, color="#9B9892", linewidth=0.16, zorder=1); PUMA.boundary.plot(ax=ax, color="#5E5B56", linewidth=0.4, zorder=2); BORO.boundary.plot(ax=ax, color="#33312E", linewidth=0.6, zorder=3)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color(GREY)
    crz.boundary.plot(ax=ax, color=INK, linewidth=0.9, zorder=6); x0, y0 = xmax - 0.05 * (xmax - xmin) - 10000 * fq, ymin + 0.04 * (ymax - ymin)
    ax.plot([x0, x0 + 10000 * fq], [y0, y0], color=INK, lw=1.1, solid_capstyle="butt", zorder=8); ax.text(x0 + 5000 * fq, y0 + 0.012 * (ymax - ymin), "10 km", ha="center", va="bottom", fontsize=6, zorder=8)
    ax.annotate("N", xy=(0.955, 0.975), xytext=(0.955, 0.905), xycoords="axes fraction", ha="center", va="center", fontsize=6.5, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, shrinkA=0, shrinkB=2.5), zorder=8)
    plabel(ML if k == "bus" else ML + mw + MG, y_map + mh, s); fig.text(((ML if k == "bus" else ML + mw + MG) + 5.5) / W, (y_map + mh + 1.6) / Hh, ("Subway stations" if k == "subway" else "Bus routes") + f"  {HEAD[k]:+.1f}% overall", fontsize=7, fontweight="bold", color=DARK[k], va="bottom", ha="left")
size = lambda v: 5.0 + 62 * (np.asarray(v) / 120000.0); o = S.eff.abs().argsort()
axb.scatter(S.geometry.x.iloc[o], S.geometry.y.iloc[o], s=size(S.vol.iloc[o]), c=S.eff.iloc[o], cmap=CMAP, norm=NORM, linewidth=0, zorder=5)
hs = [axb.scatter([], [], s=size(v), facecolor="#8a8a8a", linewidth=0) for v in (5000, 30000, 100000)]
lg = axb.legend(hs, ["5,000", "30,000", "100,000"], title="Entries per day", loc="upper left", fontsize=6, title_fontsize=6, handletextpad=0.6, labelspacing=0.6, borderpad=0.5, frameon=True, framealpha=0.92, edgecolor="none", facecolor="white"); lg._legend_box.align = "left"
bxl.plot(ax=axa, color="#8C8984", linewidth=0.35, linestyle=(0, (2, 2)), zorder=4); B.assign(a=B.eff.abs()).sort_values("a").plot(ax=axa, column="eff", cmap=CMAP, norm=NORM, linewidth=1.15, zorder=5)
axa.legend([Line2D([], [], color="#B64342", lw=1.2), Line2D([], [], color="#8C8984", lw=0.6, ls=(0, (2, 2))), Line2D([], [], color=INK, lw=0.9), Line2D([], [], color="#33312E", lw=0.6), Line2D([], [], color="#5E5B56", lw=0.4), Line2D([], [], color="#9B9892", lw=0.25)], ["Bus route (coloured by effect)", "Bronx local routes (not analysed)", "Congestion Relief Zone", "Borough", "Public Use Microdata Area", "Census tract"], loc="upper left", fontsize=6, handletextpad=0.6, labelspacing=0.45, borderpad=0.5, frameon=True, framealpha=0.92, edgecolor="none", facecolor="white")
cax = add(W / 2 - 30.0, y_map - 6.0, 60.0, 2.0); cb = fig.colorbar(mpl.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal", extend="both", ticks=[-12, -6, 0, 6, 12])
cb.outline.set_linewidth(0.4); cb.ax.tick_params(length=1.8, width=0.4, labelsize=6); cb.set_ticklabels(["−12", "−6", "0", "+6", "+12"]); cb.set_label("Effect on ridership, 5 Jan – 30 Apr 2025 (%)", fontsize=6.5, labelpad=1.5)
fig.text((W / 2 - 33.0) / W, (y_map - 5.0) / Hh, "Decrease", ha="right", va="center", fontsize=6.5, color=BLUE); fig.text((W / 2 + 33.0) / W, (y_map - 5.0) / Hh, "Increase", ha="left", va="center", fontsize=6.5, color=RED)
# ---- c, d: group effects (dot + 90 % interval)
def forest(ax, rows, kind, xlim):
    yy = []; y = 0.0
    for lab, t in rows:
        if t is None: y -= 0.55; ax.text(xlim[0] - 0.02 * (xlim[1] - xlim[0]), y - 0.05, lab, fontsize=6, color=GREY, va="center", ha="right", style="italic", clip_on=False); y -= 0.95; continue
        ax.plot([t.lo, t.hi], [y, y], color=COL[kind], lw=1.1, solid_capstyle="butt", zorder=2); ax.scatter([t.est], [y], s=16, color=COL[kind], edgecolor="white", linewidth=0.4, zorder=3)
        ax.text(xlim[1], y, mn(f"{t.est:+.1f}%"), fontsize=6.5, fontweight="bold", color=DARK[kind], va="center", ha="right"); ax.text(xlim[0] - 0.02 * (xlim[1] - xlim[0]), y, lab, fontsize=6.5, color=INK, va="center", ha="right", clip_on=False); yy.append(y); y -= 1.0
    ax.axvline(0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1); ax.axvline(HEAD[kind], color=COL[kind], lw=0.6, ls=(0, (1, 1.5)), zorder=1); ax.set_xlim(*xlim); ax.set_ylim(y + 0.3, 0.75); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlabel("Effect over 116 days (%)")
    ax.text(HEAD[kind], 0.72, "Overall", fontsize=5.5, color=DARK[kind], va="top", ha="left" if kind == "bus" else "right", rotation=0, transform=ax.get_xaxis_transform() if False else ax.transData)
axc = add(33.0, y2, 31.0, ROW2); forest(axc, [(r, ST["bus"].loc[r]) for r in ST["bus"].index], "bus", (-3, 10)); axc.set_xticks([0, 4, 8])
axd = add(96.0, y2, 31.0, ROW2); forest(axd, [(r, ST["subway_io"].loc[r]) for r in ST["subway_io"].index] + [("Outside, by distance", None)] + [(r, ST["subway_band"].loc[r]) for r in ST["subway_band"].index], "subway", (-3, 10)); axd.set_xticks([0, 4, 8])
plabel(4.0, y2 + ROW2, "c"); fig.text(9.5 / W, (y2 + ROW2 + 1.6) / Hh, "Bus routes", fontsize=7, fontweight="bold", color=DARK["bus"], va="bottom", ha="left")
plabel(70.0, y2 + ROW2, "d"); fig.text(75.5 / W, (y2 + ROW2 + 1.6) / Hh, "Subway stations", fontsize=7, fontweight="bold", color=DARK["subway"], va="bottom", ha="left")
# ---- e: station effect vs distance
axe = add(147.0, y2, 32.5, ROW2); T = S.sort_values("dist_km"); rng = np.random.default_rng(1); xj = np.where(T.dist_km <= 0, rng.uniform(-0.9, -0.1, len(T)), T.dist_km)
axe.axvspan(-1.2, 0, color="#EFEBE4", lw=0, zorder=0); axe.scatter(xj, T.eff, s=1.2 + 10 * T.vol / 120000.0, color=COL["subway"], alpha=0.35, linewidth=0, zorder=2)
w = T.cfc.to_numpy(); num = pd.Series(T.actual.to_numpy() - w).rolling(61, center=True, min_periods=25, win_type="triang").sum(); den = pd.Series(w).rolling(61, center=True, min_periods=25, win_type="triang").sum(); out = (T.dist_km > 0).to_numpy()
axe.plot(T.dist_km.to_numpy()[out], (100 * num / den).to_numpy()[out], color=DARK["subway"], lw=1.3, zorder=4); axe.axhline(0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
axe.set_xlim(-1.2, 22); axe.set_ylim(-20, 25); axe.set_yticks([-20, -10, 0, 10, 20]); axe.set_yticklabels(["−20", "−10", "0", "10", "20"]); axe.set_xticks([0, 5, 10, 15, 20]); axe.set_xlabel("Distance from the zone (km)"); axe.set_ylabel("Station effect (%)")
axe.text(-0.6, 24.2, "Inside", fontsize=5.5, color=GREY, va="top", ha="center", rotation=90); axe.text(0.97, 0.05, "Line: running mean\nweighted by entries", transform=axe.transAxes, fontsize=5.8, color=DARK["subway"], va="bottom", ha="right", linespacing=1.3)
plabel(136.0, y2 + ROW2, "e"); fig.text(141.5 / W, (y2 + ROW2 + 1.6) / Hh, "Subway, by distance", fontsize=7, fontweight="bold", color=DARK["subway"], va="bottom", ha="left"); print("stations outside the plotted range:", int(((T.eff < -20) | (T.eff > 25)).sum()))
for e_, kw in (("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}), ("png", {"dpi": 300})): fig.savefig(FIG / f"fig2_spatial{SUF}.{e_}", **kw)
O = pd.concat(ST); O.to_csv(U / f"paper_strata_fig2{SUF}.csv"); pd.set_option("display.width", 220); print(O.round(2).to_string()); print(f"stations {len(S)}, routes mapped {len(B)}; figure {W:.0f} x {Hh:.0f} mm ->", FIG / f"fig2_spatial{SUF}.png")
