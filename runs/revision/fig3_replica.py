"""Figure 3 (Replica, all travel) - Nature-style, 183 mm, Arial, editable text. Weekly data, 17 weeks from 4 Jan 2025, raw model (no calibration).
Row 1 (a-d)  trips counted at the ORIGIN tract: citywide, starting inside the zone, starting outside; and weekday transit trips into the zone.
Row 2 (e-g)  trips counted at the DESTINATION tract: citywide, ending inside the zone, ending outside. and h, the effect by area (inside the zone; outside it by borough), origin and destination side by side, 90 % interval.
             Coloured line = real policy (red increase, blue decrease), grey = the 19 pseudo-policy dates of 2024.
Row 3 (i, j) spatially pooled effect by origin tract (i) and destination tract (j): each tract pooled with queen neighbours up to second order;
             full colour = outside all 19 pseudo-policy dates, faded = within their range.
Usage: python runs/revision/fig3_replica.py [timesfm|chronos_bolt]  -> main/fig3_replica[_chronos_bolt].{pdf,svg,tiff,png}
"""
import glob, os, sys
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import contextily as cx, geopandas as gpd, numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT
for f in glob.glob(os.path.expanduser("~/.fonts/arial*.ttf")): fm.fontManager.addfont(f)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "chronos_bolt"; SUF = "" if MODEL == "chronos_bolt" else "_" + MODEL      # main model: Chronos-Bolt (2026-11-26)
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; FIG = REPO_ROOT / "outputs" / "figures" / "paper" / "main"; G = REPO_ROOT / "geo_data"; MM = 1 / 25.4
POLICY, B0, H = pd.Timestamp("2025-01-04"), pd.Timestamp("2024-09-07"), 17
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2, "ytick.major.size": 2.2, "legend.frameon": False, "axes.spines.right": False, "axes.spines.top": False,
                     "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.labelpad": 2.5})
RED, BLUE, INK, GREY, LIGHT = "#B3282D", "#1F5FA8", "#1C1C1C", "#6E6E6E", "#C4C4C4"; mn = lambda s: s.replace("-", "−")
# softer scale than Fig. 2 (user, 2026-11-25): the pooled tract map saturates under the strong ramp
CMAP = LinearSegmentedColormap.from_list("bwr_nat", ["#0F4D92", "#7FA6CF", "#F2F2F2", "#E9A6A1", "#B64342"]); NORM = TwoSlopeNorm(vmin=-12, vcenter=0, vmax=12)
# ------------------------------------------------------------------ data
TOT = {d: pd.read_csv(U / f"densewk_replica{'' if d == 'O' else '_D'}_{MODEL}.csv", parse_dates=["origin", "date"]) for d in "OD"}; MODES = pd.read_csv(U / f"densewk_modes_{MODEL}.csv", parse_dates=["origin", "date"])
def paths(d):
    d = d.groupby(["origin", "h"], as_index=False)[["actual", "cf"]].sum().sort_values(["origin", "h"]); P = {o: 100 * (g.actual.cumsum().to_numpy() / g.cf.cumsum().to_numpy() - 1) for o, g in d.groupby("origin") if len(g) >= H}
    return P[POLICY][:H], {o: v[:H] for o, v in P.items() if o >= pd.Timestamp("2024-01-01") and o + pd.Timedelta(weeks=H - 1) < B0}
PAN = [[("O", "all", "All trips, citywide", 6), ("O", "inside", "Starting inside the zone", 8), ("O", "outside", "Starting outside the zone", 6), ("M", "public_transit", "Transit trips into the zone", 6)],
       [("D", "all", "All trips, citywide", 6), ("D", "inside", "Ending inside the zone", 8), ("D", "outside", "Ending outside the zone", 6), None]]
TRX = gpd.read_file(G / "NYC_Census_Tracts_2020" / "NYC_Census_Tracts_2020.shp").to_crs(3857); TRX["geoid"] = TRX.geoid.astype(str); BORO = TRX.dissolve("boroname"); PUMA = gpd.read_file(G / "NYC_PUMA" / "NYC_Public_Use_Microdata_Areas_PUMAs_2010.shp").to_crs(3857)
crz = gpd.read_file(G / "nyc_cp_boundary_poly_json.geojson").to_crs(3857)
MAP = {d: TRX.merge(pd.read_csv(U / f"replica_tract_effects{'' if d == 'O' else '_D'}_{MODEL}.csv", dtype={"unit": str}).rename(columns={"unit": "geoid"}), on="geoid", suffixes=("", "_t")) for d in "OD"}
# area totals for the inset bars: inside the zone and, outside it, by borough; 90 % interval from the centred error over the 2024 pseudo-policy dates
AREA = TRX.set_index("geoid").boroname.replace({"Manhattan": "Manhattan, outside"}); _z = gpd.read_file(G / "nyc_cp_boundary_poly_json.geojson").to_crs(3857).geometry.union_all()
AREA[(TRX.set_index("geoid").geometry.intersection(_z).area / TRX.set_index("geoid").geometry.area > 0.5)] = "Inside the zone"; ORDER = ["Inside the zone", "Manhattan, outside", "Bronx", "Brooklyn", "Queens", "Staten Island"]
def area_effects(d):
    u = pd.read_csv(U / f"denseunitwk_replica{'' if d == 'O' else '_D'}_{MODEL}.csv", parse_dates=["origin"], dtype={"unit": str}); u = u[u.h == H]; u["area"] = u.unit.map(AREA); u = u.dropna(subset=["area"])
    g = u.groupby(["origin", "area"])[["actual", "cf"]].sum(); e = (100 * (g.actual / g.cf - 1)).unstack("area"); nul = e[(e.index >= "2024-01-01") & (e.index + pd.Timedelta(weeks=H - 1) < B0)]; c = nul - nul.mean()
    return pd.DataFrame({"est": e.loc[POLICY], "lo": e.loc[POLICY] - c.quantile(0.95), "hi": e.loc[POLICY] - c.quantile(0.05), "n_below": nul.le(e.loc[POLICY], axis=1).sum(), "n_above": nul.ge(e.loc[POLICY], axis=1).sum(), "n": len(nul)}).loc[ORDER]
AE = {d: area_effects(d) for d in "OD"}
xmin, ymin, xmax, ymax = TRX.total_bounds; pad = 0.015 * (xmax - xmin); xmin -= pad; xmax += pad; ymin -= pad; ymax += pad
cx.set_cache_dir(str(REPO_ROOT / "outputs" / "_cache_tiles_osm")); cx.tile.USER_AGENT = "transit-figure-renderer/1.0 (one-off static research figure)"
img, ext = cx.bounds2img(xmin, ymin, xmax, ymax, ll=False, source=cx.providers.OpenStreetMap.Mapnik, zoom=12)      # credited in the caption, not on the figure
base = np.clip(255 * 0.66 + 0.34 * (img[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])), 0, 255).astype(np.uint8)
# enlarged view of the zone for the map insets
zb = crz.total_bounds; zcx, zcy = (zb[0] + zb[2]) / 2, (zb[1] + zb[3]) / 2; zh = 0.60 * (zb[3] - zb[1]); zw = zh / 1.45; ZB = (zcx - zw, zcy - zh, zcx + zw, zcy + zh)
zimg, zext = cx.bounds2img(*ZB, ll=False, source=cx.providers.OpenStreetMap.Mapnik, zoom=13); zbase = np.clip(255 * 0.66 + 0.34 * (zimg[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])), 0, 255).astype(np.uint8)
# ------------------------------------------------------------------ layout (mm)
W = 183.0; L, R, PG = 12.0, 3.0, 9.5; pw = (W - L - R - 3 * PG) / 4; PX = [L + i * (pw + PG) for i in range(4)]; ML, MG = 4.0, 7.0; mw = (W - 2 * ML - MG) / 2; mh = mw * (ymax - ymin) / (xmax - xmin)
TOP, PH, RG, XG, CB = 9.0, 29.0, 17.0, 17.0, 17.0; Hh = TOP + 2 * PH + RG + XG + mh + CB; fig = plt.figure(figsize=(W * MM, Hh * MM)); YP = [Hh - TOP - PH, Hh - TOP - 2 * PH - RG]; y_map = CB
def add(x, y, w, h): return fig.add_axes([x / W, y / Hh, w / W, h / Hh])
def plabel(x, ytop, s): fig.text(x / W, (ytop + 1.4) / Hh, s, fontsize=8, fontweight="bold", va="bottom", ha="left")
letters = iter("abcdefghij"); x = np.arange(1, H + 1); stats = []
for r, row in enumerate(PAN):
    fig.text((L - 9.0) / W, (YP[r] + PH + 5.6) / Hh, "Trips counted where they start" if r == 0 else "Trips counted where they end", fontsize=7, fontweight="bold", color=INK, va="bottom", ha="left")
    for c, p in enumerate(row):
        if p is None: continue
        src, key, title, lim = p; d = MODES[MODES["mode"] == key] if src == "M" else TOT[src][TOT[src].grp == key]; real, nul = paths(d); ends = pd.Series({o: v[-1] for o, v in nul.items()}); col = RED if real[-1] > 0 else BLUE
        ax = add(PX[c], YP[r], pw, PH)
        for v in nul.values(): ax.plot(x, v, color=LIGHT, lw=0.45, zorder=2)
        ax.plot(x, real, color=col, lw=1.3, zorder=4); ax.axhline(0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1); ax.set_xlim(0, 17.6); ax.set_xticks([0, 4, 8, 12, 17]); ax.set_ylim(-lim, lim)
        tk = np.arange(-lim, lim + 0.1, 2 if lim <= 6 else 4); ax.set_yticks(tk); ax.set_yticklabels([mn(f"{int(t)}") for t in tk]); ax.tick_params(axis="x", pad=2)
        ax.text(0.0, 1.035, title, transform=ax.transAxes, fontsize=6.5, color=INK, va="bottom", ha="left"); up = real[-1] > 0
        ax.text(0.97, 0.06 if up else 0.94, mn(f"{real[-1]:+.1f}%"), transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=col, va="bottom" if up else "top", ha="right")
        if c == 0: ax.set_ylabel("Cumulative effect (%)")
        if r == 1: ax.set_xlabel("Weeks since start")
        plabel(PX[c] - 9.0, YP[r] + PH, next(letters))
        stats.append(dict(row="origin" if r == 0 else "destination", panel=title, real=real[-1], placebo_min=ends.min(), placebo_max=ends.max(), n=len(ends), n_at_or_above=int((ends >= real[-1]).sum()), n_at_or_below=int((ends <= real[-1]).sum()),
                          weeks_outside_all=int(sum((real[h] > max(v[h] for v in nul.values())) or (real[h] < min(v[h] for v in nul.values())) for h in range(H)))))
# h: effect by area, origin (solid) and destination (light), in the free slot of row 2
ax = add(PX[3] + 13.0, YP[1], pw - 13.0, PH); yy = np.arange(len(ORDER))[::-1].astype(float)
for d, off, al in (("O", 0.19, 1.0), ("D", -0.19, 0.45)):
    t = AE[d]; ax.barh(yy + off, t.est, height=0.34, color=[RED if v > 0 else BLUE for v in t.est], alpha=al, linewidth=0, zorder=2); ax.hlines(yy + off, t.lo, t.hi, color=INK, lw=0.55, zorder=3)
    stats.extend(dict(row="origin" if d == "O" else "destination", panel="area: " + k, real=r.est, placebo_min=r.lo, placebo_max=r.hi, n=r.n, n_at_or_above=r.n_above, n_at_or_below=r.n_below) for k, r in t.iterrows())
ax.axvline(0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1); ax.set_yticks(yy); ax.set_yticklabels(["Inside the zone", "Manhattan,\noutside the zone", "Bronx", "Brooklyn", "Queens", "Staten Island"], fontsize=6, linespacing=0.95); ax.tick_params(axis="y", length=0, pad=2); ax.spines["left"].set_visible(False)
ax.set_xlim(-12, 12); ax.set_xticks([-8, 0, 8]); ax.set_xticklabels(["−8", "0", "+8"]); ax.tick_params(axis="x", pad=2); ax.set_ylim(-0.6, len(ORDER) - 0.4); ax.set_xlabel("Effect (%)")
fig.text(PX[3] / W, (YP[1] + PH + 1.35) / Hh, "By area", fontsize=6.5, color=INK, va="bottom", ha="left"); plabel(PX[3] - 9.0, YP[1] + PH, next(letters))
ax.legend(handles=[mpl.patches.Patch(facecolor="#555555", edgecolor="none", label="By origin"), mpl.patches.Patch(facecolor="#555555", alpha=0.45, edgecolor="none", label="By destination")], loc="lower right", bbox_to_anchor=(1.02, 1.0), fontsize=5.5, handlelength=1.0, handleheight=0.7, labelspacing=0.25, borderpad=0, borderaxespad=0.2)
fig.legend(handles=[Line2D([], [], color=RED, lw=1.3, label="Real policy, increase"), Line2D([], [], color=BLUE, lw=1.3, label="Real policy, decrease"), Line2D([], [], color=LIGHT, lw=0.8, label="Pseudo-policy dates in 2024 (no policy, n = 19)")],
           loc="lower right", bbox_to_anchor=((W - R) / W, (YP[0] + PH + 5.0) / Hh), ncol=3, fontsize=6.5, handlelength=1.8, columnspacing=1.8, borderpad=0, borderaxespad=0)
# ---- maps
fq = 1 / np.cos(np.deg2rad(40.7))
for j, d in enumerate("OD"):
    ax = add(ML + j * (mw + MG), y_map, mw, mh); M = MAP[d]
    ax.imshow(base, extent=ext, cmap="gray", vmin=0, vmax=255, interpolation="bilinear", zorder=0, rasterized=True); ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color(GREY)
    M.plot(ax=ax, column="effect_smooth", cmap=CMAP, norm=NORM, linewidth=0, alpha=0.92, zorder=2, rasterized=True); M[~M.smooth_outside.astype(bool)].plot(ax=ax, color="white", alpha=0.72, linewidth=0, zorder=3, rasterized=True)
    PUMA.boundary.plot(ax=ax, color="#5E5B56", linewidth=0.3, zorder=4); BORO.boundary.plot(ax=ax, color="#33312E", linewidth=0.55, zorder=5); crz.boundary.plot(ax=ax, color=INK, linewidth=1.1, zorder=6)
    x0, y0 = xmax - 0.05 * (xmax - xmin) - 10000 * fq, ymin + 0.04 * (ymax - ymin); ax.plot([x0, x0 + 10000 * fq], [y0, y0], color=INK, lw=1.1, solid_capstyle="butt", zorder=8); ax.text(x0 + 5000 * fq, y0 + 0.012 * (ymax - ymin), "10 km", ha="center", va="bottom", fontsize=6, zorder=8)
    ax.annotate("N", xy=(0.955, 0.975), xytext=(0.955, 0.905), xycoords="axes fraction", ha="center", va="center", fontsize=6.5, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, shrinkA=0, shrinkB=2.5), zorder=8)
    plabel(ML + j * (mw + MG), y_map + mh, next(letters)); fig.text((ML + j * (mw + MG) + 5.5) / W, (y_map + mh + 1.6) / Hh, "By origin tract" if d == "O" else "By destination tract", fontsize=7, fontweight="bold", color=INK, va="bottom", ha="left")
    if j == 0:
        ax.legend([mpl.patches.Patch(facecolor="#B64342", edgecolor="none"), mpl.patches.Patch(facecolor="#EBCFCD", edgecolor="none"), Line2D([], [], color=INK, lw=1.1), Line2D([], [], color="#33312E", lw=0.55), Line2D([], [], color="#5E5B56", lw=0.35)],
                  ["Outside all pseudo-policy dates", "Within the pseudo-policy range", "Congestion Relief Zone", "Borough", "Public Use Microdata Area"], loc="upper left", fontsize=6, handletextpad=0.6, labelspacing=0.45, borderpad=0.5, frameon=True, framealpha=0.92, edgecolor="none", facecolor="white")
    # inset: the zone enlarged (same colours and fading)
    iw = 0.285; ins = ax.inset_axes([0.045, 0.375, iw, iw * 1.45 * mw / mh]); ins.set_zorder(9)
    ins.imshow(zbase, extent=zext, cmap="gray", vmin=0, vmax=255, interpolation="bilinear", zorder=0, rasterized=True); Mz = M.cx[ZB[0]:ZB[2], ZB[1]:ZB[3]]
    Mz.plot(ax=ins, column="effect_smooth", cmap=CMAP, norm=NORM, linewidth=0.15, edgecolor="white", alpha=0.92, zorder=2, rasterized=True); Mz[~Mz.smooth_outside.astype(bool)].plot(ax=ins, color="white", alpha=0.72, linewidth=0, zorder=3, rasterized=True)
    BORO.boundary.plot(ax=ins, color="#33312E", linewidth=0.5, zorder=5); crz.boundary.plot(ax=ins, color=INK, linewidth=1.1, zorder=6); ins.set_xlim(ZB[0], ZB[2]); ins.set_ylim(ZB[1], ZB[3]); ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values(): sp.set_visible(True); sp.set_linewidth(0.7); sp.set_color(INK)
    sx, sy = ZB[2] - 0.07 * (ZB[2] - ZB[0]) - 2000 * fq, ZB[1] + 0.035 * (ZB[3] - ZB[1]); ins.plot([sx, sx + 2000 * fq], [sy, sy], color=INK, lw=1.0, solid_capstyle="butt", zorder=8); ins.text(sx + 1000 * fq, sy + 0.012 * (ZB[3] - ZB[1]), "2 km", ha="center", va="bottom", fontsize=5.5, zorder=8)
    ax.indicate_inset_zoom(ins, edgecolor=INK, linewidth=0.6, alpha=1.0)
    inz = M.in_zone.astype(bool); stats.append(dict(row="origin" if d == "O" else "destination", panel="map: tracts inside zone below all placebos (%)", real=100 * (M.effect_smooth < M.smooth_placebo_min)[inz].mean()))
cax = add(W / 2 - 30.0, y_map - 6.5, 60.0, 2.0); cb = fig.colorbar(mpl.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal", extend="both", ticks=[-12, -8, -4, 0, 4, 8, 12])
cb.outline.set_linewidth(0.4); cb.ax.tick_params(length=1.8, width=0.4, labelsize=6); cb.set_ticklabels(["−12", "−8", "−4", "0", "+4", "+8", "+12"]); cb.set_label("Effect on trips in the tract and its neighbours, 4 Jan – 26 Apr 2025 (%)", fontsize=6.5, labelpad=1.5)
fig.text((W / 2 - 33.0) / W, (y_map - 5.5) / Hh, "Decrease", ha="right", va="center", fontsize=6.5, color=BLUE); fig.text((W / 2 + 33.0) / W, (y_map - 5.5) / Hh, "Increase", ha="left", va="center", fontsize=6.5, color=RED)
for e_, kw in (("pdf", {"dpi": 600}), ("svg", {"dpi": 600}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}), ("png", {"dpi": 300})): fig.savefig(FIG / f"fig3_replica{SUF}.{e_}", **kw)
pd.set_option("display.width", 220); print(pd.DataFrame(stats).round(2).to_string(index=False)); print(f"figure {W:.0f} x {Hh:.0f} mm ->", FIG / f"fig3_replica{SUF}.png")
