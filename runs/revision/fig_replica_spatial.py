"""Spatial distribution of the Replica effect: all trips by origin census tract, 4 Jan - 26 Apr 2025 (17 weeks), raw foundation model
(no calibration, as in the other Replica results). a, citywide; b, the Congestion Relief Zone and its surroundings.
Red = increase, blue = decrease; same colour scale as the subway / bus map. Tracts with negligible volume are left blank.
Also prints, per area, the share of tracts whose effect lies outside all 2024 pseudo-policy dates.
Usage: python runs/revision/fig_replica_spatial.py [timesfm|chronos_bolt]   -> spatial/fig_replica_spatial_<model>.{pdf,svg,tiff,png}, uq/replica_tract_effects_<model>.csv
"""
import sys
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import contextily as cx, geopandas as gpd, numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT
model = sys.argv[1] if len(sys.argv) > 1 else "timesfm"; DIR = sys.argv[2] if len(sys.argv) > 2 else "O"; SFX = "" if DIR == "O" else "_D"; WORD = "starting" if DIR == "O" else "ending"
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; FIG = REPO_ROOT / "outputs" / "figures" / "paper" / "spatial"; G = REPO_ROOT / "geo_data"; MM = 1 / 25.4; POLICY = pd.Timestamp("2025-01-04")
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.size": 7, "axes.linewidth": 0.8, "legend.frameon": False, "axes.spines.right": False, "axes.spines.top": False})
RED, BLUE, INK, GREY = "#B64342", "#0F4D92", "#1a1a1a", "#6b6b6b"
CMAP = LinearSegmentedColormap.from_list("bwr_nat", [BLUE, "#7FA6CF", "#F2F2F2", "#E9A6A1", RED]); NORM = TwoSlopeNorm(vmin=-12, vcenter=0, vmax=12)
d = pd.read_csv(U / f"denseunitwk_replica{SFX}_{model}.csv", parse_dates=["origin"], dtype={"unit": str}); d = d[(d.h == 17) & (d.cf > 0)]
E = (100 * (d.set_index(["unit", "origin"]).actual / d.set_index(["unit", "origin"]).cf - 1)).unstack("origin"); real = d[d.origin == POLICY].set_index("unit")
nul = [o for o in E.columns if o >= pd.Timestamp("2024-01-01") and o + pd.Timedelta(weeks=16) < pd.Timestamp("2024-09-07")]
T = pd.DataFrame({"effect_pct": E[POLICY], "trips_per_day_cf": real.cf / 17, "extra_trips_per_day": (real.actual - real.cf) / 17, "placebo_min": E[nul].min(axis=1), "placebo_max": E[nul].max(axis=1)}).dropna(subset=["effect_pct"])
T = T[T.trips_per_day_cf > T.trips_per_day_cf.sum() * 2e-5]; T["outside_placebos"] = (T.effect_pct > T.placebo_max) | (T.effect_pct < T.placebo_min)
tr = gpd.read_file(G / "NYC_Census_Tracts_2020" / "NYC_Census_Tracts_2020.shp").to_crs(3857); tr["geoid"] = tr.geoid.astype(str); crz = gpd.read_file(G / "nyc_cp_boundary_poly_json.geojson").to_crs(3857); zone = crz.geometry.union_all()
tr["in_zone"] = tr.geometry.intersection(zone).area / tr.geometry.area > 0.5
# spatial smoothing: pool each tract with its neighbours up to second order (queen contiguity): effect = sum(actual) / sum(cf) - 1 over the pool.
# The same pooling is applied to every pseudo-policy date, so the pooled effect can be ranked against its own placebos.
from libpysal.weights import Queen
n_all = len(T); T = T[T.index.isin(tr.geoid)]; print(f"tracts in the five boroughs: {len(T)} of {n_all} (destination files also hold tracts outside the city, not mapped)")
gT = tr.set_index("geoid").loc[T.index]; wq = Queen.from_dataframe(gT.reset_index(), use_index=False, silence_warnings=True); K = np.eye(len(gT))
for i, nb in wq.neighbors.items(): K[i, nb] = 1
K = ((K @ K) > 0).astype(float); Aw = d.pivot(index="unit", columns="origin", values="actual").reindex(T.index).fillna(0); Cw = d.pivot(index="unit", columns="origin", values="cf").reindex(T.index).fillna(0)
Es = pd.DataFrame(100 * ((K @ Aw.to_numpy()) / (K @ Cw.to_numpy()) - 1), index=T.index, columns=Aw.columns)
T["effect_smooth"] = Es[POLICY]; T["smooth_placebo_min"] = Es[nul].min(axis=1); T["smooth_placebo_max"] = Es[nul].max(axis=1); T["pool_size"] = K.sum(1)
T["smooth_outside"] = (T.effect_smooth > T.smooth_placebo_max) | (T.effect_smooth < T.smooth_placebo_min); T["smooth_n_abs_ge"] = (Es[nul].abs().to_numpy() >= T.effect_smooth.abs().to_numpy()[:, None]).sum(1)
M = tr.merge(T, left_on="geoid", right_index=True); T = T.join(tr.set_index("geoid")[["boroname", "in_zone", "ntaname"]]); 
T.to_csv(U / f"replica_tract_effects{SFX}_{model}.csv"); T["area"] = np.where(T.in_zone, "Inside the zone", T.boroname.where(T.boroname != "Manhattan", "Manhattan, outside the zone"))
print(f"{model}: {len(T)} tracts, placebo dates {len(nul)}; volume-weighted effect {100 * T.extra_trips_per_day.sum() / T.trips_per_day_cf.sum():+.2f}%")
print(T.groupby("area").apply(lambda q: pd.Series(dict(tracts=len(q), trips_per_day=q.trips_per_day_cf.sum(), effect=100 * q.extra_trips_per_day.sum() / q.trips_per_day_cf.sum(), median_tract=q.effect_pct.median(), share_negative=100 * (q.effect_pct < 0).mean(),
      below_all_placebos=100 * (q.effect_pct < q.placebo_min).mean(), above_all_placebos=100 * (q.effect_pct > q.placebo_max).mean()))).round(1).to_string())
print(f"\nSmoothed (pool of tract + neighbours up to 2nd order, median {int(T.pool_size.median())} tracts); chance rate of falling below or above all {len(nul)} placebos = {100 / (len(nul) + 1):.0f}% each")
print(T.groupby("area").apply(lambda q: pd.Series(dict(tracts=len(q), median_smoothed=q.effect_smooth.median(), below_all_placebos=100 * (q.effect_smooth < q.smooth_placebo_min).mean(), above_all_placebos=100 * (q.effect_smooth > q.smooth_placebo_max).mean()))).round(1).to_string())
nta = T.groupby(["area", "ntaname"]).apply(lambda q: pd.Series(dict(tracts=len(q), trips=q.trips_per_day_cf.sum(), effect=100 * q.extra_trips_per_day.sum() / q.trips_per_day_cf.sum()))).reset_index(); nta = nta[nta.trips >= 20000].sort_values("effect")
print("\nNeighbourhoods, largest decreases"); print(nta.head(12).round(1).to_string(index=False)); print("\nNeighbourhoods, largest increases"); print(nta.tail(12).round(1).to_string(index=False))
xmin, ymin, xmax, ymax = tr.total_bounds; pad = 800; xmin, xmax, ymin, ymax = xmin - pad, xmax + pad, ymin - pad, ymax + pad
cx.set_cache_dir(str(REPO_ROOT / "outputs" / "_cache_tiles_osm")); cx.tile.USER_AGENT = "transit-figure-renderer/1.0 (one-off static research figure)"
def basemap(b, zoom):
    img, ext = cx.bounds2img(*b, ll=False, source=cx.providers.OpenStreetMap.Mapnik, zoom=zoom); grey = img[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114]); return np.clip(255 * 0.58 + 0.42 * grey, 0, 255).astype(np.uint8), ext
W = 183 * MM; asp = (ymax - ymin) / (xmax - xmin); pw = (W - 14 * MM) / 2; H = pw * asp + 24 * MM
zb = zone.bounds; cxm, cym = (zb[0] + zb[2]) / 2, (zb[1] + zb[3]) / 2 + 1500; hw = 8200; B2 = (cxm - hw, cym - hw * asp, cxm + hw, cym + hw * asp)
def draw(col, sig, suffix, label):
    fig = plt.figure(figsize=(W, H)); f = 1 / np.cos(np.deg2rad(40.7))
    for j, (b, zoom, km) in enumerate((((xmin, ymin, xmax, ymax), 12, 10), (B2, 13, 2))):
        ax = fig.add_axes([(4 * MM + j * (pw + 6 * MM)) / W, 18 * MM / H, pw / W, pw * asp / H]); base, ext = basemap(b, zoom)
        ax.imshow(base, extent=ext, cmap="gray", vmin=0, vmax=255, interpolation="bilinear", zorder=0, rasterized=True); ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3]); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color(GREY)
        M.plot(ax=ax, column=col, cmap=CMAP, norm=NORM, linewidth=(0.15 if j else 0.05) if sig is None else 0, edgecolor="white", alpha=0.92, zorder=4, rasterized=True)
        if sig is not None and (~M[sig]).any(): M[~M[sig]].plot(ax=ax, color="white", alpha=0.72, linewidth=0, zorder=5, rasterized=True)
        crz.boundary.plot(ax=ax, color=INK, linewidth=1.0, zorder=6)
        ax.text(0.0, 1.015, "ab"[j], transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left"); dx, dy = b[2] - b[0], b[3] - b[1]
        x0, y0 = b[2] - 0.05 * dx - km * 1000 * f, b[1] + 0.045 * dy; ax.add_patch(mpl.patches.Rectangle((x0 - 0.03 * dx, b[1]), b[2] - x0 + 0.03 * dx, 0.105 * dy, facecolor="white", edgecolor="none", alpha=0.88, zorder=7)); ax.plot([x0, x0 + km * 1000 * f], [y0, y0], color=INK, lw=1.2, solid_capstyle="butt", zorder=8)
        for xx in (x0, x0 + km * 500 * f, x0 + km * 1000 * f): ax.plot([xx, xx], [y0, y0 + 0.008 * dy], color=INK, lw=0.8, zorder=8)
        for xx, s in ((x0, "0"), (x0 + km * 500 * f, f"{km // 2}"), (x0 + km * 1000 * f, f"{km} km")): ax.text(xx, y0 + 0.014 * dy, s, ha="center", va="bottom", fontsize=5.5, zorder=8)
        ax.annotate("N", xy=(0.955, 0.965), xytext=(0.955, 0.875), xycoords="axes fraction", ha="center", va="center", fontsize=6.5, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, shrinkA=0, shrinkB=3), zorder=8)
        ax.text(0.995, 0.004, "Basemap © OpenStreetMap contributors", transform=ax.transAxes, ha="right", va="bottom", fontsize=5, color=GREY, zorder=8)
        if j == 0:
            ax.add_patch(mpl.patches.Rectangle((B2[0], B2[1]), B2[2] - B2[0], B2[3] - B2[1], fill=False, edgecolor=INK, linewidth=0.6, linestyle=(0, (3, 2)), zorder=7))
            ax.legend([Line2D([], [], color=INK, lw=1.0), Line2D([], [], color=INK, lw=0.6, ls=(0, (3, 2)))] + ([] if sig is None else [mpl.patches.Patch(facecolor=RED, edgecolor="none"), mpl.patches.Patch(facecolor="#EBCFCD", edgecolor="none")]), ["Congestion Relief Zone", "Area shown in b"] + ([] if sig is None else ["Outside all pseudo-policy dates", "Within the pseudo-policy range"]), loc="upper left", fontsize=6, handletextpad=0.6, labelspacing=0.5, borderpad=0.5, frameon=True, framealpha=0.9, edgecolor="none", facecolor="white")
    cax = fig.add_axes([(W / 2 - 32 * MM) / W, 10.5 * MM / H, 64 * MM / W, 2.2 * MM / H]); cb = fig.colorbar(mpl.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal", extend="both", ticks=[-12, -8, -4, 0, 4, 8, 12])
    cb.outline.set_linewidth(0.5); cb.ax.tick_params(length=2, width=0.5, labelsize=6); cb.set_ticklabels(["−12", "−8", "−4", "0", "+4", "+8", "+12"]); cb.set_label(label, fontsize=7, labelpad=2)
    fig.text((W / 2 - 34 * MM) / W, 11.6 * MM / H, "Decrease", ha="right", va="center", fontsize=6.5, color=BLUE); fig.text((W / 2 + 34 * MM) / W, 11.6 * MM / H, "Increase", ha="left", va="center", fontsize=6.5, color=RED)
    for e, kw in (("pdf", {"dpi": 600}), ("svg", {"dpi": 600}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}), ("png", {"dpi": 300})): fig.savefig(FIG / f"fig_replica_spatial{suffix}{SFX}_{model}.{e}", **kw)
    plt.close(fig); print("wrote", FIG / f"fig_replica_spatial{suffix}{SFX}_{model}.png")
draw("effect_pct", None, "", f"Effect on trips {WORD} in the tract, 4 Jan – 26 Apr 2025 (%)")
draw("effect_smooth", "smooth_outside", "_smooth", f"Effect on trips {WORD} in the tract and its neighbours, 4 Jan – 26 Apr 2025 (%)")
