"""Spatial regression of the final TimesFM unit effects on neighbourhood characteristics (same targets and features as sociodemo_shap.py).
Units stay native: subway stations, bus routes (Bronx local excluded), Replica origin tracts.
Spatial weights (row-standardised): stations = 6 nearest stations; routes = 6 nearest routes by line-to-line distance;
tracts = queen contiguity (islands attached to the nearest tract).
y and X are z-scored, so coefficients read "s.d. of effect per s.d. of the variable" and are comparable across modes.
Variables are dropped stepwise while any VIF > 10. Models: OLS (with Moran's I of residuals and LM tests), spatial lag (ML), spatial error (ML).
Writes uq/sociodemo_spatial_{coef,fit}.csv, figure fig_sociodemo_spatial.{pdf,svg,tiff,png}
"""
import warnings
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd, numpy as np, pandas as pd, libpysal
from spreg import OLS, ML_Error, ML_Lag
from statsmodels.stats.outliers_influence import variance_inflation_factor
from nyc_cp.config import REPO_ROOT, load_paths
warnings.filterwarnings("ignore")
src = open(REPO_ROOT / "runs/revision/sociodemo_shap.py").read().split("def model()")[0]; exec(src)      # U, FIG, MM, VARS, F, LAB, targets()
G = REPO_ROOT / "geo_data"; paths = load_paths()
def geoms(mode):
    if mode == "subway":
        st = gpd.read_file(G / "MTA_Subway_Stations_20251029.geojson").to_crs(2263); st["complex_id"] = st.complex_id.astype(int); cx = st.dissolve("complex_id"); cx["geometry"] = cx.geometry.centroid
        mp = pd.read_csv(paths["data_root"] + "/subway/patterns/station_mapping.csv"); mp["unit"] = mp.station_index.astype(str); return gpd.GeoDataFrame(mp.merge(cx[["geometry"]], left_on="station_id", right_index=True), geometry="geometry", crs=2263).set_index("unit").geometry
    if mode == "bus":
        br = gpd.read_file(G / "bus_routes" / "bus_routes_nyc_dec2019.shp").to_crs(2263).dissolve("route_id"); geo = pd.read_csv(U / "unit_geo_bus_ns.csv"); geo["key"] = geo.unit.str.replace("+", "-SBS", regex=False)
        geo["rid"] = np.where(geo.unit.isin(br.index), geo.unit, np.where(geo.key.isin(br.index), geo.key, None)); g2 = geo.dropna(subset=["rid"]); return gpd.GeoSeries(br.geometry.reindex(g2.rid).to_numpy(), index=g2.unit.to_numpy(), crs=2263)
    tr = gpd.read_file(G / "NYC_Census_Tracts_2020" / "NYC_Census_Tracts_2020.shp").to_crs(2263); idc = [c for c in tr.columns if c.lower() == "geoid"][0]; return tr.set_index(tr[idc].astype(str)).geometry
def weights(mode, g):
    if mode == "subway": w = libpysal.weights.KNN.from_dataframe(gpd.GeoDataFrame(geometry=g.reset_index(drop=True)), k=6)
    elif mode == "bus":
        a = g.reset_index(drop=True); Dm = np.vstack([a.distance(a.iloc[i]).to_numpy() for i in range(len(a))]); np.fill_diagonal(Dm, np.inf); nb = {i: list(np.argsort(Dm[i])[:6]) for i in range(len(a))}; w = libpysal.weights.W(nb)
    else:
        gg = gpd.GeoDataFrame(geometry=g.reset_index(drop=True)); w = libpysal.weights.Queen.from_dataframe(gg, use_index=False)
        if w.islands: w = libpysal.weights.util.attach_islands(w, libpysal.weights.KNN.from_dataframe(gpd.GeoDataFrame(geometry=gg.geometry.centroid), k=1))
    w.transform = "r"; return w
def vif_filter(Z, thr=10.0):
    cols = list(Z.columns); dropped = []
    while True:
        v = [variance_inflation_factor(np.column_stack([np.ones(len(Z)), Z[cols].to_numpy()]), i + 1) for i in range(len(cols))]
        if max(v) < thr: return cols, dropped
        dropped.append(cols.pop(int(np.argmax(v))))
COEF, FIT = [], []
for mode, (title, y, vol) in targets().items():
    S = pd.read_csv(U / f"sociodemo_{mode}.csv", dtype={"unit": str}).set_index("unit"); D = S[F].join(y.rename("y"), how="inner").join(vol.rename("vol")).dropna(subset=["y"])
    if mode == "bus": D = D[D.vol >= 300]
    g = geoms(mode); D = D[D.index.isin(g.index)]; lo, hi = D.y.quantile([0.01, 0.99]); D["y"] = D.y.clip(lo, hi); X = D[F].astype(float); X = X.fillna(X.median()); Z = (X - X.mean()) / X.std(ddof=0)
    keep, dropped = vif_filter(Z); yz = ((D.y - D.y.mean()) / D.y.std(ddof=0)).to_numpy().reshape(-1, 1); Xm = Z[keep].to_numpy(); w = weights(mode, g.reindex(D.index))
    ols = OLS(yz, Xm, w=w, spat_diag=True, moran=True, name_y="effect", name_x=keep); lag = ML_Lag(yz, Xm, w=w, name_y="effect", name_x=keep); err = ML_Error(yz, Xm, w=w, name_y="effect", name_x=keep)
    def tab(m, names, stat): return pd.DataFrame({"variable": names, "coef": np.array(m.betas).flatten(), "se": np.array(m.std_err).flatten(), "p": [s[1] for s in stat]})
    T = {"OLS": tab(ols, ["CONSTANT", *keep], ols.t_stat), "Spatial lag": tab(lag, ["CONSTANT", *keep, "rho (spatial lag of effect)"], lag.z_stat), "Spatial error": tab(err, ["CONSTANT", *keep, "lambda (spatial error)"], err.z_stat)}
    for k, t in T.items(): COEF.append(t.assign(mode=title, model=k))
    rho, lam = T["Spatial lag"].iloc[-1], T["Spatial error"].iloc[-1]
    FIT.append(dict(mode=title, n=len(D), sd_effect_pp=D.y.std(ddof=0), dropped_for_vif=", ".join(LAB[d] for d in dropped) or "none", ols_r2=ols.r2, ols_adj_r2=ols.ar2, moran_resid=ols.moran_res[0], moran_p=ols.moran_res[2], lm_lag_p=ols.lm_lag[1], lm_error_p=ols.lm_error[1],
                    rho=rho.coef, rho_p=rho.p, lag_pseudo_r2=lag.pr2, lag_aic=lag.aic, lam=lam.coef, lam_p=lam.p, err_pseudo_r2=err.pr2, err_aic=err.aic, ols_aic=ols.aic))
COEF, FIT = pd.concat(COEF), pd.DataFrame(FIT); COEF["label"] = COEF.variable.map(LAB).fillna(COEF.variable); COEF["ci_lo"] = COEF.coef - 1.96 * COEF.se; COEF["ci_hi"] = COEF.coef + 1.96 * COEF.se
COEF.to_csv(U / f"sociodemo_spatial_coef{SUF}.csv", index=False); FIT.to_csv(U / f"sociodemo_spatial_fit{SUF}.csv", index=False)
pd.set_option("display.width", 250); print(FIT.round(3).T.to_string())
star = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""
for k in ("Spatial lag", "OLS"):
    t = COEF[(COEF.model == k) & (COEF.variable != "CONSTANT")].copy(); t["cell"] = [f"{c:+.2f}{star(p)}" for c, p in zip(t.coef, t.p)]
    print(f"\n{k}: standardised coefficients (* p<0.05, ** p<0.01, *** p<0.001, . p<0.1)"); print(t.pivot(index="label", columns="mode", values="cell").reindex([l for _, l in VARS] + ["rho (spatial lag of effect)"]).fillna("dropped / n.a.").to_string())
# ---------------- figure: spatial-lag coefficients with 95 % intervals
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2, "ytick.major.size": 2.2, "legend.frameon": False, "axes.spines.right": False, "axes.spines.top": False, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6})
RED, BLUE, INK, GREY, LIGHT = "#B64342", "#0F4D92", "#272727", "#767676", "#BDBDBD"; rows = [l for _, l in VARS] + ["rho (spatial lag of effect)"]; ylab = [l for _, l in VARS] + ["Effect in neighbouring units"]
fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 80 * MM), sharey=True); L = COEF[COEF.model == "Spatial lag"]
for ax, (title, short) in zip(axes, (("Subway stations", "Subway stations"), ("Bus routes", "Bus routes"), ("All trips by origin tract (Replica)", "Origin tracts (Replica)"))):
    t = L[L["mode"] == title].set_index("label"); f = FIT.set_index("mode").loc[title]
    for i, r in enumerate(rows):
        y = -i - (0.5 if i == len(rows) - 1 else 0)
        if r not in t.index: ax.text(0.04, y, "dropped (collinear)", fontsize=5.2, color=GREY, ha="left", va="center"); continue
        c, lo, hi, p = t.loc[r, ["coef", "ci_lo", "ci_hi", "p"]]; sig = p < 0.05; col = (RED if c > 0 else BLUE) if sig else LIGHT
        ax.plot([lo, hi], [y, y], color=col, lw=0.9, solid_capstyle="butt", zorder=2); ax.scatter([c], [y], marker="D", s=13, color=col if sig else "white", edgecolor=col, linewidth=0.7, zorder=3)
    ax.axvline(0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=0); ax.set_xlim(-0.75, 0.75); ax.set_xticks([-0.5, 0, 0.5]); ax.set_ylim(-len(rows) - 0.1, 0.6); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.text(0.0, 1.075, short, transform=ax.transAxes, fontsize=6.5, color=INK, va="bottom", ha="left"); ax.text(0.0, 1.015, f"n = {int(f.n):,}, pseudo-R² = {f.lag_pseudo_r2:.2f} (OLS R² = {f.ols_r2:.2f})", transform=ax.transAxes, fontsize=5.8, color=GREY, va="bottom", ha="left")
axes[0].set_yticks([-i - (0.5 if i == len(rows) - 1 else 0) for i in range(len(rows))]); axes[0].set_yticklabels(ylab); axes[1].set_xlabel("Standardised coefficient, spatial lag model (s.d. of effect per s.d. of the variable)")
fig.legend(handles=[plt.Line2D([], [], marker="D", ls="-", color=RED, ms=3.2, lw=0.9, label="Positive, P < 0.05"), plt.Line2D([], [], marker="D", ls="-", color=BLUE, ms=3.2, lw=0.9, label="Negative, P < 0.05"),
                    plt.Line2D([], [], marker="D", ls="-", color=LIGHT, mfc="white", ms=3.2, lw=0.9, label="Not significant")], loc="lower center", ncol=3, fontsize=6, columnspacing=2.0, bbox_to_anchor=(0.57, 0.0), title="Point estimate and 95 % interval", title_fontsize=6)
fig.subplots_adjust(left=0.175, right=0.99, top=0.87, bottom=0.24, wspace=0.10)
for ax, s in zip(axes, "abc"): bb = ax.get_position(); fig.text(bb.x0 - 0.012, bb.y1 + 0.06, s, fontsize=8, fontweight="bold", va="bottom", ha="right")
for e, kw in (("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}), ("png", {"dpi": 300})): fig.savefig(FIG / f"fig_sociodemo_spatial{SUF}.{e}", **kw)
print("wrote", FIG / f"fig_sociodemo_spatial{SUF}.png")
