"""Machine-learning model of the unit-level policy effect on neighbourhood characteristics, explained with SHAP values.
Target: final TimesFM estimate per unit (116 days / 17 weeks), same specification as the headline results:
  subway  = station effect with the system seasonal calibration passed down (uq/station_effects_subway_timesfm.csv)
  bus     = route effect excl. Bronx local, same calibration factor passed down (raw route forecast x k, k = raw / calibrated system cf)
  Replica = origin-tract effect, uncalibrated (as in the Replica results)
Features: ACS 2023 profile of the unit's catchment (station 800 m, route corridor 400 m, the tract itself) + distance to the zone.
Model: gradient-boosted trees (XGBoost, shallow). Accuracy: repeated 5-fold cross-validation, against a linear model and the mean.
Explanation: exact TreeSHAP (XGBoost pred_contribs) from the model fitted on all units.
Writes uq/sociodemo_shap_{cv,importance}.csv, uq/sociodemo_shap_values_<mode>.csv, figure fig_sociodemo_shap.{pdf,svg,tiff,png}
"""
import warnings
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from nyc_cp.config import REPO_ROOT
warnings.filterwarnings("ignore")
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; FIG = REPO_ROOT / "outputs" / "figures" / "paper" / "main"; MM = 1 / 25.4
VARS = [("income_k", "Median household income"), ("pct_bachelor_plus", "Bachelor's degree or higher"), ("pct_wfh", "Work from home"), ("pct_car_commute", "Commute by car"), ("pct_transit_commute", "Commute by transit"),
        ("pop_density_k", "Population density"), ("pct_white", "White residents"), ("pct_black", "Black residents"), ("pct_hispanic", "Hispanic residents"), ("pct_asian", "Asian residents"), ("age_median", "Median age"), ("dist_km", "Distance to the zone")]
F = [v for v, _ in VARS]; LAB = dict(VARS); SHORT = {**LAB, "income_k": "Household income", "pct_bachelor_plus": "Bachelor's or higher"}
import sys; MODEL = sys.argv[1] if len(sys.argv) > 1 else "timesfm"; SUF = "" if MODEL == "timesfm" else "_" + MODEL      # TimesFM keeps the original file names; other models get a suffix
def targets():
    out = {}
    s = pd.read_csv(U / f"station_effects_subway_{MODEL}.csv", dtype={"station_index": str}).set_index("station_index"); out["subway"] = ("Subway stations", s.effect_pct, s.entries_per_day_cf)
    r = pd.read_csv(U / f"route_effects_bus_{MODEL}.csv").set_index("route"); r = r[~r.bronx_local]; sc = pd.read_csv(U / "round9_scores.csv", parse_dates=["origin"])
    sc = sc[(sc.model == MODEL) & (sc.group == "non-Bronx") & (sc.h == 116) & (sc.origin == "2025-01-05")].set_index("variant").score; k = (1 + sc["raw"] / 100) / (1 + sc["V3"] / 100)
    out["bus"] = ("Bus routes", 100 * ((1 + r.effect_pct / 100) / k - 1), r.riders_per_day_cf * k)
    d = pd.read_csv(U / f"denseunitwk_replica_{MODEL}.csv", parse_dates=["origin"], dtype={"unit": str}); d = d[(d.h == 17) & (d.origin == "2025-01-04")].set_index("unit"); d = d[d.cf > d.cf.sum() * 2e-5]
    out["replica"] = ("All trips by origin tract (Replica)", 100 * (d.actual / d.cf - 1), d.cf / 17)
    return out
def model(): return xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=8, random_state=0, verbosity=0)
CV, IMP, SH = [], [], {}
for mode, (title, y, vol) in targets().items():
    S = pd.read_csv(U / f"sociodemo_{mode}.csv", dtype={"unit": str}).set_index("unit"); D = S[F].join(y.rename("y"), how="inner").join(vol.rename("vol")).dropna(subset=["y"])
    if mode == "bus": D = D[D.vol >= 300]
    lo, hi = D.y.quantile([0.01, 0.99]); D["y"] = D.y.clip(lo, hi); X = D[F].astype(float); X = X.fillna(X.median()); yv = D.y.to_numpy()
    sysw = 100 * ((vol.reindex(D.index) * (1 + y.reindex(D.index) / 100)).sum() / vol.reindex(D.index).sum() - 1)
    r2 = {"xgboost": [], "linear": []}
    for rep in range(5):
        px, pl = np.zeros(len(D)), np.zeros(len(D))
        for tr, te in RepeatedKFold(n_splits=5, n_repeats=1, random_state=rep).split(X):
            px[te] = model().fit(X.iloc[tr], yv[tr]).predict(X.iloc[te]); pl[te] = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(X.iloc[tr], yv[tr]).predict(X.iloc[te])
        r2["xgboost"].append(r2_score(yv, px)); r2["linear"].append(r2_score(yv, pl))
    m = model().fit(X, yv); sv = m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]; SH[mode] = (title, X, sv, D.y)
    CV.append(dict(mode=title, units=len(D), effect_volume_weighted=sysw, mean_unit_effect=yv.mean(), sd_unit_effect=yv.std(), r2_cv_xgboost=np.mean(r2["xgboost"]), r2_cv_xgboost_sd=np.std(r2["xgboost"]), r2_cv_linear=np.mean(r2["linear"]), r2_in_sample=r2_score(yv, m.predict(X))))
    for j, f in enumerate(F):
        IMP.append(dict(mode=title, variable=LAB[f], mean_abs_shap=np.abs(sv[:, j]).mean(), direction=np.corrcoef(X[f].rank(), sv[:, j])[0, 1], shap_low_fifth=sv[X[f] <= X[f].quantile(0.2), j].mean(), shap_high_fifth=sv[X[f] >= X[f].quantile(0.8), j].mean()))
    pd.DataFrame(sv, index=D.index, columns=F).join(X, rsuffix="_value").join(D.y).to_csv(U / f"sociodemo_shap_values_{mode}{SUF}.csv")
CV, IMP = pd.DataFrame(CV), pd.DataFrame(IMP); CV.to_csv(U / f"sociodemo_shap_cv{SUF}.csv", index=False); IMP.to_csv(U / f"sociodemo_shap_importance{SUF}.csv", index=False)
pd.set_option("display.width", 220); print(CV.round(3).to_string(index=False))
for t, g in IMP.groupby("mode", sort=False): print("\n" + t); print(g.drop(columns="mode").sort_values("mean_abs_shap", ascending=False).round(2).to_string(index=False))
# ---------------- figure: SHAP summary (beeswarm) per mode
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2, "ytick.major.size": 2.2, "legend.frameon": False, "axes.spines.right": False, "axes.spines.top": False, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6})
INK = "#272727"; cmap = mpl.colors.LinearSegmentedColormap.from_list("lowhigh", ["#0F4D92", "#C9C9C9", "#B64342"])
fig = plt.figure(figsize=(183 * MM, 88 * MM)); gs = fig.add_gridspec(1, 3, left=0.135, right=0.925, top=0.86, bottom=0.14, wspace=0.98); rng = np.random.default_rng(0)
for i, mode in enumerate(("subway", "bus", "replica")):
    title, X, sv, y = SH[mode]; ax = fig.add_subplot(gs[i]); order = np.argsort(-np.abs(sv).mean(0)); r2 = CV.set_index("mode").loc[title, "r2_cv_xgboost"]
    for row, j in enumerate(order):
        s = sv[:, j]; v = X.iloc[:, j].rank(pct=True).to_numpy(); n = len(s)
        # beeswarm: vertical offset proportional to local density
        bins = np.digitize(s, np.linspace(s.min(), s.max() + 1e-9, 40)); off = np.zeros(n)
        for b in np.unique(bins): idx = np.where(bins == b)[0]; k = len(idx); off[idx] = (rng.permutation(k) - (k - 1) / 2) / max(k, 1) * min(0.8, 0.06 * k)
        ax.scatter(s, -row + off, c=v, cmap=cmap, vmin=0, vmax=1, s=1.6 if n > 600 else 3.5, linewidths=0, alpha=0.85, rasterized=True)
    ax.axvline(0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=0); ax.set_yticks(-np.arange(len(F))); ax.set_yticklabels([SHORT[F[j]] for j in order]); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); ax.set_ylim(-len(F) + 0.4, 0.6)
    lim = np.percentile(np.abs(sv), 99.7) * 1.15; ax.set_xlim(-lim, lim); ax.set_xlabel("SHAP value (percentage points)")
    ax.text(-0.02, 1.075, {"subway": "Subway stations", "bus": "Bus routes", "replica": "Origin tracts (Replica)"}[mode], transform=ax.transAxes, fontsize=6.5, color=INK, va="bottom", ha="left"); ax.text(-0.02, 1.015, f"n = {len(X):,}, cross-validated R² = {r2:.2f}", transform=ax.transAxes, fontsize=5.8, color="#767676", va="bottom", ha="left")
    bb = ax.get_position(); fig.text(bb.x0 - 0.105, bb.y1 + 0.06, "abc"[i], fontsize=8, fontweight="bold", va="bottom")
cax = fig.add_axes([0.952, 0.30, 0.008, 0.45]); cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 1)), cax=cax, ticks=[0, 1]); cb.ax.set_yticklabels(["Low", "High"]); cb.set_label("Value of the variable", labelpad=-6); cb.outline.set_linewidth(0.4)
for e, kw in (("pdf", {"dpi": 600}), ("svg", {"dpi": 600}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}), ("png", {"dpi": 300})): fig.savefig(FIG / f"fig_sociodemo_shap{SUF}.{e}", **kw)
print("wrote", FIG / f"fig_sociodemo_shap{SUF}.png")
