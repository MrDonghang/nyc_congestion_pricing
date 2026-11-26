"""Appendix: 116-day effect under four calibration designs, quantile-regression (median) version on the current replay (denseq_*.csv), both models.
  none | holiday + day of week | + overall level and lead | + seasonal (target-month) profile [main specification]
Writes uq/qr_variants_scores.csv, uq/qr_variants_summary.csv and outputs/figures/paper/robustness/appendix_calibration_variants.{png,pdf}
"""
import glob, os, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import QuantileRegressor
from nyc_cp.config import REPO_ROOT
import importlib.util
spec = importlib.util.spec_from_file_location("qn", REPO_ROOT / "runs" / "revision" / "qr_native.py"); qn = importlib.util.module_from_spec(spec); spec.loader.exec_module(qn)
warnings.filterwarnings("ignore"); U, POLICY, B0, H, hol = qn.U, qn.POLICY, qn.B0, qn.H, qn.hol; FIG = REPO_ROOT / "outputs" / "figures" / "paper" / "robustness"
VAR = [("V0", "No calibration"), ("Q1", "Holiday + day of week"), ("Q2", "+ overall level"), ("Q3", "+ seasonal profile")]
def design(d, v):
    dw = np.eye(7)[d.date.dt.dayofweek.to_numpy()]; X = [d.date.isin(hol).to_numpy(float)[:, None], dw[:, 1:] - 1 / 7 if v == "Q1" else dw[:, 1:]]
    if v in ("Q2", "Q3"): X.append((d.h.to_numpy() / 116.0)[:, None])
    if v == "Q3": X.append(np.eye(12)[d.date.dt.month.to_numpy() - 1][:, 1:])
    return np.hstack(X)
def one(panel, model, d, o):
    te = d[d.origin == o]; tr = d[(d.date < o) & ~d.straddle]; a, c = te.actual.to_numpy(), te.cf.to_numpy(); y = np.log(tr.actual.to_numpy() / tr.cf.to_numpy()); rows = [dict(panel=panel, model=model, variant="V0", origin=o, score=100 * (a.sum() / c.sum() - 1))]
    for v in ("Q1", "Q2", "Q3"):
        k = QuantileRegressor(quantile=0.5, alpha=0.0, fit_intercept=(v != "Q1"), solver="highs").fit(design(tr, v), y).predict(design(te, v)); rows.append(dict(panel=panel, model=model, variant=v, origin=o, score=100 * (a.sum() / (c * np.exp(k)).sum() - 1)))
    return rows
if __name__ == "__main__":
    tasks = []
    for panel in ("subway", "bus"):
        for model in ("timesfm", "chronos_bolt"):
            d = qn.load(panel, model); origins = pd.DatetimeIndex(sorted(d.origin.unique())); tasks += [(panel, model, d, o) for i, o in enumerate(origins) if o == POLICY or (i % 2 == 0 and o >= pd.Timestamp("2024-01-01") and o + pd.Timedelta(days=H - 1) < B0)]
    S = pd.DataFrame([r for rs in Parallel(n_jobs=44)(delayed(one)(*t) for t in tasks) for r in rs]); S.to_csv(U / "qr_variants_scores.csv", index=False)
    for f in glob.glob(os.path.expanduser("~/.fonts/arial*.ttf")): fm.fontManager.addfont(f)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
    BLUE, PL, INK, MUTED = "#2a78d6", "#9a9892", "#1f1f1f", "#5f5e5a"; fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.6)); rng = np.random.default_rng(1); out = []
    for r, (panel, pt) in enumerate((("subway", "Subway entries"), ("bus", "Bus (excl. Bronx local routes)"))):
        for c, (model, ml) in enumerate((("timesfm", "TimesFM-2.0"), ("chronos_bolt", "Chronos-Bolt"))):
            ax = axes[r, c]
            for i, (v, lab) in enumerate(VAR):
                s = S[(S.panel == panel) & (S.model == model) & (S.variant == v)].set_index("origin").score; real = s[POLICY]; nul = s.drop(index=POLICY); y = -i; n_ge = int((nul >= real).sum()); rms = float(np.sqrt((nul ** 2).mean()))
                ax.scatter(nul, y - 0.2 + rng.uniform(-0.12, 0.12, len(nul)), s=20, color=PL, edgecolor="white", linewidth=0.5, zorder=2); ax.plot([nul.mean()], [y - 0.2], marker="|", ms=18, mew=2.2, color=INK, zorder=3)
                ax.plot([real], [y + 0.2], marker="D", ms=9, color=BLUE, mec="white", mew=1.2, zorder=4); ax.annotate(f"{real:+.1f}%", (real, y + 0.2), xytext=(10, 0), textcoords="offset points", va="center", fontweight="bold", color=INK)
                ax.text(11.6, y, f"placebo mean {nul.mean():+.1f}, error {rms:.1f} pp\n{n_ge} of {len(nul)} at or above real", ha="right", va="center", fontsize=8, color=MUTED, linespacing=1.3)
                out.append(dict(panel=panel, model=model, variant=v, label=lab, estimate=real, placebo_mean=nul.mean(), placebo_sd=nul.std(ddof=1), placebo_rms=rms, n_placebo=len(nul), n_ge=n_ge))
            ax.axvline(0, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=1); ax.set_yticks([-i for i in range(4)]); ax.set_yticklabels([l + ("  (main)" if v == "Q3" else "") for v, l in VAR]); ax.tick_params(axis="y", length=0); ax.set_ylim(-3.6, 0.6); ax.set_xlim(-6.5, 12)
            ax.set_xticks(range(-6, 9, 2)); ax.grid(axis="x", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True); ax.set_title(f"{pt} — {ml}", loc="left", fontweight="bold", fontsize=10.5, color=INK)
            if r == 1: ax.set_xlabel("116-day cumulative effect (%)", color=MUTED)
    hd = [plt.Line2D([], [], marker="o", ls="", color=PL, ms=6, label="pseudo-policy dates in 2024 (no policy)"), plt.Line2D([], [], marker="|", ls="", color=INK, ms=13, mew=2.2, label="their mean"), plt.Line2D([], [], marker="D", ls="", color=BLUE, ms=7, label="real policy, 5 Jan 2025")]
    fig.legend(handles=hd, loc="lower center", ncol=3, frameon=False); fig.tight_layout(rect=(0, 0.045, 1, 1), w_pad=2.5)
    for e in ("png", "pdf"): fig.savefig(FIG / f"appendix_calibration_variants.{e}", dpi=200, bbox_inches="tight")
    R = pd.DataFrame(out); R.to_csv(U / "qr_variants_summary.csv", index=False); pd.set_option("display.width", 200); print(R.drop(columns="label").round(2).to_string(index=False))
