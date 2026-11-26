"""Appendix figure: synthetic check of the main specification (Chronos-Bolt; simple and varied data-generating processes, 80 series, no policy).
a  estimate against injected effect, one dot per series   b  share of series above all 21 pseudo-policy dates   c  coverage of the 90 % interval
Reads uq/synth_bolt_qr_series.csv (runs/revision/synth_bolt_qr.py). Writes outputs/figures/paper/robustness/appendix_synthetic_chronos_bolt.{pdf,png}
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
import glob, os, numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT
for f in glob.glob(os.path.expanduser("~/.fonts/arial*.ttf")): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
INK, MUTED, COL = "#1f1f1f", "#5f5e5a", {"simple": "#2a78d6", "varied": "#d9822b"}; LAB = {"simple": "Fixed parameters", "varied": "Parameters vary by series"}
S = pd.read_csv(REPO_ROOT / "outputs/_summary/revision/uq/synth_bolt_qr_series.csv"); S = S[(S.variant == "main specification") & (S.dgp != "rw")]; taus = sorted(S.tau.unique()); rng = np.random.default_rng(1)
fig, ax = plt.subplots(1, 3, figsize=(183 / 25.4, 62 / 25.4), gridspec_kw={"width_ratios": [1.5, 1.15, 0.75]})
for k, tau in enumerate(taus):
    ax[0].plot([k - 0.42, k + 0.42], [tau, tau], color=INK, lw=0.9, zorder=4)
    for j, d in enumerate(("simple", "varied")):
        e = S[(S.tau == tau) & (S.dgp == d)].estimate.to_numpy(); x = k + (-0.2 if j == 0 else 0.2)
        ax[0].scatter(x + rng.uniform(-0.13, 0.13, len(e)), e, s=5, color=COL[d], alpha=0.55, linewidth=0, zorder=3, label=LAB[d] if k == 0 else None); ax[0].plot([x - 0.15, x + 0.15], [e.mean()] * 2, color=COL[d], lw=1.6, zorder=5)
ax[0].axhline(0, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1); ax[0].set_xticks(range(len(taus))); ax[0].set_xticklabels([f"{t:g}" for t in taus]); ax[0].set_xlabel("Injected effect (%)"); ax[0].set_ylabel("Estimated 116-day effect (%)")
ax[0].legend(frameon=False, loc="upper left", handletextpad=0.2, borderpad=0, markerscale=1.6); ax[0].text(0.98, 0.04, "black bar: injected value\ncoloured bar: mean estimate", transform=ax[0].transAxes, ha="right", va="bottom", color=MUTED, fontsize=6)
D = S.groupby(["dgp", "tau"]).detected.mean().unstack(0) * 100; P = S.groupby("tau").detected.mean() * 100
for d in ("simple", "varied"): ax[1].plot(D.index, D[d], "-o", color=COL[d], ms=3, lw=1.1)
ax[1].plot(P.index, P.values, "-o", color=INK, ms=3, lw=1.3)
ax[1].axhline(100 / 22, color=MUTED, lw=0.5, ls=(0, (3, 2))); ax[1].text(5.05, 100 / 22 + 2.5, "nominal 1/22", ha="right", color=MUTED, fontsize=6); ax[1].annotate(f"{P.loc[0.0]:.0f} % with no effect", xy=(0, P.loc[0.0]), xytext=(0.8, 64), color=INK, fontsize=6, ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkB=3))
ax[1].text(4.85, 80, "both\ngroups", ha="center", va="top", color=INK, fontsize=6, linespacing=1.1); ax[1].set_xticks(taus); ax[1].set_xticklabels([f"{t:g}" for t in taus]); ax[1].set_ylim(0, 104); ax[1].set_xlabel("Injected effect (%)"); ax[1].set_ylabel("Series above all 21\npseudo-policy dates (%)")
C = S[S.tau == 0].groupby("dgp").covered.mean() * 100; allc = S[S.tau == 0].covered.mean() * 100; names = ["Fixed", "Varying", "Both"]; vals = [C["simple"], C["varied"], allc]
ax[2].bar(range(3), vals, color=[COL["simple"], COL["varied"], INK], width=0.62); ax[2].axhline(90, color=MUTED, lw=0.5, ls=(0, (3, 2))); ax[2].text(2.45, 91.5, "nominal 90", ha="right", color=MUTED, fontsize=6)
for i, v in enumerate(vals): ax[2].text(i, v - 6, f"{v:.0f}", ha="center", color="white", fontsize=6.5, fontweight="bold")
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(names); ax[2].set_ylim(0, 104); ax[2].set_ylabel("90 % interval covers\nthe injected effect (%)")
for a, l in zip(ax, "abc"): a.text({"a": -0.17, "b": -0.3, "c": -0.5}[l], 1.04, l, transform=a.transAxes, fontsize=9, fontweight="bold")
fig.tight_layout(w_pad=1.2); out = REPO_ROOT / "outputs/figures/paper/robustness"
for e, kw in (("pdf", {}), ("png", {"dpi": 300})): fig.savefig(out / f"appendix_synthetic_chronos_bolt.{e}", **kw)
print("wrote", out / "appendix_synthetic_chronos_bolt.png", P.round(1).to_dict(), [round(v, 1) for v in vals])
