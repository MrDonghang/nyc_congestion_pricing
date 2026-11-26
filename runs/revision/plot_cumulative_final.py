"""Cumulative effect in absolute riders under the final specification (system level).
cum(h) = Σ_{t≤h} (actual − calibrated counterfactual). 90 % interval: the cumulative relative effect at h minus the 95th / 5th
percentile of the centred no-policy cumulative relative error at h across the 2024 pseudo-policy origins, times the cumulative counterfactual.
Writes main_cumulative_timesfm.{png,pdf}, appendix_cumulative_chronos_bolt.{png,pdf}, uq/paper_cumulative_<model>.csv
"""
import importlib.util, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT
warnings.filterwarnings("ignore")
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; FIG = REPO_ROOT / "outputs" / "figures" / "paper" / "robustness"
src = (REPO_ROOT / "runs" / "revision" / "plot_daily_final.py").read_text().split("PAN = [")[0]; ns = {}; exec(src, ns)   # reuse load(), calibrated(), constants
load, calibrated, POLICY, B0, H = ns["load"], ns["calibrated"], ns["POLICY"], ns["B0"], ns["H"]
C_MAIN, C_BAND, INK, MUTED = "#2a78d6", "#cfe0f5", "#1f1f1f", "#5f5e5a"
PAN = [("subway", "Subway entries", "entries"), ("bus", "Bus (excl. Bronx local routes)", "boardings")]
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
for model, mlab, fname in (("chronos_bolt", "Chronos-Bolt", "main_cumulative_chronos_bolt"), ("timesfm", "TimesFM-2.0", "appendix_cumulative_timesfm")):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4)); out = []
    for ax, (kind, title, unit) in zip(axes, PAN):
        d = load(kind, model); origins = pd.DatetimeIndex(sorted(d.origin.unique())); real = calibrated(d, POLICY)
        cA, cC = real.actual.cumsum().to_numpy(), real.cfc.cumsum().to_numpy(); rel = cA / cC - 1
        P = []
        for i, o in enumerate(origins):
            if o >= pd.Timestamp("2024-01-01") and o + pd.Timedelta(days=H - 1) < B0:
                t = calibrated(d, o); P.append(t.actual.cumsum().to_numpy() / t.cfc.cumsum().to_numpy() - 1)
        P = np.vstack(P); pm = P.mean(axis=0); q05, q95 = np.percentile(P - pm, 5, axis=0), np.percentile(P - pm, 95, axis=0)   # centred: dispersion of the no-policy error
        est = (cA - cC) / 1e6; lo = (rel - q95) * cC / 1e6; hi = (rel - q05) * cC / 1e6
        ax.fill_between(real.date, lo, hi, color=C_BAND, lw=0, label="90 % interval"); ax.plot(real.date, est, color=C_MAIN, lw=2.2, label="cumulative effect")
        ax.axhline(0, color=INK, lw=0.8); ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
        ax.set_title(f"{title} — {mlab}", loc="left", fontweight="bold", fontsize=10, color=INK); ax.set_ylabel(f"cumulative additional {unit} (millions)", color=MUTED)
        ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")); ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
        ax.annotate(f"{est[-1]:+.1f} M  ({100 * rel[-1]:+.1f}%)\n90 % interval {lo[-1]:+.1f} … {hi[-1]:+.1f} M", (real.date.iloc[-1], est[-1]), xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=8.5, fontweight="bold", color=INK, linespacing=1.3)
        first_pos = next((k + 1 for k in range(H) if (lo[k:] > 0).all()), None)
        print(f"   placebo mean cum rel at 116: {100 * pm[-1]:+.2f}%"); print(f"{model:13s} {kind:7s} total {est[-1]:+.2f} M ({100 * rel[-1]:+.2f}%)  CI {lo[-1]:+.2f}…{hi[-1]:+.2f} M  per day {1e3 * est[-1] / H:+.1f} k  CI {1e3 * lo[-1] / H:+.1f}…{1e3 * hi[-1] / H:+.1f} k  cf total {cC[-1] / 1e6:.1f} M  actual {cA[-1] / 1e6:.1f} M  interval stays above zero from day {first_pos}  at day 28/56/84: " + " ".join(f"{est[k - 1]:+.2f} [{lo[k - 1]:+.2f},{hi[k - 1]:+.2f}]" for k in (28, 56, 84)))
        out.append(pd.DataFrame({"series": title, "model": mlab, "date": real.date.to_numpy(), "h": range(1, H + 1), "cum_actual": cA, "cum_counterfactual": cC, "cum_effect": cA - cC, "ci_lo": lo * 1e6, "ci_hi": hi * 1e6, "cum_rel_pct": 100 * rel}))
    hh, ll = axes[0].get_legend_handles_labels(); fig.legend(hh[::-1], ll[::-1], loc="lower center", ncol=2, frameon=False, fontsize=8.5); fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.5)
    for e in ("png", "pdf"): fig.savefig(FIG / f"{fname}.{e}", dpi=200, bbox_inches="tight")
    pd.concat(out).to_csv(U / f"paper_cumulative_qrfig_{model}.csv", index=False); plt.close(fig)
