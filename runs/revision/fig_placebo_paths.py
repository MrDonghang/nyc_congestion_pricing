"""In-time placebo paths under the final specification: cumulative relative effect by days since the (pseudo-)policy start.
Real policy (5 Jan 2025) vs the 2024 pseudo-policy dates whose 116-day window ends before 29 Aug 2024.
Subway: entries, history from 2022. Bus: routes other than Bronx local. Pooled rolling-origin calibration with seasonal profile.
Writes outputs/figures/paper/main/fig_placebo_paths_<model>.{pdf,svg,tiff,png} and uq/paper_placebo_paths_<model>.csv
"""
import warnings
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT
warnings.filterwarnings("ignore")
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; FIG = REPO_ROOT / "outputs" / "figures" / "paper" / "main"; FIG.mkdir(parents=True, exist_ok=True)
ns = {}; exec((REPO_ROOT / "runs" / "revision" / "plot_daily_final.py").read_text().split("PAN = [")[0], ns); load, calibrated, POLICY, B0, H = ns["load"], ns["calibrated"], ns["POLICY"], ns["B0"], ns["H"]
MM = 1 / 25.4
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2, "ytick.major.size": 2.2, "legend.frameon": False, "axes.spines.right": False, "axes.spines.top": False, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6})
TEAL, VIOLET, INK, GREY, LIGHT = "#42949E", "#9A4D8E", "#272727", "#767676", "#BDBDBD"; COL = {"subway": TEAL, "bus": VIOLET}; TTL = {"subway": "Subway", "bus": "Bus"}
for model, tag in (("timesfm", "timesfm"), ("chronos_bolt", "chronos_bolt")):
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 62 * MM), sharey=True); out = []
    for ax, kind in zip(axes, ("subway", "bus")):
        d = load(kind, model); origins = pd.DatetimeIndex(sorted(d.origin.unique()))
        path = lambda t: 100 * (t.actual.cumsum().to_numpy() / t.cfc.cumsum().to_numpy() - 1)
        real = path(calibrated(d, POLICY)); pl = {o: path(calibrated(d, o)) for i, o in enumerate(origins) if o >= pd.Timestamp("2024-01-01") and o + pd.Timedelta(days=H - 1) < B0}
        x = np.arange(1, H + 1); ends = pd.Series({o: p[-1] for o, p in pl.items()})
        for o, p in pl.items(): ax.plot(x, p, color=LIGHT, lw=0.6, zorder=2)
        ax.plot(x, real, color=COL[kind], lw=1.4, zorder=4); ax.axhline(0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.text(H + 2, real[-1], f"+{real[-1]:.1f}%", color=COL[kind], fontsize=6.5, fontweight="bold", va="center", ha="left")
        for o in (ends.idxmax(), ends.idxmin()):
            dy = 0.9 if (o == ends.idxmax() and real[-1] - ends[o] < 1.3) else 0
            ax.text(H + 2, ends[o] - dy, f"{o:%d %b %Y}  {ends[o]:+.1f}%", color=GREY, fontsize=5.5, va="center", ha="left")
        ax.set_xlim(0, 152); ax.set_xticks([0, 28, 56, 84, 116]); ax.set_ylim(-9, 9); ax.set_xlabel("Days since policy or pseudo-policy start")
        ax.text(0.02, 0.97, TTL[kind], transform=ax.transAxes, fontsize=7, fontweight="bold", color=COL[kind], va="top"); ax.tick_params(labelleft=True)
        n_ge = int((ends >= real[-1]).sum()); print(f"{model:13s} {kind:7s} real {real[-1]:+.2f}  placebo n={len(ends)} mean {ends.mean():+.2f} sd {ends.std():.2f} min {ends.min():+.2f} ({ends.idxmin().date()}) max {ends.max():+.2f} ({ends.idxmax().date()})  at/above real: {n_ge}"
              f" | share of days real path above every placebo: {100 * np.mean([real[h] > max(p[h] for p in pl.values()) for h in range(H)]):.0f}% ; from day {next((h + 1 for h in range(H) if all(real[j] > max(p[j] for p in pl.values()) for j in range(h, H))), None)} onward always above")
        out.append(pd.DataFrame({"series": TTL[kind], "day": x, "real": real, **{f"placebo_{o:%Y-%m-%d}": p for o, p in pl.items()}}))
    axes[0].set_ylabel("Cumulative effect (%)")
    h = [plt.Line2D([], [], color=INK, lw=1.4, label="Real policy, 5 Jan 2025"), plt.Line2D([], [], color=LIGHT, lw=0.8, label="Pseudo-policy dates in 2024 (no policy, n = 21)")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=6, handlelength=1.8, columnspacing=2.0)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.94, bottom=0.27, wspace=0.10)
    for ax, s in zip(axes, "ab"): bb = ax.get_position(); fig.text(bb.x0 - 0.035, bb.y1 + 0.005, s, fontsize=8, fontweight="bold", va="bottom", ha="left")
    for e, kw in (("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}), ("png", {"dpi": 300})): fig.savefig(FIG / f"fig_placebo_paths_{tag}.{e}", **kw)
    pd.concat(out).to_csv(U / f"paper_placebo_paths_qrfig_{tag}.csv", index=False); plt.close(fig)
