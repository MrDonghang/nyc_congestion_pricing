"""Replica in-time placebo paths (uncalibrated forecasts; weekly; 17-week horizon).
Four panels: all trips by origin tract — citywide, originating inside the zone, originating outside — and weekday public-transit
trips into the zone. Coloured line = real policy (4 Jan 2025); grey = 2024 pseudo-policy origins
whose window ends before 7 Sep 2024. Red / blue by the sign of the real 17-week estimate.
Writes fig_replica_placebo_<model>.{pdf,svg,tiff,png}, uq/replica_placebo_paths_<model>.csv
"""
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; FIG = REPO_ROOT / "outputs" / "figures" / "paper" / "main"
POLICY, B0, H, MM = pd.Timestamp("2025-01-04"), pd.Timestamp("2024-09-07"), 17, 1 / 25.4
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2, "ytick.major.size": 2.2, "legend.frameon": False, "axes.spines.right": False, "axes.spines.top": False, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6})
RED, BLUE, INK, GREY, LIGHT = "#B64342", "#0F4D92", "#272727", "#767676", "#BDBDBD"
import sys; DIR = sys.argv[1] if len(sys.argv) > 1 else "O"; SFX = "" if DIR == "O" else "_D"; W = "starting" if DIR == "O" else "ending"      # O = trips by origin tract (default), D = by destination tract
PAN = [("tot", "all", "All trips, citywide"), ("tot", "inside", f"Trips {W} inside the zone"), ("tot", "outside", f"Trips {W} outside the zone"), ("mode", "public_transit", "Transit trips into the zone")]
# User decision (2026-11-24): keep these four panels only; the other mode panels are too noisy to report.
for model in ("timesfm", "chronos_bolt"):
    T = pd.read_csv(U / f"densewk_replica{SFX}_{model}.csv", parse_dates=["origin", "date"]); M = pd.read_csv(U / f"densewk_modes_{model}.csv", parse_dates=["origin", "date"]); M = M[M["mode"] != "other"]
    def series(kind, key):
        if kind == "tot": d = T[T.grp == key]
        elif key == "ALL": d = M
        elif key == "ALL_AUTO": d = M[M["mode"].isin(["private_auto", "auto_passenger", "on_demand_auto"])]
        else: d = M[M["mode"] == key]
        return d.groupby(["origin", "h"], as_index=False)[["actual", "cf"]].sum().sort_values(["origin", "h"])
    fig, axes = plt.subplots(1, 4, figsize=(183 * MM, 52 * MM)); axes = axes.reshape(1, 4); out = []; x = np.arange(1, H + 1)
    for ax, p in zip(axes.ravel(), PAN):
        if p is None: ax.set_axis_off(); continue
        kind, key, title = p; d = series(kind, key); paths = {o: 100 * (g.actual.cumsum().to_numpy() / g.cf.cumsum().to_numpy() - 1) for o, g in d.groupby("origin") if len(g) >= H}
        real = paths[POLICY][:H]; nul = {o: v[:H] for o, v in paths.items() if o >= pd.Timestamp("2024-01-01") and o + pd.Timedelta(weeks=H - 1) < B0}; ends = pd.Series({o: v[-1] for o, v in nul.items()}); col = RED if real[-1] > 0 else BLUE
        for v in nul.values(): ax.plot(x, v, color=LIGHT, lw=0.5, zorder=2)
        ax.plot(x, real, color=col, lw=1.3, zorder=4); ax.axhline(0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
        lim = float(np.ceil(max(6.0, np.abs(real).max(), ends.abs().max(), np.percentile(np.abs(np.vstack(list(nul.values()))[:, 3:]), 98)) / 2.0) * 2.0); ax.set_ylim(-lim, lim); ax.set_yticks(np.arange(-lim, lim + 0.1, 2 if lim <= 6 else 4)); ax.set_xlim(0, 17.6); ax.set_xticks([0, 4, 8, 12, 17])
        ax.text(0.0, 1.03, title, transform=ax.transAxes, fontsize=6.5, color=INK, va="bottom", ha="left")
        n_hi, n_lo = int((ends >= real[-1]).sum()), int((ends <= real[-1]).sum())
        ax.text(0.98, 0.05 if real[-1] > 0 else 0.95, f"{real[-1]:+.1f}%   placebo {ends.min():+.1f} to {ends.max():+.1f}", transform=ax.transAxes, fontsize=5.8, color=col, ha="right", va="bottom" if real[-1] > 0 else "top", fontweight="bold")
        out.append(dict(model=model, panel=title, real=real[-1], n=len(ends), placebo_mean=ends.mean(), placebo_sd=ends.std(), placebo_min=ends.min(), placebo_max=ends.max(), n_at_or_above=n_hi, n_at_or_below=n_lo,
                        weeks_outside_all=int(sum((real[h] > max(v[h] for v in nul.values())) or (real[h] < min(v[h] for v in nul.values())) for h in range(H)))))
    fig.text(0.525, 0.115, "Weeks since policy or pseudo-policy start", ha="center", va="bottom", fontsize=6.5)
    for ax in axes[:, 0]: ax.set_ylabel("Cumulative effect (%)")
    fig.legend(handles=[plt.Line2D([], [], color=RED, lw=1.3, label="Real policy, increase"), plt.Line2D([], [], color=BLUE, lw=1.3, label="Real policy, decrease"), plt.Line2D([], [], color=LIGHT, lw=0.8, label="Pseudo-policy dates in 2024 (no policy, n = 19)")], loc="lower center", ncol=3, fontsize=6, handlelength=1.6, columnspacing=2.0, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.27, wspace=0.30)
    for ax, s in zip([a for a, p in zip(axes.ravel(), PAN) if p is not None], "abcdefghijk"): bb = ax.get_position(); fig.text(bb.x0 - 0.035, bb.y1 + 0.03, s, fontsize=8, fontweight="bold", va="bottom")
    for e, kw in (("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}), ("png", {"dpi": 300})): fig.savefig(FIG / f"fig_replica_placebo{SFX}_{model}.{e}", **kw)
    plt.close(fig); O = pd.DataFrame(out); O.to_csv(U / f"replica_placebo_paths{SFX}_{model}.csv", index=False); pd.set_option("display.width", 230); print(O.round(2).to_string(index=False))
