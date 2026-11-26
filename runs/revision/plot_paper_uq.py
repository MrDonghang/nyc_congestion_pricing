"""Paper figures for the effect estimates (decided with the user, 2026-11-24).
Specification: student fares excluded; subway = station entries with history from 2022; bus = routes other than Bronx local;
pooled rolling-origin calibration with holiday, day of week, level and seasonal profile (V3); null = 2024 pseudo-policy
dates whose window ends before 29 Aug 2024; horizon 116 days (5 Jan – 30 Apr 2025).
  main_effect_timesfm            main text: TimesFM-2.0
  appendix_effect_chronos_bolt   robustness: Chronos-Bolt
  appendix_calibration_variants  four calibration variants × two models × subway / bus
Tables: uq/paper_effects.csv, uq/paper_calibration_variants.csv
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; FIG = REPO_ROOT / "outputs" / "figures" / "paper" / "robustness"
POLICY, B0 = pd.Timestamp("2025-01-05"), pd.Timestamp("2024-08-29"); HS = (28, 56, 84, 116)
C_REAL, C_PL, C_BAND, INK, MUTED = "#2a78d6", "#9a9892", "#e3e1d9", "#1f1f1f", "#5f5e5a"
s8 = pd.read_csv(U / "round8_scores.csv", parse_dates=["origin"]); s9 = pd.read_csv(U / "round9_scores.csv", parse_dates=["origin"])
VAR = [("V0", "raw", "No calibration"), ("V1", "V1", "Holiday + day of week"), ("V2", "V2", "+ overall level"), ("V3", "V3", "+ seasonal profile")]
MOD = {"timesfm": "TimesFM-2.0", "chronos_bolt": "Chronos-Bolt"}; TTL = {"subway": "Subway entries", "bus": "Bus (excl. Bronx local routes)"}
QS = pd.read_csv(U / "qr_final_scores.csv", parse_dates=["origin"])      # quantile-regression (median) calibration, seasonal design, current replay
def get(kind, model, h, vi=3):
    s = QS[(QS.panel == kind) & (QS.model == model)].set_index("origin")[f"e{h}"]; return s[POLICY], s.drop(index=POLICY)
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
def stats(real, nul): return f"placebo mean {nul.mean():+.1f}, SD {nul.std():.1f}\n{int((nul >= real).sum())} of {len(nul)} at or above real"
rows = []
# ---------- one-model figures (main text and Chronos-Bolt appendix)
for model, fname in (("chronos_bolt", "main_effect_chronos_bolt"), ("timesfm", "appendix_effect_timesfm")):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1.2, 1]}); rng = np.random.default_rng(1)
    ax = axes[0]
    for i, kind in enumerate(("subway", "bus")):
        real, nul = get(kind, model, 116); y = -i
        ax.scatter(nul, y - 0.15 + rng.uniform(-0.12, 0.12, len(nul)), s=22, color=C_PL, edgecolor="white", linewidth=0.5, zorder=2)
        ax.plot([nul.mean()], [y - 0.15], marker="|", ms=18, mew=2, color=INK, zorder=3); ax.plot([real], [y + 0.22], marker="D", ms=9, color=C_REAL, mec="white", mew=1.2, zorder=4)
        ax.annotate(f"{real:+.1f}%", (real, y + 0.22), xytext=(9, 0), textcoords="offset points", va="center", fontsize=9.5, fontweight="bold", color=INK)
        ax.text(1.0, y, stats(real, nul), transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.5, color=MUTED, linespacing=1.3)
    ax.axvline(0, color=INK, lw=0.9, ls="--"); ax.set_yticks([0, -1]); ax.set_yticklabels([TTL["subway"], "Bus\n(excl. Bronx local routes)"]); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_ylim(-1.6, 0.6); ax.set_xlim(-5.5, 9.5); ax.grid(axis="x", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    ax.set_title("116-day cumulative effect (5 Jan – 30 Apr 2025)", loc="left", fontweight="bold", fontsize=10, color=INK); ax.set_xlabel("effect on ridership (%)", color=MUTED)
    ax = axes[1]
    for j, kind in enumerate(("subway", "bus")):
        for k, h in enumerate(HS):
            x = k + (j - 0.5) * 0.36; real, nul = get(kind, model, h); lo, hi = nul.quantile(0.05), nul.quantile(0.95)
            ax.plot([x, x], [lo, hi], color=C_BAND, lw=9, solid_capstyle="butt", zorder=1); ax.plot([x - 0.1, x + 0.1], [nul.mean()] * 2, color=INK, lw=1.6, zorder=2)
            ax.plot([x], [real], marker="D" if j == 0 else "o", ms=8, color=C_REAL, mec="white", mew=1.2, zorder=3); ax.annotate(f"{real:+.1f}", (x, real), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=INK)
            rows.append(dict(model=MOD[model], series=TTL[kind], h=h, effect_pct=real, placebo_mean=nul.mean(), placebo_sd=nul.std(), placebo_q05=lo, placebo_q95=hi, placebo_min=nul.min(), placebo_max=nul.max(),
                             n_placebo=len(nul), n_at_or_above=int((nul >= real).sum()), rank_p_one_sided=(1 + (nul >= real).sum()) / (len(nul) + 1), z=(real - nul.mean()) / nul.std()))
    ax.axhline(0, color=INK, lw=0.9, ls="--"); ax.set_xticks(range(4)); ax.set_xticklabels([f"first {h} days" for h in HS]); ax.set_ylim(-6, 8.5); ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    ax.set_title("By length of the window", loc="left", fontweight="bold", fontsize=10, color=INK); ax.set_ylabel("effect on ridership (%)", color=MUTED)
    hd = [plt.Line2D([], [], marker="o", ls="none", color=C_PL, ms=6, label="pseudo-policy dates in 2024 (no policy)"), plt.Line2D([], [], color=C_BAND, lw=8, label="their 5–95 % range"), plt.Line2D([], [], marker="|", ls="none", color=INK, ms=12, mew=2, label="their mean"),
          plt.Line2D([], [], marker="D", ls="none", color=C_REAL, ms=7, label="real policy — subway"), plt.Line2D([], [], marker="o", ls="none", color=C_REAL, ms=7, label="real policy — bus (right panel)")]
    fig.legend(handles=hd, loc="lower center", ncol=5, frameon=False, fontsize=8.5); fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=2.5)
    for e in ("png", "pdf"): fig.savefig(FIG / f"{fname}.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)
pd.DataFrame(rows).to_csv(U / "paper_effects.csv", index=False)
# The calibration-variant comparison (appendix_calibration_variants) is produced by qr_variants.py.
