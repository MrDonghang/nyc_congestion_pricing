"""Method flowchart: rolling forecast origins (a) and the estimation / inference steps applied at each origin (b).
Describes the pipeline behind Fig. 1 under the main specification (dense_q.py -> qr_native.py -> paper_tables_qr.py -> fig1_overview.py).
Writes outputs/figures/paper/main/fig_method_flow.{pdf,png}
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import glob, os
import matplotlib.font_manager as fm
import numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT
FIG = REPO_ROOT / "outputs" / "figures" / "paper" / "main"; W, Hh = 183.0, 132.0
INK, MUTED, BLUE, PL, LIGHT, FILL, EDGE = "#1f1f1f", "#5f5e5a", "#2a78d6", "#8d8b85", "#cfcdc6", "#f6f5f1", "#c9c7bf"
for f in glob.glob(os.path.expanduser("~/.fonts/arial*.ttf")): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 6.5, "pdf.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
fig = plt.figure(figsize=(W / 25.4, Hh / 25.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, Hh); ax.axis("off")
T = pd.Timestamp; POLICY, B0, H = T("2025-01-05"), T("2024-08-29"), 116; t0, t1 = T("2022-01-01"), T("2025-05-15")
fx = lambda d: 16 + (T(d) - t0).days / (t1 - t0).days * 162

# ---------------- a: timeline
ax.text(3, 128, "a", fontsize=8, fontweight="bold", va="center"); ax.text(8, 128, "Rolling forecast origins", fontsize=7, fontweight="bold", va="center", color=INK)
Y = 104.0; ax.plot([fx(t0), fx(t1)], [Y, Y], color=INK, lw=0.8)
for y in (2022, 2023, 2024, 2025): ax.plot([fx(f"{y}-01-01")] * 2, [Y, Y - 1.4], color=INK, lw=0.8); ax.text(fx(f"{y}-01-01"), Y - 3.6, str(y), ha="center", va="center", color=INK)
origins = pd.date_range(t0 + pd.Timedelta(days=180), POLICY - pd.Timedelta(days=14), freq="3D")
pseudo = [o for i, o in enumerate(origins) if i % 2 == 0 and o >= T("2024-01-01") and o + pd.Timedelta(days=H - 1) < B0]
for o in origins: ax.plot([fx(o)] * 2, [Y, Y + 1.8], color=LIGHT, lw=0.3)
for k, o in enumerate(pseudo):
    ax.plot([fx(o)] * 2, [Y, Y + 3.0], color=PL, lw=0.5); ax.plot([fx(o), fx(o + pd.Timedelta(days=H - 1))], [Y + 5 + 0.55 * k] * 2, color=PL, lw=0.7, solid_capstyle="butt")
ax.text(fx(pseudo[0]) - 2, Y + 10.5, f"{len(pseudo)} pseudo-policy windows\n(116 days each, no policy)\norigins {pseudo[0]:%-d %b} – {pseudo[-1]:%-d %b %Y}", ha="right", va="center", color=MUTED, linespacing=1.35)
ax.plot([fx(B0)] * 2, [Y - 1.5, Y + 18.5], color=INK, lw=0.6, ls=(0, (3, 2))); ax.text(fx(B0), Y + 21.5, "break in fare data\n29 Aug 2024", ha="center", va="center", color=MUTED, linespacing=1.35)
ax.plot([fx(POLICY)] * 2, [Y, Y + 5], color=BLUE, lw=1.2); ax.plot([fx(POLICY), fx("2025-04-30")], [Y + 5] * 2, color=BLUE, lw=2.4, solid_capstyle="butt")
ax.text((fx(POLICY) + fx("2025-04-30")) / 2, Y + 10.5, "policy window\n5 Jan – 30 Apr 2025", ha="center", va="center", color=BLUE, fontweight="bold", linespacing=1.35)
def bracket(x0, x1, y, s):
    ax.plot([x0, x0, x1, x1], [y + 1, y, y, y + 1], color=MUTED, lw=0.5); ax.text((x0 + x1) / 2, y - 2.2, s, ha="center", va="center", color=MUTED)
bracket(fx(origins[0]), fx(origins[-1]), Y - 7.5, "one forecast origin every 3 days; each issues a 116-day forecast. Errors of forecasts already observed feed the calibration of later origins")
bracket(fx(t0), fx(POLICY) - 0.4, Y - 14.5, "model input for the policy-date forecast: all observations before 5 Jan 2025 (for any other origin: all observations before that origin)")

# ---------------- b: steps
ax.text(3, 81, "b", fontsize=8, fontweight="bold", va="center"); ax.text(8, 81, "Steps applied identically at the policy date and at every pseudo-policy date", fontsize=7, fontweight="bold", va="center", color=INK)
XS, BW, R1, R2, BH = (3, 64, 125), 55, 49, 12, 28
def box(x, y, n, title, body, accent=INK):
    ax.add_patch(FancyBboxPatch((x, y), BW, BH, boxstyle="round,pad=0,rounding_size=1.6", fc=FILL, ec=EDGE, lw=0.7))
    ax.text(x + 2.6, y + BH - 3.4, f"{n}", fontsize=7, fontweight="bold", color=accent, va="center"); ax.text(x + 6.2, y + BH - 3.4, title, fontsize=7, fontweight="bold", color=INK, va="center")
    ax.text(x + 2.6, y + BH - 7.0, body, va="top", color=INK, linespacing=1.42)
def arrow(p, q): ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=7))
box(XS[0], R1, 1, "Data", "Daily ridership per unit, 1 Jan 2022 – 30 Apr 2025\nSubway: entries per station\nBus: boardings per route (Bronx local routes\nanalysed separately)\nStudent fare classes excluded")
box(XS[1], R1, 2, "Zero-shot forecast", "Pretrained time-series model, no fitting:\nChronos-Bolt (main), TimesFM-2.0 (robustness)\nInput: each unit's history before origin $D$\nOutput: 116-day point forecast and 10 % / 90 %\nquantiles per unit\nSum over units: daily counterfactual $\\hat{y}_{D,h}$")
box(XS[2], R1, 3, "Calibration (past data only)", "Log error of every earlier forecast whose target\ndate precedes $D$:  $r = \\log(y_t\\,/\\,\\hat{y}_{D',h})$\nMedian (quantile) regression of $r$ on holiday,\nday of week, lead time and month\nCalibrated counterfactual $\\tilde{y}_{D,h} = \\hat{y}_{D,h}\\,\\exp(\\hat{r}_{D,h})$\nDaily 90 % band: model quantiles, same regression at 5 % / 95 %")
box(XS[0], R2, 4, "Effect", "Daily:  $y_t\\,/\\,\\tilde{y}_t - 1$\n116-day cumulative:\n   $\\hat{\\tau}_D = \\sum_h y_{D+h-1}\\,/\\,\\sum_h \\tilde{y}_{D,h} - 1$\nAdditional riders:  $\\sum_h\\,(y_{D+h-1} - \\tilde{y}_{D,h})$")
box(XS[1], R2, 5, "Placebo distribution", "Steps 2–4 at the 21 pseudo-policy dates give\n$\\hat{\\tau}_1,\\dots,\\hat{\\tau}_{21}$: the error of the method when\nno policy is present")
box(XS[2], R2, 6, "Inference", "Significance: number of placebo values at or\nabove the policy estimate (0 of 21: one-sided\n$p \\approx 1/22$)\n90 % interval: $[\\hat{\\tau} - q_{95},\\ \\hat{\\tau} - q_{05}]$, $q$ = quantiles\nof the mean-centred placebo values", accent=BLUE)
ym1, ym2 = R1 + BH / 2, R2 + BH / 2
arrow((XS[0] + BW, ym1), (XS[1], ym1)); arrow((XS[1] + BW, ym1), (XS[2], ym1)); arrow((XS[0] + BW, ym2), (XS[1], ym2)); arrow((XS[1] + BW, ym2), (XS[2], ym2))
xm3, xm4, yg = XS[2] + BW / 2, XS[0] + BW / 2, (R1 + R2 + BH) / 2
ax.plot([xm3, xm3, xm4], [R1, yg, yg], color=INK, lw=0.8); arrow((xm4, yg), (xm4, R2 + BH))
# mini strip plot in box 5
rng = np.random.default_rng(4); bx, by = XS[1] + 6, R2 + 5.2; ax.plot([bx, bx + 43], [by, by], color=EDGE, lw=0.6); ax.plot([bx + 14] * 2, [by - 0.9, by + 3.6], color=INK, lw=0.5, ls=(0, (2, 1.5))); ax.text(bx + 14, by - 2.4, "0", ha="center", va="center", color=MUTED, fontsize=5.6)
ax.scatter(bx + 14 + rng.normal(-1.5, 4.2, 21), by + 1.4 + rng.uniform(-0.7, 0.7, 21), s=5, color=PL, edgecolor="white", linewidth=0.3, zorder=3); ax.scatter([bx + 34], [by + 1.4], s=20, marker="D", color=BLUE, edgecolor="white", linewidth=0.5, zorder=4)
ax.text(bx + 36.3, by + 1.4, "$\\hat{\\tau}$ policy", va="center", color=BLUE, fontsize=5.8); ax.text(bx + 43, by - 2.4, "116-day effect (%)", ha="right", va="center", color=MUTED, fontsize=5.6)
ax.text(3, 5.5, "The calibration at origin $D$ never uses observations on or after $D$. Forecast errors from windows that cross the 29 Aug 2024 break are excluded\nfrom calibration and from the placebo set.", color=MUTED, va="center", linespacing=1.4)
FIG.mkdir(parents=True, exist_ok=True)
for e, kw in (("pdf", {}), ("png", {"dpi": 300})): fig.savefig(FIG / f"fig_method_flow.{e}", **kw)
print("wrote", FIG / "fig_method_flow.png", len(pseudo), pseudo[0].date(), pseudo[-1].date())
