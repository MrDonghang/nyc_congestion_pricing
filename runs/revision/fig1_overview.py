"""Figure 1 (overview) - Nature-style, 183 mm double column, Arial 5-8 pt, no titles, editable text. Four-column grid.
Hero row (large):
  a, b  one example station / route: observed vs forecast (7-day means) with the calibrated 90 % model interval, in a no-policy rehearsal
        window one year earlier and in the policy window. Number = effect on the window total with its calibrated 90 % interval
        (split-conformal on windows ending before the origin; no pseudo-policy information). See example_forecasts.py.
Evidence block (small multiples; top row subway, bottom row bus, routes other than Bronx local):
  c  daily effect: 7-day mean, 90 % interval, thin line = daily values
  d  cumulative additional riders with 90 % interval
  e  116-day effect (diamond) against the 2024 pseudo-policy dates (dots; tick = their mean)
  f  cumulative effect path of the real policy vs every 2024 pseudo-policy date
Usage: python runs/revision/fig1_overview.py [timesfm|chronos_bolt]  -> main/fig1_overview[_chronos_bolt].{pdf,svg,tiff,png}
"""
import glob, os, sys
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.font_manager as fm
import numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT
for f in glob.glob(os.path.expanduser("~/.fonts/arial*.ttf")): fm.fontManager.addfont(f)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "chronos_bolt"; SUF = "" if MODEL == "chronos_bolt" else "_" + MODEL      # main model: Chronos-Bolt (user decision 2026-11-26); TimesFM = robustness
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; FIG = REPO_ROOT / "outputs" / "figures" / "paper" / "main"; MM = 1 / 25.4
POLICY, B0, H = pd.Timestamp("2025-01-05"), pd.Timestamp("2024-08-29"), 116
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2, "ytick.major.size": 2.2, "legend.frameon": False, "axes.spines.right": False, "axes.spines.top": False,
                     "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.labelpad": 2.5})
# palette: one hue per mode (cool / warm), ink for observations, warm-grey wash for the policy period, greys for no-policy references
COL = {"subway": "#16788A", "bus": "#D4722A"}; DARK = {"subway": "#0F5563", "bus": "#A8551A"}; TINT = {"subway": "#BFE0E4", "bus": "#F6D4B4"}
INK, GREY, MID, LIGHT, WASH = "#1C1C1C", "#6E6E6E", "#9C9C9C", "#C9C9C9", "#F5F1EA"; KINDS = ("subway", "bus"); NAME = {"subway": "Subway", "bus": "Bus"}; EXNAME = {"subway": "Northern Blvd station", "bus": "B11 bus route"}
mn = lambda s: s.replace("-", "−")
# ------------------------------------------------------------------ data
ex = pd.read_csv(U / f"example_forecasts_{MODEL}.csv", parse_dates=["origin", "date"])
SPEC = os.environ.get("FIG1_SPEC", "qr")   # "qr": quantile-regression calibration + model quantiles (qr_final.py, paper_tables_qr.py); "huber": earlier Huber version
if SPEC == "qr":
    sc = pd.read_csv(U / "qr_final_scores.csv", parse_dates=["origin"]); SC = {k: sc[(sc.panel == k) & (sc.model == MODEL)].set_index("origin").e116 for k in ("subway", "bus")}; Q = "_qr"
else:
    s8 = pd.read_csv(U / "round8_scores.csv", parse_dates=["origin"]); s9 = pd.read_csv(U / "round9_scores.csv", parse_dates=["origin"]); Q = ""
    SC = {"subway": s8[(s8.panel == "subway_ns22") & (s8.model == MODEL) & (s8.h == H) & (s8.variant == "V3")].set_index("origin").score, "bus": s9[(s9.group == "non-Bronx") & (s9.model == MODEL) & (s9.h == H) & (s9.variant == "V3")].set_index("origin").score}
daily = pd.read_csv(U / f"paper_daily{Q}_{MODEL}.csv", parse_dates=["date"]); cum = pd.read_csv(U / f"paper_cumulative{Q}_{MODEL}.csv", parse_dates=["date"]); pp = pd.read_csv(U / f"paper_placebo_paths{Q}_{MODEL}.csv")
pick = lambda d, k: d[d.series.str.startswith("Subway" if k == "subway" else "Bus")]
# ------------------------------------------------------------------ layout (mm): 4 columns; each hero panel spans two of them
W = 183.0; L, R, G = 11.5, 3.0, 10.0; cw = (W - L - R - 3 * G) / 4; X = [L + i * (cw + G) for i in range(4)]
TOP, HERO, GAP1, HEAD, ROW, RG, BOT = 6.0, 47.0, 11.0, 5.0, 27.0, 5.0, 18.0; Hh = TOP + HERO + GAP1 + HEAD + 2 * ROW + RG + BOT; fig = plt.figure(figsize=(W * MM, Hh * MM))
y_hero = Hh - TOP - HERO; YR = {"subway": y_hero - GAP1 - HEAD - ROW, "bus": y_hero - GAP1 - HEAD - 2 * ROW - RG}
def add(x, y, w, h): return fig.add_axes([x / W, y / Hh, w / W, h / Hh])
def plabel(x, ytop, s): fig.text((x - 9.0) / W, (ytop + 1.6) / Hh, s, fontsize=8, fontweight="bold", va="bottom", ha="left")
def months(ax, d0, d1, labels=True):
    ax.set_xlim(d0, d1); ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b")); ax.tick_params(axis="x", pad=2, labelbottom=labels)
stats = []; r7 = lambda s: s.rolling(7, center=True).mean() / 1e3
# ================================================================== hero: a, b
sw = cw + G / 2 - 1.5      # two sub-axes fill the two-column span with a 3 mm break between them
for i, kind in enumerate(KINDS):
    e = ex[ex["mode"] == kind]; name = EXNAME[kind]; axs = []; lim = [np.inf, -np.inf]
    for j, win in enumerate(("rehearsal", "policy")):
        ax = add(X[2 * i] + j * (sw + 3.0), y_hero, sw, HERO); axs.append(ax); t = e[e.window == win].set_index("date").sort_index(); o = t.origin.iloc[0]; f = t.dropna(subset=["cf"])
        obs, cf, lo, hi = r7(t.actual), r7(f.cf), f.lower7 / 1e3, f.upper7 / 1e3; d1 = pd.Timestamp(o.year, 5, 2)
        if win == "policy": ax.axvspan(o, d1, color=WASH, lw=0, zorder=0)
        ax.fill_between(cf.index, lo, hi, color=TINT[kind], lw=0, zorder=1); ax.plot(cf.index, cf, color=COL[kind], lw=1.1, zorder=3); ax.plot(obs.index, obs, color=INK, lw=1.0, zorder=4)
        ax.axvline(o, color=GREY, lw=0.6, ls=(0, (2, 2)), zorder=2); ax.set_xlim(pd.Timestamp(o.year - 1, 12, 3), d1); ax.set_xticks([pd.Timestamp(o.year, m, 1) for m in (1, 2, 3, 4)]); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr"]); ax.tick_params(axis="x", pad=2); lim = [min(lim[0], lo.min(), obs.min()), max(lim[1], hi.max(), obs.max())]
        eff, hw = f.effect_pct.iloc[0], f.effect_halfwidth.iloc[0]; sig = abs(eff) > hw
        stats.append(dict(panel=name, window=win, origin=o.date(), effect=eff, ci_lo=eff - hw, ci_hi=eff + hw, days_above_band=f.days_above_pct.iloc[0], days_below_band=f.days_below_pct.iloc[0]))
        ax.text(0.965, 0.965, f"No policy, {o.year}" if win == "rehearsal" else f"Policy, {o.year}", transform=ax.transAxes, fontsize=6.5, color=GREY if win == "rehearsal" else DARK[kind], fontweight="normal" if win == "rehearsal" else "bold", va="top", ha="right")
        ax.text(0.965, 0.115, mn(f"{eff:+.1f}%"), transform=ax.transAxes, fontsize=8 if sig else 7, color=DARK[kind] if sig else GREY, fontweight="bold" if sig else "normal", va="bottom", ha="right")
        ax.text(0.965, 0.04, mn(f"90% interval {eff - hw:+.1f} to {eff + hw:+.1f}"), transform=ax.transAxes, fontsize=5.8, color=DARK[kind] if sig else GREY, va="bottom", ha="right")
        if j: ax.spines["left"].set_visible(False); ax.tick_params(axis="y", left=False, labelleft=False)
    pad = 0.08 * (lim[1] - lim[0]); lo_y = max(0.0, lim[0] - 3.2 * pad)
    for ax in axs: ax.set_ylim(lo_y, lim[1] + 2.4 * pad); ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    axs[0].set_ylabel(("Entries" if kind == "subway" else "Boardings") + " per day (thousands)"); axs[0].text(0.0, 1.03, name, transform=axs[0].transAxes, fontsize=7, color=DARK[kind], fontweight="bold", va="bottom", ha="left"); plabel(X[2 * i], y_hero + HERO, "ab"[i])
# ================================================================== evidence block: c-f (rows = modes)
HEADS = ["Daily effect (%)", "Additional riders (millions)", "116-day effect (%)", "Cumulative effect (%)"]; ytop = YR["subway"] + ROW
for c, s in enumerate(HEADS): fig.text(X[c] / W, (ytop + 1.8) / Hh, s, fontsize=6.5, color=INK, va="bottom", ha="left"); plabel(X[c], ytop, "cdef"[c])
rng = np.random.default_rng(3); D0, D1 = POLICY - pd.Timedelta(days=2), POLICY + pd.Timedelta(days=H + 1)
for kind in KINDS:
    last = kind == "bus"; y0 = YR[kind]
    # c: effect vs pseudo-policy dates
    ax = add(X[2], y0, cw, ROW); s = SC[kind]; real = s[POLICY]; nul = s[(s.index >= "2024-01-01") & (s.index + pd.Timedelta(days=H - 1) < B0)]
    ax.scatter(nul, rng.uniform(-0.2, 0.2, len(nul)) - 0.25, s=8, color=LIGHT, edgecolor="white", linewidth=0.3, zorder=2); ax.plot([nul.mean()] * 2, [-0.6, 0.1], color=MID, lw=0.9, zorder=3)
    ax.scatter([real], [-0.25], marker="D", s=22, color=COL[kind], edgecolor="white", linewidth=0.4, zorder=4); ax.axvline(0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(0.0, 0.97, NAME[kind], transform=ax.transAxes, fontsize=7, fontweight="bold", color=DARK[kind], va="top", ha="left"); ax.text(1.0, 0.97, f"{real:+.1f}%", transform=ax.transAxes, fontsize=9, fontweight="bold", color=DARK[kind], va="top", ha="right")
    ax.set_xlim(-5, 5); ax.set_ylim(-0.85, 1.25); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_xticklabels(["−4", "−2", "0", "2", "4"]); ax.tick_params(axis="x", labelbottom=last, pad=2)
    stats.append(dict(panel=kind, window="116-day effect", effect=real, placebo_mean=nul.mean(), placebo_sd=nul.std(), n=len(nul), n_at_or_above=int((nul >= real).sum())))
    # d: daily effect
    ax = add(X[0], y0, cw, ROW); d = pick(daily, kind)
    ax.fill_between(d.date, d.ci7_lo, d.ci7_hi, color=TINT[kind], lw=0, zorder=1); ax.plot(d.date, d.eff, color=COL[kind], lw=0.3, alpha=0.5, zorder=2); ax.plot(d.date, d.eff_7d, color=COL[kind], lw=1.1, zorder=3)
    ax.axhline(0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1); months(ax, D0, D1, last); ax.set_ylim(-12, 18); ax.set_yticks([-10, 0, 10]); ax.set_yticklabels(["−10", "0", "10"])
    # e: cumulative additional riders
    ax = add(X[1], y0, cw, ROW); c_ = pick(cum, kind); est, lo, hi = c_.cum_effect / 1e6, c_.ci_lo / 1e6, c_.ci_hi / 1e6
    ax.fill_between(c_.date, lo, hi, color=TINT[kind], lw=0, zorder=1); ax.plot(c_.date, est, color=COL[kind], lw=1.1, zorder=3); ax.axhline(0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1); months(ax, D0, D1, last); ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.text(0.04, 0.95, mn(f"{est.iloc[-1]:+.1f} M"), transform=ax.transAxes, fontsize=7, color=DARK[kind], fontweight="bold", va="top", ha="left"); ax.text(0.04, 0.80, mn(f"{lo.iloc[-1]:+.1f} to {hi.iloc[-1]:+.1f}"), transform=ax.transAxes, fontsize=5.8, color=DARK[kind], va="top", ha="left")
    stats.append(dict(panel=kind, window="cumulative riders (millions)", effect=est.iloc[-1], ci_lo=lo.iloc[-1], ci_hi=hi.iloc[-1]))
    # f: real path vs pseudo-policy paths
    ax = add(X[3], y0, cw, ROW); p = pp[pp.series == NAME[kind]].dropna(axis=1, how="all").set_index("day"); P = p[[k for k in p.columns if k.startswith("placebo_")]]
    for k in P.columns: ax.plot(P.index, P[k].clip(-9.5, 9.5), color=LIGHT, lw=0.45, zorder=2)
    ax.plot(p.index, p.real.clip(-9.5, 9.5), color=COL[kind], lw=1.2, zorder=4); ax.axhline(0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(0, 119); ax.set_xticks([0, 28, 56, 84, 116]); ax.tick_params(axis="x", labelbottom=last, pad=2); ax.set_ylim(-8, 8); ax.set_yticks([-8, -4, 0, 4, 8]); ax.set_yticklabels(["−8", "−4", "0", "4", "8"])
    if last: ax.set_xlabel("Days since start")
# ---- one legend for the whole figure
hd = [plt.Line2D([], [], color=INK, lw=1.0, label="Observed"), plt.Line2D([], [], color=COL["subway"], lw=1.2, label="Subway: forecast or effect"), plt.Line2D([], [], color=COL["bus"], lw=1.2, label="Bus: forecast or effect"),
      mpl.patches.Patch(facecolor="#DCDCDC", edgecolor="none", label="90% interval"), plt.Line2D([], [], marker="o", ls="none", color=LIGHT, ms=3, label="Pseudo-policy dates in 2024 (no policy)")]
fig.legend(handles=hd, loc="lower center", ncol=5, fontsize=6.5, handlelength=1.6, columnspacing=2.0, bbox_to_anchor=(0.5, 1.0 / Hh))
for e_, kw in (("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}), ("png", {"dpi": 300})): fig.savefig(FIG / f"fig1_overview{SUF}.{e_}", **kw)
pd.set_option("display.width", 220); print(pd.DataFrame(stats).round(2).to_string(index=False)); print(f"figure {W:.0f} x {Hh:.0f} mm ->", FIG / f"fig1_overview{SUF}.png")
