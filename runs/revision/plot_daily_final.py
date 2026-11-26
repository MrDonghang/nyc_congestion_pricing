"""Daily effect under the final specification (quantile-regression calibration, seasonal design), system level.
For each day of 5 Jan – 30 Apr 2025: effect = actual / calibrated counterfactual − 1.
Bands: 90 % intervals from the models' native quantiles, calibrated by quantile regression (see paper_tables_qr.py).
Writes main_daily_timesfm.{png,pdf}, appendix_daily_chronos_bolt.{png,pdf}, uq/paper_daily_<model>.csv
"""
import glob, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import numpy as np, pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.linear_model import HuberRegressor
from nyc_cp.config import REPO_ROOT
warnings.filterwarnings("ignore")
U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; FIG = REPO_ROOT / "outputs" / "figures" / "paper" / "robustness"
POLICY, B0 = pd.Timestamp("2025-01-05"), pd.Timestamp("2024-08-29"); H = 116
C_MAIN, C_PL, INK, MUTED = "#2a78d6", "#d9d7cf", "#1f1f1f", "#5f5e5a"
hol = set(USFederalHolidayCalendar().holidays("2021-12-01", "2026-12-31"))
# Specification of 2026-11-26: quantile-regression (median) calibration with the seasonal design; daily bands from the models' native 10 % / 90 % quantiles
# calibrated at tau = 0.05 / 0.95 (qr_native.py -> uq/qr_native_daily.csv, paper_tables_qr.py). The table holds the policy date and the 21 pseudo-policy dates only.
QD = pd.read_csv(U / "qr_native_daily.csv", parse_dates=["origin", "date"])
def load(kind, model): return QD[(QD.panel == kind) & (QD.model == model)].rename(columns={"cc": "cfc"}).sort_values(["origin", "h"]).reset_index(drop=True)
def calibrated(d, o): te = d[d.origin == o].copy(); te["eff"] = 100 * (te.actual / te.cfc - 1); return te
PAN = [("subway", "Subway entries"), ("bus", "Bus (excl. Bronx local routes)")]
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
for model, mlab, fname in (("chronos_bolt", "Chronos-Bolt", "main_daily_chronos_bolt"), ("timesfm", "TimesFM-2.0", "appendix_daily_timesfm")):
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True); out = []
    for ax, (kind, title) in zip(axes, PAN):
        d = load(kind, model); origins = pd.DatetimeIndex(sorted(d.origin.unique()))
        real = calibrated(d, POLICY)
        raw = [calibrated(d, o).eff.to_numpy() for i, o in enumerate(origins) if o >= pd.Timestamp("2024-01-01") and o + pd.Timedelta(days=H - 1) < B0]
        Pd = np.vstack(raw); pl = raw                                                   # K × 116 daily no-policy "effects"
        Ps = np.vstack([pd.Series(r).rolling(7, center=True, min_periods=4).mean().to_numpy() for r in raw])
        # 90 % interval for the effect on day h: estimate minus the 95th / 5th percentile of the no-policy error at horizon
        # days h−3 … h+3 (pooled over the K pseudo-policy dates) — the same placebo logic as the cumulative estimate.
        def band(P):
            lo, hi = np.empty(H), np.empty(H)
            for h in range(H):
                w = P[:, max(0, h - 3):min(H, h + 4)].ravel(); w = w - w.mean(); lo[h], hi[h] = np.percentile(w, 5), np.percentile(w, 95)   # centred
            return lo, hi
        qlo, qhi = band(Pd); slo, shi = band(Ps)
        sm = real.eff.rolling(7, center=True, min_periods=4).mean()
        pq = pd.read_csv(U / f"paper_daily_qr_{model}.csv"); pq = pq[pq.series == title]; ci_lo, ci_hi, sci_lo, sci_hi = (pq[c].to_numpy() for c in ("ci_lo", "ci_hi", "ci7_lo", "ci7_hi"))      # model-quantile bands (paper_tables_qr.py)
        lo, hi, mid = slo, shi, Ps.mean(axis=0)
        ax.fill_between(real.date, ci_lo, ci_hi, color="#cfe0f5", lw=0, label="90 % interval, daily estimate")
        ax.plot(real.date, real.eff, color=C_MAIN, lw=0.7, alpha=0.6, label="daily estimate"); ax.plot(real.date, sm, color="#0d366b", lw=2.0, label="7-day mean")
        for hd in real.date[real.date.isin(hol)]: ax.axvline(hd, color="#b9b7ae", lw=0.8, ls="--", zorder=0)
        ax.axhline(0, color=INK, lw=0.8); ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True); ax.set_ylim(-14, 18)
        cum = 100 * (real.actual.sum() / real.cfc.sum() - 1); above = float((sm.to_numpy() > hi).mean()); below = float((sm.to_numpy() < lo).mean())
        ax.set_title(f"{title} — {mlab}", loc="left", fontweight="bold", fontsize=10, color=INK); ax.set_ylabel("effect on ridership (%)", color=MUTED); ax.tick_params(labelbottom=True)
        sig_d = float((ci_lo > 0).mean()); sig_s = float((sci_lo > 0).mean()); neg_d = float((ci_hi < 0).mean())
        ax.text(0.995, 0.96, f"116-day cumulative {cum:+.1f}%   ·   interval above zero on {100 * sig_d:.0f}% of days", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK)
        ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")); ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
        blk = [(real.eff.iloc[a:b].mean()) for a, b in ((0, 29), (29, 58), (58, 87), (87, 116))]
        print(f"   daily CI>0 {100*sig_d:.0f}%  CI<0 {100*neg_d:.0f}%  7d CI>0 {100*sig_s:.0f}%  mean half-width daily {np.mean(ci_hi-ci_lo)/2:.1f}  7d {np.mean(sci_hi-sci_lo)/2:.1f}")
        print(f"{model:13s} {kind:7s} cum {cum:+.2f}  mean-daily {real.eff.mean():+.2f}  median-daily {real.eff.median():+.2f}  days>0 {100 * (real.eff > 0).mean():.0f}%  7d-mean above band {100 * above:.0f}% / below {100 * below:.0f}%  blocks " + " ".join(f"{b:+.1f}" for b in blk) + f"  holidays {real.eff[real.date.isin(hol)].round(1).tolist()}  band half-width {np.mean(hi - lo) / 2:.1f}")
        out.append(real.assign(series=title, model=mlab, ci_lo=ci_lo, ci_hi=ci_hi, eff_7d=sm.to_numpy(), ci7_lo=sci_lo, ci7_hi=sci_hi)[["series", "model", "date", "h", "actual", "cfc", "eff", "ci_lo", "ci_hi", "eff_7d", "ci7_lo", "ci7_hi"]])
    hh, ll = axes[0].get_legend_handles_labels(); order = [1, 0, 2]
    hh.append(plt.Line2D([], [], color="#b9b7ae", lw=0.8, ls="--")); ll.append("federal holiday")
    fig.legend([hh[k] for k in order] + [hh[-1]], [ll[k] for k in order] + [ll[-1]], loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.8)
    for e in ("png", "pdf"): fig.savefig(FIG / f"{fname}.{e}", dpi=200, bbox_inches="tight")
    pd.concat(out).to_csv(U / f"paper_daily_{model}.csv", index=False); plt.close(fig)
