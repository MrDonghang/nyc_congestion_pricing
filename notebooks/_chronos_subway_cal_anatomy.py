"""Anatomy of QR-cal modifications on subway/O and subway/D val window.

For each direction, breaks the cal delta = mu_post − mu_pre down by:
  (a) day of week         — when does cal push mu up vs down?
  (b) month               — seasonal correction
  (c) per-unit log level  — does cal adjust large stations differently?
  (d) the fitted median QR coefs (re-fit once to dump them)

Saves a 4-panel figure per direction + a CSV of coefs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nyc_cp.calibration import build_features, fit_calibration, residuals_long
from nyc_cp.config import get_window, load_mode, load_paths, output_dir
from nyc_cp.data import load_actual

DIRECTIONS = ["O", "D"]
ID_COL = "station_id"
DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_ORDER = list(range(1, 13))


def gather(direction: str):
    paths = load_paths()
    cfg = load_mode("subway")
    actual = load_actual("subway", direction=direction, mode_cfg=cfg, paths=paths)
    val_w = get_window("subway", "val")
    a = actual.loc[(actual.index >= pd.Timestamp(val_w.test_start)) & (actual.index <= pd.Timestamp(val_w.test_end))]

    suffix = f"_{direction}"
    pre_dir = output_dir("subway", "chronos", direction=direction, paths=paths)
    post_dir = output_dir("subway", "chronos_qrcal", direction=direction, paths=paths)
    mu_pre = pd.read_csv(pre_dir / f"subway_chronos_val{suffix}_mu.csv", index_col=0, parse_dates=[0])
    mu_post = pd.read_csv(post_dir / f"subway_chronos_qrcal_val{suffix}_mu.csv", index_col=0, parse_dates=[0])

    cols = a.columns.intersection(mu_pre.columns).intersection(mu_post.columns)
    a = a[cols].reindex(mu_pre.index)
    mu_pre = mu_pre[cols]
    mu_post = mu_post[cols]
    return a, mu_pre, mu_post, val_w


def fit_q50_coefs(direction: str) -> pd.Series:
    """Re-fit only q=0.5 to dump its linear coefs (subway is ~50k×18, ~1 min)."""
    paths = load_paths()
    cfg = load_mode("subway")
    actual = load_actual("subway", direction=direction, mode_cfg=cfg, paths=paths)
    val_w = get_window("subway", "val")
    val_train_end = pd.Timestamp(val_w.train_end)
    history_for_levels = actual.loc[actual.index <= val_train_end]
    a_val = actual.loc[(actual.index >= pd.Timestamp(val_w.test_start)) & (actual.index <= pd.Timestamp(val_w.test_end))]

    pre_dir = output_dir("subway", "chronos", direction=direction, paths=paths)
    suffix = f"_{direction}"
    mu_pre = pd.read_csv(pre_dir / f"subway_chronos_val{suffix}_mu.csv", index_col=0, parse_dates=[0])

    feat = build_features(history_for_levels, mu_pre, id_col=ID_COL).rename(columns={ID_COL: "unit_id"})
    res = residuals_long(a_val, mu_pre, id_col=ID_COL)
    feat = feat.set_index(["date", "unit_id"])
    common = feat.index.intersection(res.index)
    feat = feat.loc[common].reset_index()
    res = res.loc[common]

    cal = fit_calibration(feat, res, quantiles=(0.5,), alpha=1e-4)
    m = cal.models[0.5]
    coefs = pd.Series(m.coef_, index=cal.feature_cols)
    coefs["__intercept__"] = m.intercept_
    return coefs


def render(direction: str, fig_dir: Path):
    a, mu_pre, mu_post, val_w = gather(direction)

    delta = mu_post - mu_pre  # wide
    actual_per_unit = a.mean(axis=0)
    log_level = np.log(actual_per_unit.clip(lower=1.0))

    long = delta.reset_index().melt(id_vars=delta.index.name or "index", var_name="unit", value_name="delta")
    long = long.rename(columns={delta.index.name or "index": "date"})
    long["date"] = pd.to_datetime(long["date"])
    long["dow"] = long["date"].dt.day_name().str[:3]
    long["month"] = long["date"].dt.month
    long["log_level"] = long["unit"].map(log_level)
    mu_pre_long = mu_pre.reset_index().melt(id_vars=mu_pre.index.name or "index", var_name="unit", value_name="mu_pre")
    mu_pre_long = mu_pre_long.rename(columns={mu_pre.index.name or "index": "date"})
    mu_pre_long["date"] = pd.to_datetime(mu_pre_long["date"])
    long = long.merge(mu_pre_long, on=["date", "unit"], how="left")
    long["delta_rel"] = long["delta"] / long["mu_pre"].replace(0, np.nan) * 100

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # (a) DOW
    g = long.groupby("dow")["delta"].mean().reindex(DOW_ORDER)
    g_rel = long.groupby("dow")["delta_rel"].mean().reindex(DOW_ORDER)
    ax = axes[0, 0]
    bars = ax.bar(DOW_ORDER, g.values, color=["#1f77b4" if v > 0 else "#d62728" for v in g.values])
    ax.axhline(0, color="black", lw=0.6)
    for i, (v, rv) in enumerate(zip(g.values, g_rel.values)):
        ax.text(i, v, f"{v:+.0f}\n({rv:+.1f}%)", ha="center", va="bottom" if v > 0 else "top", fontsize=8)
    ax.set_ylabel("mean Δμ (rides/day)")
    ax.set_title(f"subway/{direction} — Δμ by DOW")
    ax.grid(alpha=0.3, axis="y")

    # (b) Month
    g = long.groupby("month")["delta"].mean().reindex(MONTH_ORDER).dropna()
    g_rel = long.groupby("month")["delta_rel"].mean().reindex(MONTH_ORDER).dropna()
    ax = axes[0, 1]
    bars = ax.bar([f"{m:02d}" for m in g.index], g.values, color=["#1f77b4" if v > 0 else "#d62728" for v in g.values])
    ax.axhline(0, color="black", lw=0.6)
    for i, (m, v, rv) in enumerate(zip(g.index, g.values, g_rel.values)):
        ax.text(i, v, f"{v:+.0f}\n({rv:+.1f}%)", ha="center", va="bottom" if v > 0 else "top", fontsize=8)
    ax.set_ylabel("mean Δμ (rides/day)")
    ax.set_title(f"subway/{direction} — Δμ by month (val)")
    ax.grid(alpha=0.3, axis="y")

    # (c) per-unit Δμ vs log_level (scatter)
    per_unit_delta = delta.mean(axis=0)
    per_unit_rel = (delta.mean(axis=0) / mu_pre.mean(axis=0)) * 100
    ax = axes[1, 0]
    sc = ax.scatter(log_level.reindex(per_unit_delta.index), per_unit_rel, c=per_unit_delta, cmap="RdBu_r",
                    s=14, alpha=0.7, vmin=-abs(per_unit_delta).max(), vmax=abs(per_unit_delta).max())
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xlabel("log(actual_mean)  [station size]")
    ax.set_ylabel("mean Δμ / mu_pre  [%]")
    ax.set_title(f"subway/{direction} — per-station Δμ vs size")
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Δμ abs (rides/day)")

    # (d) Distribution of relative Δμ across all (date, unit) cells
    ax = axes[1, 1]
    ax.hist(long["delta_rel"].dropna(), bins=80, color="#9467bd", alpha=0.8)
    ax.axvline(0, color="black", lw=0.6)
    ax.axvline(long["delta_rel"].mean(), color="red", lw=1, ls="--", label=f"mean={long['delta_rel'].mean():+.1f}%")
    ax.axvline(long["delta_rel"].median(), color="orange", lw=1, ls="--", label=f"median={long['delta_rel'].median():+.1f}%")
    ax.set_xlabel("Δμ / mu_pre per (date, station)  [%]")
    ax.set_ylabel("count")
    ax.set_title(f"subway/{direction} — Δμ distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"subway/{direction} — anatomy of QR-cal corrections (val window)", fontsize=12, y=1.00)
    plt.tight_layout()
    out = fig_dir / f"subway_cal_anatomy_{direction}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    fig_dir = Path("outputs") / "figures" / "chronos_qrcal_anatomy"
    csv_dir = Path("outputs") / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    coefs_all = {}
    for d in DIRECTIONS:
        render(d, fig_dir)
        print(f"Re-fitting q=0.5 coefs for subway/{d} (~1 min)...")
        coefs_all[d] = fit_q50_coefs(d)

    coef_df = pd.DataFrame(coefs_all).round(2)
    out = csv_dir / "subway_chronos_qrcal_q50_coefs.csv"
    coef_df.to_csv(out)
    print(f"\nFitted q=0.5 coefs:\n{coef_df.to_string()}")
    print(f"\nSaved coefs to {out}")


if __name__ == "__main__":
    main()
