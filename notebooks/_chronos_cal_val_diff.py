"""Visualize chronos pre- vs post-calibration on the validation window.

Top-line question: is chronos systematically over- or under-predicting,
and does QR calibration fix it? Two columns per mode/dir:

  Left  — daily aggregate (sum across units): actual / mu_pre / mu_post
  Right — mean residual (actual - mu) by DOW, pre vs post

Saves a single PNG to ``outputs/figures/chronos_cal_val_diff.png``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nyc_cp.config import get_window, load_mode, load_paths, output_dir
from nyc_cp.data import load_actual

COMBOS = [
    ("bus", "all"),
    ("subway", "O"),
    ("subway", "D"),
    ("citibike", "O"),
    ("citibike", "D"),
]
DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _load_mu(out_dir: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(out_dir / f"{prefix}_mu.csv", index_col=0, parse_dates=[0])
    df.index.name = "date"
    return df


def _slice_val(df: pd.DataFrame, val_w) -> pd.DataFrame:
    return df.loc[(df.index >= pd.Timestamp(val_w.test_start)) & (df.index <= pd.Timestamp(val_w.test_end))]


def gather(mode: str, direction: str):
    paths = load_paths()
    mode_cfg = load_mode(mode)
    actual = load_actual(mode, direction=direction, mode_cfg=mode_cfg, paths=paths)
    val_w = get_window(mode, "val")
    a = _slice_val(actual, val_w)

    suffix = f"_{direction}" if direction != "all" else ""
    pre_dir = output_dir(mode, "chronos", direction=direction, paths=paths)
    post_dir = output_dir(mode, "chronos_qrcal", direction=direction, paths=paths)
    mu_pre = _load_mu(pre_dir, f"{mode}_chronos_val{suffix}")
    mu_post = _load_mu(post_dir, f"{mode}_chronos_qrcal_val{suffix}")

    # Align columns
    cols = a.columns.intersection(mu_pre.columns).intersection(mu_post.columns)
    a, mu_pre, mu_post = a[cols], mu_pre.reindex(a.index)[cols], mu_post.reindex(a.index)[cols]
    return a, mu_pre, mu_post


def render() -> Path:
    fig_dir = Path("outputs") / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    n = len(COMBOS)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.6 * n), gridspec_kw={"width_ratios": [3, 1.2]})

    for i, (mode, direction) in enumerate(COMBOS):
        a, mu_pre, mu_post = gather(mode, direction)
        agg_a = a.sum(axis=1)
        agg_pre = mu_pre.sum(axis=1)
        agg_post = mu_post.sum(axis=1)

        ax = axes[i, 0]
        ax.plot(agg_a.index, agg_a.values, color="black", lw=1.0, label="actual")
        ax.plot(agg_pre.index, agg_pre.values, color="#d62728", lw=1.0, alpha=0.85, label="mu (pre-cal)")
        ax.plot(agg_post.index, agg_post.values, color="#1f77b4", lw=1.0, alpha=0.85, label="mu (post-cal)")
        label = mode if direction == "all" else f"{mode}/{direction}"
        ax.set_title(f"{label} — daily aggregate (sum over units)", fontsize=10)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

        # DOW residual bars (relative to actual, %)
        r_pre = (a - mu_pre).mean(axis=1)
        r_post = (a - mu_post).mean(axis=1)
        cf_pre = mu_pre.mean(axis=1)
        cf_post = mu_post.mean(axis=1)
        df = pd.DataFrame({"r_pre": r_pre, "r_post": r_post, "cf_pre": cf_pre, "cf_post": cf_post})
        df["dow"] = df.index.day_name().str[:3]
        rel_pre = (df.groupby("dow")["r_pre"].mean() / df.groupby("dow")["cf_pre"].mean()).reindex(DOW_ORDER) * 100
        rel_post = (df.groupby("dow")["r_post"].mean() / df.groupby("dow")["cf_post"].mean()).reindex(DOW_ORDER) * 100

        ax2 = axes[i, 1]
        x = np.arange(len(DOW_ORDER))
        w = 0.4
        ax2.bar(x - w / 2, rel_pre.values, w, color="#d62728", alpha=0.8, label="pre")
        ax2.bar(x + w / 2, rel_post.values, w, color="#1f77b4", alpha=0.8, label="post")
        ax2.axhline(0, color="black", lw=0.6)
        ax2.set_xticks(x)
        ax2.set_xticklabels(DOW_ORDER, fontsize=8)
        ax2.set_title(f"{label} — mean (actual−mu)/mu by DOW [%]", fontsize=10)
        ax2.grid(alpha=0.3, axis="y")
        if i == 0:
            ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out = fig_dir / "chronos_cal_val_diff.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = render()
    print(f"Saved: {p}")
