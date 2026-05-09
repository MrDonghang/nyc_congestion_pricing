"""Per-unit residual diagnostic on the val window: for each route / station /
tract, compute the mean relative bias (actual − mu) / mean(mu) before and
after QR calibration. Saves one figure per (mode, dir) and a master CSV.

Goal: see whether the global QR calibration leaves systematic per-unit bias,
which would justify adding unit fixed effects.
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

COMBOS = [("bus", "all"), ("subway", "O"), ("subway", "D"), ("citibike", "O"), ("citibike", "D")]


def gather(mode: str, direction: str):
    paths = load_paths()
    cfg = load_mode(mode)
    actual = load_actual(mode, direction=direction, mode_cfg=cfg, paths=paths)
    val_w = get_window(mode, "val")
    a = actual.loc[(actual.index >= pd.Timestamp(val_w.test_start)) & (actual.index <= pd.Timestamp(val_w.test_end))]

    suffix = f"_{direction}" if direction != "all" else ""
    pre_dir = output_dir(mode, "chronos", direction=direction, paths=paths)
    post_dir = output_dir(mode, "chronos_qrcal", direction=direction, paths=paths)
    mu_pre = pd.read_csv(pre_dir / f"{mode}_chronos_val{suffix}_mu.csv", index_col=0, parse_dates=[0])
    mu_post = pd.read_csv(post_dir / f"{mode}_chronos_qrcal_val{suffix}_mu.csv", index_col=0, parse_dates=[0])

    cols = a.columns.intersection(mu_pre.columns).intersection(mu_post.columns)
    a = a[cols].reindex(mu_pre.index)
    mu_pre = mu_pre[cols]
    mu_post = mu_post[cols]
    return a, mu_pre, mu_post


def per_unit(a: pd.DataFrame, mu: pd.DataFrame) -> pd.DataFrame:
    """Per-column: actual mean, mu mean, abs bias, relative bias."""
    actual_mean = a.mean(axis=0)
    mu_mean = mu.mean(axis=0)
    abs_bias = (a - mu).mean(axis=0)
    # relative bias against mu (so positive = under-prediction)
    rel = abs_bias / mu_mean.replace(0, np.nan)
    return pd.DataFrame({
        "actual_mean": actual_mean,
        "mu_mean": mu_mean,
        "abs_bias": abs_bias,
        "rel_bias": rel,
    })


def render_one(mode: str, direction: str, fig_dir: Path, csv_dir: Path) -> dict:
    a, mu_pre, mu_post = gather(mode, direction)
    pre = per_unit(a, mu_pre).rename(columns=lambda c: f"{c}_pre")
    post = per_unit(a, mu_post).rename(columns=lambda c: f"{c}_post")
    df = pre.join(post)
    df["log_level"] = np.log(df["actual_mean_pre"].clip(lower=1.0))

    label = mode if direction == "all" else f"{mode}/{direction}"
    suffix = f"_{direction}" if direction != "all" else ""
    csv_path = csv_dir / f"{mode}_chronos_qrcal_per_unit_bias{suffix}.csv"
    df.sort_values("rel_bias_pre").to_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    pre_pct = df["rel_bias_pre"] * 100
    post_pct = df["rel_bias_post"] * 100

    bins = np.linspace(min(pre_pct.min(), post_pct.min()) - 1,
                       max(pre_pct.max(), post_pct.max()) + 1, 50)
    axes[0].hist(pre_pct.dropna(), bins=bins, alpha=0.6, color="#d62728", label="pre-cal")
    axes[0].hist(post_pct.dropna(), bins=bins, alpha=0.6, color="#1f77b4", label="post-cal")
    axes[0].axvline(0, color="black", lw=0.6)
    axes[0].set_xlabel("Per-unit relative bias (actual − mu) / mu  [%]")
    axes[0].set_ylabel("# units")
    axes[0].set_title(f"{label} — bias distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    sc = axes[1].scatter(pre_pct, post_pct, c=df["log_level"], s=14, cmap="viridis", alpha=0.7)
    lim = max(abs(pre_pct).max(), abs(post_pct).max()) * 1.05
    axes[1].plot([-lim, lim], [-lim, lim], "k--", lw=0.5, alpha=0.5)
    axes[1].axhline(0, color="black", lw=0.4)
    axes[1].axvline(0, color="black", lw=0.4)
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[1].set_xlabel("pre-cal bias [%]")
    axes[1].set_ylabel("post-cal bias [%]")
    axes[1].set_title(f"{label} — pre vs post (per unit)")
    axes[1].grid(alpha=0.3)
    cbar = plt.colorbar(sc, ax=axes[1])
    cbar.set_label("log(actual_mean)", fontsize=8)

    # Bias vs level: does cal residual depend on unit size?
    axes[2].scatter(df["log_level"], post_pct, s=14, color="#1f77b4", alpha=0.6, label="post")
    axes[2].scatter(df["log_level"], pre_pct, s=14, color="#d62728", alpha=0.4, label="pre")
    axes[2].axhline(0, color="black", lw=0.4)
    axes[2].set_xlabel("log(actual_mean)  [unit size]")
    axes[2].set_ylabel("relative bias [%]")
    axes[2].set_title(f"{label} — residual vs log_level")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = fig_dir / f"per_unit_bias_{mode}{suffix}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {
        "combo": label,
        "n_units": int(df.shape[0]),
        "pre_median": float(pre_pct.median()),
        "pre_p10": float(pre_pct.quantile(0.10)),
        "pre_p90": float(pre_pct.quantile(0.90)),
        "pre_iqr": float(pre_pct.quantile(0.75) - pre_pct.quantile(0.25)),
        "post_median": float(post_pct.median()),
        "post_p10": float(post_pct.quantile(0.10)),
        "post_p90": float(post_pct.quantile(0.90)),
        "post_iqr": float(post_pct.quantile(0.75) - post_pct.quantile(0.25)),
    }


def main() -> None:
    fig_dir = Path("outputs") / "figures" / "chronos_qrcal_per_unit"
    csv_dir = Path("outputs") / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode, direction in COMBOS:
        rows.append(render_one(mode, direction, fig_dir, csv_dir))

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:+.1f}"))
    summary.to_csv(csv_dir / "chronos_qrcal_per_unit_bias_summary.csv", index=False)
    print(f"\nSaved figures to {fig_dir}")
    print(f"Saved CSVs    to {csv_dir}")


if __name__ == "__main__":
    main()
