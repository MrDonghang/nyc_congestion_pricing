"""Robustness check figures.

Section 1 — Daily ATT comparison across three forecasters:
  * TimesFM (raw)
  * TimesFM-HQC (per-unit intercept + pooled quantile calibration)
  * Chronos-HQC

One panel per (mode, direction); each panel overlays three lines with shaded
90% prediction intervals on the test window.
"""

# %% Auto-reload + publication rcParams
try:
    get_ipython().run_line_magic("load_ext", "autoreload")     # type: ignore[name-defined]
    get_ipython().run_line_magic("autoreload", "2")            # type: ignore[name-defined]
except (NameError, ImportError):
    pass

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from nyc_cp.config import REPO_ROOT, output_dir, load_paths

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# %% Configuration
WINDOW = "test"

# The three forecasters we compare. Order matters for legend / z-order.
MODELS = [
    {"key": "timesfm",                  "label": "TimesFM",      "color": "#1F77B4"},
    {"key": "timesfm_qrcal_intercept",  "label": "TimesFM-HQC",  "color": "#D55E00"},
    {"key": "chronos_qrcal_intercept",  "label": "Chronos-HQC",  "color": "#009E73"},
]

# Each combo gets one figure. Bus has only "all" direction.
COMBOS = [
    {"mode": "bus",     "direction": "all",   "title": "Bus"},
    {"mode": "subway",  "direction": "O",     "title": "Subway (Origin)"},
    {"mode": "subway",  "direction": "D",     "title": "Subway (Destination)"},
    {"mode": "subway",  "direction": "total", "title": "Subway"},
    {"mode": "replica", "direction": "O",     "title": "Replica / Overall travel"},
    {"mode": "replica", "direction": "D",     "title": "Replica / Overall travel (Destination)"},
]

POLICY_DATE = pd.Timestamp("2025-01-05")

SAVE = True
OUT_DIR = REPO_ROOT / "outputs" / "figures" / "paper" / "robustness"


# %% Helper to load one daily csv
paths = load_paths()


def load_daily(mode: str, model: str, direction: str) -> pd.DataFrame:
    """Return per-day ATT table for ``(mode, model, direction)`` on the test window."""
    base = output_dir(mode, model, direction=direction, paths=paths)
    sfx = "" if direction == "all" else f"_{direction}"
    csv = base / "effects" / f"{mode}_{model}_{WINDOW}{sfx}_daily.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    df = pd.read_csv(csv, parse_dates=["date"])
    return df.sort_values("date")


# %% Plot one (mode, direction)
def plot_combo(ax, mode: str, direction: str, title: str) -> None:
    for spec in MODELS:
        df = load_daily(mode, spec["key"], direction)
        ax.fill_between(
            df["date"], df["mean_eff_lo"], df["mean_eff_hi"],
            color=spec["color"], alpha=0.13, lw=0, zorder=1,
        )
        ax.plot(
            df["date"], df["mean_tau"],
            color=spec["color"], lw=1.6, label=spec["label"], zorder=3,
        )

    # Weekend shading (every Sat–Sun pair).
    first_df = load_daily(mode, MODELS[0]["key"], direction)
    date_min, date_max = first_df["date"].min(), first_df["date"].max()
    for d in pd.date_range(date_min, date_max, freq="D"):
        if d.weekday() == 5:
            ax.axvspan(d, d + pd.Timedelta(days=2), color="0.92", alpha=0.5, lw=0, zorder=0)

    ax.axhline(0, color="0.25", lw=0.8, ls=(0, (3, 2)), alpha=0.8, zorder=0)
    ax.set_title(title, loc="left", fontweight="bold", pad=4)
    ax.set_ylabel("Daily ATT")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", color="0.92", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)


# %% Render — one figure per combo
if SAVE:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

for cfg in COMBOS:
    fig, ax = plt.subplots(figsize=(11, 3), dpi=200)
    plot_combo(ax, cfg["mode"], cfg["direction"], cfg["title"])
    ax.legend(loc="upper right", frameon=False, ncol=3, handlelength=2.0, columnspacing=1.4)
    fig.tight_layout()

    if SAVE:
        stem = f"daily_att_compare_{cfg['mode']}" + ("" if cfg["direction"] == "all" else f"_{cfg['direction']}")
        fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight", dpi=300)
        print(f"saved → {OUT_DIR / stem}.{{pdf,png}}")
    plt.show()

print("Done.")

# %%
