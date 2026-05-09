"""Three-model overlay of daily ATT (chronos / timesfm / tft) for each (mode,
direction) combo on the test window. Saves PNGs to
``outputs/figures/_compare_tft_chronos_timesfm/``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from nyc_cp.config import load_paths, output_dir

WINDOW = "test"
MODELS = ["chronos", "timesfm"]
COLORS = {"chronos": "#1f77b4", "timesfm": "#2ca02c"}
COMBOS = [
    ("bus", "all"),
    ("subway", "O"),
    ("subway", "D"),
    ("citibike", "O"),
    ("citibike", "D"),
]


def load_daily(mode: str, direction: str, model: str) -> pd.DataFrame | None:
    paths = load_paths()
    out_dir = output_dir(mode, model, direction=direction, paths=paths)
    suffix = f"_{direction}" if direction != "all" else ""
    f = out_dir / "effects" / f"{mode}_{model}_{WINDOW}{suffix}_daily.csv"
    if not f.exists():
        return None
    return pd.read_csv(f, parse_dates=["date"])


def render_pair(mode: str, direction: str, fig_dir: Path) -> None:
    series = {m: load_daily(mode, direction, m) for m in MODELS}
    series = {m: d for m, d in series.items() if d is not None}
    if not series:
        print(f"SKIP {mode}/{direction}")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for m, df in series.items():
        c = COLORS[m]
        ax.plot(df["date"], df["mean_tau"], color=c, label=m, linewidth=1.5)
        ax.fill_between(df["date"], df["mean_eff_lo"], df["mean_eff_hi"],
                        color=c, alpha=0.12, linewidth=0)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Daily mean ATT (rides/unit/day)")
    ax.set_xlabel("Date")
    ax.set_title(f"{mode}/{direction}/{WINDOW} — daily ATT (chronos vs timesfm, 90% PI shaded)")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = fig_dir / f"compare_{mode}_{WINDOW}{('_' + direction) if direction != 'all' else ''}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[{mode}/{direction}] → {fname}  (models: {list(series)})")


if __name__ == "__main__":
    fig_dir = Path("outputs") / "figures" / "_compare_chronos_timesfm"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for mode, direction in COMBOS:
        render_pair(mode, direction, fig_dir)
    print("Done.")
