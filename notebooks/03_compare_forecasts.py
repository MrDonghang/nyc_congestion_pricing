"""Visualise all 9 forecasters on the same (mode, direction, window, series).

Open as a notebook in VS Code / Jupyter (cells delimited by ``# %%``) — each
cell is independent so you can re-run just the plotting cells after tweaking
``MODE`` / ``WINDOW`` / ``SERIES`` below. Reads forecasts saved by
``scripts/train_forecast.py`` under ``<output_root>/<mode>/<model>/``.

The two figures it produces:
1. **Overlay** — observed + every model's μ on one axis (no PI bands; readable
   when many models are present).
2. **Small multiples** — one subplot per model showing μ + 90% PI band against
   the ground truth, so PI calibration is comparable.
"""

# %% Imports + config
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nyc_cp.analysis.effects import load_forecast_triplet
from nyc_cp.config import get_window, load_mode, load_paths, output_dir
from nyc_cp.data import load_actual

# ---- Edit these to point at the (mode, direction, window, series) you want.
MODE = "bus"          # "bus" | "subway" | "citibike" | "replica"
DIRECTION = "all"          # "all" | "O" | "D"
WINDOW = "val"          # "val" | "test"
SERIES: str | int = 1    # column name (str) or positional index (int)
MODELS = ["arima", "prophet", "deepar", "pcn",
          "chronos", "timesfm", "nhits", "tft", "bsts"]
HISTORY_TAIL_DAYS = 180  # how many days of pre-window history to draw

# %% Load truth and every available forecast triplet
paths = load_paths()
mode_cfg = load_mode(MODE)
window = get_window(MODE, WINDOW, mode_cfg=mode_cfg)
test_start, test_end = pd.Timestamp(window.test_start), pd.Timestamp(window.test_end)

actual = load_actual(MODE, direction=DIRECTION, mode_cfg=mode_cfg, paths=paths)
if isinstance(SERIES, int):
    series_name = actual.columns[SERIES]
else:
    series_name = SERIES
print(f"Series: {series_name}  |  test window: {test_start.date()} → {test_end.date()}")

forecasts: dict[str, "tuple[pd.Series, pd.Series, pd.Series]"] = {}
for m in MODELS:
    out_dir = output_dir(MODE, m, direction=DIRECTION, paths=paths)
    prefix = f"{MODE}_{m}_{WINDOW}" + (f"_{DIRECTION}" if DIRECTION != "all" else "")
    try:
        res = load_forecast_triplet(out_dir, prefix)
    except FileNotFoundError:
        print(f"  (skip {m}: no forecasts at {out_dir / (prefix + '_mu.csv')})")
        continue
    if series_name not in res.mu.columns:
        print(f"  (skip {m}: column {series_name!r} missing — has {len(res.mu.columns)} cols)")
        continue
    forecasts[m] = (res.mu[series_name], res.lower[series_name], res.upper[series_name])
print(f"Loaded {len(forecasts)} models: {list(forecasts)}")

# Truth + history slice
history_start = test_start - pd.Timedelta(days=HISTORY_TAIL_DAYS)
hist = actual.loc[(actual.index >= history_start) & (actual.index < test_start), series_name]
truth = actual.loc[(actual.index >= test_start) & (actual.index <= test_end), series_name]

# %% Figure 1 — Overlay (μ only, no PI bands; readable with many models)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(hist.index, hist.values, color="black", lw=1.0, label="Observed (history)")
ax.plot(truth.index, truth.values, color="black", lw=1.5, label="Observed (test)")
ax.axvline(test_start, color="red", linestyle="--", lw=1)

cmap = plt.colormaps.get_cmap("tab10")
for i, (m, (mu, _, _)) in enumerate(forecasts.items()):
    ax.plot(mu.index, mu.values, lw=1.2, alpha=0.85, color=cmap(i % 10), label=m)
ax.set_title(f"{MODE}/{DIRECTION}/{WINDOW} — series {series_name} — μ overlay")
ax.set_xlabel("Date")
ax.set_ylabel("Ridership")
ax.legend(ncol=3, fontsize=9, loc="best")
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% Figure 2 — Small multiples (one panel per model with PI band)
n = len(forecasts)
ncols = 3
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), sharex=True)
axes = axes.flatten() if n > 1 else [axes]

for ax, (m, (mu, lo, hi)) in zip(axes, forecasts.items()):
    ax.plot(hist.index, hist.values, color="black", lw=0.8, alpha=0.6)
    ax.plot(truth.index, truth.values, color="black", lw=1.4, label="truth")
    ax.plot(mu.index, mu.values, color="C1", lw=1.4, label="μ")
    ax.fill_between(mu.index, lo.values, hi.values, alpha=0.2, color="C1", label="90% PI")
    ax.axvline(test_start, color="red", linestyle="--", lw=0.8)
    # Lightweight per-panel WMAPE so calibration vs accuracy is visible.
    aligned = truth.reindex(mu.index)
    wmape = float((mu - aligned).abs().sum() / aligned.abs().sum()) if aligned.abs().sum() > 0 else float("nan")
    ax.set_title(f"{m}  (WMAPE={wmape:.3f})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.tick_params(labelrotation=20)

# Hide leftover empty axes if n isn't a multiple of ncols.
for ax in axes[len(forecasts):]:
    ax.set_visible(False)

axes[0].legend(loc="best", fontsize=8)
fig.suptitle(f"{MODE}/{DIRECTION}/{WINDOW} — series {series_name}  (90% PI shaded)",
             y=1.0, fontsize=12)
fig.tight_layout()
plt.show()

# %% Save figures (optional — uncomment to write to outputs/figures/)
# fig_dir = Path(paths["output_root"]).parent / "outputs" / "figures"
# fig_dir.mkdir(parents=True, exist_ok=True)
# fig.savefig(fig_dir / f"compare_{MODE}_{DIRECTION}_{WINDOW}_{series_name}.png", dpi=150, bbox_inches="tight")
# print(f"saved to {fig_dir}")
