"""Visualise forecasts against ground truth on a single series.

Plots one figure with:
    * 2 months (configurable) of actual history *before* the forecast window
    * the actual ground truth across the forecast window
    * one or more model forecasts (μ + 90% PI band) overlaid on the window

Configurable via the constants below — works for any model whose triplet
lives at ``<output_root>/<mode>/<model>/[<dir>/]``. To compare calibration
before / after, set ``MODELS = ["chronos", "chronos_qrcal"]``; for a single
model just pass a 1-element list.

Open as a notebook in VS Code / Jupyter (cells delimited by ``# %%``).
"""

# %% Imports + config
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from nyc_cp.analysis.effects import load_forecast_triplet
from nyc_cp.config import get_window, load_paths, output_dir
from nyc_cp.data import load_actual

# ---- Edit these ------------------------------------------------------------
MODE = "bus"                                 # "bus" | "subway" | "citibike" | "replica"
DIRECTION = "all"                            # "all" for bus, "O" / "D" for OD modes
MODELS = ["timesfm_qrcal"]        # one or more model names — overlaid in the same axes
WINDOW = "test"                               # "val" | "test"
SERIES: str | int = 10                        # column name (str) or positional index (int)
HISTORY_MONTHS = 2                           # how much actual history to show before the forecast window
# ---------------------------------------------------------------------------

# %% Load actual ridership + forecast triplets
paths = load_paths()
suffix = f"_{DIRECTION}" if DIRECTION != "all" else ""
window = get_window(MODE, WINDOW)

actual = load_actual(MODE, direction=DIRECTION)
forecasts: dict[str, object] = {}
for model in MODELS:
    try:
        forecasts[model] = load_forecast_triplet(
            output_dir(MODE, model, direction=DIRECTION, paths=paths),
            f"{MODE}_{model}_{WINDOW}{suffix}",
        )
    except FileNotFoundError as e:
        print(f"[skip] {model}: {e}")

if not forecasts:
    raise RuntimeError("no forecast triplets loaded — check MODELS / WINDOW / DIRECTION")

ref = next(iter(forecasts.values()))
series_name = ref.mu.columns[SERIES] if isinstance(SERIES, int) else SERIES
if series_name not in actual.columns:
    raise KeyError(f"series {series_name!r} not in actual columns")

print(f"window : {window.test_start} -> {window.test_end}")
print(f"series : {series_name}")
print(f"loaded : {list(forecasts.keys())}")

# %% Plot — actual history + window + every model overlaid
forecast_start = pd.Timestamp(window.test_start)
forecast_end = pd.Timestamp(window.test_end)
history_start = forecast_start - pd.DateOffset(months=HISTORY_MONTHS)

actual_slice = actual.loc[(actual.index >= history_start) & (actual.index <= forecast_end), series_name]

fig, ax = plt.subplots(figsize=(12, 4.8))
ax.plot(actual_slice.index, actual_slice.values, color="black", lw=1.4, label="actual", zorder=3)
ax.axvline(forecast_start, color="grey", ls="--", lw=0.8, alpha=0.7)
ax.text(forecast_start, actual_slice.max(),
        "  forecast start", color="grey", fontsize=8, va="top")

palette = ["C0", "C1", "C2", "C3", "C4"]
for color, (model, fc) in zip(palette, forecasts.items()):
    idx = fc.mu.index
    ax.fill_between(idx, fc.lower[series_name], fc.upper[series_name],
                    color=color, alpha=0.18, label=f"{model} 90% PI")
    ax.plot(idx, fc.mu[series_name], color=color, lw=1.6, label=f"{model} μ")

ax.set_title(f"{MODE} {DIRECTION} | {WINDOW} | series = {series_name}")
ax.set_ylabel("ridership")
ax.tick_params(axis="x", rotation=30)
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=8, ncol=2)
fig.tight_layout()
plt.show()

# %% Per-model metrics on the chosen series
rows = []
for model, fc in forecasts.items():
    mu = fc.mu[series_name]
    lo = fc.lower[series_name]
    hi = fc.upper[series_name]
    a = actual[series_name].reindex(mu.index)
    err = a - mu
    rows.append({
        "model":    model,
        "rmse":     float((err ** 2).mean() ** 0.5),
        "ecr":      float(((a >= lo) & (a <= hi)).mean()),
        "pi_width": float((hi - lo).mean()),
    })
print(pd.DataFrame(rows).to_string(index=False))

# %%
