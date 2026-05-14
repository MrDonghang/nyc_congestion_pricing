"""Visualise forecasts against ground truth on a single series.

Plots validation and test figures separately. Each figure contains:
    * 2 months (configurable) of actual history before the forecast window
    * the actual ground truth across the forecast window
    * one or more model forecasts (mu + 90% PI band) overlaid on the window

Validation forecasts are shown in blue, and test forecasts are shown in orange.

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
MODE = "bus"                           # "bus" | "subway" | "citibike" | "replica"
DIRECTION = "all"                      # "all" for bus, "O" / "D" for OD modes
MODELS = ["timesfm_qrcal_intercept"]   # one or more model names
WINDOWS = ["val", "test"]              # generate both validation and test figures
SERIES: str | int = "BX12+"            # column name (str) or positional index (int)
HISTORY_MONTHS = 2                     # actual history before the forecast window
# ---------------------------------------------------------------------------

# %% Load actual ridership
paths = load_paths()
suffix = f"_{DIRECTION}" if DIRECTION != "all" else ""

actual = load_actual(MODE, direction=DIRECTION)

def get_common_ylim() -> tuple[float, float]:
    y_values = []

    for window_name in WINDOWS:
        window = get_window(MODE, window_name)
        forecast_start = pd.Timestamp(window.test_start)
        forecast_end = pd.Timestamp(window.test_end)
        history_start = forecast_start - pd.DateOffset(months=HISTORY_MONTHS)

        ref_forecasts = {}
        for model in MODELS:
            fc = load_forecast_triplet(
                output_dir(MODE, model, direction=DIRECTION, paths=paths),
                f"{MODE}_{model}_{window_name}{suffix}",
            )
            ref_forecasts[model] = fc

        ref = next(iter(ref_forecasts.values()))
        series_name = ref.mu.columns[SERIES] if isinstance(SERIES, int) else SERIES

        actual_slice = actual.loc[
            (actual.index >= history_start) & (actual.index <= forecast_end),
            series_name,
        ]

        y_values.extend(actual_slice.dropna().values)

        for fc in ref_forecasts.values():
            y_values.extend(fc.lower[series_name].dropna().values)
            y_values.extend(fc.upper[series_name].dropna().values)

    y_min = min(y_values)
    y_max = max(y_values)
    padding = 0.05 * (y_max - y_min)

    return y_min - padding, y_max + padding

# %% Helper function
def plot_forecast_window(window_name: str, color: str, common_ylim: tuple[float, float]) -> pd.DataFrame:
    window = get_window(MODE, window_name)

    forecasts: dict[str, object] = {}
    for model in MODELS:
        try:
            forecasts[model] = load_forecast_triplet(
                output_dir(MODE, model, direction=DIRECTION, paths=paths),
                f"{MODE}_{model}_{window_name}{suffix}",
            )
        except FileNotFoundError as e:
            print(f"[skip] {window_name} | {model}: {e}")

    if not forecasts:
        raise RuntimeError(
            f"no forecast triplets loaded for {window_name} — check MODELS / DIRECTION"
        )

    ref = next(iter(forecasts.values()))
    series_name = ref.mu.columns[SERIES] if isinstance(SERIES, int) else SERIES

    if series_name not in actual.columns:
        raise KeyError(f"series {series_name!r} not in actual columns")

    print(f"\nwindow : {window_name}")
    print(f"period : {window.test_start} -> {window.test_end}")
    print(f"series : {series_name}")
    print(f"loaded : {list(forecasts.keys())}")

    forecast_start = pd.Timestamp(window.test_start)
    forecast_end = pd.Timestamp(window.test_end)
    history_start = forecast_start - pd.DateOffset(months=HISTORY_MONTHS)

    actual_slice = actual.loc[
        (actual.index >= history_start) & (actual.index <= forecast_end),
        series_name,
    ]

    # Compute metrics first so ECR can be shown on the figure
    rows = []
    for model, fc in forecasts.items():
        mu = fc.mu[series_name]
        lo = fc.lower[series_name]
        hi = fc.upper[series_name]
        a = actual[series_name].reindex(mu.index)
        err = a - mu

        rows.append({
            "window":   window_name,
            "model":    model,
            "rmse":     float((err ** 2).mean() ** 0.5),
            "ecr":      float(((a >= lo) & (a <= hi)).mean()),
            "pi_width": float((hi - lo).mean()),
        })

    metrics = pd.DataFrame(rows)
    ecr_text = metrics["ecr"].iloc[0]

    fig, ax = plt.subplots(figsize=(10.0, 3.0), dpi=300)

    window_bg = {
        "val": "#EAF2FF",
        "test": "#FFF1E6",
    }

    #ax.axvspan(
    #    forecast_start,
    #    forecast_end,
    #    color=window_bg[window_name],
    #    alpha=0.9,
    #    lw=0,
    #    zorder=0,
    #)

    ax.plot(
        actual_slice.index,
        actual_slice.values,
        color="black",
        lw=1.2,
        label="Actual",
        zorder=3,
    )

    ax.axvline(
        forecast_start,
        color=color,
        ls="--",
        lw=1.5,
        alpha=0.45,
        zorder=4,
    )

    ax.text(
        forecast_start,
        actual_slice.max(),
        "  Forecast start",
        color="0.25",
        fontsize=10,
        va="top",
    )

    for i, (model, fc) in enumerate(forecasts.items()):
        idx = fc.mu.index

        ax.fill_between(
            idx,
            fc.lower[series_name],
            fc.upper[series_name],
            color=color,
            alpha=0.18,
            lw=0,
            label="90% Prediction Interval" if i == 0 else None,
            zorder=1,
        )

        ax.plot(
            idx,
            fc.mu[series_name],
            color=color,
            lw=1.4,
            label="Forecast" if i == 0 else None,
            zorder=2,
        )

    ax.set_ylabel("Daily ridership", fontsize=12)
    ax.set_ylim(common_ylim)

    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="both", length=3, width=0.8)

    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.legend(
        loc="lower left",
        fontsize=10,
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.0,
    )

    fig.tight_layout()
    plt.show()

    print(metrics.to_string(index=False))

    return metrics

# %% Plot validation and test figures
all_metrics = []
common_ylim = get_common_ylim()

for window_name, color in {
    "val": "C0",
    "test": "C1",
}.items():
    metrics = plot_forecast_window(window_name, color, common_ylim)
    all_metrics.append(metrics)

all_metrics = pd.concat(all_metrics, ignore_index=True)

# %% Combined metrics
print("\nCombined metrics:")
print(all_metrics.to_string(index=False))

# %%