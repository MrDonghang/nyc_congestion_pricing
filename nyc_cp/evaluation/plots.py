"""Single-series forecast plot used by notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast(
    actual: pd.DataFrame,
    mu: pd.DataFrame,
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    column: str | int = 0,
    coverage_level: float = 0.9,
    title_prefix: str = "",
    ylabel: str = "Ridership",
    figsize: tuple[float, float] = (14, 5),
):
    if isinstance(column, int):
        column = actual.columns[column]

    actual = actual.copy()
    actual.index = pd.to_datetime(actual.index)
    mu = mu.copy()
    mu.index = pd.to_datetime(mu.index)

    forecast_start = mu.index[0]
    history = actual.loc[actual.index < forecast_start, column]
    truth_future = actual.loc[actual.index.isin(mu.index), column]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(history.index, history.values, label="Observed (history)")
    ax.plot(truth_future.index, truth_future.values, label="Observed (future)")
    ax.plot(mu.index, mu[column].values, "--", label="Forecast")
    ax.fill_between(
        mu.index,
        lower[column].values,
        upper[column].values,
        alpha=0.25,
        label=f"{int(coverage_level * 100)}% PI",
    )
    ax.axvline(forecast_start, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix} Forecast — {column}".strip())
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax
