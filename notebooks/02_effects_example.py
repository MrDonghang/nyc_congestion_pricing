"""Example: counterfactual ATT pipeline for the bus test window.

Assumes ``scripts.train_forecast`` (or notebook ``01_forecast_example.py``)
has produced a forecast triplet under ``<output_root>/bus/pcn/``.
"""

# %% Setup
import pandas as pd

from nyc_cp.analysis import (
    build_long_df,
    compute_effects,
    load_forecast_triplet,
    summarize_by_unit,
    summarize_over_time,
    summarize_overall,
)
from nyc_cp.analysis.geospatial import plot_effects_over_time, plot_significance_calendar
from nyc_cp.config import get_window, load_paths, output_dir
from nyc_cp.data import load_actual

# %% Load actuals and forecast over the test window
paths = load_paths()
window = get_window("bus", "test")
test_start = pd.Timestamp(window.test_start)
test_end = pd.Timestamp(window.test_end)

actual = load_actual("bus", paths=paths)
actual = actual.loc[(actual.index >= test_start) & (actual.index <= test_end)]

result = load_forecast_triplet(output_dir("bus", "pcn"), prefix="bus_pcn_test")

common = list(actual.columns.intersection(result.mu.columns))
print(f"Aligned series: {len(common)}")

# %% Long-format effects
long = build_long_df(actual[common], result.mu[common], result.lower[common], result.upper[common], id_col="route_id", columns=common)
eff = compute_effects(long, id_col="route_id")
print(eff.head())

# %% Summaries
unit = summarize_by_unit(eff, id_col="route_id")
daily = summarize_over_time(eff)
overall = summarize_overall(unit, id_col="route_id")

print("\nOverall:")
print(overall.to_string(index=False))

# %% Time-series plot
plot_effects_over_time(daily, mode="daily_att", title_prefix="Bus")
plot_effects_over_time(daily, mode="cum_rel", title_prefix="Bus")

# %% Calendar of significant days
plot_significance_calendar(daily, year=2025, start_month=1, end_month=4)
