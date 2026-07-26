"""Self-contained demo: counterfactual forecasting + treatment effects.

Runs the core pipeline end-to-end on the real processed MTA bus panel shipped
in ``demo_data/`` (246 routes, 2022-01-07 → 2025-04-30, the same panel used in
the paper) — no external data, no config edits, no GPU required:

1. load the daily ridership panel,
2. forecast the post-policy window (2025-01-05 → 2025-04-30) with Chronos-2
   zero-shot, using only pre-policy history,
3. score forecast accuracy on a held-out pre-policy window (2024-01-05 →
   2024-04-30, no policy contamination),
4. compute per-route and overall treatment effects (ATT) by comparing
   post-policy actuals to the counterfactual, and
5. save forecast CSVs, an effects summary, and a per-route plot under
   ``demo_output/``.

The demo uses the (uncalibrated) Chronos-2 robustness model because its
weights are a small download; the paper's main specification (HQC-TimesFM)
is run via the CLIs documented in README §4.

Run from the repo root (cells delimited by ``# %%`` also open as a notebook):

    python notebooks/00_demo.py

First run downloads the Chronos-2 weights (~460 MB) from Hugging Face.
"""

# %% Setup
import time
from pathlib import Path

import pandas as pd

from nyc_cp.analysis.effects import build_long_df, compute_effects, summarize_by_unit, summarize_overall
from nyc_cp.evaluation import evaluate_per_series, plot_forecast
from nyc_cp.models import build_forecaster
from nyc_cp.utils import set_seed

t0 = time.time()
set_seed(42)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "demo_data" / "bus_data_2022_2025_daily_final.csv"
OUT = REPO / "demo_output"
OUT.mkdir(exist_ok=True)

POLICY_START = pd.Timestamp("2025-01-05")
TEST_END = pd.Timestamp("2025-04-30")
VAL_START, VAL_END = pd.Timestamp("2024-01-05"), pd.Timestamp("2024-04-30")

MODEL_CFG = {
    "type": "chronos",
    "model_id": "amazon/chronos-2",
    "device": "cpu",        # demo is CPU-only so it runs on any machine
    "dtype": "float32",
    "batch_size": 32,
    "coverage_level": 0.9,
}

# %% 1. Load the daily ridership panel
actual = pd.read_csv(DATA, index_col=0, parse_dates=True)
actual.index.name = "date"
print(f"Panel: {actual.shape[0]} days x {actual.shape[1]} routes "
      f"({actual.index[0].date()} → {actual.index[-1].date()})")

# %% 2. Zero-shot counterfactual forecast of the post-policy window
history = actual.loc[actual.index < POLICY_START]
horizon = pd.date_range(POLICY_START, TEST_END, freq="D")

forecaster = build_forecaster(dict(MODEL_CFG))
result = forecaster.fit_predict(
    history,
    start=POLICY_START,
    end=TEST_END,
    train_end=history.index[-1],
    prediction_length=len(horizon),
)
print(f"Counterfactual forecast: {result.mu.shape[0]} days x {result.mu.shape[1]} routes")

# %% 3. Forecast accuracy on a held-out pre-policy window (no policy in val)
val_history = actual.loc[actual.index < VAL_START]
val_horizon = pd.date_range(VAL_START, VAL_END, freq="D")

val_forecaster = build_forecaster(dict(MODEL_CFG))
val_result = val_forecaster.fit_predict(
    val_history,
    start=VAL_START,
    end=VAL_END,
    train_end=val_history.index[-1],
    prediction_length=len(val_horizon),
)
val_metrics = evaluate_per_series(
    actual.loc[VAL_START:VAL_END], val_result.mu, val_result.lower, val_result.upper
)
print(f"\nValidation accuracy (pre-policy window, medians across {actual.shape[1]} routes):")
print(val_metrics.median().round(3).to_string())

# %% 4. Treatment effects: post-policy actuals vs. counterfactual
df_long = build_long_df(actual.loc[POLICY_START:TEST_END], result.mu, result.lower, result.upper)
df_eff = compute_effects(df_long)
unit_summary = summarize_by_unit(df_eff)
overall = summarize_overall(unit_summary)

print("\nPer-route cumulative effects (first 5):")
print(unit_summary.head().round(2).to_string())
print("\nOverall ATT summary:")
print(overall.round(3).to_string())

# %% 5. Save outputs
result.mu.to_csv(OUT / "demo_cf_mu.csv")
result.lower.to_csv(OUT / "demo_cf_lower.csv")
result.upper.to_csv(OUT / "demo_cf_upper.csv")
val_metrics.to_csv(OUT / "demo_val_metrics.csv")
unit_summary.to_csv(OUT / "demo_effects_by_route.csv")
overall.to_csv(OUT / "demo_effects_overall.csv", index=False)

busiest = int(actual.mean().to_numpy().argmax())   # plot the busiest route
route_name = str(actual.columns[busiest])
fig, _ = plot_forecast(actual.loc["2024-07-01":], result.mu, result.lower, result.upper,
                       column=busiest, title_prefix="Demo:")
fig.savefig(OUT / f"demo_forecast_{route_name.replace('/', '-')}.png", dpi=150, bbox_inches="tight")

print(f"\nOutputs written to {OUT}/")
print(f"Total wall time: {time.time() - t0:.0f} s")
