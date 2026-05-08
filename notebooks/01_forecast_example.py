"""Example: forecast bus ridership for the 2025 test window with PCN.

Open this file in Jupyter / VS Code as a notebook (cells delimited by ``# %%``)
or run it as a script. Either way it expects the package to be installed
(``pip install -e .``) and the actual data at the paths in
``configs/paths.yaml``.
"""

# %% Setup
import pandas as pd

from nyc_cp.config import get_window, load_model, output_dir
from nyc_cp.data import load_actual
from nyc_cp.evaluation import evaluate_per_series, plot_forecast
from nyc_cp.models import build_forecaster
from nyc_cp.utils import set_seed

set_seed(42)

# %% Load actual ridership and define the forecast window
actual = load_actual("bus")
window = get_window("bus", "test")

train_end = pd.Timestamp(window.train_end)
test_start = pd.Timestamp(window.test_start)
test_end = pd.Timestamp(window.test_end)
freq = "D"
prediction_length = len(pd.date_range(test_start, test_end, freq=freq))

history = actual.loc[actual.index <= train_end]
print(f"History: {history.shape}  |  horizon: {prediction_length} days")

# %% Build a forecaster and fit
model_cfg = load_model("pcn")
model_cfg.setdefault("freq", freq)
forecaster = build_forecaster(model_cfg)

result = forecaster.fit_predict(
    history,
    start=test_start,
    end=test_end,
    train_end=train_end,
    prediction_length=prediction_length,
    freq=freq,
)

# %% Save the triplet to disk in the same place train_forecast.py uses
out_dir = output_dir("bus", "pcn", direction="all")
prefix = "bus_pcn_test_demo"
result.save(out_dir, prefix)
print(f"Wrote {out_dir}/{prefix}_(mu|lower|upper).csv")

# %% Per-series metrics on the test window
truth = actual.loc[result.mu.index]
metrics = evaluate_per_series(truth, result.mu, result.lower, result.upper, coverage_level=result.coverage_level)
print(metrics.mean(numeric_only=True).to_string())

# %% Visualise one route
plot_forecast(actual, result.mu, result.lower, result.upper, column=0, title_prefix="Bus PCN")
