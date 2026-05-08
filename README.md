# NYC Congestion Pricing

Counterfactual evaluation of New York City's Central Business District Tolling Program (the **Congestion Relief Zone**, effective **2025-01-05**) on transit and mobility patterns.

The pipeline:

1. Builds **daily / weekly ridership matrices** for bus, subway OD, Citibike OD, and Replica OD from the raw open-data feeds.
2. Trains **forecasters** (ARIMA, Prophet, DeepAR, PCN) on pre-policy data and produces "no-policy" counterfactual forecasts with prediction intervals.
3. Computes **per-unit treatment effects** (ATT) by comparing actuals to counterfactuals.
4. Maps effects to **census-tract demographics** through spatial regression and ML to identify who's most affected.

## Installation

```bash
git clone <repo>
cd nyc_congestion_pricing
pip install -e .            # installs nyc_cp + console scripts
```

Optional: `pip install -e ".[dev]"` adds pytest, ruff, ipykernel.

If your data lives somewhere other than the defaults in `configs/paths.yaml`, drop a `configs/paths.local.yaml` (gitignored) with overrides — only the keys you change are required.

## Quick start

```python
import pandas as pd
from nyc_cp.config import get_window, load_model
from nyc_cp.data import load_actual
from nyc_cp.models import build_forecaster

actual = load_actual("bus")                                # date × route
window = get_window("bus", "test")                         # 2025-01-05 → 2025-04-30
history = actual.loc[actual.index <= window.train_end]

forecaster = build_forecaster(load_model("pcn"))
result = forecaster.fit_predict(
    history,
    start=pd.Timestamp(window.test_start),
    end=pd.Timestamp(window.test_end),
    train_end=pd.Timestamp(window.train_end),
    prediction_length=len(pd.date_range(window.test_start, window.test_end, freq="D")),
)
result.mu, result.lower, result.upper      # three aligned date × route DataFrames
```

## End-to-end workflow (CLIs)

```bash
# 1. Raw → daily/weekly matrices  (skip if you already have data_processed/)
python -m scripts.process_data  --mode bus      --hourly-2020-2024 ...  --hourly-2025 ...
python -m scripts.process_data  --mode subway   --od-estimate ...        --hourly ...
python -m scripts.process_data  --mode citibike --raw-dir ...            --census-shp ...
python -m scripts.process_data  --mode replica  --raw-dir ...            --geo-csv ...

# 2. Train + forecast (validation tunes; test is the counterfactual horizon)
python -m scripts.train_forecast --mode bus      --model pcn     --window test
python -m scripts.train_forecast --mode subway   --model deepar  --window test --direction O
python -m scripts.train_forecast --mode citibike --model prophet --window test --direction D

# 3. Compare model accuracy
python -m scripts.compare_models  --mode bus --window test

# 4. Counterfactual treatment effects
python -m scripts.compute_effects --mode bus --model pcn --window test

# 5. Geospatial + causal regression
python -m scripts.geospatial_analysis --mode bus --model pcn --window test
```

## Repository layout

```
nyc_congestion_pricing/
├── configs/
│   ├── paths.yaml                # data/output roots + forecast windows
│   ├── modes/{bus,subway,citibike,replica}.yaml
│   └── models/{arima,prophet,deepar,pcn}.yaml
│
├── nyc_cp/                       # the package
│   ├── config.py                 # config loaders + path resolution
│   ├── utils/                    # seed, normalize, logging
│   ├── data/                     # raw → matrix processors + actual-CSV loader
│   ├── models/                   # BaseForecaster + 4 forecasters
│   ├── evaluation/               # metrics + plot helpers
│   └── analysis/                 # ATT pipeline + spatial + causal regression
│
├── scripts/                      # thin CLIs (process_data, train_forecast, ...)
├── notebooks/                    # example workflows
├── geo_data/                     # NYC shapefiles + CRZ polygon
├── tests/                        # smoke tests for metrics + effects
└── pyproject.toml
```

## Modes & windows

| Mode      | Frequency | Directions  | Source                                          |
|-----------|-----------|-------------|-------------------------------------------------|
| bus       | daily     | —           | MTA Bus Hourly Ridership                        |
| subway    | daily     | O / D       | MTA Hourly + OD Estimate (per-month patterns)   |
| citibike  | daily     | O / D       | Citibike trip-data zips, joined to census tract |
| replica   | weekly    | O / D       | Replica weekly trip-count zips                  |

Forecast windows are defined per-mode (see `configs/modes/<mode>.yaml`) because citibike and replica do not share the year-displaced validation window that bus/subway use.

| Mode      | Validation                              | Test (counterfactual)                   |
|-----------|-----------------------------------------|-----------------------------------------|
| bus       | 2024-01-05 → 2024-04-30                 | 2025-01-05 → 2025-04-30                 |
| subway    | 2024-01-05 → 2024-04-30                 | 2025-01-05 → 2025-04-30                 |
| citibike  | 2024-10-01 → 2025-01-04                 | 2025-01-05 → 2025-04-30                 |
| replica   | 2024-09-28 → 2024-12-28 (W-SAT)         | 2025-01-04 → 2025-04-26 (W-SAT)         |

## Models

All four forecasters implement `nyc_cp.models.base.BaseForecaster` (`fit(history) → predict(start, end)`):

| Model    | Library                     | Notes                                                               |
|----------|-----------------------------|---------------------------------------------------------------------|
| ARIMA    | `pmdarima` + `statsmodels`  | One model per series; auto-order with fallback ARIMA(1,1,1).        |
| Prophet  | `gluonts.ext.prophet`       | 100 sample paths per series.                                        |
| DeepAR   | `gluonts.torch.DeepAREstimator` | Optional Optuna search over context_length / layers / dropout. |
| PCN      | Custom PyTorch (`nyc_cp.models.pcn`) | `unidirectional` (default) or `bidirectional` variant.       |

Switch PCN variants with `variant: bidirectional` in `configs/models/pcn.yaml`.

## Configuration

Three layers of config compose at runtime:

* `configs/paths.yaml` — where data lives, where outputs land, the train/test windows.
* `configs/modes/<mode>.yaml` — per-mode preprocessing (column drops, weekday filter, …).
* `configs/models/<model>.yaml` — per-model hyperparameters and (optionally) Optuna search spaces.

For machine-specific overrides, write `configs/paths.local.yaml` (gitignored) with only the keys you want to change. The override is deep-merged on top of `paths.yaml`.

## Outputs

Forecast triplets land at `<output_root>/<mode>/<model>/[<direction>/]`:

* `<mode>_<model>_<window>[_<direction>]_mu.csv`     — predicted mean
* `<mode>_<model>_<window>[_<direction>]_lower.csv`  — lower PI bound
* `<mode>_<model>_<window>[_<direction>]_upper.csv`  — upper PI bound
* `<mode>_<model>_<window>[_<direction>]_evaluation.csv` — per-series metrics

ATT outputs land under `.../effects/`; spatial-regression outputs under `.../causal/`.

## Tests

```bash
pytest tests/
```
