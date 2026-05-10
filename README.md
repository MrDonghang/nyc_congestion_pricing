# NYC Congestion Pricing

Counterfactual evaluation of New York City's Central Business District Tolling Program (the **Congestion Relief Zone**, effective **2025-01-05**) on transit and mobility patterns.

The pipeline:

1. Builds **daily / weekly ridership matrices** for bus, subway OD, Citibike OD, and Replica OD from the raw open-data feeds.
2. Runs **forecasters** on pre-policy data and produces "no-policy" counterfactual forecasts with prediction intervals. Headline models are the time-series foundation models **Chronos** and **TimesFM** (zero-shot); ARIMA, Prophet, DeepAR, PCN, BSTS, NHITS, and TFT are also wired up as baselines / comparisons.
3. **Calibrates** the foundation-model outputs with per-quantile regression on the val window, then re-applies the calibration to the test horizon.
4. Computes **per-unit treatment effects** (ATT) by comparing actuals to (calibrated) counterfactuals.
5. Maps effects to **census-tract demographics** through spatial regression and ML to identify who's most affected.

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

forecaster = build_forecaster(load_model("chronos"))       # or timesfm / pcn / deepar / ...
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

# 2. Train + forecast (val tunes / calibrates; test is the counterfactual horizon)
#    --window accepts val / validation / test  (val and validation are aliases)
#    Run val and test for the same model so step 3 has both windows to work with.
python -m scripts.train_forecast --mode bus      --model chronos --window val
python -m scripts.train_forecast --mode bus      --model chronos --window test
python -m scripts.train_forecast --mode subway   --model timesfm --window test --direction O
python -m scripts.train_forecast --mode citibike --model prophet --window val  --direction D

# 3. Calibrate foundation-model forecasts (quantile regression on val → applied to test)
#    Writes a new model named "<base>_qrcal" alongside the original outputs.
python -m scripts.calibrate_forecast --mode bus      --base-model chronos --insample-val
python -m scripts.calibrate_forecast --mode subway   --base-model timesfm --direction O --insample-val
python -m scripts.calibrate_forecast --mode citibike --base-model chronos --direction D --insample-val
# Replica is intentionally not calibrated (weekly W-SAT cadence + small panel).

# 4. Compare model accuracy
python -m scripts.compare_models  --mode bus --window test

# 5. Counterfactual treatment effects (use the calibrated model for foundation models)
python -m scripts.compute_effects --mode bus --model chronos_qrcal --window test

# 6. Geospatial + causal regression
python -m scripts.geospatial_analysis --mode bus --model chronos_qrcal --window test
```

## Repository layout

```
nyc_congestion_pricing/
├── configs/
│   ├── paths.yaml                # data/output roots
│   ├── modes/{bus,subway,citibike,replica}.yaml      # per-mode windows + preprocessing
│   └── models/{arima,prophet,deepar,pcn,chronos,timesfm,nhits,tft,bsts}.yaml
│
├── nyc_cp/                       # the package
│   ├── config.py                 # config loaders + path resolution + get_window
│   ├── calibration.py            # per-quantile residual regression on val
│   ├── utils/                    # seed, normalize, logging
│   ├── data/                     # raw → matrix processors + actual-CSV loader
│   ├── models/                   # BaseForecaster + all forecasters
│   ├── tuning/                   # Optuna search helpers
│   ├── evaluation/               # metrics + plot helpers
│   └── analysis/                 # ATT pipeline + spatial + causal regression
│
├── scripts/                      # thin CLIs (process_data, train_forecast,
│                                 #   calibrate_forecast, compare_models,
│                                 #   compute_effects, geospatial_analysis)
├── notebooks/                    # example workflows + calibration diagnostics
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

Forecast windows are defined per-mode (see `configs/modes/<mode>.yaml`) because citibike and replica do not share the year-displaced val window that bus/subway use. The CLI flag `--window` accepts `val` (or `validation` as an alias) and `test`; output filenames use the short form `_val_` / `_test_`.

| Mode      | val                                     | test (counterfactual)                   |
|-----------|-----------------------------------------|-----------------------------------------|
| bus       | 2024-01-05 → 2024-04-30                 | 2025-01-05 → 2025-04-30                 |
| subway    | 2024-01-05 → 2024-04-30                 | 2025-01-05 → 2025-04-30                 |
| citibike  | 2024-10-01 → 2025-01-04                 | 2025-01-05 → 2025-04-30                 |
| replica   | 2024-09-28 → 2024-12-28 (W-SAT)         | 2025-01-04 → 2025-04-26 (W-SAT)         |

## Models

All forecasters implement `nyc_cp.models.base.BaseForecaster` (`fit(history) → predict(start, end)`) and are dispatched by the `type:` key in `configs/models/<model>.yaml`.

**Headline (foundation models, zero-shot):**

| Model    | Library                                 | Notes                                                          |
|----------|-----------------------------------------|----------------------------------------------------------------|
| Chronos  | `chronos-forecasting`                   | Default headline. Bfloat16. Quantile-regression calibrated.    |
| TimesFM  | `timesfm`                               | Second headline. Quantile-regression calibrated.               |

**Baselines / comparisons:**

| Model    | Library                              | Notes                                                          |
|----------|--------------------------------------|----------------------------------------------------------------|
| ARIMA    | `pmdarima` + `statsmodels`           | One model per series; auto-order with fallback ARIMA(1,1,1).   |
| Prophet  | `gluonts.ext.prophet`                | 100 sample paths per series.                                   |
| DeepAR   | `gluonts.torch.DeepAREstimator`      | Optional Optuna search over context_length / layers / dropout. |
| PCN      | Custom PyTorch (`nyc_cp.models.pcn`) | `unidirectional` (default) or `bidirectional` variant.         |
| NHITS    | `neuralforecast`                     | Hierarchical interpolation; configured in `nhits.yaml`.        |
| TFT      | `neuralforecast`                     | Temporal Fusion Transformer.                                   |
| BSTS     | Custom (`nyc_cp.models.bsts`)        | Bayesian structural time series.                               |

Switch PCN variants with `variant: bidirectional` in `configs/models/pcn.yaml`.

## Calibration

`scripts/calibrate_forecast.py` reads the val + test forecast triplets for a base model, fits per-quantile residual regressions on val (lower / median / upper), and applies them to test. The calibrated triplet is written under a new model name `<base>_<suffix>` (default suffix `qrcal`):

```bash
python -m scripts.calibrate_forecast \
  --mode subway --base-model chronos --direction O \
  --coverage 0.9 --insample-val
```

Key flags: `--coverage` (target PI level, default 0.9), `--alpha` (L1 reg on the quantile regressors), `--insample-val` (apply the val-fitted calibrator back to val, no CV) vs. `--val-kfold K` (date-stratified out-of-sample val output; mutually exclusive). After calibration runs, downstream `compute_effects` / `geospatial_analysis` should be invoked with `--model chronos_qrcal` (or `timesfm_qrcal`) instead of the raw base model. **Replica is intentionally not calibrated** (weekly W-SAT cadence + small panel).

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

Calibrated outputs live alongside the originals under `<output_root>/<mode>/<base>_qrcal/[<direction>/]` with the same `_mu / _lower / _upper` triplet naming (no `_evaluation.csv` from the calibration step itself).

ATT outputs land under `.../effects/`; spatial-regression outputs under `.../causal/`.

## Tests

```bash
pytest tests/
```
