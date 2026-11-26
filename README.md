# NYC Congestion Pricing

Counterfactual evaluation of New York City's Central Business District Tolling Program (the **Congestion Relief Zone**, effective **2025-01-05**) on transit and mobility patterns.

Code accompanying the manuscript *"Public transit gains and spatially uneven travel demand changes after NYC congestion pricing"*.

The analysis:

1. Builds **daily / weekly ridership matrices** for bus, subway and Replica from the raw open-data feeds. The main transit panels exclude student fare classes (`bus_ns`, `subway_ns22`; see [Modes & windows](#modes--windows)).
2. Produces zero-shot **"no-policy" counterfactual forecasts** with time-series foundation models, per station / route, from a dense grid of **rolling forecast origins** (one every 3 days, 116-day horizon). The main model is **Chronos-Bolt**; **TimesFM-2.0** is the robustness check. Both were released before the policy took effect.
3. **Calibrates** each forecast with a **quantile regression** of past forecast errors on holiday, day of week, lead time and month, using only forecasts whose target dates precede the forecast origin. The median gives the calibrated counterfactual; the models' own 10 % / 90 % quantiles, calibrated at the 5 % / 95 % level, give the daily band.
4. Estimates **daily and cumulative effects** as deviations of observed ridership from the calibrated counterfactual, and judges them against **pseudo-policy dates** in 2024, when no policy was in place (in-time placebo).
5. Relates station-, route- and tract-level effects to **location and census-tract demographics**.

The scripts for steps 2–5 are in [`runs/revision/`](runs/revision/README.md), which documents the specification and the order in which to run them. The package `nyc_cp` and the CLIs in `scripts/` provide the data processing, the forecasters (including ARIMA, Prophet, DeepAR, PCN, BSTS, NHITS and TFT as comparisons) and the earlier single-window pipeline with hierarchical quantile calibration (HQC), described in §4 below.

---

## 1. System requirements

**Operating systems.** Any OS with Python ≥ 3.10 (Linux, macOS, Windows). The code is pure Python; no compilation is required.

**Tested on.**

* macOS 15.6.1 (Apple M2, 16 GB RAM), Python 3.11.0 — installation, full test suite, and demo verified end-to-end.

**Software dependencies.** Declared in [pyproject.toml](pyproject.toml); key packages with the exact versions the demo and tests were verified against:

| Package | Version | | Package | Version |
|---|---|---|---|---|
| numpy | 2.4.6 | | chronos-forecasting | 2.3.1 |
| pandas | 2.3.3 | | timesfm | 1.3.0 |
| scipy | 1.15.3 | | gluonts | 0.16.3 |
| scikit-learn | 1.9.0 | | neuralforecast | 3.2.0 |
| statsmodels | 0.14.6 | | pmdarima | 2.1.1 |
| torch | 2.13.0 | | prophet | 1.3.0 |
| lightning | 2.4.0 | | optuna | 4.9.0 |
| geopandas | 1.1.4 | | libpysal / spreg | 4.14.1 / 1.9.0 |
| xgboost | 3.2.0 | | lightgbm | 4.7.0 |

The complete frozen environment (122 pinned packages) is in [requirements-freeze.txt](requirements-freeze.txt).

**Hardware.** No non-standard hardware is required. The demo and test suite run CPU-only on a laptop (16 GB RAM is ample). A CUDA GPU is optional and only accelerates the full-scale foundation-model and neural-baseline runs; the model configs in `configs/models/*.yaml` default to `device: cuda` for full runs — set `device: cpu` (as the demo does) when no GPU is available.

## 2. Installation guide

```bash
git clone https://github.com/MrDonghang/nyc_congestion_pricing.git
cd nyc_congestion_pricing
python -m venv .venv && source .venv/bin/activate   # recommended
pip install -e .            # installs nyc_cp + console scripts
```

Optional: `pip install -e ".[dev]"` adds pytest, ruff, ipykernel. To reproduce the exact tested environment instead, use `pip install -r requirements-freeze.txt`.

**Typical install time:** ~3 minutes on a normal desktop computer (measured: 157 s in a fresh Python 3.11 virtual environment on an Apple M2 MacBook with a broadband connection; the bulk is downloading PyTorch and the forecasting libraries).

Verify the installation with the test suite (a few seconds):

```bash
pip install pytest && pytest tests/
```

## 3. Demo

A small real dataset ships with the repo at [demo_data/bus_data_2022_2025_daily_final.csv](demo_data/bus_data_2022_2025_daily_final.csv) (2 MB): the processed MTA bus panel used in the paper — 246 routes × 1,210 days (2022-01-07 → 2025-04-30), aggregated from the public MTA Bus Hourly Ridership feed. The demo runs the paper's bus counterfactual analysis on it end-to-end, with no external data, config edits, or GPU.

**Run** (from the repo root):

```bash
python notebooks/00_demo.py
```

The demo (1) zero-shot forecasts the post-policy window (2025-01-05 → 2025-04-30) with Chronos-2 using only pre-policy history, (2) scores forecast accuracy on a held-out pre-policy validation window (2024-01-05 → 2024-04-30), and (3) computes per-route and overall policy effects (deviations from the no-policy counterfactual). The demo is a quick, CPU-only check of the installation and uses Chronos-2 with a single forecast window; it is not the specification used for the results in the manuscript (see [`runs/revision/`](runs/revision/README.md)).

**Expected output.** Console output reports per-route validation accuracy (median R² ≈ 0.80, 90%-PI coverage ≈ 0.92 across 246 routes) and the policy-effect summary (overall bus ridership ≈ +2% versus the no-policy counterfactual, outside the 90% prediction interval). Files land in `demo_output/`:

```
demo_cf_mu.csv / demo_cf_lower.csv / demo_cf_upper.csv   # counterfactual triplet
demo_val_metrics.csv                                     # per-route accuracy metrics
demo_effects_by_route.csv / demo_effects_overall.csv     # policy-effect summaries
demo_forecast_<route>.png                                # forecast plot, busiest route
```

**Expected run time** on a normal desktop computer (CPU-only): **~2 minutes on the first run** (dominated by a one-time ~460 MB download of the Chronos-2 weights from Hugging Face), **~20 seconds on subsequent runs** (measured: 19 s on an Apple M2 MacBook).

## 4. Instructions for use

### Running on your own data

The CLIs below run the single-window pipeline (one validation and one test window per mode) with HQC calibration. The results in the manuscript come from the rolling-origin pipeline in [`runs/revision/`](runs/revision/README.md); the CLIs remain useful for building the data panels (step 1) and for the baseline-model comparisons.

Point `configs/paths.yaml` (or a gitignored machine-local override `configs/paths.local.yaml` — only changed keys required) at your data roots, then run the CLIs below. Raw input feeds are the public MTA sources and the Replica subscription data listed in the paper's Data Availability statement.

```bash
# 1. Raw → daily/weekly matrices  (skip if you already have data_processed/)
python -m scripts.process_data  --mode bus      --hourly-2020-2024 ...  --hourly-2025 ...
python -m scripts.process_data  --mode subway   --od-estimate ...        --hourly ...
python -m scripts.process_data  --mode replica  --raw-dir ...            --geo-csv ...

# 2. Train + forecast (val tunes / calibrates; test is the counterfactual horizon)
#    --window accepts val / validation / test  (val and validation are aliases)
#    Run val and test for the same model so step 3 has both windows to work with.
python -m scripts.train_forecast --mode bus      --model timesfm --window val
python -m scripts.train_forecast --mode bus      --model timesfm --window test
python -m scripts.train_forecast --mode subway   --model timesfm --window val  --direction O
python -m scripts.train_forecast --mode subway   --model timesfm --window test --direction O

# 3. Calibrate with HQC (--per-unit-intercept).
#    Writes a new model named "<base>_qrcal_intercept" alongside the original outputs.
#    NOTE: running calibrate_forecast WITHOUT --per-unit-intercept gives the global
#    QR baseline ("<base>_qrcal"), a comparison variant.
python -m scripts.calibrate_forecast --mode bus    --base-model timesfm --per-unit-intercept --insample-val
python -m scripts.calibrate_forecast --mode subway --base-model timesfm --per-unit-intercept --direction O --insample-val

# 4. Compare model accuracy
python -m scripts.compare_models  --mode bus --window test

# 5. Policy effects (observed vs. counterfactual) — use the HQC-calibrated model
python -m scripts.compute_effects --mode bus --model timesfm_qrcal_intercept --window test

# 6. Geospatial + demographic regression
python -m scripts.geospatial_analysis --mode bus --model timesfm_qrcal_intercept --window test
```

### Reproducing the results in the manuscript

1. **Data.** Download the raw feeds listed in the paper's Data Availability statement (MTA Bus Hourly Ridership, MTA Subway Hourly Ridership, Replica weekly OD) and build the processed panels: step 1 above, then the student-fare-excluded panels with `runs/revision/fetch_bus_nostudent.py`, `finalize_bus_nostudent.py`, `build_subway_nostudent.py` and `build_subway_nostudent22.py`.
2. **Counterfactuals, calibration, inference and figures.** Follow the reproduction table in [`runs/revision/README.md`](runs/revision/README.md): rolling-origin forecasts for `chronos_bolt` (main) and `timesfm` (robustness), quantile-regression calibration, pseudo-policy-date inference, and the figure scripts.
3. **Baseline-model comparisons.** Run steps 2–5 of the CLIs above for the comparison models (`arima`, `prophet`, `deepar`, `pcn`, `bsts`, `nhits`, `tft`).
4. **Determinism.** All stochastic components are seeded (`nyc_cp.utils.set_seed(42)`); the foundation-model forecasts are zero-shot (no training), so results are reproducible up to library/hardware numerics.

Forecast windows per mode for the single-window CLIs (defined in `configs/modes/<mode>.yaml`); the rolling-origin pipeline sets its own origins:

| Mode      | val                                     | test (counterfactual)                   |
|-----------|-----------------------------------------|-----------------------------------------|
| bus       | 2024-01-05 → 2024-04-30                 | 2025-01-05 → 2025-04-30                 |
| subway    | 2024-01-05 → 2024-04-30                 | 2025-01-05 → 2025-04-30                 |
| replica   | 2024-09-28 → 2024-12-28 (W-SAT)         | 2025-01-04 → 2025-04-26 (W-SAT)         |

---

## Quick start (library API)

```python
import pandas as pd
from nyc_cp.config import get_window, load_model
from nyc_cp.data import load_actual
from nyc_cp.models import build_forecaster

actual = load_actual("bus")                                # date × route
window = get_window("bus", "test")                         # 2025-01-05 → 2025-04-30
history = actual.loc[actual.index <= window.train_end]

forecaster = build_forecaster(load_model("timesfm"))       # or chronos / pcn / deepar / ...
result = forecaster.fit_predict(
    history,
    start=pd.Timestamp(window.test_start),
    end=pd.Timestamp(window.test_end),
    train_end=pd.Timestamp(window.train_end),
    prediction_length=len(pd.date_range(window.test_start, window.test_end, freq="D")),
)
result.mu, result.lower, result.upper      # three aligned date × route DataFrames
```

## Repository layout

```
nyc_congestion_pricing/
├── configs/
│   ├── paths.yaml                # data/output roots
│   ├── modes/{bus,bus_ns,subway,subway_ns,subway_ns22,replica}.yaml   # per-mode windows + preprocessing
│   └── models/{arima,prophet,deepar,pcn,chronos,chronos_bolt,timesfm,nhits,tft,bsts}.yaml
│
├── nyc_cp/                       # the package
│   ├── config.py                 # config loaders + path resolution + get_window
│   ├── calibration.py            # HQC + QR calibration (fit on val, apply to test)
│   ├── utils/                    # seed, normalize, logging
│   ├── data/                     # raw → matrix processors + actual-CSV loader
│   ├── models/                   # BaseForecaster + all forecasters
│   ├── tuning/                   # Optuna search helpers
│   ├── evaluation/               # metrics + plot helpers
│   └── analysis/                 # policy-effect pipeline + spatial/demographic regression
│
├── scripts/                      # thin CLIs (process_data, train_forecast,
│                                 #   calibrate_forecast, compare_models,
│                                 #   compute_effects, geospatial_analysis)
├── runs/revision/                # rolling-origin pipeline behind the manuscript's results
│                                 #   (forecast replay, calibration, placebo inference, figures)
├── notebooks/                    # 00_demo.py + example workflows + diagnostics
├── demo_data/                    # real MTA bus panel for the demo (see §3)
├── geo_data/                     # NYC shapefiles + CRZ polygon
├── tests/                        # smoke tests for metrics + effects
├── requirements-freeze.txt       # exact tested environment (pip freeze)
├── LICENSE                       # MIT
└── pyproject.toml
```

## Modes & windows

| Mode      | Frequency | Directions  | Source                                          |
|-----------|-----------|-------------|-------------------------------------------------|
| bus       | daily     | —           | MTA Bus Hourly Ridership                        |
| subway    | daily     | O / D       | MTA Hourly + OD Estimate (per-month patterns)   |
| bus_ns    | daily     | —           | Bus, student fare classes excluded (main bus panel) |
| subway_ns | daily     | O           | Subway entries, student fare classes excluded, from 2023 |
| subway_ns22 | daily   | O           | As `subway_ns`, history from 2022 (main subway panel) |
| replica   | weekly    | O / D       | Replica weekly trip-count zips                  |

Forecast windows are defined per-mode (see `configs/modes/<mode>.yaml`) because replica does not share the year-displaced val window that bus/subway use. The CLI flag `--window` accepts `val` (or `validation` as an alias) and `test`; output filenames use the short form `_val_` / `_test_`.

## Models

All forecasters implement `nyc_cp.models.base.BaseForecaster` (`fit(history) → predict(start, end)`) and are dispatched by the `type:` key in `configs/models/<model>.yaml`.

**Foundation models (zero-shot):**

| Model        | Config                | Library               | Notes                                                              |
|--------------|-----------------------|-----------------------|--------------------------------------------------------------------|
| Chronos-Bolt | `chronos_bolt.yaml`   | `chronos-forecasting` | **Main model in the manuscript.** Released 2024-11-26, before the policy. |
| TimesFM-2.0  | `timesfm.yaml`        | `timesfm`             | Robustness check. Released 2024-12-30, before the policy.          |
| Chronos-2    | `chronos.yaml`        | `chronos-forecasting` | Used by the demo only: released after the policy took effect, so not used for the results. |

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

The manuscript's results use the quantile-regression calibration on rolling forecast origins implemented in `runs/revision/qr_native.py` (see [`runs/revision/README.md`](runs/revision/README.md)).

The single-window CLIs use **HQC**, implemented in `nyc_cp/calibration.py` and driven by `scripts/calibrate_forecast.py`. HQC (hierarchical quantile calibration: a per-unit residual intercept plus a pooled quantile regression) is selected with `--per-unit-intercept`:

```bash
python -m scripts.calibrate_forecast \
  --mode subway --base-model timesfm --direction O \
  --per-unit-intercept --insample-val
```

This writes a calibrated model named `<base>_qrcal_intercept`, which downstream `compute_effects` / `geospatial_analysis` take via `--model timesfm_qrcal_intercept`. Running without `--per-unit-intercept` gives the global-QR baseline (`<base>_qrcal`); `--per-unit` gives the fully per-unit variant (`<base>_qrcal_perunit`) — both are comparison variants.

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

Calibrated outputs live alongside the originals under `<output_root>/<mode>/<base>_qrcal_intercept/[<direction>/]` (HQC; `_qrcal` / `_qrcal_perunit` for the comparison calibrations) with the same `_mu / _lower / _upper` triplet naming (no `_evaluation.csv` from the calibration step itself).

Policy-effect outputs land under `.../effects/`; spatial-regression outputs under `.../causal/`.

## Tests

```bash
pytest tests/
```

14 tests covering the metrics and effects pipelines (a few seconds; measured 1.4 s on an Apple M2 MacBook).

## License

Released under the [MIT License](LICENSE).
