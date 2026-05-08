# NYC Congestion Pricing — Project Summary

A counterfactual evaluation of NYC's Central Business District Tolling Program (Congestion Relief Zone, "CRZ", effective **2025-01-05**) on multi-modal transit and mobility patterns. The pipeline (1) builds daily / weekly ridership matrices for several transit modes, (2) trains time-series forecasters on pre-policy data and produces "no-policy" counterfactual forecasts for 2025 with prediction intervals, (3) compares actuals vs. counterfactuals to estimate per-route / per-station / per-tract treatment effects (ATT), and (4) links those effects to census-tract demographics through spatial regression and ML.

This document is the **architectural overview**. For installation and usage, see `README.md`.

---

## 1. Modes & Data Sources

| Mode      | Source                                            | Granularity                       | Period             |
|-----------|---------------------------------------------------|------------------------------------|--------------------|
| Bus       | MTA Bus Hourly Ridership                          | daily × bus route (~397 routes)    | 2022 → 2025        |
| Subway    | MTA Subway Hourly + OD Estimate                   | daily × station × O/D              | 2023 → 2025        |
| Citibike  | Citibike trip-data (NYC + JC)                     | daily × census tract × O/D         | 2022 → 2025        |
| Replica   | Replica synthetic OD trip counts                  | weekly × census tract / PUMA × O/D | 2022 → 2025        |
| Geo / ACS | NYC shapefiles, ACS demographics, CRZ polygon     | tract / PUMA                       | —                  |

---

## 2. Repository Layout

```
nyc_congestion_pricing/
├── pyproject.toml                # package metadata + console scripts
├── README.md                     # installation + workflow
├── PROJECT_SUMMARY.md            # this file (architecture overview)
│
├── configs/                      # all run-time parameters
│   ├── paths.yaml                # data/output roots + forecast windows
│   ├── modes/{bus,subway,citibike,replica}.yaml
│   └── models/{arima,prophet,deepar,pcn}.yaml
│
├── nyc_cp/                       # the importable package
│   ├── config.py                 # config loaders + path resolution
│   ├── utils/                    # seed / normalize / logging
│   ├── data/                     # raw → matrix processors + actual-CSV loader
│   ├── models/                   # BaseForecaster + ARIMA / Prophet / DeepAR / PCN
│   ├── evaluation/               # metrics + plot helpers
│   └── analysis/                 # ATT pipeline + geospatial + causal regression
│
├── scripts/                      # thin CLIs over the package
│   ├── process_data.py
│   ├── train_forecast.py
│   ├── compare_models.py
│   ├── compute_effects.py
│   └── geospatial_analysis.py
│
├── notebooks/                    # example workflows (jupytext-style .py)
├── geo_data/                     # NYC shapefiles + CRZ polygon
└── tests/                        # pytest suite (metrics + effects)
```

---

## 3. Configuration Layers

A run is described by three configs that compose at start-up:

* **`configs/paths.yaml`** — single source of truth for `raw_root`, `data_root`, `output_root`, `geo_root`, plus the validation / test forecast windows. For machine-specific overrides, write `configs/paths.local.yaml` (gitignored) — the override is deep-merged on top.
* **`configs/modes/<mode>.yaml`** — per-mode preprocessing knobs (column drops, weekday-only filter, sparsity threshold).
* **`configs/models/<model>.yaml`** — per-model hyperparameters and (optionally) Optuna search spaces.

`nyc_cp.config` exposes `load_paths`, `load_mode`, `load_model`, `get_window`, `actual_csv`, `output_dir`. All path resolution is centralised here — no hard-coded paths anywhere else in the package.

---

## 4. Data Processing (`nyc_cp/data/`)

Each mode owns a single module exposing a `process(...)` function that turns raw open-data files into a daily / weekly wide-format CSV indexed by date.

| Module                        | Inputs                                              | Output                                            |
|-------------------------------|-----------------------------------------------------|---------------------------------------------------|
| `nyc_cp/data/bus.py`          | Two MTA hourly CSVs                                 | `bus_data_2022_2025_daily.csv`                    |
| `nyc_cp/data/subway.py`       | OD-estimate CSV + hourly CSVs (+ optional region map) | `subway_2023_2025_daily_O.csv` / `_D.csv` + per-month `(7, n, n)` patterns under `patterns/` |
| `nyc_cp/data/citibike.py`     | Monthly trip-data zips + tract shapefile            | `(T, O, D)` `.npz` + per-tract O / D CSVs         |
| `nyc_cp/data/replica.py`      | Weekly trip-count zips + GEOID→index CSV            | `(T, O, D)` `.npz` (optionally per mode)          |

The subway pipeline is split into three callable functions (`build_patterns`, `infer_daily_od`, `aggregate_od`); the per-month dow-only pattern shape `(7, n_stations, n_stations)` and pattern-file naming match the original `pattern_<year>_<month>.npz` schema exactly.

`nyc_cp/data/loaders.py` provides the read-side: `load_actual(mode, direction)` returns a wide-format DataFrame already filtered per the mode config.

CLI: `python -m scripts.process_data --mode <bus|subway|citibike|replica> ...`.

---

## 5. Forecasting (`nyc_cp/models/`)

A single abstraction — `BaseForecaster` — with `fit(history) → predict(start, end) -> ForecastResult(mu, lower, upper, coverage_level)`. All four models implement it.

| Model    | Library                                      | Notes                                                                                                              |
|----------|----------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| ARIMA    | `pmdarima.auto_arima` + `statsmodels`        | One model per series. Falls back to ARIMA(1,1,1) on failure.                                                       |
| Prophet  | `gluonts.ext.prophet.ProphetPredictor`       | 100 sample paths.                                                                                                   |
| DeepAR   | `gluonts.torch.model.deepar.DeepAREstimator` | Optuna-tunable (`context_length`, `num_layers`, `dropout_rate`).                                                    |
| PCN      | Custom PyTorch (`nyc_cp.models.pcn`)         | Per-series z-score; Gaussian NLL + boundary-continuity (`alpha`) + first-derivative-smoothness (`beta`) penalties. |

PCN ships in two flavours (`MultiLayerPCN` unidirectional with iterative refinement; `MultiLayerPCNBi` bidirectional with iterative inference). Switch via `variant: unidirectional | bidirectional` in `configs/models/pcn.yaml`.

A factory — `nyc_cp.models.build_forecaster(model_cfg)` — instantiates the right class. Imports are lazy: ARIMA does not pull in GluonTS, PCN does not pull in Prophet, etc.

CLI: `python -m scripts.train_forecast --mode <mode> --model <model> --window <validation|test> [--direction O|D]`. It saves the `mu / lower / upper` triplet plus a per-series evaluation table.

---

## 6. Evaluation (`nyc_cp/evaluation/`)

`evaluate_per_series(actual, mu, lower, upper, coverage_level)` returns one row per series with **RMSE / MAE / MAPE / WMAPE / SMAPE / R² / Coverage**. `evaluate_forecasts(...)` returns the column-wise mean. There is also a single-series `plot_forecast` helper.

CLI: `python -m scripts.compare_models --mode <mode> --window <window>` reads every model's saved triplet and tabulates accuracy across `arima / prophet / deepar / pcn`.

---

## 7. Counterfactual Effects (`nyc_cp/analysis/effects.py`)

The same long-format effects pipeline is parameterised by `id_col`, so one implementation serves bus routes, subway stations, and Citibike tracts.

* `build_long_df(actual, mu, lower, upper, id_col)` — melts the four DataFrames and joins on `(id_col, date)`.
* `compute_effects(long, id_col)` — adds `tau = actual - cf_mean`, PI-based effect bounds (`eff_lo / eff_hi`), `signif`, and cumulative absolute / relative effects per unit.
* `summarize_by_unit / summarize_over_time / summarize_overall` — aggregations consumed by downstream geospatial and causal stages.

CLI: `python -m scripts.compute_effects --mode <mode> --model <model> --window <window>` writes `_long.csv`, `_unit.csv`, `_daily.csv`, `_overall.csv` under `<output_root>/<mode>/<model>/effects/`.

---

## 8. Geospatial & Causal Analysis (`nyc_cp/analysis/`)

* **`geospatial.py`** — `classify_crz(gdf, crz_polygon, kind)` (three-class for routes, binary for tracts / stations); `map_units_to_tracts(...)` does the spatial join + tract aggregation; plus `plot_choropleth`, `plot_significance_calendar`, `plot_effects_over_time`. All work in `EPSG:2263` (NY State Plane).
* **`demographics.py`** — `derive_percentages(dem)` builds rate variables from the raw ACS counts (`pct_male`, `pct_white/black/asian/hispanic`, `pct_driving / public_transit / taxi / cycle / walk / wfh`, `pct_bachelor/master/phd`, `vacancy_rate`, `unemployment_rate`); `build(tracts_gdf, acs_geojson, year)` is the convenience wrapper that joins onto NYC tract polygons.
* **`causal.py`** — `compute_vif`, `stepwise_vif_filter`; `run_spatial_regression` returns OLS, ML_Lag, ML_Error from `spreg`; `run_tree_models` returns Random Forest / XGBoost / LightGBM with feature importances.

CLI: `python -m scripts.geospatial_analysis --mode <mode> --model <model> --window <window>` runs the full pipeline and saves `tract_effects.geojson`, `spatial_regression.txt`, `ml_models.csv` under `<output_root>/<mode>/<model>/causal/`.

---

## 9. End-to-End Flow

```
raw MTA / Citibike / Replica feeds
        │
        ▼  scripts/process_data.py             (nyc_cp/data/<mode>.py)
  daily ridership / weekly OD matrices
        │
        ▼  scripts/train_forecast.py           (nyc_cp/models/<model>.py)
   (mu, lower, upper) forecasts                — validation (2024) + test (2025) windows
        │
        ├─ scripts/compare_models.py           cross-model accuracy table
        │
        ▼  scripts/compute_effects.py          (nyc_cp/analysis/effects.py)
  per-unit ATT, cumulative effects, calendar significance
        │
        ▼  scripts/geospatial_analysis.py      (nyc_cp/analysis/{geospatial,demographics,causal}.py)
  spatial regression (OLS / ML_Lag / ML_Error) +
  ML feature-importance vs demographics
```

---

## 10. Tests

```bash
pytest tests/
```

Two suites:

* `tests/test_metrics.py` — perfect-forecast → zero error, constant offset → recovered MAE, dominating PI → coverage 1, mean of per-series matches `evaluate_forecasts`.
* `tests/test_effects.py` — `tau` recovers a known counterfactual offset, significance correctly flagged when PI excludes zero, per-unit summary is one row per unit, daily / overall summaries are mutually consistent.

These run without GPU or heavy ML deps — only `numpy`, `pandas`, and `pytest`.
