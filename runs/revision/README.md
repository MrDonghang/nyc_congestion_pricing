# Rolling-origin analysis pipeline (`runs/revision/`)

Scripts behind the results in the manuscript: zero-shot counterfactual forecasts from rolling origins, quantile-regression calibration,
pseudo-policy-date (in-time placebo) inference, and the figures.

## Specification

- **Models.** Chronos-Bolt (main) and TimesFM-2.0 (robustness check), zero-shot. Each station / route is forecast separately and the forecasts are summed to the system.
- **Data.** Subway: station entries, student fare classes excluded, history from 2022 (`subway_ns22`, direction `O`). Bus: boardings per route, student fare classes excluded (`bus_ns`); Bronx local routes are analysed separately.
- **Forecast origins.** One origin every 3 days, each issuing a 116-day forecast; the policy date is 2025-01-05.
- **Calibration.** Median quantile regression of the log forecast error, log(observed / forecast), on holiday, day of week, lead time and target month. The regression at origin *D* uses only forecasts whose target dates precede *D*; forecasts that cross the break in the fare data on 2024-08-29 are excluded.
- **Daily 90 % band.** The models' own 10 % / 90 % quantile forecasts, calibrated with the same regression at the 5 % / 95 % levels.
- **Inference.** The same steps are applied at 21 pseudo-policy dates between January and May 2024 (116-day windows that end before 2024-08-29), when no policy was in place. The policy estimate is compared with these values, and the 90 % interval of the cumulative effect comes from their mean-centred quantiles.
- **Synthetic check.** Series with known properties and no policy, with effects of known size added afterwards (`synth_calib.py`, `synth_bolt_qr.py`, `synth_timesfm_qr.py`).

## Reproduction

Run from the repository root with the package installed (`pip install -e .`) and `PYTHONPATH=$PWD`. Steps marked GPU run the foundation models.
Tables are written to `outputs/_summary/revision/uq/`, figures to `outputs/figures/paper/{main,robustness}/` (both git-ignored).
`<model>` is `chronos_bolt` or `timesfm`; run both.

| Step | Scripts | Main outputs |
|---|---|---|
| 1 Forecast replay (GPU) | `dense_q.py --mode subway_ns22 --direction O --model <model>`; `dense_q.py --mode bus_ns --model <model> --groups uq/bus_groups.csv` | `denseq_*.csv`, `denseunitq_*.csv` |
| 2 Calibration | `qr_final.py`, then `qr_native.py` | `qr_native_daily.csv`, `qr_native_summary.csv`, `qr_final_scores.csv` |
| 3 Tables for Figure 1 | `paper_tables_qr.py` | `paper_{daily,cumulative,placebo_paths}_qr_<model>.csv` |
| 4 Example station / route (GPU) | `EX_MIN_SUBWAY=2500 example_forecasts.py <model>` | `example_forecasts_<model>.csv`, `example_screen_<model>.csv` |
| 5 Station- and route-level effects | `unit_effects.py <model>` | `station_effects_subway_<model>.csv`, `route_effects_bus_<model>.csv` |
| 6 Main figures | `fig_method_flow.py`; `fig1_overview.py [timesfm]`; `fig2_spatial.py [timesfm]`; `fig3_replica.py [timesfm]`; `fig4_drivers.py [timesfm]` | `main/fig*.png` (no suffix: Chronos-Bolt; `_timesfm`: TimesFM) |
| 7 Robustness figures | `plot_daily_final.py`, `plot_cumulative_final.py`, `plot_paper_uq.py`, `fig_placebo_paths.py`, `qr_variants.py` | `robustness/main_*_chronos_bolt.png`, `appendix_*_timesfm.png`, `appendix_calibration_variants.png` |
| 8 Synthetic check | `synth_calib.py replay <model>` (GPU); `synth_bolt_qr.py`; `synth_timesfm_qr.py`; `fig_synth_check.py` | `synth_{bolt,timesfm}_qr_summary.csv`, `robustness/appendix_synthetic_chronos_bolt.png` |

Each script states its inputs, outputs and usage in the docstring at the top of the file; check it before running, as not every command line above has been re-run from a clean checkout.
`qr_final.py`, `round8_prc.py` and `round9_bus.py` read the output of two earlier replay scripts, `dense_pi.py` and `dense_origins.py`, which therefore have to be run before step 2.

## Scripts

### 1. Data preparation

| Script | Purpose (first line of the script's docstring) |
|---|---|
| `fetch_bus_nostudent.py` | Rebuild the daily route × ridership bus matrix EXCLUDING student fare classes, |
| `finalize_bus_nostudent.py` | Align the student-excluded bus matrix with the existing 'final' bus matrix: |
| `build_subway_nostudent.py` | Station × day subway entries EXCLUDING student fare classes (Metrocard - Students, OMNY - Students), |
| `build_subway_nostudent22.py` | Station × day subway entries EXCLUDING student fare classes (Metrocard - Students, OMNY - Students), |
| `build_subway_ext.py` | Extend the subway station-day panel back to 2022-01-01 |
| `fetch_extension.py` | Fetch post-April-2025 data, student fare classes excluded, aggregated server-side (data.ny.gov SODA API): |
| `subway_fare_class.py` | Daily system-wide subway ridership by fare class from the raw MTA hourly files (2023-01 → 2025-04) |
| `build_unit_geo.py` | Unit geography for the spatial contrast: relation of each subway station / bus route to the Congestion Relief … |
| `build_sociodemo.py` | Socio-demographic profile of each analysis unit (ACS 5-year, 2023 vintage, census tracts; geo_data/nyc_census_… |

### 2. Rolling-origin forecasts (GPU)

| Script | Purpose (first line of the script's docstring) |
|---|---|
| `dense_q.py` | Dense rolling-origin replay keeping the models' NATIVE 10 % / 50 % / 90 % quantiles (no normal approximation),… |
| `dense_weekly.py` | Weekly rolling-origin replay for Replica (all trips by origin tract), raw foundation model |
| `dense_weekly_modes.py` | Weekly rolling-origin replay for Replica trips INTO the Congestion Relief Zone, by travel mode |
| `example_forecasts.py` | Example units for Fig. 1a,b: observed vs forecast with a CALIBRATED model interval, in a no-policy rehearsal w… |

### 3. Calibration, effects and inference

| Script | Purpose (first line of the script's docstring) |
|---|---|
| `qr_native.py` | Quantile-regression calibration on the models' NATIVE quantiles (denseq_*.csv; no normal approximation). Syste… |
| `qr_final.py` | Original experiment with quantile-regression calibration and the models' own predictive quantiles (user reques… |
| `paper_tables_qr.py` | Tables behind Fig. 1 (panels c-f) under the quantile-regression specification (qr_final.py): same schema as th… |
| `unit_effects.py` | Station-level (subway) and route-level (bus) effect tables for a given model; same definitions as station_leve… |
| `qr_variants.py` | Appendix: 116-day effect under four calibration designs, quantile-regression (median) version on the current r… |
| `round8_prc.py` | Round 8 — pooled rolling-origin calibration (PRC), system level |
| `round9_bus.py` | Round 9 — repairing the bus null. Per route group (Bronx local / serves CRZ / other / non-Bronx / all), raw an… |

### 4. Main figures

| Script | Purpose (first line of the script's docstring) |
|---|---|
| `fig_method_flow.py` | Method flowchart: rolling forecast origins (a) and the estimation / inference steps applied at each origin (b) |
| `fig1_overview.py` | Figure 1 (overview) - Nature-style, 183 mm double column, Arial 5-8 pt, no titles, editable text. Four-column … |
| `fig2_spatial.py` | Figure 2 (where the gains are) - Nature-style, 183 mm, Arial, editable text |
| `fig3_replica.py` | Figure 3 (Replica, all travel) - Nature-style, 183 mm, Arial, editable text. Weekly data, 17 weeks from 4 Jan … |
| `fig4_drivers.py` | Figure 4 (who responded, and where people drove before the policy) - Nature-style, 183 mm, Arial, editable tex… |

### 5. Robustness / supplementary figures

| Script | Purpose (first line of the script's docstring) |
|---|---|
| `plot_daily_final.py` | Daily effect under the final specification (quantile-regression calibration, seasonal design), system level |
| `plot_cumulative_final.py` | Cumulative effect in absolute riders under the final specification (system level) |
| `plot_paper_uq.py` | Paper figures for the effect estimates (decided with the user, 2026-11-24) |
| `fig_placebo_paths.py` | In-time placebo paths under the final specification: cumulative relative effect by days since the (pseudo-)pol… |
| `fig_synth_check.py` | Appendix figure: synthetic check of the main specification (Chronos-Bolt; simple and varied data-generating pr… |

### 6. Analyses behind Figures 3 and 4 (Replica, socio-demographics)

| Script | Purpose (first line of the script's docstring) |
|---|---|
| `replica_car_share.py` | Where did people travel INTO the Congestion Relief Zone mainly by car before the policy? (Replica, typical wee… |
| `station_level.py` | Station-level subway effects (TimesFM, entries, student fares excluded, history from 2022, 116 days) |
| `route_level.py` | Route-level bus effects (TimesFM, student fares excluded, 116 days). Each route is compared with its own 2024 |
| `sociodemo_lib.py` | Shared loader for the socio-demographic figures: per-origin standardised slopes of the unit-level effect on ea… |
| `sociodemo_analysis.py` | Policy effect vs socio-demographics (ACS 2023), with the same in-time placebo logic as the headline estimates |
| `sociodemo_shap.py` | Machine-learning model of the unit-level policy effect on neighbourhood characteristics, explained with SHAP v… |
| `sociodemo_spatial.py` | Spatial regression of the final TimesFM unit effects on neighbourhood characteristics (same targets and featur… |

### 7. Synthetic check

| Script | Purpose (first line of the script's docstring) |
|---|---|
| `synth_calib.py` | Synthetic comparison of calibration methods under the placebo framework (truth known: no policy effect in the … |
| `synth_bolt_qr.py` | Synthetic check of the main specification (Chronos-Bolt, median quantile-regression calibration with holiday, … |
| `synth_timesfm_qr.py` | Synthetic check of the main specification (Chronos-Bolt, median quantile-regression calibration with holiday, … |

### Also required

| Script | Purpose |
|---|---|
| `dense_origins.py`, `dense_pi.py` | Earlier replay scripts whose output is still read by `round8_prc.py` / `round9_bus.py` and `qr_final.py` |
| `fig_replica_spatial.py`, `fig_replica_placebo.py` | Replica tract-level effect table (read by Figure 3) and Replica placebo paths |
