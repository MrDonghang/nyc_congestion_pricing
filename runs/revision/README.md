# 返修实验索引（`runs/revision/`）

最后整理：2026-11-27。本目录是返修的最终分析流程：滚动起点的零样本预测、分位数回归校准、伪政策日推断和出图。

## 当前主设定（2026-11-26 定稿）

- **主模型**：Chronos-Bolt；**稳健性模型**：TimesFM-2.0。零样本，逐站 / 逐线路预测后求和。
- **数据**：地铁 `subway_ns22`（进站量，不含学生票，2022 年起）；公交 `bus_ns`（不含学生票），布朗克斯本地线路单独分析。
- **滚动起点**：每 3 天一个起点，每个起点预测 116 天；政策日 2025-01-05。
- **校准**：对过去预测的对数误差做中位数分位数回归，自变量为节假日、星期、提前期、目标月份；只用起点之前的数据。
- **逐日 90 % 区间**：模型原生的 10 % / 90 % 分位线，用同一回归在 5 % / 95 % 处校准。
- **显著性与累积区间**：2024 年 1–5 月的 21 个伪政策日（窗口在 2024-08-29 数据断点之前结束）。

| 116 天累积效应 | Chronos-Bolt（主） | TimesFM（稳健性） |
|---|---|---|
| 地铁 | +2.6 % [+0.1, +4.2]，0/21，+9.6 M 人次 | +2.5 % [−0.4, +4.6]，1/21 |
| 公交（不含布朗克斯本地线） | +3.5 % [+0.5, +6.0]，0/21，+3.3 M 人次 | +2.5 % [+0.9, +3.8]，0/21 |

"0/21" 表示 21 个伪政策日中没有一个达到或超过政策日的估计。

写作时需要带上的限制：

- 伪政策窗口高度重叠，"0/21" 是描述性证据，不能写成 p ≈ 0.045。
- Bolt 的伪政策日平均约 −2 %（季节校准把 2024 年伪政策日整体压低）。
- 合成数据检验（Bolt，两组 80 条序列）：点估计基本无偏，+2.6 % / +3.5 % / +5 % 的检出率为 72 / 88 / 96 %，但无效应时误报率 12 %（名义约 4.5 %）。TimesFM 的误报率为 51 %，不能作为合成检验的证据。
- 图 1a 的示例站 Northern Blvd 是为了看得清楚挑选的（+7.2 %），高于系统平均。

## 复现顺序

环境：项目的 Python 环境（`pip install -e .`），在仓库根目录运行并设 `PYTHONPATH=$PWD`；滚动预测需要 GPU。
表输出到 `outputs/_summary/revision/uq/`，图输出到 `outputs/figures/paper/{main,robustness}/`。

| 步骤 | 命令（两个模型各跑一次，`<model>` = `chronos_bolt` 或 `timesfm`） | 主要输出 |
|---|---|---|
| 1 滚动预测（GPU） | `dense_q.py --mode subway_ns22 --direction O --model <model>`；`dense_q.py --mode bus_ns --model <model> --groups uq/bus_groups.csv` | `denseq_*.csv`、`denseunitq_*.csv` |
| 2 校准 | `qr_final.py`，然后 `qr_native.py` | `qr_native_daily.csv`、`qr_native_summary.csv`、`qr_final_scores.csv` |
| 3 图 1 用表 | `paper_tables_qr.py` | `paper_{daily,cumulative,placebo_paths}_qr_<model>.csv` |
| 4 示例站 / 线路（GPU） | `EX_MIN_SUBWAY=2500 example_forecasts.py <model>` | `example_forecasts_<model>.csv`、`example_screen_<model>.csv` |
| 5 单元效应 | `unit_effects.py <model>` | `station_effects_subway_<model>.csv`、`route_effects_bus_<model>.csv` |
| 6 正文图 | `fig_method_flow.py`；`fig1_overview.py [timesfm]`；`fig2_spatial.py [timesfm]`；`fig3_replica.py [timesfm]`；`fig4_drivers.py [timesfm]` | `main/fig*.png`（不带后缀 = Bolt，`_timesfm` = TimesFM） |
| 7 稳健性图 | `plot_daily_final.py`、`plot_cumulative_final.py`、`plot_paper_uq.py`、`fig_placebo_paths.py`、`qr_variants.py` | `robustness/main_*_chronos_bolt.png`、`appendix_*_timesfm.png`、`appendix_calibration_variants.png` |
| 8 合成检验 | `synth_calib.py replay <model>`（GPU）；`synth_bolt_qr.py`；`synth_timesfm_qr.py`；`fig_synth_check.py` | `synth_{bolt,timesfm}_qr_summary[_norw].csv`、`robustness/appendix_synthetic_chronos_bolt.png` |

步骤 2、3、7 中没有写参数的脚本，命令行用法未逐个核对，运行前先看脚本开头的注释。

## 脚本分类

这里只保留最终分析用到的脚本。

### 1. 数据准备

| 脚本 | 说明（脚本首行注释） |
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

### 2. 滚动预测（需要 GPU）

| 脚本 | 说明（脚本首行注释） |
|---|---|
| `dense_q.py` | Dense rolling-origin replay keeping the models' NATIVE 10 % / 50 % / 90 % quantiles (no normal approximation),… |
| `dense_weekly.py` | Weekly rolling-origin replay for Replica (all trips by origin tract), raw foundation model |
| `dense_weekly_modes.py` | Weekly rolling-origin replay for Replica trips INTO the Congestion Relief Zone, by travel mode |
| `example_forecasts.py` | Example units for Fig. 1a,b: observed vs forecast with a CALIBRATED model interval, in a no-policy rehearsal w… |

### 3. 校准、效应与显著性（主设定）

| 脚本 | 说明（脚本首行注释） |
|---|---|
| `qr_native.py` | Quantile-regression calibration on the models' NATIVE quantiles (denseq_*.csv; no normal approximation). Syste… |
| `qr_final.py` | Original experiment with quantile-regression calibration and the models' own predictive quantiles (user reques… |
| `paper_tables_qr.py` | Tables behind Fig. 1 (panels c-f) under the quantile-regression specification (qr_final.py): same schema as th… |
| `unit_effects.py` | Station-level (subway) and route-level (bus) effect tables for a given model; same definitions as station_leve… |
| `qr_variants.py` | Appendix: 116-day effect under four calibration designs, quantile-regression (median) version on the current r… |
| `round8_prc.py` | Round 8 — pooled rolling-origin calibration (PRC), system level |
| `round9_bus.py` | Round 9 — repairing the bus null. Per route group (Bronx local / serves CRZ / other / non-Bronx / all), raw an… |

### 4. 正文图

| 脚本 | 说明（脚本首行注释） |
|---|---|
| `fig_method_flow.py` | Method flowchart: rolling forecast origins (a) and the estimation / inference steps applied at each origin (b) |
| `fig1_overview.py` | Figure 1 (overview) - Nature-style, 183 mm double column, Arial 5-8 pt, no titles, editable text. Four-column … |
| `fig2_spatial.py` | Figure 2 (where the gains are) - Nature-style, 183 mm, Arial, editable text |
| `fig3_replica.py` | Figure 3 (Replica, all travel) - Nature-style, 183 mm, Arial, editable text. Weekly data, 17 weeks from 4 Jan … |
| `fig4_drivers.py` | Figure 4 (who responded, and where people drove before the policy) - Nature-style, 183 mm, Arial, editable tex… |

### 5. 稳健性 / 附录图

| 脚本 | 说明（脚本首行注释） |
|---|---|
| `plot_daily_final.py` | Daily effect under the final specification (quantile-regression calibration, seasonal design), system level |
| `plot_cumulative_final.py` | Cumulative effect in absolute riders under the final specification (system level) |
| `plot_paper_uq.py` | Paper figures for the effect estimates (decided with the user, 2026-11-24) |
| `fig_placebo_paths.py` | In-time placebo paths under the final specification: cumulative relative effect by days since the (pseudo-)pol… |
| `fig_synth_check.py` | Appendix figure: synthetic check of the main specification (Chronos-Bolt; simple and varied data-generating pr… |

### 6. 图 3、图 4 的上游分析（Replica、社会人口）

| 脚本 | 说明（脚本首行注释） |
|---|---|
| `replica_car_share.py` | Where did people travel INTO the Congestion Relief Zone mainly by car before the policy? (Replica, typical wee… |
| `station_level.py` | Station-level subway effects (TimesFM, entries, student fares excluded, history from 2022, 116 days) |
| `route_level.py` | Route-level bus effects (TimesFM, student fares excluded, 116 days). Each route is compared with its own 2024 |
| `sociodemo_lib.py` | Shared loader for the socio-demographic figures: per-origin standardised slopes of the unit-level effect on ea… |
| `sociodemo_analysis.py` | Policy effect vs socio-demographics (ACS 2023), with the same in-time placebo logic as the headline estimates |
| `sociodemo_shap.py` | Machine-learning model of the unit-level policy effect on neighbourhood characteristics, explained with SHAP v… |
| `sociodemo_spatial.py` | Spatial regression of the final TimesFM unit effects on neighbourhood characteristics (same targets and featur… |

### 7. 合成数据检验

| 脚本 | 说明（脚本首行注释） |
|---|---|
| `synth_calib.py` | Synthetic comparison of calibration methods under the placebo framework (truth known: no policy effect in the … |
| `synth_bolt_qr.py` | Synthetic check of the main specification (Chronos-Bolt, median quantile-regression calibration with holiday, … |
| `synth_timesfm_qr.py` | Synthetic check of the main specification (Chronos-Bolt, median quantile-regression calibration with holiday, … |

### 另外保留

| 脚本 | 说明 |
|---|---|
| `dense_origins.py`、`dense_pi.py` | 早期的滚动预测脚本，`round8_prc.py` / `round9_bus.py` 和 `qr_final.py` 仍读取它们的输出 |
| `fig_replica_spatial.py`、`fig_replica_placebo.py` | Replica 的空间效应表（图 3 读取）和伪政策路径图 |
