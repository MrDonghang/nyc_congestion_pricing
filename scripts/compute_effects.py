"""Compute counterfactual ATT for one (mode, model, window, direction).

Loads the saved forecast triplet + actuals, runs the effects pipeline, and
writes three artefacts under ``<output_root>/<mode>/<model>/effects/``:

  * ``<prefix>_long.csv``     — long-format actual + cf_mean / lo / hi + tau / signif
  * ``<prefix>_unit.csv``     — per-unit summary
  * ``<prefix>_daily.csv``    — daily mean / cumulative ATT with 90% PI
  * ``<prefix>_overall.csv``  — one-row global summary

Examples
--------
    python -m scripts.compute_effects --mode bus      --model pcn --window test
    python -m scripts.compute_effects --mode subway   --model pcn --window test --direction D
    python -m scripts.compute_effects --mode citibike --model pcn --window test --direction O
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from nyc_cp.analysis.effects import (
    build_long_df,
    compute_effects,
    load_forecast_triplet,
    summarize_by_unit,
    summarize_over_time,
    summarize_overall,
)
from nyc_cp.config import get_window, load_mode, load_paths, normalize_window_name, output_dir
from nyc_cp.data import load_actual
from nyc_cp.utils import setup_logging

log = logging.getLogger("compute_effects")

ID_COL_BY_MODE = {
    "bus": "route_id",
    "subway": "station_id",
    "citibike": "tract_id",
    "replica": "tract_id",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["bus", "subway", "citibike", "replica"])
    p.add_argument("--model", required=True, choices=["arima", "prophet", "deepar", "pcn", "chronos", "timesfm", "nhits", "tft", "bsts", "chronos_qrcal", "timesfm_qrcal"])
    p.add_argument("--window", required=True, choices=["val", "validation", "test"], help="Forecast window (val and validation are aliases).")
    p.add_argument("--direction", choices=["all", "O", "D"], default="all")
    p.add_argument("--id-col", default=None, help="Override the unit-id column name (defaults per mode).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.window = normalize_window_name(args.window)
    paths = load_paths()
    setup_logging(f"effects_{args.mode}_{args.model}_{args.window}", log_root=paths["log_root"])

    id_col = args.id_col or ID_COL_BY_MODE[args.mode]
    actual = load_actual(args.mode, direction=args.direction, mode_cfg=load_mode(args.mode), paths=paths)

    window = get_window(args.mode, args.window)
    test_start = pd.Timestamp(window.test_start)
    test_end = pd.Timestamp(window.test_end)
    actual = actual.loc[(actual.index >= test_start) & (actual.index <= test_end)]

    out_dir = output_dir(args.mode, args.model, direction=args.direction, paths=paths)
    prefix = f"{args.mode}_{args.model}_{args.window}" + (f"_{args.direction}" if args.direction != "all" else "")
    result = load_forecast_triplet(out_dir, prefix)

    common = list(actual.columns.intersection(result.mu.columns))
    log.info("Common series after alignment: %d", len(common))
    cov = result.coverage_level
    long = build_long_df(
        actual[common], result.mu[common], result.lower[common], result.upper[common], id_col=id_col, columns=common
    )
    eff = compute_effects(long, id_col=id_col, coverage_level=cov)
    unit = summarize_by_unit(eff, id_col=id_col, coverage_level=cov)
    daily = summarize_over_time(eff, coverage_level=cov)
    overall = summarize_overall(unit, id_col=id_col, coverage_level=cov)

    eff_dir = out_dir / "effects"
    eff_dir.mkdir(exist_ok=True)
    eff.to_csv(eff_dir / f"{prefix}_long.csv", index=False)
    unit.to_csv(eff_dir / f"{prefix}_unit.csv", index=False)
    daily.to_csv(eff_dir / f"{prefix}_daily.csv", index=False)
    overall.to_csv(eff_dir / f"{prefix}_overall.csv", index=False)

    log.info("Wrote effects under %s", eff_dir)
    print("\nOverall summary:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
