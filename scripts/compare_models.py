"""Tabulate forecast accuracy across all models for one (mode, window, direction).

Reads the saved ``*_mu.csv / *_lower.csv / *_upper.csv`` triplets that
``train_forecast.py`` writes, computes per-series metrics for each model, and
prints + saves a one-row-per-model summary.

Examples
--------
    python -m scripts.compare_models --mode bus      --window val
    python -m scripts.compare_models --mode subway   --window val --direction O
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from nyc_cp.analysis.effects import load_forecast_triplet
from nyc_cp.config import Direction, get_window, load_mode, load_paths, normalize_window_name, output_dir
from nyc_cp.data import load_actual
from nyc_cp.evaluation.metrics import evaluate_per_series
from nyc_cp.utils import setup_logging

log = logging.getLogger("compare_models")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["bus", "subway", "replica"])
    p.add_argument("--window", required=True, choices=["val", "validation", "test"], help="Forecast window (val and validation are aliases).")
    p.add_argument("--direction", choices=["all", "O", "D", "total"], default="all")
    p.add_argument("--models", nargs="+", default=["arima", "prophet", "deepar", "pcn", "chronos", "timesfm", "nhits", "tft", "bsts"])
    p.add_argument("--coverage-level", type=float, default=0.9)
    p.add_argument("--suffix", default=None,
                   help="Optional tag appended to the output filename, e.g. 'IS' or 'OOS'. "
                        "Default: no suffix (overwrites the canonical compare_<mode>_<window>[_<dir>].csv).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.window = normalize_window_name(args.window)
    paths = load_paths()
    setup_logging(f"compare_{args.mode}_{args.window}", log_root=paths["log_root"])

    actual = load_actual(args.mode, direction=args.direction, mode_cfg=load_mode(args.mode), paths=paths)
    window = get_window(args.mode, args.window)
    test_start = pd.Timestamp(window.test_start)
    test_end = pd.Timestamp(window.test_end)
    truth = actual.loc[(actual.index >= test_start) & (actual.index <= test_end)]
    log.info("Truth window: %s → %s (%d days × %d series)", test_start.date(), test_end.date(), *truth.shape)

    rows: list[pd.DataFrame] = []
    for model in args.models:
        out_dir = output_dir(args.mode, model, direction=args.direction, paths=paths)
        prefix = f"{args.mode}_{model}_{args.window}" + (f"_{args.direction}" if args.direction != "all" else "")
        try:
            result = load_forecast_triplet(out_dir, prefix)
        except FileNotFoundError:
            log.warning("Missing forecasts for %s (prefix=%s) under %s — skipping.", model, prefix, out_dir)
            continue

        common = truth.columns.intersection(result.mu.columns)
        per_series = evaluate_per_series(
            truth[common], result.mu[common], result.lower[common], result.upper[common],
            coverage_level=args.coverage_level,
        )
        avg = per_series.mean(numeric_only=True).to_frame().T
        avg.insert(0, "model", model)
        avg.insert(0, "direction", args.direction)
        avg.insert(0, "mode", args.mode)
        rows.append(avg)

    if not rows:
        raise SystemExit("No forecasts found for any model — run train_forecast.py first.")

    table = pd.concat(rows, ignore_index=True)
    summary_dir = Path(paths["output_root"]) / args.mode / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    fname = f"compare_{args.mode}_{args.window}" + (f"_{args.direction}" if args.direction != "all" else "") + (f"_{args.suffix}" if args.suffix else "") + ".csv"
    out = summary_dir / fname
    table.to_csv(out, index=False)
    log.info("Wrote %s", out)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
