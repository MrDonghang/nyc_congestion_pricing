"""Train one model on one mode, save mu/lower/upper + per-series metrics.

Examples
--------
    python -m scripts.train_forecast --mode bus      --model pcn     --window test
    python -m scripts.train_forecast --mode subway   --model deepar  --window validation --direction O
    python -m scripts.train_forecast --mode citibike --model prophet --window validation --direction D
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from nyc_cp.config import (
    Direction,
    Window,
    actual_csv,
    get_window,
    load_mode,
    load_model,
    load_paths,
    output_dir,
)
from nyc_cp.data.loaders import load_actual
from nyc_cp.evaluation.metrics import evaluate_per_series
from nyc_cp.models import build_forecaster
from nyc_cp.utils import set_seed, setup_logging

log = logging.getLogger("train_forecast")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["bus", "subway", "citibike", "replica"])
    p.add_argument("--model", required=True, choices=["arima", "prophet", "deepar", "pcn"])
    p.add_argument("--window", required=True, choices=["validation", "test"])
    p.add_argument("--direction", choices=["all", "O", "D"], default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-optuna", action="store_true", help="Disable Optuna search even if config sets it.")
    return p.parse_args()


def run(mode: str, model_name: str, window_name: str, direction: Direction, seed: int, disable_optuna: bool) -> None:
    set_seed(seed)
    paths = load_paths()
    setup_logging(f"{mode}_{model_name}_{window_name}", log_root=paths["log_root"])

    mode_cfg = load_mode(mode)
    model_cfg = load_model(model_name)
    if disable_optuna:
        model_cfg.pop("optuna", None)
    model_cfg.setdefault("freq", paths["modes"][mode]["freq"])

    window: Window = get_window(mode, window_name, mode_cfg=mode_cfg)
    train_end = pd.Timestamp(window.train_end)
    test_start = pd.Timestamp(window.test_start)
    test_end = pd.Timestamp(window.test_end)
    freq = model_cfg["freq"]
    pred_length = len(pd.date_range(test_start, test_end, freq=freq))

    log.info("Mode=%s model=%s window=%s direction=%s", mode, model_name, window_name, direction)
    log.info("Train end=%s | Test=[%s, %s] (%d steps, freq=%s)", train_end.date(), test_start.date(), test_end.date(), pred_length, freq)

    actual = load_actual(mode, direction=direction, mode_cfg=mode_cfg, paths=paths)
    history = actual.loc[actual.index <= train_end]
    log.info("History shape: %s | Full actual shape: %s", history.shape, actual.shape)

    forecaster = build_forecaster(model_cfg)
    forecaster.fit(history, train_end=train_end, prediction_length=pred_length)
    result = forecaster.predict(test_start, test_end, freq=freq)

    out_dir = output_dir(mode, model_name, direction=direction, paths=paths)
    prefix = f"{mode}_{model_name}_{window_name}" + (f"_{direction}" if direction != "all" else "")
    result.save(out_dir, prefix)

    truth = actual.loc[result.mu.index]
    metrics = evaluate_per_series(truth, result.mu, result.lower, result.upper, coverage_level=result.coverage_level)
    metrics.to_csv(out_dir / f"{prefix}_evaluation.csv")
    log.info("Saved forecasts and metrics to %s (prefix=%s)", out_dir, prefix)
    log.info("Mean metrics:\n%s", metrics.mean(numeric_only=True).to_string())


def main() -> None:
    args = parse_args()
    run(
        mode=args.mode,
        model_name=args.model,
        window_name=args.window,
        direction=args.direction,
        seed=args.seed,
        disable_optuna=args.no_optuna,
    )


if __name__ == "__main__":
    main()
