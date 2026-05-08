"""Train one model on one mode, save mu/lower/upper + per-series metrics.

Examples
--------
    python -m scripts.train_forecast --mode bus      --model pcn     --window test
    python -m scripts.train_forecast --mode subway   --model deepar  --window val  --direction O
    python -m scripts.train_forecast --mode citibike --model prophet --window val  --direction D
    python -m scripts.train_forecast --mode bus      --model pcn     --window test --from-checkpoint
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
    normalize_window_name,
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
    p.add_argument("--model", required=True, choices=["arima", "prophet", "deepar", "pcn", "chronos", "timesfm", "nhits", "tft", "bsts"])
    p.add_argument("--window", required=True, choices=["val", "validation", "test"], help="Forecast window (val and validation are aliases).")
    p.add_argument("--direction", choices=["all", "O", "D"], default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-optuna", action="store_true", help="Disable Optuna search even if config sets it.")
    p.add_argument(
        "--from-checkpoint",
        action="store_true",
        help="Skip training; load weights saved by a previous run from <output_dir>/checkpoints/<prefix>.pt.",
    )
    return p.parse_args()


def run(
    mode: str,
    model_name: str,
    window_name: str,
    direction: Direction,
    seed: int,
    disable_optuna: bool,
    from_checkpoint: bool = False,
) -> None:
    set_seed(seed)
    window_name = normalize_window_name(window_name)  # "validation" -> "val"
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

    out_dir = output_dir(mode, model_name, direction=direction, paths=paths)
    prefix = f"{mode}_{model_name}_{window_name}" + (f"_{direction}" if direction != "all" else "")
    ckpt_path = out_dir / "checkpoints" / f"{prefix}.pt"

    forecaster = build_forecaster(model_cfg)

    if from_checkpoint:
        if not forecaster.supports_checkpoints:
            raise SystemExit(f"--from-checkpoint is not supported for model {model_name!r}.")
        if not ckpt_path.exists():
            raise SystemExit(f"--from-checkpoint requested but no checkpoint at {ckpt_path}. Run without the flag first.")
        log.info("Loading checkpoint from %s — skipping training.", ckpt_path)
        forecaster.load_checkpoint(ckpt_path, history=history, train_end=train_end, prediction_length=pred_length)
    else:
        # ``actual`` is passed to fit() so Prophet/DeepAR can correctly
        # identify the forecast horizon at predict time (GluonTS strips the
        # last ``prediction_length`` timesteps of the input series, so the
        # series must end at test_end). ARIMA / PCN ignore this kwarg.
        forecaster.fit(history, train_end=train_end, prediction_length=pred_length, actual=actual)
        if forecaster.supports_checkpoints:
            forecaster.save_checkpoint(ckpt_path)

    result = forecaster.predict(test_start, test_end, freq=freq)
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
        from_checkpoint=args.from_checkpoint,
    )


if __name__ == "__main__":
    main()
