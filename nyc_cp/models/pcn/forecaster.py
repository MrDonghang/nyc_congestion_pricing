"""PCN as a BaseForecaster.

Each series is z-scored and trained independently. Optuna tunes hyperparameters
on the first column (``reuse_first_column_params``) by default — matching the
original per-mode scripts — and then those parameters are reused for every
remaining series.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from tqdm import tqdm

from nyc_cp.models.base import BaseForecaster, ForecastResult
from nyc_cp.models.pcn.network import MultiLayerPCN
from nyc_cp.models.pcn.trainer import TrainConfig, train_pcn
from nyc_cp.utils.normalize import zscore

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Bumped from v1: the broken unidirectional + missing-gamma checkpoints from
# the previous iteration are no longer compatible.
CHECKPOINT_SCHEMA_VERSION = 2

_Z_TABLE = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}

_ACTIVATIONS = {"tanh": torch.tanh, "relu": torch.relu}


def _suggest(trial: optuna.Trial, search: dict[str, dict]):
    out: dict[str, Any] = {}
    for name, spec in search.items():
        kind = spec["type"]
        if kind == "int":
            out[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif kind == "uniform":
            out[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif kind == "loguniform":
            out[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif kind == "categorical":
            out[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown Optuna type: {kind}")
    return out


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PCNForecaster(BaseForecaster):
    name = "pcn"
    supports_checkpoints = True

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.batch_length: int = int(config.get("batch_length", 360))
        self.hidden_sizes: list[int] = list(config.get("hidden_sizes", [256, 128]))
        self.iterations: int = int(config.get("iterations", 10))
        self.inference_lr: float = float(config.get("inference_lr", 0.05))
        self.activation_name: str = str(config.get("activation", "tanh"))
        self.training_cfg: dict = config.get("training", {})
        self.prediction_cfg: dict = config.get("prediction", {"method": "analytic", "num_samples": 100})
        self.optuna_cfg: dict | None = config.get("optuna", None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._models: dict[str, torch.nn.Module] = {}
        self._params_per_series: dict[str, dict] = {}
        self._history: pd.DataFrame | None = None
        self._train_end: pd.Timestamp | None = None
        self._pred_length: int | None = None
        self._best_params: dict | None = None

    # ------------------------------------------------------------------ build ---

    def _activation(self):
        return _ACTIVATIONS.get(self.activation_name, torch.tanh)

    def _build(self, batch_length: int, hidden_sizes: list[int], iterations: int, pred_length: int) -> MultiLayerPCN:
        return MultiLayerPCN(
            layer_sizes=[batch_length, *hidden_sizes],
            iterations=iterations,
            activation=self._activation(),
            pred_length=pred_length,
            inference_lr=self.inference_lr,
        )

    # ------------------------------------------------------------------ fit ---

    def _make_train_cfg(self, params: dict, epochs_cap: int | None = None) -> TrainConfig:
        epochs = int(self.training_cfg.get("epochs", 200))
        if epochs_cap is not None:
            epochs = min(epochs, epochs_cap)
        return TrainConfig(
            epochs=epochs,
            batch_size=int(params.get("batch_size", self.training_cfg.get("batch_size", 32))),
            lr=float(params.get("lr", self.training_cfg.get("lr", 1e-3))),
            weight_decay=float(params.get("weight_decay", self.training_cfg.get("weight_decay", 1e-4))),
            alpha=float(params.get("alpha", self.training_cfg.get("alpha", 10.0))),
            beta=float(params.get("beta", self.training_cfg.get("beta", 0.5))),
            gamma=float(params.get("gamma", self.training_cfg.get("gamma", 0.1))),
            test_split=float(self.training_cfg.get("test_split", 0.2)),
            val_split=float(self.training_cfg.get("val_split", 0.1)),
            patience=int(self.training_cfg.get("patience", 10)),
        )

    def _arch_from_params(self, params: dict) -> tuple[int, list[int], int]:
        bl = int(params.get("batch_length", self.batch_length))
        h1 = int(params.get("hidden1", self.hidden_sizes[0]))
        h2 = int(params.get("hidden2", self.hidden_sizes[1] if len(self.hidden_sizes) > 1 else self.hidden_sizes[0]))
        it = int(params.get("iterations", self.iterations))
        return bl, [h1, h2], it

    def _train_one(self, series: np.ndarray, params: dict, pred_length: int) -> torch.nn.Module:
        bl, hidden, it = self._arch_from_params(params)
        net = self._build(bl, hidden, it, pred_length)
        cfg = self._make_train_cfg(params)
        train_pcn(net, series, batch_length=bl, pred_length=pred_length, cfg=cfg, device=self.device)
        return net

    def _tune(self, series: np.ndarray, pred_length: int) -> dict:
        cfg = self.optuna_cfg or {}
        n_trials = int(cfg.get("n_trials", 30))
        search = cfg["search_space"]

        def objective(trial: optuna.Trial) -> float:
            # Match pcn_model_new.py: re-seed each trial so trials differ only
            # by their hyperparameters, not by RNG state.
            _seed_all(42)
            params = _suggest(trial, search)
            bl, hidden, it = self._arch_from_params(params)
            short_cfg = self._make_train_cfg(params, epochs_cap=50)
            try:
                net = self._build(bl, hidden, it, pred_length)
                return train_pcn(net, series, bl, pred_length, short_cfg, device=self.device)
            except (ValueError, RuntimeError) as e:
                # Common failures: series too short for this batch_length, or OOM.
                # Steer Optuna away from this region without aborting the study.
                log.warning("Optuna trial failed (%s); returning sentinel loss.", e)
                return 1e9

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        log.info("PCN best params: %s", study.best_params)
        return study.best_params

    def fit(
        self,
        history: pd.DataFrame,
        train_end: pd.Timestamp | None = None,
        prediction_length: int | None = None,
        **_,
    ) -> "PCNForecaster":
        if prediction_length is None:
            raise ValueError("PCN.fit() requires prediction_length.")
        self._history = history
        self._train_end = pd.Timestamp(train_end) if train_end is not None else history.index.max()
        self._pred_length = prediction_length

        self._models = {}
        self._params_per_series = {}
        self._best_params = None

        reuse = bool((self.optuna_cfg or {}).get("reuse_first_column_params", True))

        for i, col in enumerate(tqdm(history.columns, desc="PCN fit")):
            series = history[col].dropna().to_numpy(dtype=float)
            if isinstance(history.index, pd.DatetimeIndex):
                mask = history.index <= self._train_end
                series_train = series[mask] if len(mask) == len(series) else series
            else:
                series_train = series
            norm, _ = zscore(series_train)

            if i == 0 and self.optuna_cfg is not None:
                self._best_params = self._tune(norm, prediction_length)

            if reuse and self._best_params is not None:
                params = self._best_params
            else:
                params = self._best_params or {}

            self._models[col] = self._train_one(norm, params, prediction_length)
            self._params_per_series[col] = dict(params)

        return self

    # -------------------------------------------------------------- predict ---

    def _predict_one(
        self, series_full: np.ndarray, model: torch.nn.Module, pred_length: int, batch_length: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        norm, params = zscore(series_full)
        x = torch.tensor(norm[-batch_length:], dtype=torch.float32).unsqueeze(0).to(self.device)
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-6
        x_norm = (x - x_mean) / x_std

        model.eval()
        with torch.no_grad():
            mu_n, sigma_n, _ = model(x_norm.float())

        mu_local = mu_n * x_std.expand_as(mu_n) + x_mean.expand_as(mu_n)
        sigma_local = sigma_n * x_std.expand_as(sigma_n)
        mu = mu_local * params.std + params.mean
        sigma = sigma_local * params.std

        method = self.prediction_cfg.get("method", "analytic")
        ci = self.coverage_level
        if method == "analytic":
            z = _Z_TABLE.get(round(ci, 2), 1.645)
            mu_np = mu.squeeze().cpu().numpy()
            sigma_np = sigma.squeeze().cpu().numpy()
            return mu_np, mu_np - z * sigma_np, mu_np + z * sigma_np
        if method == "sampling":
            ns = int(self.prediction_cfg.get("num_samples", 100))
            samples = torch.normal(mu.expand(ns, -1, -1), sigma.expand(ns, -1, -1)).squeeze(1).cpu().numpy()
            return (
                samples.mean(axis=0),
                np.percentile(samples, (1 - ci) / 2 * 100, axis=0),
                np.percentile(samples, (1 - (1 - ci) / 2) * 100, axis=0),
            )
        raise ValueError(f"Unknown prediction method: {method}")

    def predict(self, start: pd.Timestamp, end: pd.Timestamp, freq: str = "D") -> ForecastResult:
        if self._history is None or not self._models:
            raise RuntimeError("Call fit() first.")
        idx = pd.date_range(start=start, end=end, freq=freq)
        n = len(idx)
        if n != self._pred_length:
            raise ValueError(f"predict horizon ({n}) != fit prediction_length ({self._pred_length}).")

        mu_df = pd.DataFrame(index=idx, columns=self._history.columns, dtype=float)
        lo_df = pd.DataFrame(index=idx, columns=self._history.columns, dtype=float)
        hi_df = pd.DataFrame(index=idx, columns=self._history.columns, dtype=float)

        for col, model in self._models.items():
            params = self._params_per_series.get(col, {})
            bl, _, _ = self._arch_from_params(params)
            series = self._history[col].dropna().to_numpy(dtype=float)
            mu, lo, hi = self._predict_one(series, model, n, bl)
            mu_df[col] = mu
            lo_df[col] = lo
            hi_df[col] = hi

        return ForecastResult(mu=mu_df, lower=lo_df, upper=hi_df, coverage_level=self.coverage_level)

    # ------------------------------------------------------------ checkpoint ---

    def save_checkpoint(self, path: Path) -> None:
        if not self._models:
            raise RuntimeError("Call fit() before save_checkpoint().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state": {col: m.state_dict() for col, m in self._models.items()},
            "params_per_series": self._params_per_series,
            "best_params": self._best_params,
            "columns": list(self._models.keys()),
            "config_snapshot": {
                "inference_lr": self.inference_lr,
                "activation": self.activation_name,
                "default_batch_length": self.batch_length,
                "default_hidden_sizes": list(self.hidden_sizes),
                "default_iterations": self.iterations,
            },
            "pred_length": self._pred_length,
            "train_end": str(self._train_end) if self._train_end is not None else None,
            "coverage_level": self.coverage_level,
        }
        torch.save(state, path)
        log.info("PCN checkpoint saved to %s (%d series)", path, len(state["columns"]))

    def load_checkpoint(
        self,
        path: Path,
        history: pd.DataFrame,
        train_end: pd.Timestamp,
        prediction_length: int,
    ) -> "PCNForecaster":
        path = Path(path)
        # weights_only=False because the checkpoint contains plain Python dicts
        # (params_per_series, best_params) alongside the tensor state_dicts.
        state = torch.load(path, map_location=self.device, weights_only=False)

        if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Checkpoint schema {state.get('schema_version')} != expected "
                f"{CHECKPOINT_SCHEMA_VERSION}. Re-train this model from scratch."
            )

        ckpt_pred_length = int(state["pred_length"])
        if ckpt_pred_length != prediction_length:
            raise ValueError(
                f"Checkpoint was trained for prediction_length={ckpt_pred_length} "
                f"but caller asked for {prediction_length}. Re-train for this horizon."
            )

        ckpt_train_end = state.get("train_end")
        req_train_end = pd.Timestamp(train_end)
        if ckpt_train_end is not None and pd.Timestamp(ckpt_train_end) != req_train_end:
            log.warning(
                "Checkpoint train_end (%s) differs from requested (%s); using checkpoint's.",
                ckpt_train_end, req_train_end,
            )

        self._best_params = state["best_params"]
        self._params_per_series = state["params_per_series"]
        self._pred_length = ckpt_pred_length
        self._train_end = pd.Timestamp(ckpt_train_end) if ckpt_train_end else req_train_end
        self._history = history

        missing = [c for c in state["columns"] if c not in history.columns]
        if missing:
            raise ValueError(
                f"Checkpoint has {len(missing)} series not present in current history "
                f"(first few: {missing[:3]}). History/source data has changed since training."
            )

        self._models = {}
        for col in state["columns"]:
            params = self._params_per_series.get(col, {})
            bl, hidden, it = self._arch_from_params(params)
            net = self._build(bl, hidden, it, self._pred_length)
            net.load_state_dict(state["model_state"][col])
            net.to(self.device)
            net.eval()
            self._models[col] = net

        log.info("PCN checkpoint loaded from %s (%d series)", path, len(self._models))
        return self
