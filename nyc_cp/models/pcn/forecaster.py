"""PCN as a BaseForecaster.

Each series is z-scored and trained independently. Optuna tunes hyperparameters
on the first column (``reuse_first_column_params``) by default — matching the
original per-mode scripts — and then those parameters are reused for every
remaining series.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from tqdm import tqdm

from nyc_cp.models.base import BaseForecaster, ForecastResult
from nyc_cp.models.pcn.network import MultiLayerPCN, MultiLayerPCNBi
from nyc_cp.models.pcn.trainer import TrainConfig, train_pcn
from nyc_cp.utils.normalize import zscore

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

_Z_TABLE = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


def _build_network(variant: str, batch_length: int, hidden_sizes: list[int], iterations: int, pred_length: int):
    layer_sizes = [batch_length, *hidden_sizes]
    activation = torch.relu
    if variant == "bidirectional":
        return MultiLayerPCNBi(layer_sizes, iterations=iterations, activation=activation, pred_length=pred_length)
    return MultiLayerPCN(layer_sizes, iterations=iterations, activation=activation, pred_length=pred_length)


def _suggest(trial: optuna.Trial, search: dict[str, dict]):
    out = {}
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


class PCNForecaster(BaseForecaster):
    name = "pcn"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.variant: str = config.get("variant", "unidirectional")
        self.batch_length: int = int(config.get("batch_length", 360))
        self.hidden_sizes: list[int] = list(config.get("hidden_sizes", [256, 128]))
        self.iterations: int = int(config.get("iterations", 10))
        self.training_cfg = config.get("training", {})
        self.prediction_cfg = config.get("prediction", {"method": "analytic", "num_samples": 100})
        self.optuna_cfg = config.get("optuna", None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._models: dict[str, torch.nn.Module] = {}
        self._params_per_series: dict[str, dict] = {}
        self._history: pd.DataFrame | None = None
        self._train_end: pd.Timestamp | None = None
        self._pred_length: int | None = None
        self._best_params: dict | None = None

    # ------------------------------------------------------------------ fit ---

    def _train_one(self, series: np.ndarray, params: dict, pred_length: int) -> torch.nn.Module:
        net = _build_network(
            self.variant,
            batch_length=self.batch_length,
            hidden_sizes=[params.get("hidden1", self.hidden_sizes[0]), params.get("hidden2", self.hidden_sizes[1])],
            iterations=params.get("iterations", self.iterations),
            pred_length=pred_length,
        )
        cfg = TrainConfig(
            epochs=int(self.training_cfg.get("epochs", 200)),
            batch_size=int(params.get("batch_size", self.training_cfg.get("batch_size", 32))),
            lr=float(params.get("lr", self.training_cfg.get("lr", 1e-3))),
            weight_decay=float(params.get("weight_decay", self.training_cfg.get("weight_decay", 1e-4))),
            alpha=float(params.get("alpha", self.training_cfg.get("alpha", 10.0))),
            beta=float(params.get("beta", self.training_cfg.get("beta", 0.5))),
            test_split=float(self.training_cfg.get("test_split", 0.2)),
            val_split=float(self.training_cfg.get("val_split", 0.1)),
            patience=int(self.training_cfg.get("patience", 10)),
        )
        train_pcn(net, series, batch_length=self.batch_length, pred_length=pred_length, cfg=cfg, device=self.device)
        return net

    def _tune(self, series: np.ndarray, pred_length: int) -> dict:
        cfg = self.optuna_cfg
        n_trials = int(cfg.get("n_trials", 30))
        search = cfg["search_space"]

        def objective(trial: optuna.Trial) -> float:
            params = _suggest(trial, search)
            short_cfg = TrainConfig(
                epochs=min(50, int(self.training_cfg.get("epochs", 200))),
                batch_size=int(params.get("batch_size", 32)),
                lr=float(params.get("lr", 1e-3)),
                weight_decay=float(params.get("weight_decay", 1e-4)),
                alpha=float(params.get("alpha", 10.0)),
                beta=float(params.get("beta", 0.5)),
                test_split=float(self.training_cfg.get("test_split", 0.2)),
                val_split=float(self.training_cfg.get("val_split", 0.1)),
                patience=int(self.training_cfg.get("patience", 10)),
            )
            net = _build_network(
                self.variant,
                batch_length=self.batch_length,
                hidden_sizes=[params.get("hidden1", 256), params.get("hidden2", 128)],
                iterations=params.get("iterations", 10),
                pred_length=pred_length,
            )
            return train_pcn(net, series, self.batch_length, pred_length, short_cfg, device=self.device)

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
            series_train = series[history.index <= self._train_end] if isinstance(history.index, pd.DatetimeIndex) else series
            norm, _ = zscore(series_train)

            if i == 0 and self.optuna_cfg is not None:
                self._best_params = self._tune(norm, prediction_length)
            params = self._best_params if reuse and self._best_params is not None else (self._best_params or {})
            self._models[col] = self._train_one(norm, params, prediction_length)
            self._params_per_series[col] = params
        return self

    # -------------------------------------------------------------- predict ---

    def _predict_one(self, series_full: np.ndarray, model: torch.nn.Module, pred_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Use the last ``batch_length`` of *normalised, full* series to predict.
        norm, params = zscore(series_full)
        x = torch.tensor(norm[-self.batch_length :], dtype=torch.float32).unsqueeze(0).to(self.device)
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-6
        x_norm = (x - x_mean) / x_std

        model.eval()
        with torch.no_grad():
            mu_n, sigma_n = model(x_norm.float())
        # Two-stage denormalisation: per-window then global.
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
            series = self._history[col].dropna().to_numpy(dtype=float)
            mu, lo, hi = self._predict_one(series, model, n)
            mu_df[col] = mu
            lo_df[col] = lo
            hi_df[col] = hi

        return ForecastResult(mu=mu_df, lower=lo_df, upper=hi_df, coverage_level=self.coverage_level)
