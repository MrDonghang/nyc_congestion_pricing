"""DeepAR forecaster — GluonTS' torch DeepAREstimator with optional Optuna tuning."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from gluonts.torch.model.deepar import DeepAREstimator

from nyc_cp.models._gluonts import forecast_to_dfs, predict_with, to_listdataset
from nyc_cp.models.base import BaseForecaster, ForecastResult

log = logging.getLogger(__name__)


class DeepARForecaster(BaseForecaster):
    name = "deepar"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.context_length: int = int(config.get("context_length", 180))
        self.num_layers: int = int(config.get("num_layers", 2))
        self.dropout_rate: float = float(config.get("dropout_rate", 0.1))
        self.max_epochs: int = int(config.get("max_epochs", 100))
        self.num_samples: int = int(config.get("num_samples", 100))
        self.optuna_cfg = config.get("optuna")

        self._predictor = None
        self._history: pd.DataFrame | None = None
        self._train_end: pd.Timestamp | None = None
        self._prediction_length: int | None = None

    def _train_predictor(
        self,
        history: pd.DataFrame,
        train_end: pd.Timestamp,
        prediction_length: int,
        context_length: int,
        num_layers: int,
        dropout_rate: float,
    ):
        train_ds = to_listdataset(history, end=train_end, freq=self.config.get("freq", "D"))
        estimator = DeepAREstimator(
            freq=self.config.get("freq", "D"),
            context_length=context_length,
            prediction_length=prediction_length,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            cardinality=[len(history.columns)],
            trainer_kwargs={"max_epochs": self.max_epochs},
        )
        return estimator.train(training_data=train_ds, num_workers=4, cache_data=True)

    def _tune(self, history: pd.DataFrame, train_end: pd.Timestamp, prediction_length: int) -> dict:
        import optuna

        from nyc_cp.evaluation.metrics import evaluate_per_series

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = int(self.optuna_cfg.get("n_trials", 30))
        space = self.optuna_cfg["search_space"]

        def objective(trial: "optuna.Trial") -> float:
            ctx = trial.suggest_int("context_length", space["context_length"]["low"], space["context_length"]["high"])
            nl = trial.suggest_int("num_layers", space["num_layers"]["low"], space["num_layers"]["high"])
            dr = trial.suggest_float("dropout_rate", space["dropout_rate"]["low"], space["dropout_rate"]["high"])
            predictor = self._train_predictor(history, train_end, prediction_length, ctx, nl, dr)
            forecasts = predict_with(predictor, history, end=train_end + pd.Timedelta(prediction_length, unit="D"), freq=self.config.get("freq", "D"), num_samples=self.num_samples)
            mu, lo, hi = forecast_to_dfs(forecasts, list(history.columns), train_end + pd.Timedelta(1, unit="D"), train_end + pd.Timedelta(prediction_length, unit="D"), self.config.get("freq", "D"), self.coverage_level)
            truth = history.tail(prediction_length).copy()
            truth.index = mu.index
            metrics = evaluate_per_series(truth, mu, lo, hi, coverage_level=self.coverage_level)
            return float(metrics["RMSE"].mean())

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        log.info("DeepAR best params: %s", study.best_params)
        return study.best_params

    def fit(
        self,
        history: pd.DataFrame,
        train_end: pd.Timestamp | None = None,
        prediction_length: int | None = None,
        **_,
    ) -> "DeepARForecaster":
        self._history = history
        self._train_end = pd.Timestamp(train_end) if train_end is not None else history.index.max()
        if prediction_length is None:
            raise ValueError("DeepAR.fit() requires prediction_length (matches forecast horizon).")
        self._prediction_length = prediction_length

        if self.optuna_cfg is not None:
            best = self._tune(history, self._train_end, prediction_length)
            self.context_length = int(best.get("context_length", self.context_length))
            self.num_layers = int(best.get("num_layers", self.num_layers))
            self.dropout_rate = float(best.get("dropout_rate", self.dropout_rate))

        self._predictor = self._train_predictor(
            history, self._train_end, prediction_length, self.context_length, self.num_layers, self.dropout_rate
        )
        return self

    def predict(self, start: pd.Timestamp, end: pd.Timestamp, freq: str = "D") -> ForecastResult:
        if self._predictor is None or self._history is None:
            raise RuntimeError("Call fit() first.")
        forecasts = predict_with(
            self._predictor, self._history, end=end, freq=freq, num_samples=self.num_samples
        )
        mu, lo, hi = forecast_to_dfs(forecasts, list(self._history.columns), start, end, freq, self.coverage_level)
        return ForecastResult(mu=mu, lower=lo, upper=hi, coverage_level=self.coverage_level)
