"""Prophet forecaster — wraps GluonTS' ProphetPredictor."""

from __future__ import annotations

from typing import Any

import pandas as pd
from gluonts.ext.prophet import ProphetPredictor

from nyc_cp.models._gluonts import forecast_to_dfs, predict_with, to_listdataset
from nyc_cp.models.base import BaseForecaster, ForecastResult


class ProphetForecaster(BaseForecaster):
    name = "prophet"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.num_samples: int = int(config.get("num_samples", 100))
        self._history: pd.DataFrame | None = None
        self._train_end: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame, train_end: pd.Timestamp | None = None, **_) -> "ProphetForecaster":
        # Prophet doesn't need a separate fit step — fitting happens at predict time.
        self._history = history
        self._train_end = pd.Timestamp(train_end) if train_end is not None else history.index.max()
        return self

    def predict(self, start: pd.Timestamp, end: pd.Timestamp, freq: str = "D") -> ForecastResult:
        if self._history is None:
            raise RuntimeError("Call fit() first.")
        prediction_length = len(pd.date_range(start=start, end=end, freq=freq))
        predictor = ProphetPredictor(prediction_length=prediction_length)
        # Prophet predicts off the history; cap history at train_end.
        forecasts = predict_with(predictor, self._history, end=self._train_end, freq=freq, num_samples=self.num_samples)
        mu, lo, hi = forecast_to_dfs(forecasts, list(self._history.columns), start, end, freq, self.coverage_level)
        return ForecastResult(mu=mu, lower=lo, upper=hi, coverage_level=self.coverage_level)
