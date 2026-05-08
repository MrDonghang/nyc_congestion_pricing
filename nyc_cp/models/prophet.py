"""Prophet forecaster — wraps GluonTS' ProphetPredictor."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from gluonts.ext.prophet import ProphetPredictor

from nyc_cp.models._gluonts import forecast_to_dfs, predict_with
from nyc_cp.models.base import BaseForecaster, ForecastResult

log = logging.getLogger(__name__)


class ProphetForecaster(BaseForecaster):
    name = "prophet"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.num_samples: int = int(config.get("num_samples", 100))
        self._history: pd.DataFrame | None = None
        self._actual: pd.DataFrame | None = None
        self._train_end: pd.Timestamp | None = None

    def fit(
        self,
        history: pd.DataFrame,
        train_end: pd.Timestamp | None = None,
        actual: pd.DataFrame | None = None,
        **_,
    ) -> "ProphetForecaster":
        """Prophet does not pre-fit — the model is built per series at predict
        time inside GluonTS' ``ProphetPredictor``.

        ``actual`` (the full date-indexed wide frame, including the future
        target window) is required for correct forecasting:
        ``make_evaluation_predictions`` strips the last ``prediction_length``
        timesteps of each series to use as the forecast horizon, and fits
        Prophet on what remains. We need that strip to align with the test
        window — which only happens when the input series ends at ``test_end``,
        not ``train_end``. The values inside the strip are *never used* by
        Prophet's fit, so no information leaks.
        """
        self._history = history
        self._train_end = pd.Timestamp(train_end) if train_end is not None else history.index.max()
        self._actual = actual if actual is not None else history
        if actual is None:
            log.warning(
                "ProphetForecaster.fit() called without `actual=` — falling back to "
                "history. Forecast horizon will be the wrong window unless the caller "
                "passes the full series through the prediction end-date."
            )
        return self

    def predict(self, start: pd.Timestamp, end: pd.Timestamp, freq: str = "D") -> ForecastResult:
        if self._history is None or self._actual is None:
            raise RuntimeError("Call fit() first.")

        end_ts = pd.Timestamp(end)
        if self._actual.index.max() < end_ts:
            raise ValueError(
                f"`actual` ends at {self._actual.index.max().date()} but predict was "
                f"asked for {end_ts.date()}. Pass the full series through ``end`` to "
                "fit() so GluonTS can correctly identify the forecast horizon."
            )

        prediction_length = len(pd.date_range(start, end, freq=freq))
        predictor = ProphetPredictor(prediction_length=prediction_length)
        # Pass the series ending at `end` (= test_end). GluonTS strips the last
        # `prediction_length` values (= the test window) and fits Prophet on
        # the remainder (= ≤ train_end), then forecasts the stripped horizon.
        forecasts = predict_with(predictor, self._actual, end=end, freq=freq, num_samples=self.num_samples)
        mu, lo, hi = forecast_to_dfs(forecasts, list(self._history.columns), start, end, freq, self.coverage_level)
        return ForecastResult(mu=mu, lower=lo, upper=hi, coverage_level=self.coverage_level)
