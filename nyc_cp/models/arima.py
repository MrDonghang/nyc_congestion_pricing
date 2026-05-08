"""ARIMA forecaster — one univariate model per series, auto-order via pmdarima."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

from nyc_cp.models.base import BaseForecaster, ForecastResult

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger(__name__)


class ArimaForecaster(BaseForecaster):
    name = "arima"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.seasonal: bool = bool(config.get("seasonal", False))
        self.fallback_order = tuple(config.get("fallback_order", (1, 1, 1)))
        self._fitted: dict[str, Any] = {}
        self._history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame, **kwargs) -> "ArimaForecaster":
        from pmdarima.arima import auto_arima

        self._history = history.copy()
        self._fitted = {}

        for col in tqdm(history.columns, desc="ARIMA fit"):
            series = history[col].dropna().to_numpy(dtype=float)
            try:
                auto = auto_arima(
                    series,
                    seasonal=self.seasonal,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=False,
                )
                order = auto.order
            except Exception:
                order = self.fallback_order

            try:
                self._fitted[col] = ARIMA(series, order=order).fit()
            except Exception as e:
                log.warning("ARIMA fit failed for %s (order=%s): %s; using fallback", col, order, e)
                self._fitted[col] = ARIMA(series, order=self.fallback_order).fit()
        return self

    def predict(self, start: pd.Timestamp, end: pd.Timestamp, freq: str = "D") -> ForecastResult:
        if self._history is None:
            raise RuntimeError("Call fit() first.")
        idx = pd.date_range(start=start, end=end, freq=freq)
        n = len(idx)
        alpha = 1.0 - self.coverage_level

        mu = pd.DataFrame(index=idx, columns=self._history.columns, dtype=float)
        lo = pd.DataFrame(index=idx, columns=self._history.columns, dtype=float)
        hi = pd.DataFrame(index=idx, columns=self._history.columns, dtype=float)

        for col, model in self._fitted.items():
            res = model.get_forecast(steps=n)
            ci = res.conf_int(alpha=alpha)
            mu[col] = np.asarray(res.predicted_mean)
            lo[col] = np.asarray(ci.iloc[:, 0])
            hi[col] = np.asarray(ci.iloc[:, 1])

        return ForecastResult(mu=mu, lower=lo, upper=hi, coverage_level=self.coverage_level)
