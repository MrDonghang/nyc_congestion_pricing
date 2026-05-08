"""BSTS forecaster — Bayesian Structural Time Series via statsmodels.

This is BSTS *without control regressors* — purely structural extrapolation,
not the Google CausalImpact-style "y_t = trend + season + β·X_control_t"
setup. We don't use a donor pool; each series is decomposed independently.
For this codebase that means BSTS plays the same role as Prophet/ARIMA: a
structural univariate baseline whose post-policy forecast is the
counterfactual.

One UnobservedComponents model per series. Components:
- ``level``: local linear trend by default (μ_{t+1} = μ_t + β_t + η; β random walk).
- ``seasonal``: weekly (period=7) for daily data, yearly (period=52) for weekly.
  Disable with ``seasonal_period: null`` if your series has no obvious cycle.
- ``cycle``: optional smooth cycle on top of trend+seasonal.

The Kalman filter gives analytic prediction intervals — no MCMC, fast.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents
from tqdm import tqdm

from nyc_cp.models.base import BaseForecaster, ForecastResult

log = logging.getLogger(__name__)


class BSTSForecaster(BaseForecaster):
    name = "bsts"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.level: str = str(config.get("level", "local linear trend"))
        # None / 0 disables the seasonal component.
        sp = config.get("seasonal_period")
        self.seasonal_period: int | None = int(sp) if sp not in (None, 0, "null") else None
        self.cycle: bool = bool(config.get("cycle", False))
        self.stochastic_seasonal: bool = bool(config.get("stochastic_seasonal", True))
        self.stochastic_cycle: bool = bool(config.get("stochastic_cycle", True))
        self._fitted: dict[str, Any] = {}
        self._history: pd.DataFrame | None = None

    def _auto_seasonal_period(self, freq: str) -> int | None:
        """If user didn't set seasonal_period, pick a sensible default from freq."""
        if self.seasonal_period is not None:
            return self.seasonal_period
        f = freq.upper()
        if f.startswith("D"):
            return 7
        if f.startswith("W"):
            return 52
        if f.startswith("M"):
            return 12
        return None

    def fit(self, history: pd.DataFrame, **kwargs) -> "BSTSForecaster":
        self._history = history.copy()
        self._fitted = {}
        freq = self.config.get("freq", "D")
        period = self._auto_seasonal_period(freq)
        if period is not None and period >= len(history):
            log.warning("seasonal_period=%d >= history length %d; disabling seasonality.", period, len(history))
            period = None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for col in tqdm(history.columns, desc="BSTS fit"):
                series = history[col].dropna().to_numpy(dtype=float)
                try:
                    model = UnobservedComponents(
                        series,
                        level=self.level,
                        seasonal=period,
                        stochastic_seasonal=self.stochastic_seasonal,
                        cycle=self.cycle,
                        stochastic_cycle=self.stochastic_cycle if self.cycle else False,
                    )
                    fitted = model.fit(disp=False, maxiter=200)
                except Exception as e:
                    tqdm.write(f"  {col}: fit failed ({e}); falling back to local level + no seasonal")
                    model = UnobservedComponents(series, level="local level", seasonal=None)
                    fitted = model.fit(disp=False, maxiter=200)
                self._fitted[col] = fitted

        log.info("Fitted BSTS on %d series (level=%s, seasonal=%s, cycle=%s).",
                 len(self._fitted), self.level, period, self.cycle)
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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for col, fitted in self._fitted.items():
                fc = fitted.get_forecast(steps=n)
                ci = fc.conf_int(alpha=alpha)
                ci_arr = ci.to_numpy() if hasattr(ci, "to_numpy") else np.asarray(ci)
                mu[col] = np.asarray(fc.predicted_mean)
                lo[col] = ci_arr[:, 0]
                hi[col] = ci_arr[:, 1]

        return ForecastResult(mu=mu, lower=lo, upper=hi, coverage_level=self.coverage_level)
