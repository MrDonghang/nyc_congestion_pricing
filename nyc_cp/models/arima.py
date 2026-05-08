"""ARIMA forecaster — one univariate model per series, auto-order via pmdarima."""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

from nyc_cp.models.base import BaseForecaster, ForecastResult

# pmdarima warns on every fit; statsmodels emits ConvergenceWarning /
# UserWarning ('Non-stationary AR' etc.) per series. None of these are
# actionable here — the fallback path already handles real fit failures.
warnings.filterwarnings("ignore", category=FutureWarning)

log = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1


class ArimaForecaster(BaseForecaster):
    name = "arima"
    supports_checkpoints = True

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.seasonal: bool = bool(config.get("seasonal", False))
        self.fallback_order: tuple = tuple(config.get("fallback_order", (1, 1, 1)))
        self._fitted: dict[str, Any] = {}
        self._orders: dict[str, tuple] = {}
        self._history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame, **kwargs) -> "ArimaForecaster":
        from pmdarima.arima import auto_arima

        self._history = history.copy()
        self._fitted = {}
        self._orders = {}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")  # silence every per-fit warning
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
                    fitted = ARIMA(series, order=order).fit()
                except Exception as e:
                    tqdm.write(f"  {col}: order={order} fit failed ({e}); falling back to {self.fallback_order}")
                    order = self.fallback_order
                    fitted = ARIMA(series, order=order).fit()

                self._fitted[col] = fitted
                self._orders[col] = order
                tqdm.write(f"  {col}: order={order}")

        log.info("Fitted ARIMA on %d series.", len(self._fitted))
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
            for col, model in self._fitted.items():
                res = model.get_forecast(steps=n)
                # `.conf_int` returns either DataFrame or ndarray depending on
                # statsmodels version / whether the input had a date index.
                ci = res.conf_int(alpha=alpha)
                ci_arr = ci.to_numpy() if hasattr(ci, "to_numpy") else np.asarray(ci)

                mu_arr = np.asarray(res.predicted_mean)
                mu[col] = mu_arr
                lo[col] = ci_arr[:, 0]
                hi[col] = ci_arr[:, 1]

        return ForecastResult(mu=mu, lower=lo, upper=hi, coverage_level=self.coverage_level)

    # ------------------------------------------------------------ checkpoint ---

    def save_checkpoint(self, path: Path) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before save_checkpoint().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "fitted": self._fitted,           # statsmodels ARIMAResultsWrapper objects
            "orders": self._orders,
            "columns": list(self._fitted.keys()),
            "seasonal": self.seasonal,
            "fallback_order": self.fallback_order,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("ARIMA checkpoint saved to %s (%d series)", path, len(state["columns"]))

        # Also dump the orders as a human-readable CSV alongside the .pkl —
        # the original repo printed these per-route; this preserves them.
        orders_csv = path.with_suffix(".orders.csv")
        pd.DataFrame(
            [{"series": c, "p": o[0], "d": o[1], "q": o[2]} for c, o in self._orders.items()]
        ).to_csv(orders_csv, index=False)

    def load_checkpoint(
        self,
        path: Path,
        history: pd.DataFrame,
        train_end: pd.Timestamp,
        prediction_length: int,
    ) -> "ArimaForecaster":
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)
        if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"ARIMA checkpoint schema {state.get('schema_version')} != expected "
                f"{CHECKPOINT_SCHEMA_VERSION}. Re-train this model."
            )

        missing = [c for c in state["columns"] if c not in history.columns]
        if missing:
            raise ValueError(
                f"Checkpoint has {len(missing)} series not present in current history "
                f"(first few: {missing[:3]})."
            )

        self._fitted = state["fitted"]
        self._orders = state["orders"]
        self._history = history
        log.info("ARIMA checkpoint loaded from %s (%d series)", path, len(self._fitted))
        return self
