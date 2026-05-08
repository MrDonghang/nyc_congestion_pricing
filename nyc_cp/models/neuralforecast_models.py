"""NHITS and TFT forecasters via Nixtla's neuralforecast library.

Both share the same long-format (unique_id, ds, y) wrapper. PIs come from
``DistributionLoss(distribution='Normal', level=[100*coverage_level])`` so
``coverage_level=0.9`` yields a Normal-distribution 90% PI.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from nyc_cp.models.base import BaseForecaster, ForecastResult

log = logging.getLogger(__name__)


def _wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    long = df.reset_index().melt(
        id_vars=df.index.name or "index", var_name="unique_id", value_name="y"
    ).rename(columns={df.index.name or "index": "ds"})
    long["ds"] = pd.to_datetime(long["ds"])
    long = long.dropna(subset=["y"])
    return long[["unique_id", "ds", "y"]]


class _NeuralForecastBase(BaseForecaster):
    """Shared NHITS/TFT plumbing.

    Subclasses set ``model_kind`` and ``_build_model(h, input_size, level, max_steps)``.
    """

    model_kind: str = ""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.input_size: int = int(config.get("input_size", 180))
        self.max_steps: int = int(config.get("max_steps", 1000))
        self.batch_size: int = int(config.get("batch_size", 32))
        self.learning_rate: float = float(config.get("learning_rate", 1e-3))
        self.hidden_size: int = int(config.get("hidden_size", 64))
        self.scaler_type: str = str(config.get("scaler_type", "standard"))
        self._nf = None
        self._history: pd.DataFrame | None = None
        self._horizon: int | None = None
        self._train_freq: str = "D"

    def _build_model(self, h: int, level: list[int]):
        raise NotImplementedError

    def fit(self, history: pd.DataFrame, prediction_length: int | None = None, **kwargs) -> "_NeuralForecastBase":
        from neuralforecast import NeuralForecast

        if prediction_length is None:
            raise ValueError(f"{type(self).__name__}.fit() requires prediction_length.")
        self._history = history
        self._horizon = prediction_length
        self._train_freq = self.config.get("freq", "D")

        long = _wide_to_long(history)
        # Coerce unique_id to string — neuralforecast prefers str ids.
        long["unique_id"] = long["unique_id"].astype(str)

        level_pct = [int(round(self.coverage_level * 100))]
        model = self._build_model(h=prediction_length, level=level_pct)
        log.info("Fitting %s on %d series × %d timesteps (h=%d, max_steps=%d)",
                 self.model_kind, long["unique_id"].nunique(), len(long), prediction_length, self.max_steps)
        self._nf = NeuralForecast(models=[model], freq=self._train_freq)
        self._nf.fit(df=long)
        return self

    def predict(self, start: pd.Timestamp, end: pd.Timestamp, freq: str = "D") -> ForecastResult:
        if self._nf is None or self._history is None:
            raise RuntimeError("Call fit() first.")
        idx = pd.date_range(start=start, end=end, freq=freq)
        n = len(idx)

        pred = self._nf.predict()
        pred = pred.rename(columns=str)
        pred["ds"] = pd.to_datetime(pred["ds"])

        level_pct = int(round(self.coverage_level * 100))
        # Column names depend on the model's class name in upper-case.
        kind = self.model_kind.upper()
        col_mu, col_lo, col_hi = kind, f"{kind}-lo-{level_pct}", f"{kind}-hi-{level_pct}"

        def _pivot(value_col: str) -> pd.DataFrame:
            out = pred.pivot(index="ds", columns="unique_id", values=value_col)
            out.index = pd.to_datetime(out.index)
            return out.reindex(idx).reindex(columns=[str(c) for c in self._history.columns])

        mu, lo, hi = _pivot(col_mu), _pivot(col_lo), _pivot(col_hi)
        # Restore original column dtype on the wide frame.
        mu.columns = self._history.columns
        lo.columns = self._history.columns
        hi.columns = self._history.columns
        mu, lo, hi = mu.iloc[:n], lo.iloc[:n], hi.iloc[:n]
        return ForecastResult(mu=mu, lower=lo, upper=hi, coverage_level=self.coverage_level)


class NHITSForecaster(_NeuralForecastBase):
    name = "nhits"
    model_kind = "NHITS"

    def _build_model(self, h: int, level: list[int]):
        from neuralforecast.losses.pytorch import DistributionLoss
        from neuralforecast.models import NHITS

        return NHITS(
            h=h,
            input_size=self.input_size,
            loss=DistributionLoss(distribution="Normal", level=level),
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            scaler_type=self.scaler_type,
            random_seed=int(self.config.get("seed", 42)),
        )


class TFTForecaster(_NeuralForecastBase):
    name = "tft"
    model_kind = "TFT"

    def _build_model(self, h: int, level: list[int]):
        from neuralforecast.losses.pytorch import DistributionLoss
        from neuralforecast.models import TFT

        return TFT(
            h=h,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            loss=DistributionLoss(distribution="Normal", level=level),
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            scaler_type=self.scaler_type,
            random_seed=int(self.config.get("seed", 42)),
        )
