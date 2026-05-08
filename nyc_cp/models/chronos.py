"""Chronos-2 forecaster — Amazon's zero-shot time-series foundation model.

Chronos-2 is encoder-only (~120M params) with cross-item attention, native
multivariate / covariate support, max prediction_length=1024, max context=8192.
We use ``predict_df`` which accepts the full panel at once and runs cross-item
in-context learning.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from nyc_cp.models.base import BaseForecaster, ForecastResult

log = logging.getLogger(__name__)


class ChronosForecaster(BaseForecaster):
    name = "chronos"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.model_id: str = str(config.get("model_id", "amazon/chronos-2"))
        self.device: str = str(config.get("device", "cuda"))
        self.dtype: str = str(config.get("dtype", "bfloat16"))
        self.batch_size: int = int(config.get("batch_size", 32))
        self._pipeline = None
        self._history: pd.DataFrame | None = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        import torch
        from chronos import BaseChronosPipeline

        torch_dtype = getattr(torch, self.dtype)
        log.info("Loading Chronos pipeline %s on %s (dtype=%s)", self.model_id, self.device, self.dtype)
        self._pipeline = BaseChronosPipeline.from_pretrained(
            self.model_id, device_map=self.device, torch_dtype=torch_dtype
        )
        return self._pipeline

    def fit(self, history: pd.DataFrame, **kwargs) -> "ChronosForecaster":
        """Chronos-2 is zero-shot — there's no training. We just stash the
        history and load the pipeline (downloads weights on first call)."""
        self._history = history
        self._load_pipeline()
        return self

    def predict(self, start: pd.Timestamp, end: pd.Timestamp, freq: str = "D") -> ForecastResult:
        if self._history is None or self._pipeline is None:
            raise RuntimeError("Call fit() first.")

        idx = pd.date_range(start=start, end=end, freq=freq)
        n = len(idx)
        alpha = 1.0 - self.coverage_level
        q_lo, q_hi = alpha / 2, 1.0 - alpha / 2

        # Wide → long for predict_df; one row per (date, series). We only feed
        # history (≤ train_end) — Chronos doesn't see the test window.
        long = (
            self._history.reset_index()
            .melt(id_vars=self._history.index.name or "index", var_name="item_id", value_name="target")
            .rename(columns={self._history.index.name or "index": "timestamp"})
            .dropna(subset=["target"])
        )
        long["timestamp"] = pd.to_datetime(long["timestamp"])

        log.info("Chronos predict: %d series × ~%d timesteps → horizon %d", long["item_id"].nunique(), len(long) // long["item_id"].nunique(), n)
        pred_df = self._pipeline.predict_df(
            long,
            prediction_length=n,
            quantile_levels=[q_lo, 0.5, q_hi],
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            batch_size=self.batch_size,
        )

        # Long → wide. Chronos returns columns like "0.05", "0.5", "0.95".
        col_lo = f"{q_lo}" if f"{q_lo}" in pred_df.columns else str(round(q_lo, 4))
        col_hi = f"{q_hi}" if f"{q_hi}" in pred_df.columns else str(round(q_hi, 4))
        col_mid = "0.5"

        def _pivot(value_col: str) -> pd.DataFrame:
            out = pred_df.pivot(index="timestamp", columns="item_id", values=value_col)
            out.index = pd.to_datetime(out.index)
            return out.reindex(idx).reindex(columns=self._history.columns)

        mu = _pivot(col_mid)
        lo = _pivot(col_lo)
        hi = _pivot(col_hi)
        return ForecastResult(mu=mu, lower=lo, upper=hi, coverage_level=self.coverage_level)
