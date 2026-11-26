"""Chronos-Bolt forecaster — Amazon's zero-shot time-series foundation model.

**Chronos-Bolt** (``amazon/chronos-bolt-base``, released 2024-11-26, i.e. before
the 2025-01-05 policy start) is univariate and trained to emit the 0.1 … 0.9
quantile grid with a native horizon of 64 steps; longer horizons are rolled out
autoregressively on the median. Because the model cannot emit 0.05 / 0.95
directly, we request 0.1 / 0.5 / 0.9 and extrapolate the 90% PI under a
Gaussian assumption: ``sigma = (q90 - q10) / (2 * 1.2816)``,
``lo/hi = q50 ∓ 1.6449 * sigma``. This keeps the ``coverage_level`` contract
consistent with every other forecaster (and with ``effects._recover_sigma``).
The rolling-origin pipeline in ``runs/revision/`` reads the native 0.1 / 0.9
quantiles directly instead of this approximation.

Chronos-2 (released 2025-10-20, after the policy took effect) is not supported.
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
        self.model_id: str = str(config.get("model_id", "amazon/chronos-bolt-base"))
        self.device: str = str(config.get("device", "cuda"))
        self.dtype: str = str(config.get("dtype", "bfloat16"))
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
        """Chronos-Bolt is zero-shot — there's no training. We just stash the
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

        # Wide → long for predict_df; one row per (date, series). We only feed
        # history (≤ train_end) — Chronos doesn't see the test window.
        long = (
            self._history.reset_index()
            .melt(id_vars=self._history.index.name or "index", var_name="item_id", value_name="target")
            .rename(columns={self._history.index.name or "index": "timestamp"})
            .dropna(subset=["target"])
        )
        long["timestamp"] = pd.to_datetime(long["timestamp"])

        n_series = long["item_id"].nunique()
        log.info("Chronos predict (%s): %d series × ~%d timesteps → horizon %d",
                 type(self._pipeline).__name__, n_series, len(long) // max(n_series, 1), n)

        if type(self._pipeline).__name__ != "ChronosBoltPipeline":
            raise ValueError(f"{self.model_id} is not a Chronos-Bolt checkpoint; only Chronos-Bolt is supported.")
        # Bolt: fixed 0.1..0.9 quantile grid. Ask for the 10/50/90 grid and
        # Gaussian-extrapolate to the requested PI.
        q_req = [0.1, 0.5, 0.9]

        pred_df = self._pipeline.predict_df(
            long,
            prediction_length=n,
            quantile_levels=q_req,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
        )

        def _col(q: float) -> str:
            return f"{q}" if f"{q}" in pred_df.columns else str(round(q, 4))

        def _pivot(value_col: str) -> pd.DataFrame:
            out = pred_df.pivot(index="timestamp", columns="item_id", values=value_col)
            out.index = pd.to_datetime(out.index)
            return out.reindex(idx).reindex(columns=self._history.columns)

        mu = _pivot(_col(0.5))
        from scipy.stats import norm

        q10, q90 = _pivot(_col(0.1)), _pivot(_col(0.9))
        sigma = (q90 - q10) / (2 * norm.ppf(0.9))
        z = norm.ppf(1.0 - alpha / 2)
        lo = mu - z * sigma
        hi = mu + z * sigma
        log.info("Bolt PI: extrapolated 0.1/0.9 grid to %.0f%% interval (z=%.3f)", 100 * self.coverage_level, z)
        return ForecastResult(mu=mu, lower=lo, upper=hi, coverage_level=self.coverage_level)
