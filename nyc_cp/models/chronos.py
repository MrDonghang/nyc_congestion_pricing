"""Chronos forecaster — Amazon's zero-shot time-series foundation models.

Two checkpoints are wired up through the same class (``type: chronos``):

* **Chronos-2** (``amazon/chronos-2``, released 2025-10-20) — encoder-only
  (~120M params) with cross-item attention; ``predict_df`` runs the whole
  panel at once with cross-item in-context learning. Released AFTER the
  2025-01-05 policy start, so it is no longer used for headline results.
* **Chronos-Bolt** (``amazon/chronos-bolt-base``, released 2024-11-26) —
  univariate, trained to emit the 0.1 … 0.9 quantile grid, native horizon 64
  (longer horizons are rolled out autoregressively on the median). Because
  the model cannot emit 0.05 / 0.95 directly, we request 0.1 / 0.5 / 0.9 and
  extrapolate the 90% PI under a Gaussian assumption:
  ``sigma = (q90 - q10) / (2 * 1.2816)``, ``lo/hi = q50 ∓ 1.6449 * sigma``.
  This keeps the ``coverage_level`` contract consistent with every other
  forecaster (and with ``effects._recover_sigma``). Downstream HQC
  calibration rebuilds the PI from residual quantiles anyway, so the
  approximation only affects the *raw* Bolt interval.
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

        n_series = long["item_id"].nunique()
        log.info("Chronos predict (%s): %d series × ~%d timesteps → horizon %d",
                 type(self._pipeline).__name__, n_series, len(long) // max(n_series, 1), n)

        is_bolt = type(self._pipeline).__name__ == "ChronosBoltPipeline"
        if is_bolt:
            # Bolt: fixed 0.1..0.9 quantile grid, no batch_size kwarg. Ask for
            # the 10/50/90 grid and Gaussian-extrapolate to the requested PI.
            q_req = [0.1, 0.5, 0.9]
            extra = {}
        else:
            q_req = [q_lo, 0.5, q_hi]
            extra = {"batch_size": self.batch_size}

        pred_df = self._pipeline.predict_df(
            long,
            prediction_length=n,
            quantile_levels=q_req,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            **extra,
        )

        def _col(q: float) -> str:
            return f"{q}" if f"{q}" in pred_df.columns else str(round(q, 4))

        def _pivot(value_col: str) -> pd.DataFrame:
            out = pred_df.pivot(index="timestamp", columns="item_id", values=value_col)
            out.index = pd.to_datetime(out.index)
            return out.reindex(idx).reindex(columns=self._history.columns)

        mu = _pivot(_col(0.5))
        if is_bolt:
            from scipy.stats import norm

            q10, q90 = _pivot(_col(0.1)), _pivot(_col(0.9))
            sigma = (q90 - q10) / (2 * norm.ppf(0.9))
            z = norm.ppf(1.0 - alpha / 2)
            lo = mu - z * sigma
            hi = mu + z * sigma
            log.info("Bolt PI: extrapolated 0.1/0.9 grid to %.0f%% interval (z=%.3f)", 100 * self.coverage_level, z)
        else:
            lo = _pivot(_col(q_lo))
            hi = _pivot(_col(q_hi))
        return ForecastResult(mu=mu, lower=lo, upper=hi, coverage_level=self.coverage_level)
