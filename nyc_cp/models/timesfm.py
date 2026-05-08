"""TimesFM 2.0 forecaster — Google's zero-shot time-series foundation model.

500M params, 2048 context, 10-quantile head trained at (0.1, 0.2, ..., 0.9).
We approximate arbitrary coverage levels by Gaussian-fitting the (q10, q50,
q90) triple: sigma ≈ (q90 - q10) / (2 × 1.2816). At coverage_level=0.9 this
gives PIs slightly wider than the native 80% Q10/Q90 — closer to a "true 90%"
under approximate Gaussianity. Documented for reproducibility.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from nyc_cp.models.base import BaseForecaster, ForecastResult

log = logging.getLogger(__name__)

# freq codes per timesfm: 0 = high-freq (sub-hour ~ daily), 1 = monthly/quarterly, 2 = yearly+.
_FREQ_CODE = {"D": 0, "W": 0, "B": 0, "H": 0, "T": 0, "M": 1, "Q": 1, "Y": 2, "A": 2}


def _freq_code(freq: str) -> int:
    return _FREQ_CODE.get(freq.upper()[0], 0)


class TimesFMForecaster(BaseForecaster):
    name = "timesfm"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.model_id: str = str(config.get("model_id", "google/timesfm-2.0-500m-pytorch"))
        self.context_len: int = int(config.get("context_len", 2048))
        self.num_layers: int = int(config.get("num_layers", 50))
        self.model_dims: int = int(config.get("model_dims", 1280))
        self.use_positional_embedding: bool = bool(config.get("use_positional_embedding", False))
        self.batch_size: int = int(config.get("batch_size", 32))
        self.backend: str = str(config.get("backend", "gpu"))
        self._tfm = None
        self._history: pd.DataFrame | None = None

    def _load(self, horizon: int):
        if self._tfm is not None:
            return self._tfm
        from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams

        log.info("Loading TimesFM %s (backend=%s, ctx=%d, horizon=%d)",
                 self.model_id, self.backend, self.context_len, horizon)
        self._tfm = TimesFm(
            hparams=TimesFmHparams(
                backend=self.backend,
                per_core_batch_size=self.batch_size,
                horizon_len=horizon,
                context_len=self.context_len,
                num_layers=self.num_layers,
                model_dims=self.model_dims,
                use_positional_embedding=self.use_positional_embedding,
            ),
            checkpoint=TimesFmCheckpoint(huggingface_repo_id=self.model_id),
        )
        return self._tfm

    def fit(self, history: pd.DataFrame, prediction_length: int | None = None, **kwargs) -> "TimesFMForecaster":
        """TimesFM is zero-shot. We stash history and prep the model with the
        right horizon (horizon_len is fixed at construction)."""
        self._history = history
        if prediction_length is None:
            raise ValueError("TimesFM.fit() requires prediction_length so horizon_len can be set at load time.")
        self._load(horizon=prediction_length)
        return self

    def predict(self, start: pd.Timestamp, end: pd.Timestamp, freq: str = "D") -> ForecastResult:
        if self._history is None or self._tfm is None:
            raise RuntimeError("Call fit() first.")

        idx = pd.date_range(start=start, end=end, freq=freq)
        n = len(idx)

        # Build a list of float arrays — one per series (NaN-stripped).
        cols = list(self._history.columns)
        arrays = [self._history[c].dropna().to_numpy(dtype=float) for c in cols]
        freq_codes = [_freq_code(freq)] * len(arrays)

        log.info("TimesFM forecast: %d series × ctx≤%d → horizon %d", len(arrays), self.context_len, n)
        point_forecast, quantile_forecast = self._tfm.forecast(arrays, freq=freq_codes)
        # quantile_forecast: shape (B, H, 1+9)  → indices 0=mean, 1=q10, 2=q20, ..., 9=q90
        point = np.asarray(point_forecast)         # (B, H)
        q = np.asarray(quantile_forecast)          # (B, H, 10)
        q10 = q[..., 1]
        q50 = q[..., 5]
        q90 = q[..., 9]

        # Approximate sigma from (q90 - q10) / (2 * 1.2816), then build PI at
        # the requested coverage_level around q50 (median).
        sigma = (q90 - q10) / (2.0 * norm.ppf(0.9))
        z = norm.ppf(1 - (1 - self.coverage_level) / 2)
        lo_arr = q50 - z * sigma
        hi_arr = q50 + z * sigma

        mu = pd.DataFrame(point.T[:n], index=idx, columns=cols)  # point forecast (mean per hparams default)
        lo = pd.DataFrame(lo_arr.T[:n], index=idx, columns=cols)
        hi = pd.DataFrame(hi_arr.T[:n], index=idx, columns=cols)
        return ForecastResult(mu=mu, lower=lo, upper=hi, coverage_level=self.coverage_level)
