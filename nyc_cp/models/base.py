"""BaseForecaster — the contract every model in this package satisfies.

A forecaster is fitted on a wide-format history DataFrame (index = dates,
columns = series IDs) and produces three aligned DataFrames for a future
window: predicted mean and the lower / upper quantiles of a prediction
interval at ``coverage_level``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ForecastResult:
    """Result of a forecast over a fixed horizon."""

    mu: pd.DataFrame      # date × series — predicted mean
    lower: pd.DataFrame   # date × series — lower bound of PI
    upper: pd.DataFrame   # date × series — upper bound of PI
    coverage_level: float

    def save(self, output_dir: Path, prefix: str) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.mu.to_csv(output_dir / f"{prefix}_mu.csv")
        self.lower.to_csv(output_dir / f"{prefix}_lower.csv")
        self.upper.to_csv(output_dir / f"{prefix}_upper.csv")

    @classmethod
    def load(cls, output_dir: Path, prefix: str, coverage_level: float = 0.9) -> "ForecastResult":
        return cls(
            mu=pd.read_csv(output_dir / f"{prefix}_mu.csv", index_col=0, parse_dates=[0]),
            lower=pd.read_csv(output_dir / f"{prefix}_lower.csv", index_col=0, parse_dates=[0]),
            upper=pd.read_csv(output_dir / f"{prefix}_upper.csv", index_col=0, parse_dates=[0]),
            coverage_level=coverage_level,
        )


class BaseForecaster(ABC):
    """Common interface for ARIMA, Prophet, DeepAR, and PCN."""

    name: str = "base"
    supports_checkpoints: bool = False

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.coverage_level: float = float(config.get("coverage_level", 0.9))

    @abstractmethod
    def fit(self, history: pd.DataFrame, **kwargs) -> "BaseForecaster":
        """Fit on history. ``history`` is wide-format (date × series)."""

    @abstractmethod
    def predict(self, start: pd.Timestamp, end: pd.Timestamp, freq: str = "D") -> ForecastResult:
        """Forecast over ``[start, end]`` inclusive."""

    def fit_predict(
        self,
        history: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        freq: str = "D",
        **fit_kwargs,
    ) -> ForecastResult:
        self.fit(history, **fit_kwargs)
        return self.predict(start, end, freq=freq)

    def save_checkpoint(self, path: Path) -> None:
        """Persist trained model state so a future run can skip ``fit()``.

        Override in subclasses that support it; opt in via
        ``supports_checkpoints = True`` so the CLI knows whether to offer
        ``--from-checkpoint`` for this model.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support checkpointing.")

    def load_checkpoint(
        self,
        path: Path,
        history: pd.DataFrame,
        train_end: pd.Timestamp,
        prediction_length: int,
    ) -> "BaseForecaster":
        """Restore trained state from disk. ``history`` must be the same
        date-filtered frame that was passed to the original ``fit()`` so
        ``predict()`` sees the right tail."""
        raise NotImplementedError(f"{type(self).__name__} does not support checkpointing.")
