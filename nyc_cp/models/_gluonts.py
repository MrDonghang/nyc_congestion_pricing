"""Shared GluonTS plumbing used by Prophet and DeepAR forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions

# GluonTS still calls deprecated np.bool internally on some versions.
np.bool = np.bool_  # type: ignore[attr-defined]


def to_listdataset(history: pd.DataFrame, end: pd.Timestamp, freq: str) -> ListDataset:
    """Wide DataFrame → GluonTS ListDataset of time series ending at ``end``."""
    items = []
    for col in history.columns:
        s = history[col].dropna()
        s = s[s.index <= end]
        items.append({"start": s.index[0], "target": s.to_numpy(dtype=float), "item_id": str(col)})
    return ListDataset(items, freq=freq)


def forecast_to_dfs(
    forecasts,
    columns: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str,
    coverage_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range(start=start, end=end, freq=freq)
    n = len(idx)
    alpha = 1.0 - coverage_level

    mu = pd.DataFrame(index=idx, columns=columns, dtype=float)
    lo = pd.DataFrame(index=idx, columns=columns, dtype=float)
    hi = pd.DataFrame(index=idx, columns=columns, dtype=float)

    for f, col in zip(forecasts, columns):
        mu[col] = f.mean[-n:]
        lo[col] = f.quantile(alpha / 2)[-n:]
        hi[col] = f.quantile(1 - alpha / 2)[-n:]
    return mu, lo, hi


def predict_with(predictor, history: pd.DataFrame, end: pd.Timestamp, freq: str, num_samples: int):
    ds = to_listdataset(history, end, freq=freq)
    forecast_it, _ = make_evaluation_predictions(dataset=ds, predictor=predictor, num_samples=num_samples)
    return list(forecast_it)
