"""Forecasting models.

All forecasters implement :class:`nyc_cp.models.base.BaseForecaster`. Use
:func:`build_forecaster` to construct one from a model-config dict.
"""

from __future__ import annotations

from typing import Any

from nyc_cp.models.base import BaseForecaster, ForecastResult

__all__ = ["BaseForecaster", "ForecastResult", "build_forecaster"]


def build_forecaster(model_config: dict[str, Any]) -> BaseForecaster:
    kind = model_config["type"]
    if kind == "arima":
        from nyc_cp.models.arima import ArimaForecaster

        return ArimaForecaster(model_config)
    if kind == "prophet":
        from nyc_cp.models.prophet import ProphetForecaster

        return ProphetForecaster(model_config)
    if kind == "deepar":
        from nyc_cp.models.deepar import DeepARForecaster

        return DeepARForecaster(model_config)
    if kind == "pcn":
        from nyc_cp.models.pcn import PCNForecaster

        return PCNForecaster(model_config)
    raise ValueError(f"Unknown model type: {kind!r}")
