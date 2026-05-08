"""Causal regression: VIF-filtered OLS / spatial-lag / spatial-error / ML models.

Endpoint: per-tract treatment effect (e.g. ``att_mean``); regressors are the
demographic, economic, and mobility indicators built in
:mod:`nyc_cp.analysis.demographics`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ----------------------------------------------------------------- VIF ---


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factor per column."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    return pd.DataFrame(
        {
            "variable": df.columns,
            "VIF": [variance_inflation_factor(df.values, i) for i in range(df.shape[1])],
        }
    )


def stepwise_vif_filter(df: pd.DataFrame, threshold: float = 10.0) -> tuple[list[str], list[str]]:
    """Drop the highest-VIF column repeatedly until all VIFs < ``threshold``.

    Returns ``(kept_columns, dropped_columns)``.
    """
    df = df.copy()
    dropped: list[str] = []
    while True:
        vif = compute_vif(df)
        worst = vif.loc[vif["VIF"].idxmax()]
        if worst["VIF"] < threshold:
            return df.columns.tolist(), dropped
        dropped.append(worst["variable"])
        log.info("VIF %.1f → drop %s", worst["VIF"], worst["variable"])
        df = df.drop(columns=[worst["variable"]])


# ----------------------------------------------------------- spatial reg ---


@dataclass
class SpatialResults:
    ols: Any
    ml_lag: Any
    ml_error: Any
    weights: Any
    feature_names: list[str] = field(default_factory=list)


def run_spatial_regression(
    gdf,
    y_col: str,
    x_cols: Sequence[str],
    weights: str = "Rook",
    transform: str = "r",
) -> SpatialResults:
    """Fit OLS, ML_Lag (spatial lag), and ML_Error (spatial error) models.

    ``gdf`` must be a GeoDataFrame with the regressors and outcome plus geometry.
    Rows with NaN in any of ``y_col`` / ``x_cols`` are dropped.
    """
    import libpysal
    from spreg import OLS, ML_Error, ML_Lag

    df = gdf.dropna(subset=[y_col, *x_cols]).reset_index(drop=True)
    y = df[y_col].to_numpy().reshape(-1, 1)
    X = df[list(x_cols)].to_numpy()

    if weights == "Rook":
        w = libpysal.weights.Rook.from_dataframe(df)
    elif weights == "Queen":
        w = libpysal.weights.Queen.from_dataframe(df)
    else:
        raise ValueError(f"Unknown weight matrix: {weights!r}")
    w.transform = transform

    name_x = list(x_cols)
    ols = OLS(y, X, w=w, spat_diag=True, moran=True, name_y=y_col, name_x=name_x)
    ml_lag = ML_Lag(y, X, w=w, name_y=y_col, name_x=name_x, name_w=weights)
    ml_err = ML_Error(y, X, w=w, name_y=y_col, name_x=name_x, name_w=weights)
    return SpatialResults(ols=ols, ml_lag=ml_lag, ml_error=ml_err, weights=w, feature_names=name_x)


# --------------------------------------------------------------- ML models ---


def run_tree_models(
    df: pd.DataFrame,
    y_col: str,
    x_cols: Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit Random Forest, XGBoost, LightGBM. Returns a metrics DataFrame.

    Rows: model. Columns: ``r2_train``, ``r2_test``, ``rmse_test``,
    ``feature_importance`` (dict).
    """
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor

    df = df.dropna(subset=[y_col, *x_cols]).reset_index(drop=True)
    X = df[list(x_cols)]
    y = df[y_col]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)

    models = {
        "random_forest": RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1),
        "xgboost": XGBRegressor(n_estimators=400, random_state=random_state, n_jobs=-1, verbosity=0),
        "lightgbm": LGBMRegressor(n_estimators=400, random_state=random_state, n_jobs=-1, verbose=-1),
    }
    rows = []
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred_tr = m.predict(X_tr)
        pred_te = m.predict(X_te)
        rows.append(
            dict(
                model=name,
                r2_train=r2_score(y_tr, pred_tr),
                r2_test=r2_score(y_te, pred_te),
                rmse_test=float(np.sqrt(mean_squared_error(y_te, pred_te))),
                feature_importance=dict(zip(x_cols, getattr(m, "feature_importances_", np.zeros(len(x_cols))))),
            )
        )
    return pd.DataFrame(rows).set_index("model")
