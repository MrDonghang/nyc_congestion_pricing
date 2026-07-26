"""SHAP analysis on tract-level ATT vs ACS demographics.

Loads the ``tract_effects.geojson`` produced by ``scripts.geospatial_analysis``
(per-tract ATT + ACS demographics, VIF-eligible features), refits an
XGBoost regressor, and writes the two publication-standard SHAP figures plus
a CSV of per-feature mean(|SHAP|) for table use.

Examples
--------
    python -m scripts.shap_analysis --mode bus --model chronos_qrcal_intercept --window test
    python -m scripts.shap_analysis --mode replica --model chronos_qrcal_intercept --window test --direction O
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nyc_cp.analysis import demographics
from nyc_cp.analysis.causal import stepwise_vif_filter
from nyc_cp.config import load_paths, normalize_window_name, output_dir
from nyc_cp.utils import setup_logging

log = logging.getLogger("shap_analysis")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["bus", "subway", "replica"])
    p.add_argument("--model", required=True)
    p.add_argument("--window", required=True, choices=["val", "validation", "test"])
    p.add_argument("--direction", choices=["all", "O", "D", "total"], default="all")
    p.add_argument("--y-col", default="att", help="Tract-level outcome.")
    p.add_argument("--vif-threshold", type=float, default=20.0)
    p.add_argument("--max-display", type=int, default=15, help="Top-N features in summary plots.")
    return p.parse_args()


def main() -> None:
    import geopandas as gpd  # noqa: F401  (delayed to keep import cost out of -h)
    import shap
    from lightgbm import LGBMRegressor

    args = parse_args()
    args.window = normalize_window_name(args.window)
    paths = load_paths()
    setup_logging(f"shap_{args.mode}_{args.model}_{args.window}", log_root=paths["log_root"])

    out_dir = output_dir(args.mode, args.model, direction=args.direction, paths=paths) / "causal"
    geo_path = out_dir / "tract_effects.geojson"
    if not geo_path.exists():
        raise SystemExit(f"Missing {geo_path} — run scripts.geospatial_analysis first.")

    import geopandas as gpd
    gdf = gpd.read_file(geo_path)
    log.info("Loaded %s: %d tracts", geo_path.name, len(gdf))

    # Same feature set + VIF filter as scripts/geospatial_analysis.py — keep
    # SHAP comparable to OLS/spatial regression in the paper.
    x_cols = sum([demographics.GROUPS[g] for g in
                  ["demographics", "race_ethnicity", "economics", "travel", "education", "housing"]], [])
    x_cols = [c for c in x_cols if c in gdf.columns]
    if args.y_col not in gdf.columns:
        raise SystemExit(f"y-column {args.y_col!r} not found. Available: {sorted(gdf.columns)}")

    df = gdf.dropna(subset=[args.y_col, *x_cols]).reset_index(drop=True)
    log.info("Sample size: %d", len(df))
    kept, dropped = stepwise_vif_filter(df[x_cols], threshold=args.vif_threshold)
    log.info("VIF kept %d / dropped %d", len(kept), len(dropped))

    X = df[kept]
    y = df[args.y_col]

    # LightGBM — paired with SHAP.TreeExplainer this is exact (no sampling).
    # Chosen over XGBoost only because shap 0.45.x can't parse XGBoost 3.x
    # serialised models; LGBM works seamlessly with shap.TreeExplainer.
    model = LGBMRegressor(n_estimators=400, random_state=42, n_jobs=8, verbose=-1)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # --- Figure 1: beeswarm summary (direction + magnitude per feature) ---
    plt.figure(figsize=(8, 8))
    shap.plots.beeswarm(shap_values, max_display=args.max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "shap_beeswarm.png", bbox_inches="tight", dpi=200)
    plt.close()
    log.info("Wrote %s.{pdf,png}", out_dir / "shap_beeswarm")

    # --- Figure 2: mean(|SHAP|) bar (clean feature-importance ranking) ---
    plt.figure(figsize=(8, 6))
    shap.plots.bar(shap_values, max_display=args.max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_bar.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "shap_bar.png", bbox_inches="tight", dpi=200)
    plt.close()
    log.info("Wrote %s.{pdf,png}", out_dir / "shap_bar")

    # --- Table: mean(|SHAP|) per feature, for inclusion in regression-table appendix ---
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": kept, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(out_dir / "shap_importance.csv", index=False)
    log.info("Wrote %s", out_dir / "shap_importance.csv")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
