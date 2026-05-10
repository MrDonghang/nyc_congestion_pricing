"""Calibrate a forecaster's val/test outputs via quantile regression.

Reads the saved forecast triplet for ``--base-model`` on val (calibration set)
and test (target), fits per-quantile residual regressions on val, applies them
to test, and saves a new forecast triplet under model name
``<base-model>_qrcal``. Run ``compute_effects --model <base-model>_qrcal`` to
get calibrated ATT/PI tables.

Example
-------
    python -m scripts.calibrate_forecast --mode subway --base-model chronos --direction O
    python -m scripts.compute_effects --mode subway --model chronos_qrcal --window test --direction O
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nyc_cp.analysis.effects import load_forecast_triplet
from nyc_cp.calibration import (
    apply_calibration,
    apply_intercept_plus_pooled_calibration,
    apply_per_unit_calibration,
    build_features,
    fit_calibration,
    fit_intercept_plus_pooled_calibration,
    fit_per_unit_calibration,
    predict_intercept_plus_pooled_deltas,
    predict_per_unit_deltas,
    residuals_long,
)
from nyc_cp.config import get_window, load_mode, load_paths, output_dir
from nyc_cp.data import load_actual
from nyc_cp.models.base import ForecastResult
from nyc_cp.utils import setup_logging

log = logging.getLogger("calibrate_forecast")

ID_COL_BY_MODE = {"bus": "route_id", "subway": "station_id", "citibike": "tract_id", "replica": "tract_id"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["bus", "subway", "citibike", "replica"])
    p.add_argument("--base-model", required=True, help="Base forecaster whose outputs we calibrate (e.g. chronos).")
    p.add_argument("--direction", choices=["all", "O", "D"], default="all")
    p.add_argument("--coverage", type=float, default=0.9)
    p.add_argument("--alpha", type=float, default=1e-4, help="L1 regularization for QuantileRegressor.")
    p.add_argument("--suffix", default=None,
                   help="Suffix appended to base model name. Defaults to qrcal (or qrcal_perunit with --per-unit).")
    p.add_argument("--val-kfold", type=int, default=0,
                   help="Folds for out-of-sample val calibration (date-stratified). 0 disables val output.")
    p.add_argument("--insample-val", action="store_true",
                   help="Apply the val-fitted calibration back to val in-sample (no CV). Mutually exclusive with --val-kfold>0.")
    p.add_argument("--per-unit", action="store_true",
                   help="Fit one QuantileRegressor set per unit (route/station/tract). "
                        "Units with fewer than --per-unit-min-obs val rows fall back to the pooled fit.")
    p.add_argument("--per-unit-min-obs", type=int, default=60,
                   help="Minimum val rows for a unit to be fit on its own; below this falls back to pooled.")
    p.add_argument("--per-unit-intercept", action="store_true",
                   help="Per-unit intercept (mean residual) + pooled QR on de-biased residuals. "
                        "Cheap intermediate between pooled and full per-unit. Mutually exclusive with --per-unit.")
    return p.parse_args()


def _diagnose_dow(actual: pd.DataFrame, mu: pd.DataFrame, label: str) -> None:
    common = actual.columns.intersection(mu.columns)
    tau = (actual[common] - mu[common]).mean(axis=1)
    cf = mu[common].mean(axis=1)
    df = pd.DataFrame({"tau": tau, "cf": cf})
    df["dow"] = df.index.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    g = df.groupby("dow")["tau"].mean().reindex(order)
    rel = (df.groupby("dow")["tau"].mean() / df.groupby("dow")["cf"].mean()).reindex(order)
    log.info("%s DOW residuals: %s", label, " ".join(f"{d[:3]}={v:+.1%}" for d, v in rel.items()))
    log.info("%s amplitude (max - min mean tau): %.0f", label, g.max() - g.min())


def main() -> None:
    args = parse_args()
    if args.insample_val and args.val_kfold > 0:
        raise SystemExit("--insample-val and --val-kfold>0 are mutually exclusive.")
    if args.per_unit and args.per_unit_intercept:
        raise SystemExit("--per-unit and --per-unit-intercept are mutually exclusive.")
    if args.suffix is None:
        if args.per_unit:
            args.suffix = "qrcal_perunit"
        elif args.per_unit_intercept:
            args.suffix = "qrcal_intercept"
        else:
            args.suffix = "qrcal"
    paths = load_paths()
    setup_logging(f"calibrate_{args.mode}_{args.base_model}_{args.direction}", log_root=paths["log_root"])

    id_col = ID_COL_BY_MODE[args.mode]
    mode_cfg = load_mode(args.mode)
    actual_full = load_actual(args.mode, direction=args.direction, mode_cfg=mode_cfg, paths=paths)

    val_w = get_window(args.mode, "val")
    test_w = get_window(args.mode, "test")
    val_train_end = pd.Timestamp(val_w.train_end)

    # Pre-policy history for level computation
    history_for_levels = actual_full.loc[actual_full.index <= val_train_end]
    log.info("Levels computed from %d days of pre-val history", len(history_for_levels))

    base_dir = output_dir(args.mode, args.base_model, direction=args.direction, paths=paths)
    suffix = f"_{args.direction}" if args.direction != "all" else ""

    val_prefix = f"{args.mode}_{args.base_model}_val{suffix}"
    test_prefix = f"{args.mode}_{args.base_model}_test{suffix}"
    val_fc = load_forecast_triplet(base_dir, val_prefix)
    test_fc = load_forecast_triplet(base_dir, test_prefix)

    # Slice actuals to each window
    a_val = actual_full.loc[(actual_full.index >= pd.Timestamp(val_w.test_start)) & (actual_full.index <= pd.Timestamp(val_w.test_end))]
    a_test = actual_full.loc[(actual_full.index >= pd.Timestamp(test_w.test_start)) & (actual_full.index <= pd.Timestamp(test_w.test_end))]

    # Diagnostics: BEFORE calibration
    _diagnose_dow(a_val, val_fc.mu, "val (pre-cal)")
    _diagnose_dow(a_test, test_fc.mu, "test (pre-cal)")

    common_val = a_val.columns.intersection(val_fc.mu.columns)
    e_val_pre = a_val[common_val].reindex(val_fc.mu.index) - val_fc.mu[common_val]
    rmse_val_pre = float(np.sqrt((e_val_pre ** 2).mean().mean()))
    inside_val_pre = (
        (a_val[common_val].reindex(val_fc.mu.index) >= val_fc.lower[common_val])
        & (a_val[common_val].reindex(val_fc.mu.index) <= val_fc.upper[common_val])
    ).mean().mean()
    log.info("Val pre-cal:  RMSE=%.0f  ECR=%.3f", rmse_val_pre, inside_val_pre)

    # Build features for val (fitting) and test (application)
    val_feat = build_features(history_for_levels, val_fc.mu, id_col=id_col)
    test_feat = build_features(history_for_levels, test_fc.mu, id_col=id_col)

    # val residuals (long-form, indexed by (date, unit_id))
    val_res = residuals_long(a_val, val_fc.mu, id_col=id_col)
    val_feat = val_feat.set_index(["date", id_col])
    common_idx = val_feat.index.intersection(val_res.index)
    val_feat_a = val_feat.loc[common_idx].reset_index()
    val_res_a = val_res.loc[common_idx]
    log.info("Val calibration sample: %d (date, unit) rows", len(val_res_a))

    # Map id_col → "unit_id" so the module's column conventions match
    val_feat_a = val_feat_a.rename(columns={id_col: "unit_id"})
    test_feat = test_feat.rename(columns={id_col: "unit_id"})

    alpha_pi = 1.0 - args.coverage
    q_lo, q_hi = alpha_pi / 2, 1.0 - alpha_pi / 2
    quantiles = (q_lo, 0.5, q_hi)

    cal = fit_calibration(val_feat_a, val_res_a, quantiles=quantiles, alpha=args.alpha)

    # Report fitted intercepts/coefs briefly
    for q, m in cal.models.items():
        log.info("q=%.2f  intercept=%.1f  ||coef||_1=%.1f", q, m.intercept_, np.abs(m.coef_).sum())

    # val_res_a's MultiIndex was built with the mode's native id_col name
    # (e.g. route_id). Rename to match the renamed feature panel.
    val_res_pu = val_res_a.copy()
    val_res_pu.index = val_res_pu.index.rename(["date", "unit_id"])

    pucal = ipcal = None
    if args.per_unit:
        log.info("Fitting per-unit calibration on top of pooled (fallback for sparse units).")
        pucal = fit_per_unit_calibration(
            val_feat_a, val_res_pu, fallback=cal,
            quantiles=quantiles, alpha=args.alpha,
            min_obs=args.per_unit_min_obs, id_col="unit_id",
        )
        mu_cal, lower_cal, upper_cal = apply_per_unit_calibration(
            pucal, test_feat, test_fc.mu, quantile_lo=q_lo, quantile_hi=q_hi, id_col="unit_id",
        )
    elif args.per_unit_intercept:
        log.info("Fitting per-unit intercept + pooled QR on de-biased residuals.")
        ipcal = fit_intercept_plus_pooled_calibration(
            val_feat_a, val_res_pu,
            quantiles=quantiles, alpha=args.alpha, id_col="unit_id",
        )
        mu_cal, lower_cal, upper_cal = apply_intercept_plus_pooled_calibration(
            ipcal, test_feat, test_fc.mu, quantile_lo=q_lo, quantile_hi=q_hi, id_col="unit_id",
        )
    else:
        mu_cal, lower_cal, upper_cal = apply_calibration(
            cal, test_feat, test_fc.mu, quantile_lo=q_lo, quantile_hi=q_hi, id_col="unit_id",
        )

    _diagnose_dow(a_test, mu_cal, "test (post-cal)")

    new_model = f"{args.base_model}_{args.suffix}"
    out_dir = output_dir(args.mode, new_model, direction=args.direction, paths=paths)
    new_prefix_test = f"{args.mode}_{new_model}_test{suffix}"
    ForecastResult(mu=mu_cal, lower=lower_cal, upper=upper_cal, coverage_level=args.coverage).save(out_dir, new_prefix_test)
    log.info("Saved calibrated test forecasts to %s/%s_*.csv", out_dir, new_prefix_test)

    common = a_test.columns.intersection(mu_cal.columns)
    e = (a_test[common] - mu_cal[common])
    rmse = float(np.sqrt((e ** 2).mean().mean()))
    inside = ((a_test[common] >= lower_cal[common]) & (a_test[common] <= upper_cal[common])).mean().mean()
    log.info("Test post-cal: RMSE=%.0f  ECR(empirical)=%.3f", rmse, inside)

    # ----- In-sample val calibration (no CV) -----
    if args.insample_val:
        log.info("Applying val-fitted calibration back to val in-sample (no CV)...")
        val_feat_is = build_features(history_for_levels, val_fc.mu, id_col=id_col).rename(columns={id_col: "unit_id"})
        if args.per_unit:
            v_mu, v_lo, v_hi = apply_per_unit_calibration(
                pucal, val_feat_is, val_fc.mu, quantile_lo=q_lo, quantile_hi=q_hi, id_col="unit_id",
            )
        elif args.per_unit_intercept:
            v_mu, v_lo, v_hi = apply_intercept_plus_pooled_calibration(
                ipcal, val_feat_is, val_fc.mu, quantile_lo=q_lo, quantile_hi=q_hi, id_col="unit_id",
            )
        else:
            v_mu, v_lo, v_hi = apply_calibration(
                cal, val_feat_is, val_fc.mu, quantile_lo=q_lo, quantile_hi=q_hi, id_col="unit_id",
            )
        new_prefix_val = f"{args.mode}_{new_model}_val{suffix}"
        ForecastResult(mu=v_mu, lower=v_lo, upper=v_hi, coverage_level=args.coverage).save(out_dir, new_prefix_val)
        log.info("Saved in-sample val calibrated forecasts to %s/%s_*.csv", out_dir, new_prefix_val)

        _diagnose_dow(a_val, v_mu, "val (post-cal, in-sample)")
        a_val_a = a_val.reindex(v_mu.index)[v_mu.columns]
        ev = a_val_a - v_mu
        v_rmse = float(np.sqrt((ev ** 2).mean().mean()))
        v_inside = ((a_val_a >= v_lo) & (a_val_a <= v_hi)).mean().mean()
        log.info("Val post-cal (in-sample): RMSE=%.0f  ECR=%.3f", v_rmse, v_inside)
        print(f"Val pre-cal:                RMSE={rmse_val_pre:.0f}  ECR={inside_val_pre:.3f}")
        print(f"Val post-cal (in-sample):   RMSE={v_rmse:.0f}  ECR={v_inside:.3f}")

    # ----- OOS val calibration via k-fold (date-stratified) -----
    if args.val_kfold > 0:
        from sklearn.model_selection import KFold

        log.info("Running %d-fold CV on val for honest out-of-sample calibration...", args.val_kfold)
        unique_dates = np.sort(val_fc.mu.index.unique())
        rng = np.random.default_rng(0)
        order = rng.permutation(len(unique_dates))
        folds = np.array_split(unique_dates[order], args.val_kfold)

        # We re-derive features for the WHOLE val and do per-fold prediction
        val_feat_full = build_features(history_for_levels, val_fc.mu, id_col=id_col).rename(columns={id_col: "unit_id"})
        val_res_full = residuals_long(a_val, val_fc.mu, id_col=id_col)
        val_feat_full = val_feat_full.set_index(["date", "unit_id"])
        common_idx_full = val_feat_full.index.intersection(val_res_full.index)
        val_feat_full = val_feat_full.loc[common_idx_full]
        val_res_full = val_res_full.loc[common_idx_full]

        oos_med = pd.Series(np.nan, index=common_idx_full)
        oos_lo = pd.Series(np.nan, index=common_idx_full)
        oos_hi = pd.Series(np.nan, index=common_idx_full)

        for k, fold_dates in enumerate(folds):
            test_mask = val_feat_full.index.get_level_values("date").isin(fold_dates)
            train_X = val_feat_full.loc[~test_mask].reset_index()
            train_y = val_res_full.loc[~test_mask]
            test_X = val_feat_full.loc[test_mask].reset_index()
            log.info("  fold %d/%d: train=%d  test=%d (dates=%d)", k + 1, args.val_kfold, len(train_y), test_mask.sum(), len(fold_dates))
            cal_k = fit_calibration(train_X, train_y, quantiles=quantiles, alpha=args.alpha)

            if args.per_unit:
                # Per-fold per-unit fit using cal_k (this fold's pooled cal) as fallback.
                # Units with < per_unit_min_obs train rows in this fold fall back to cal_k.
                train_y_pu = train_y.copy()
                train_y_pu.index = train_y_pu.index.rename(["date", "unit_id"])
                pucal_k = fit_per_unit_calibration(
                    train_X, train_y_pu, fallback=cal_k,
                    quantiles=quantiles, alpha=args.alpha,
                    min_obs=args.per_unit_min_obs, id_col="unit_id",
                )
                deltas_k = predict_per_unit_deltas(pucal_k, test_X, id_col="unit_id")
            elif args.per_unit_intercept:
                train_y_pu = train_y.copy()
                train_y_pu.index = train_y_pu.index.rename(["date", "unit_id"])
                ipcal_k = fit_intercept_plus_pooled_calibration(
                    train_X, train_y_pu,
                    quantiles=quantiles, alpha=args.alpha, id_col="unit_id",
                )
                deltas_k = predict_intercept_plus_pooled_deltas(ipcal_k, test_X, id_col="unit_id")
            else:
                deltas_k = cal_k.predict_deltas(test_X)

            arr = np.sort(np.stack([deltas_k[q] for q in sorted(deltas_k)], axis=0), axis=0)
            qs_sorted = sorted(deltas_k)
            d_lo = arr[qs_sorted.index(q_lo)]
            d_med = arr[qs_sorted.index(0.5)]
            d_hi = arr[qs_sorted.index(q_hi)]
            keys = list(zip(test_X["date"], test_X["unit_id"]))
            for key, dl, dm, dh in zip(keys, d_lo, d_med, d_hi):
                oos_lo.at[key] = dl
                oos_med.at[key] = dm
                oos_hi.at[key] = dh

        # Build val mu_cal/lower/upper by adding deltas to mu
        mu_long = val_feat_full[["mu_pred"]].copy()
        mu_long["mu_cal"] = mu_long["mu_pred"] + oos_med
        mu_long["lower_cal"] = mu_long["mu_pred"] + oos_lo
        mu_long["upper_cal"] = mu_long["mu_pred"] + oos_hi
        mu_long = mu_long.reset_index()

        def _pivot(col):
            wide = mu_long.pivot(index="date", columns="unit_id", values=col)
            return wide.reindex(val_fc.mu.index).reindex(columns=val_fc.mu.columns)

        v_mu, v_lo, v_hi = _pivot("mu_cal"), _pivot("lower_cal"), _pivot("upper_cal")
        new_prefix_val = f"{args.mode}_{new_model}_val{suffix}"
        ForecastResult(mu=v_mu, lower=v_lo, upper=v_hi, coverage_level=args.coverage).save(out_dir, new_prefix_val)
        log.info("Saved OOS val calibrated forecasts to %s/%s_*.csv", out_dir, new_prefix_val)

        _diagnose_dow(a_val, v_mu, "val (post-cal, OOS)")
        ev = (a_val.reindex(v_mu.index)[v_mu.columns] - v_mu)
        v_rmse = float(np.sqrt((ev ** 2).mean().mean()))
        v_inside = ((a_val.reindex(v_mu.index)[v_mu.columns] >= v_lo) & (a_val.reindex(v_mu.index)[v_mu.columns] <= v_hi)).mean().mean()
        log.info("Val post-cal (OOS): RMSE=%.0f  ECR=%.3f", v_rmse, v_inside)
        print(f"Val OOS calibrated:  RMSE={v_rmse:.0f}  ECR={v_inside:.3f}")

    print(f"\nDone. Calibrated test: RMSE={rmse:.0f}  ECR={inside:.3f}")
    print(f"Run: python -m scripts.compute_effects --mode {args.mode} --model {new_model} --window {{val,test}}"
          + (f" --direction {args.direction}" if args.direction != 'all' else ""))


if __name__ == "__main__":
    main()
