"""Mirror the key per-(mode, model, direction) artefacts produced by
``compare_models``, ``geospatial_analysis`` and ``06_residual_diagnostic``
from ``<output_root>`` into the repo's ``outputs/`` tree, so reviewers can
browse without reaching into ``/public_dataset``.

Layout produced under ``outputs/figures/<mode>/``:
    val_metrics_<window>[_<dir>].csv      ← compare_models
    causal_significance/<dir>_<model>.png ← geospatial_analysis
    causal_significance/crz_<dir>_<model>.csv
    causal_significance/regression_<dir>_<model>.txt
    residual/...                          ← already written by 06_residual_diagnostic
    att_summary.csv                       ← already written by 06

Plus a top-level ``outputs/_summary/`` index linking each file by short
name for cross-mode navigation.
"""
from __future__ import annotations

import shutil
from pathlib import Path
import pandas as pd

from nyc_cp.config import load_paths

paths = load_paths()
out_root = Path(paths["output_root"])
REPO = Path("/home/donghang/nyc_congestion_pricing")
FIG_ROOT = REPO / "outputs" / "figures"
SUMMARY = REPO / "outputs" / "_summary"
SUMMARY.mkdir(parents=True, exist_ok=True)

HEADLINE = ["chronos", "chronos_qrcal", "chronos_qrcal_intercept",
            "timesfm", "timesfm_qrcal", "timesfm_qrcal_intercept"]

MODE_DIRS = {"bus": ["all"], "subway": ["O", "D", "total"], "replica": ["O", "D"], "citibike": ["O", "D"]}

# Citibike has only raw chronos + timesfm because val/test seasonal misalignment
# (data starts 2024-01) makes calibration produce artifacts (~+148% rel_eff).
# Headline list per mode:
HEADLINE_BY_MODE = {
    "bus":      HEADLINE,
    "subway":   HEADLINE,
    "replica":  HEADLINE,
    "citibike": ["chronos", "timesfm"],
}


def _safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def mirror_compare_models() -> None:
    """Copy compare_<mode>_<win>[_<dir>].csv from output_new to outputs/_summary/."""
    for mode, dirs in MODE_DIRS.items():
        src_summary = out_root / mode / "_summary"
        if not src_summary.exists(): continue
        for f in src_summary.glob("compare_*.csv"):
            dst = SUMMARY / f.name
            shutil.copyfile(f, dst)


def mirror_geospatial() -> None:
    """For each headline model × direction, copy:
    - significance_map.png  → outputs/figures/<mode>/causal_significance/<dir>_<model>.png
    - crz_summary.csv       → outputs/figures/<mode>/causal_significance/crz_<dir>_<model>.csv
    - daily_att.png         → outputs/figures/<mode>/trends/daily_<dir>_<model>.png
    - cumulative_att.png    → outputs/figures/<mode>/trends/cum_<dir>_<model>.png
    - unit_effects_map.png  → outputs/figures/<mode>/spatial_effects/unit_<dir>_<model>.png
    - tract_choropleth.png  → outputs/figures/<mode>/spatial_effects/tract_<dir>_<model>.png
                              (only bus / subway — replica's unit map IS the tract map)
    - spatial_regression.txt → outputs/figures/<mode>/regression/<dir>_<model>.txt
    - ml_models.csv         → outputs/figures/<mode>/regression/ml_<dir>_<model>.csv
    - tract_effects.geojson too big — skip
    """
    for mode, dirs in MODE_DIRS.items():
        cs_dir = FIG_ROOT / mode / "causal_significance"
        rg_dir = FIG_ROOT / mode / "regression"
        tr_dir = FIG_ROOT / mode / "trends"
        sp_dir = FIG_ROOT / mode / "spatial_effects"
        for direction in dirs:
            d_path = "" if direction == "all" else direction
            d_label = "all" if direction == "all" else direction
            for model in HEADLINE_BY_MODE[mode]:
                src = out_root / mode / model / d_path / "causal"
                _safe_copy(src / "significance_map.png", cs_dir / f"{d_label}_{model}.png")
                _safe_copy(src / "crz_summary.csv",      cs_dir / f"crz_{d_label}_{model}.csv")
                _safe_copy(src / "daily_att.png",        tr_dir / f"daily_{d_label}_{model}.png")
                _safe_copy(src / "cumulative_att.png",   tr_dir / f"cum_{d_label}_{model}.png")
                _safe_copy(src / "trends_by_crz.png",    tr_dir / f"crz_{d_label}_{model}.png")
                _safe_copy(src / "unit_effects_map.png", sp_dir / f"unit_{d_label}_{model}.png")
                _safe_copy(src / "tract_choropleth.png", sp_dir / f"tract_{d_label}_{model}.png")
                _safe_copy(src / "spatial_regression.txt", rg_dir / f"{d_label}_{model}.txt")
                _safe_copy(src / "ml_models.csv",         rg_dir / f"ml_{d_label}_{model}.csv")


def build_unified_att_table() -> None:
    """Build a single CSV at outputs/_summary/att_unified.csv covering every
    (mode, model, direction) triple where compute_effects has produced an
    overall.csv.

    Columns:
        n_units              — number of units (routes / stations / tracts)
        daily_att_avg        — average daily ATT per unit (avg_daily_all)
        cum_att_M            — cumulative ATT in millions of trips (total_att / 1e6)
        cum_att_lo_M, cum_att_hi_M  — 90% CI on cumulative ATT, in millions
        cum_rel_eff_pct      — cumulative relative effect = sum(eff) / sum(cf), in %
        signif_days          — total (unit, day) pairs flagged signif (actual outside PI)
        avg_signif_share     — fraction of unit-days that are signif
        att_signif           — whether the cumulative ATT 90% CI excludes 0
    """
    rows = []
    # 11 headline models (5 baselines + raw chronos + raw timesfm + 4 cal variants)
    # plus the OOS sibling variants for robustness check.
    ALL = ["arima", "bsts", "prophet", "nhits", "tft",
           "chronos", "chronos_qrcal", "chronos_qrcal_intercept",
           "timesfm", "timesfm_qrcal", "timesfm_qrcal_intercept",
           "chronos_qrcal_oos", "timesfm_qrcal_oos",
           "chronos_qrcal_intercept_oos", "timesfm_qrcal_intercept_oos"]
    for mode, dirs in MODE_DIRS.items():
        for direction in dirs:
            d_path = "" if direction == "all" else direction
            suffix = "" if direction == "all" else f"_{direction}"
            for model in ALL:
                f = out_root / mode / model / d_path / "effects" / f"{mode}_{model}_test{suffix}_overall.csv"
                if not f.exists(): continue
                o = pd.read_csv(f).iloc[0]
                rows.append({
                    "mode": mode, "direction": direction, "model": model,
                    "n_units": int(o.n_units),
                    "daily_att_avg": round(float(o.avg_daily_all), 2),
                    "cum_att_M": round(float(o.total_att) / 1e6, 3),
                    "cum_att_lo_M": round(float(o.total_att_lo) / 1e6, 3),
                    "cum_att_hi_M": round(float(o.total_att_hi) / 1e6, 3),
                    "cum_rel_eff_pct": round(float(o.total_cum_relative_effect) * 100, 3),
                    "signif_days": int(o.total_signif_days),
                    "avg_signif_share": round(float(o.avg_signif_share), 3),
                    "att_signif": bool(o.total_att_signif),
                })
    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY / "att_unified.csv", index=False)
    print(f"Wrote unified ATT table: {len(df)} rows × {len(df.columns)} cols")


def build_unified_crz_table() -> None:
    """Aggregate every crz_summary.csv (headline 6 × all modes/dirs) into a
    single long-format CSV for cross-mode comparison."""
    rows = []
    for mode, dirs in MODE_DIRS.items():
        for direction in dirs:
            d_path = "" if direction == "all" else direction
            for model in HEADLINE_BY_MODE[mode]:
                f = out_root / mode / model / d_path / "causal" / "crz_summary.csv"
                if not f.exists(): continue
                df = pd.read_csv(f)
                df["mode"] = mode; df["direction"] = direction; df["model"] = model
                rows.append(df)
    if rows:
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(SUMMARY / "crz_unified.csv", index=False)
        print(f"Wrote unified CRZ table: {len(out)} rows")


def main() -> None:
    print("Mirroring compare_models tables...")
    mirror_compare_models()
    print("Mirroring geospatial outputs...")
    mirror_geospatial()
    print("Building unified ATT table...")
    build_unified_att_table()
    print("Building unified CRZ table...")
    build_unified_crz_table()
    print("Done.")


if __name__ == "__main__":
    main()
