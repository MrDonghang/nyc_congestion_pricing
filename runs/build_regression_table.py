"""Parse all spatial_regression.txt + ml_models.csv produced by
``geospatial_analysis`` into a single unified CSV.

For each (mode, model, direction):
  * OLS R² + ML_Lag pseudo-R² + ML_Error pseudo-R²
  * rho (ML_Lag spatial-lag coef, with p-value) and lambda (ML_Error spatial-
    error coef, with p-value)
  * For each demographic variable, whether it was significant (p<0.05) in
    each of the 3 regression types — stored as a comma-separated list of
    {OLS, LAG, ERR} tags so a single column captures cross-model robustness
  * ML tree-based train/test R² for RF / XGB / LightGBM

Writes ``outputs/_summary/regression_unified.csv``.
"""
from __future__ import annotations

import re
import json
from pathlib import Path

import pandas as pd

from nyc_cp.config import load_paths

paths = load_paths()
out_root = Path(paths["output_root"])
REPO = Path("/home/donghang/nyc_congestion_pricing")
SUMMARY = REPO / "outputs" / "_summary"
SUMMARY.mkdir(parents=True, exist_ok=True)

HEADLINE = ["chronos", "chronos_qrcal", "chronos_qrcal_intercept",
            "timesfm", "timesfm_qrcal", "timesfm_qrcal_intercept"]
MODE_DIRS = {"bus": ["all"], "subway": ["O", "D"], "replica": ["O", "D"], "citibike": ["O", "D"]}
HEADLINE_BY_MODE = {
    "bus":      HEADLINE,
    "subway":   HEADLINE,
    "replica":  HEADLINE,
    "citibike": ["chronos", "timesfm"],   # citibike: cal not viable, raw only
}


def _extract_section(text: str, start: str, end: str | None = None) -> str:
    parts = text.split(f"=== {start} ===")
    if len(parts) < 2: return ""
    after = parts[1]
    return after.split(f"=== {end} ===")[0] if end else after


def _parse_one_section(text: str) -> dict:
    """Return {n_obs, r2, pseudo, rho, lam, sig_vars: {name: (coef, p)}}."""
    m = re.search(r"R-squared\s*:\s*([\d.\-]+)", text)
    r2 = float(m.group(1)) if m else None
    m = re.search(r"Pseudo R-squared\s*:\s*([\d.\-]+)", text)
    pseudo = float(m.group(1)) if m else None
    m = re.search(r"Number of Observations:\s*([\d]+)", text)
    n = int(m.group(1)) if m else None
    m = re.search(r"\brho\s+([-\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([-\d.]+)", text)
    rho = (float(m.group(1)), float(m.group(4))) if m else None
    m = re.search(r"\blambda\s+([-\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([-\d.]+)", text)
    lam = (float(m.group(1)), float(m.group(4))) if m else None
    sig: dict[str, tuple[float, float]] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(\S+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$", line)
        if not m: continue
        name = m.group(1)
        if name in ("Variable", "CONSTANT", "rho", "lambda", "R-squared", "Pseudo"): continue
        try:
            coef = float(m.group(2)); p = float(m.group(5))
            if p < 0.05:
                sig[name] = (coef, p)
        except ValueError:
            pass
    return {"n_obs": n, "r2": r2, "pseudo": pseudo, "rho": rho, "lam": lam, "sig_vars": sig}


def parse_regression(txt_path: Path) -> dict:
    txt = txt_path.read_text()
    ols   = _parse_one_section(_extract_section(txt, "OLS", "ML_Lag"))
    mllag = _parse_one_section(_extract_section(txt, "ML_Lag", "ML_Error"))
    mlerr = _parse_one_section(_extract_section(txt, "ML_Error", None))
    return {"OLS": ols, "ML_Lag": mllag, "ML_Error": mlerr}


def parse_tree(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    return {row["model"]: (float(row["r2_train"]), float(row["r2_test"]), float(row["rmse_test"]))
            for _, row in df.iterrows()}


def main() -> None:
    rows = []
    for mode, dirs in MODE_DIRS.items():
        for direction in dirs:
            d_path = "" if direction == "all" else direction
            for model in HEADLINE_BY_MODE[mode]:
                base = out_root / mode / model / d_path / "causal"
                txt_p = base / "spatial_regression.txt"
                ml_p  = base / "ml_models.csv"
                if not txt_p.exists():
                    continue
                regs = parse_regression(txt_p)
                trees = parse_tree(ml_p) if ml_p.exists() else {}

                ols, lag, err = regs["OLS"], regs["ML_Lag"], regs["ML_Error"]
                # Union of all significant variables across OLS / Lag / Err
                all_sig_names = set(ols["sig_vars"]) | set(lag["sig_vars"]) | set(err["sig_vars"])
                sig_tag_list = []
                for name in sorted(all_sig_names):
                    tags = []
                    if name in ols["sig_vars"]: tags.append("OLS")
                    if name in lag["sig_vars"]: tags.append("LAG")
                    if name in err["sig_vars"]: tags.append("ERR")
                    sig_tag_list.append(f"{name}[{'+'.join(tags)}]")

                row = {
                    "mode": mode, "direction": direction, "model": model,
                    "n_obs": ols["n_obs"],
                    "ols_R2": ols["r2"],
                    "mllag_pseudoR2": lag["pseudo"],
                    "mlerr_pseudoR2": err["pseudo"],
                    "rho_coef":   lag["rho"][0] if lag["rho"] else None,
                    "rho_pvalue": lag["rho"][1] if lag["rho"] else None,
                    "lam_coef":   err["lam"][0] if err["lam"] else None,
                    "lam_pvalue": err["lam"][1] if err["lam"] else None,
                    "sig_vars_all": ", ".join(sig_tag_list),
                    "n_robust_sig": sum(1 for x in sig_tag_list if "OLS" in x and "LAG" in x and "ERR" in x),
                }
                for tree in ["random_forest", "xgboost", "lightgbm"]:
                    if tree in trees:
                        r2t, r2v, rmse = trees[tree]
                        row[f"{tree}_r2_train"] = round(r2t, 3)
                        row[f"{tree}_r2_test"] = round(r2v, 3)
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY / "regression_unified.csv", index=False)
    print(f"Wrote unified regression table: {len(df)} rows × {len(df.columns)} cols")


if __name__ == "__main__":
    main()
