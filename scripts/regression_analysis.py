"""Standardized ML_Lag for the 3 paper modes (bus/all, subway/total, replica/O).

Pipeline
--------
1. For each mode, load ``tract_effects.geojson`` produced by
   ``scripts.geospatial_analysis``.
2. Same VIF filter as the main pipeline (threshold 20).
3. **Z-score the X matrix** so coefficients are "effect of 1 SD increase in
   feature on y in raw y-units" — comparable across variables within a mode.
4. Fit ML_Lag (spatial lag, Rook weights).
5. Save per-mode coefficient CSV; emit a single 3-column LaTeX table comparing
   all three modes side-by-side.

Outputs go to ``outputs/figures/paper/causal/ml_lag_zscored/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nyc_cp.analysis import demographics
from nyc_cp.analysis.causal import stepwise_vif_filter
from nyc_cp.config import REPO_ROOT, load_paths, output_dir
from nyc_cp.utils import setup_logging

log = logging.getLogger("ml_lag_paper")

MODEL = "chronos_qrcal_intercept"
WINDOW = "test"
Y_COL = "att"
VIF_THRESHOLD = 20.0

# Fixed 3-mode setup the paper reports on.
RUNS = [
    {"mode": "bus",     "direction": "all",   "label": "Bus"},
    {"mode": "subway",  "direction": "total", "label": "Subway"},
    {"mode": "replica", "direction": "O",     "label": "Replica"},
]

OUT_DIR = REPO_ROOT / "outputs" / "figures" / "paper" / "causal" / "ml_lag_zscored"


def fit_one(mode: str, direction: str, paths: dict) -> dict:
    """Return everything we need to build the LaTeX cell for one mode."""
    import geopandas as gpd
    import libpysal
    from spreg import ML_Lag

    out = output_dir(mode, MODEL, direction=direction, paths=paths) / "causal"
    gdf = gpd.read_file(out / "tract_effects.geojson")
    log.info("[%s/%s] loaded %d tracts", mode, direction, len(gdf))

    x_cols = sum([demographics.GROUPS[g] for g in
                  ["demographics", "race_ethnicity", "economics", "travel", "education", "housing"]], [])
    x_cols = [c for c in x_cols if c in gdf.columns]

    df = gdf.dropna(subset=[Y_COL, *x_cols]).reset_index(drop=True)
    kept, dropped = stepwise_vif_filter(df[x_cols], threshold=VIF_THRESHOLD)
    log.info("[%s/%s] VIF kept %d, dropped %d", mode, direction, len(kept), len(dropped))

    # --- Fully standardized: z-score X AND y so coefficients are
    # "SD y per SD X" — directly comparable across modes with different y scales. ---
    X_raw = df[kept].astype(float)
    X_mean = X_raw.mean()
    X_std = X_raw.std(ddof=0).replace(0, np.nan)
    X_z = (X_raw - X_mean) / X_std

    y_raw = df[Y_COL].astype(float)
    y_mean = y_raw.mean()
    y_std = y_raw.std(ddof=0)
    y_z = (y_raw - y_mean) / y_std

    df_z = df.copy()
    df_z[kept] = X_z
    df_z[Y_COL] = y_z

    y = df_z[Y_COL].to_numpy().reshape(-1, 1)
    X = df_z[kept].to_numpy()

    w = libpysal.weights.Rook.from_dataframe(df_z)
    w.transform = "r"
    fit = ML_Lag(y, X, w=w, name_y=Y_COL, name_x=list(kept), name_w="Rook")

    # spreg stores coefficients with constant first; betas[-1] is the spatial
    # lag (rho/W_att). Standard errors live in ``std_err``, z-stats in
    # ``z_stat`` (list of (z, p)).
    betas = np.array(fit.betas).flatten()
    std_err = np.array(fit.std_err).flatten()
    z_stat = np.array([s[0] for s in fit.z_stat])
    p_val = np.array([s[1] for s in fit.z_stat])

    # Variable ordering produced by spreg: [CONSTANT, *x_cols, W_y].
    names = ["CONSTANT", *kept, "W_att"]
    table = pd.DataFrame({
        "variable": names,
        "coef": betas,
        "std_err": std_err,
        "z": z_stat,
        "p": p_val,
    })

    # Spatial-lag impacts (direct, indirect, total) — note these are for the
    # standardized X too, so they live in the same units as ``coef``.
    impacts = pd.DataFrame(fit.sp_multipliers) if hasattr(fit, "sp_multipliers") else None

    return {
        "mode": mode,
        "direction": direction,
        "n": int(fit.n),
        "k": int(fit.k),
        "pseudo_r2": float(fit.pr2),
        "aic": float(fit.aic),
        "schwarz": float(fit.schwarz),
        "log_lik": float(fit.logll),
        "table": table,
        "x_std": X_std,
        "x_mean": X_mean,
        "impacts": impacts,
    }


def sig_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def latex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def build_latex(results: list[dict]) -> str:
    # Union of variables across the 3 modes, preserving GROUPS ordering.
    group_order = sum([demographics.GROUPS[g] for g in
                       ["demographics", "race_ethnicity", "economics", "travel", "education", "housing"]], [])

    all_vars = set()
    for r in results:
        all_vars.update(r["table"]["variable"].tolist())
    all_vars -= {"CONSTANT", "W_att"}
    # Sort: those that appear in group_order keep that order, others appended.
    in_order = [v for v in group_order if v in all_vars]
    extras = sorted(all_vars - set(in_order))
    var_rows = in_order + extras

    n_modes = len(results)
    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{ML\_Lag (spatial lag) fully standardized $\beta$ coefficients. "
                 r"Both $X$ and $y$ are z-scored, so each coefficient is the change in ATT (in SD units of $y$) "
                 r"associated with a 1 SD increase in the predictor — directly comparable across modes. "
                 r"Significance: $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$, $^{.}p<0.1$.}")
    lines.append(r"\label{tab:ml_lag_zscored}")
    lines.append(r"\begin{tabular}{l" + "c" * n_modes + "}")
    lines.append(r"\toprule")
    head = "Variable & " + " & ".join(latex_escape(r["mode"].capitalize() + ("/" + r["direction"] if r["direction"] != "all" else "")) for r in results) + r" \\"
    lines.append(head)
    lines.append(r"\midrule")

    def coef_cell(row: pd.Series | None) -> str:
        if row is None or row.empty:
            return "—"
        coef, p = float(row["coef"]), float(row["p"])
        return f"{coef:.3f}$^{{{sig_stars(p)}}}$"

    def se_cell(row: pd.Series | None) -> str:
        if row is None or row.empty:
            return ""
        se = float(row["std_err"])
        return f"\\scriptsize({se:.3f})"

    # Two-row layout per variable: coef row + (se) row underneath.
    for v in var_rows:
        subs = [r["table"][r["table"]["variable"] == v] for r in results]
        coefs = [coef_cell(s.iloc[0]) if not s.empty else "—" for s in subs]
        ses = [se_cell(s.iloc[0]) if not s.empty else "" for s in subs]
        lines.append(latex_escape(v) + " & " + " & ".join(coefs) + r" \\")
        lines.append(" & " + " & ".join(ses) + r" \\[2pt]")

    # W_att row (same two-line layout).
    lines.append(r"\midrule")
    subs_w = [r["table"][r["table"]["variable"] == "W_att"] for r in results]
    coefs_w = [coef_cell(s.iloc[0]) if not s.empty else "—" for s in subs_w]
    ses_w = [se_cell(s.iloc[0]) if not s.empty else "" for s in subs_w]
    lines.append(r"$W \cdot \text{ATT}$ (spatial lag) & " + " & ".join(coefs_w) + r" \\")
    lines.append(" & " + " & ".join(ses_w) + r" \\[2pt]")

    # Summary rows.
    lines.append(r"\midrule")
    lines.append("Observations $n$ & " + " & ".join(f"{r['n']:,}" for r in results) + r" \\")
    lines.append("Pseudo $R^2$ & " + " & ".join(f"{r['pseudo_r2']:.3f}" for r in results) + r" \\")
    lines.append("AIC & " + " & ".join(f"{r['aic']:,.1f}" for r in results) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    paths = load_paths()
    setup_logging("ml_lag_paper", log_root=paths["log_root"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for spec in RUNS:
        r = fit_one(spec["mode"], spec["direction"], paths)
        r["label"] = spec["label"]
        out_csv = OUT_DIR / f"coef_{spec['mode']}_{spec['direction']}.csv"
        r["table"].to_csv(out_csv, index=False)
        log.info("Wrote %s", out_csv)
        results.append(r)

    latex = build_latex(results)
    tex_path = OUT_DIR / "table_ml_lag_zscored.tex"
    tex_path.write_text(latex)
    log.info("Wrote %s", tex_path)
    print("\n" + "=" * 70)
    print(latex)
    print("=" * 70)
    print(f"\nFiles under {OUT_DIR}")


if __name__ == "__main__":
    main()
