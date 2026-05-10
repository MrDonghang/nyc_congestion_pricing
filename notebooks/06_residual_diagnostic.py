# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # Val residual diagnostic — shared across modes (bus / subway / replica)
#
# For the **6 headline models** (Chronos × {raw, qrcal, qrcal_intercept} +
# TimesFM × {raw, qrcal, qrcal_intercept}) on the val window, produces a
# standardized panel of diagnostics:
#
# 1. Per-unit summary table (bias, RMSE, ECR, PI width, level)
# 2. Pooled residual histogram per model
# 3. Per-unit ECR distribution per model
# 4. Spatial bias map per model (points for subway / lines for bus / polygons for replica)
# 5. Spatial low-coverage map (units with ECR < 0.80) per model
# 6. Worst-bias unit table
# 7. Test ATT summary across the 6 model variants
#
# Run with the ``MODE`` environment variable to select which mode to render
# (default: ``subway``). Outputs land at ``outputs/figures/<mode>/residual/``.

# %%
from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm

from nyc_cp.analysis.effects import load_forecast_triplet
from nyc_cp.config import get_window, load_paths, output_dir
from nyc_cp.data import load_actual

MODE = os.environ.get("MODE", "subway").lower()
WINDOW = "val"
ECR_LOW = 0.80
COVERAGE_LEVEL = 0.9

# 6 headline models — same across all modes
MODELS = [
    ("chronos (raw)",            "chronos"),
    ("chronos_qrcal",            "chronos_qrcal"),
    ("chronos_qrcal_intercept",  "chronos_qrcal_intercept"),
    ("timesfm (raw)",            "timesfm"),
    ("timesfm_qrcal",            "timesfm_qrcal"),
    ("timesfm_qrcal_intercept",  "timesfm_qrcal_intercept"),
]

paths = load_paths()
REPO_ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path("/home/donghang/nyc_congestion_pricing")

# %% [markdown]
# ## Mode-specific config: directions, id column, geometry source/kind

# %%
SUBWAY_GEOJSON = "/home/donghang/nyc_congestion_pricing/geo_data/MTA_Subway_Stations_20251029.geojson"
SUBWAY_MAPPING = "/public_dataset/donghang/nyc_congestion_pricing/data_processed/subway/patterns/station_mapping.csv"
BUS_SHP        = "/home/donghang/nyc_congestion_pricing/geo_data/bus_routes/bus_routes_nyc_dec2019.shp"
TRACT_SHP      = "/home/donghang/nyc_congestion_pricing/geo_data/NYC_Census_Tracts_2020/NYC_Census_Tracts_2020.shp"


def _load_subway_geo() -> gpd.GeoDataFrame:
    g = gpd.read_file(SUBWAY_GEOJSON).to_crs(4326)
    m = pd.read_csv(SUBWAY_MAPPING).astype({"station_id": "Int64", "station_index": "Int64"})
    g["station_id"] = pd.to_numeric(g["station_id"], errors="coerce").astype("Int64")
    out = g.merge(m, on="station_id", how="inner")[["station_index", "stop_name", "borough", "daytime_routes", "geometry"]]
    out = out.rename(columns={"station_index": "unit_id", "stop_name": "name", "daytime_routes": "line"})
    out["unit_id"] = out["unit_id"].astype(str)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=4326)


def _load_bus_geo() -> gpd.GeoDataFrame:
    g = gpd.read_file(BUS_SHP).to_crs(4326)
    g = g.dissolve(by="route_id", as_index=False)
    out = g[["route_id", "route_long", "geometry"]].rename(columns={"route_id": "unit_id", "route_long": "name"})
    out["borough"] = ""
    out["line"] = out["unit_id"].astype(str)
    out["unit_id"] = out["unit_id"].astype(str)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=4326)


def _load_replica_geo() -> gpd.GeoDataFrame:
    g = gpd.read_file(TRACT_SHP).to_crs(4326)
    if "GEOID" not in g.columns:
        g = g.rename(columns={c: "GEOID" for c in g.columns if c.lower() == "geoid"})
    g["GEOID"] = g["GEOID"].astype(str)
    out = g[["GEOID", "geometry"]].rename(columns={"GEOID": "unit_id"})
    out["name"] = ""; out["borough"] = ""; out["line"] = ""
    return gpd.GeoDataFrame(out, geometry="geometry", crs=4326)


def _load_citibike_geo() -> gpd.GeoDataFrame:
    # Citibike units are integer indices 0–1529 mapped via pkl to 6-digit
    # ct2020 codes. Multiple physical tracts (one per borough) can share the
    # same ct2020, so each unit_id maps to ≥1 polygon.
    import pickle
    pkl_path = Path("/public_dataset/donghang/nyc_congestion_pricing/data_processed/citibike/census/censustract_idx_mapping.pkl")
    with open(pkl_path, "rb") as f:
        idx_map = pickle.load(f)            # ct2020 (str) -> idx (int)
    inv = {str(v): k for k, v in idx_map.items()}  # idx (str) -> ct2020 (str)
    g = gpd.read_file(TRACT_SHP).to_crs(4326)
    g["ct2020"] = g["ct2020"].astype(str)
    rows = []
    for unit_id_str, ct in inv.items():
        match = g[g["ct2020"] == ct]
        for _, r in match.iterrows():
            rows.append({"unit_id": unit_id_str, "name": "", "borough": "", "line": "", "geometry": r["geometry"]})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


MODE_CONFIG = {
    "bus":      {"directions": ["all"],   "id_col": "route_id",      "geo_kind": "line",    "geo_loader": _load_bus_geo,      "freq_label": "daily"},
    "subway":   {"directions": ["O", "D"], "id_col": "station_index", "geo_kind": "point",   "geo_loader": _load_subway_geo,   "freq_label": "daily"},
    "replica":  {"directions": ["O", "D"], "id_col": "tract_id",      "geo_kind": "polygon", "geo_loader": _load_replica_geo,  "freq_label": "weekly"},
    "citibike": {"directions": ["O", "D"], "id_col": "tract_id",      "geo_kind": "polygon", "geo_loader": _load_citibike_geo, "freq_label": "daily"},
}
if MODE not in MODE_CONFIG:
    raise SystemExit(f"Unknown MODE={MODE!r}. Pick one of {list(MODE_CONFIG)}.")
cfg = MODE_CONFIG[MODE]

fig_root = REPO_ROOT / "outputs" / "figures" / MODE / "residual"
fig_root.mkdir(parents=True, exist_ok=True)
print(f"[06_residual_diagnostic] mode={MODE}  directions={cfg['directions']}  →  {fig_root}")

# %% [markdown]
# ## Loaders + per-unit summary

# %%
def load_panel(direction: str) -> dict[str, dict]:
    win = get_window(MODE, WINDOW)
    a_full = load_actual(MODE, direction=direction, paths=paths)
    a = a_full.loc[(a_full.index >= pd.Timestamp(win.test_start)) & (a_full.index <= pd.Timestamp(win.test_end))]
    a.columns = a.columns.astype(str)
    suffix = f"_{direction}" if direction != "all" else ""
    out = {}
    for label, m in MODELS:
        d = output_dir(MODE, m, direction=direction, paths=paths)
        prefix = f"{MODE}_{m}_val{suffix}"
        try:
            fc = load_forecast_triplet(d, prefix)
        except Exception as e:
            print(f"  SKIP {label} (no triplet): {e}")
            continue
        common = a.columns.intersection(fc.mu.columns)
        a_aln = a[common].reindex(fc.mu.index)
        out[label] = {
            "actual": a_aln, "mu": fc.mu[common],
            "lower": fc.lower[common], "upper": fc.upper[common],
            "residual": a_aln - fc.mu[common],
        }
    return out


def per_unit_summary(p: dict, label: str) -> pd.DataFrame:
    a, lo, hi, r = p[label]["actual"], p[label]["lower"], p[label]["upper"], p[label]["residual"]
    level = a.mean(axis=0)
    bias = r.mean(axis=0)
    rmse = np.sqrt((r ** 2).mean(axis=0))
    ecr = ((a >= lo) & (a <= hi)).mean(axis=0)
    pi_w = (hi - lo).mean(axis=0)
    return pd.DataFrame({
        "unit_id": a.columns.astype(str), "model": label,
        "level": level.values, "bias": bias.values,
        "rel_bias": (bias / level.replace(0, np.nan)).values,
        "rmse": rmse.values, "ecr": ecr.values, "pi_width": pi_w.values,
    })

# %%
panels = {d: load_panel(d) for d in cfg["directions"]}
per_unit_dfs = {
    d: pd.concat([per_unit_summary(panels[d], lbl) for lbl, _ in MODELS if lbl in panels[d]], ignore_index=True)
    for d in cfg["directions"] if panels[d]
}

# Save per-unit raw + panel-level recap
combined = pd.concat([df.assign(direction=d) for d, df in per_unit_dfs.items()], ignore_index=True)
combined.to_csv(fig_root / "per_unit_summary.csv", index=False)
recap = combined.groupby(["direction", "model"], sort=False).agg(
    n_units=("unit_id", "nunique"),
    rmse_panel=("rmse", lambda s: float(np.sqrt((s ** 2).mean()))),
    bias_mean=("bias", "mean"), bias_std=("bias", "std"),
    ecr_mean=("ecr", "mean"), ecr_p10=("ecr", lambda s: s.quantile(0.10)),
    n_below_80=("ecr", lambda s: (s < ECR_LOW).sum()),
    pi_width_mean=("pi_width", "mean"),
).round(4)
recap.to_csv(fig_root / "panel_summary.csv")
print("\nPanel-level recap:")
print(recap.to_string())

# %% [markdown]
# ## Pooled residual histograms per model

# %%
def plot_residual_hists(direction: str):
    if direction not in panels or not panels[direction]: return
    p = panels[direction]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    all_r = np.concatenate([p[lbl]["residual"].values.flatten() for lbl, _ in MODELS if lbl in p])
    all_r = all_r[np.isfinite(all_r)]
    xmin, xmax = np.quantile(all_r, [0.005, 0.995])
    for i, (lbl, _) in enumerate(MODELS):
        ax = axes[i]
        if lbl not in p:
            ax.set_visible(False); continue
        r = p[lbl]["residual"].values.flatten(); r = r[np.isfinite(r)]
        rt = r[(r >= xmin) & (r <= xmax)]
        ax.hist(rt, bins=80, color="steelblue", alpha=0.75, density=True)
        ax.axvline(0, color="k", lw=0.8)
        ax.axvline(r.mean(), color="C3", lw=1.2, label=f"mean={r.mean():+.0f}")
        ax.set_title(f"{lbl}\nμ={r.mean():+.0f}  σ={r.std():.0f}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{MODE} {direction}: pooled residual distribution per model (val, {cfg['freq_label']})", fontsize=13, y=1.0)
    fig.supxlabel("residual (actual - mu)")
    fig.tight_layout()
    fig.savefig(fig_root / f"residual_hist_{direction}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

for d in cfg["directions"]:
    plot_residual_hists(d)

# %% [markdown]
# ## Per-unit ECR distribution

# %%
def plot_ecr_dist(direction: str):
    if direction not in per_unit_dfs: return
    df = per_unit_dfs[direction]
    fig, ax = plt.subplots(figsize=(11, 5))
    palette = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for i, (lbl, _) in enumerate(MODELS):
        sub = df[df.model == lbl]
        if sub.empty: continue
        ax.hist(sub["ecr"].dropna(), bins=30, alpha=0.4, color=palette[i],
                label=f"{lbl}  med={sub['ecr'].median():.2f}  <80%: {(sub['ecr']<ECR_LOW).sum()}")
    ax.axvline(0.9, color="k", lw=0.8, ls="--", label="nominal 0.9")
    ax.set_xlabel("per-unit ECR (90% PI empirical coverage on val)")
    ax.set_title(f"{MODE} {direction}: ECR distribution across 6 headline models")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_root / f"ecr_dist_{direction}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

for d in cfg["directions"]:
    plot_ecr_dist(d)

# %% [markdown]
# ## Spatial bias map per model (geometry-kind aware)

# %%
def _plot_geom(ax, gdf, **kwargs):
    if cfg["geo_kind"] == "point":
        kwargs.setdefault("markersize", 12)
    elif cfg["geo_kind"] == "line":
        kwargs.setdefault("linewidth", 1.4)
    else:  # polygon
        kwargs.setdefault("edgecolor", "white")
        kwargs.setdefault("linewidth", 0.05)
    gdf.plot(ax=ax, **kwargs)


geo_gdf = cfg["geo_loader"]()
print(f"Loaded geometry: {len(geo_gdf)} {cfg['geo_kind']}s")


def plot_spatial_bias(direction: str):
    if direction not in per_unit_dfs: return
    df = per_unit_dfs[direction]
    bias_vmax = float(np.nanpercentile(np.abs(df["bias"].values), 95))
    bias_norm = Normalize(vmin=-bias_vmax, vmax=bias_vmax)

    fig, axes = plt.subplots(2, 3, figsize=(16, 13))
    axes = axes.ravel()
    for i, (lbl, _) in enumerate(MODELS):
        ax = axes[i]
        if lbl not in df["model"].unique():
            ax.set_visible(False); continue
        sub = df[df.model == lbl].merge(geo_gdf, on="unit_id", how="inner")
        if sub.empty:
            ax.set_title(f"{lbl}: no spatial join"); ax.set_axis_off(); continue
        sub = gpd.GeoDataFrame(sub, geometry="geometry", crs=4326)
        _plot_geom(ax, geo_gdf, color="lightgrey", alpha=0.35)
        _plot_geom(ax, sub, column="bias", cmap="RdBu_r", norm=bias_norm)
        full = df[df.model == lbl]
        ax.set_title(f"{lbl}\nμ_bias={full['bias'].mean():+.1f}  σ={full['bias'].std():.1f}", fontsize=10)
        ax.set_axis_off()
    fig.suptitle(f"{MODE} {direction}: per-unit mean residual on val (red=under-pred, blue=over-pred)",
                 fontsize=13, y=0.995)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.08, wspace=0.02, hspace=0.10)
    cax = fig.add_axes([0.30, 0.04, 0.40, 0.018])
    fig.colorbar(ScalarMappable(norm=bias_norm, cmap="RdBu_r"), cax=cax,
                 orientation="horizontal", label="mean residual (actual - mu)")
    fig.savefig(fig_root / f"spatial_bias_{direction}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

for d in cfg["directions"]:
    plot_spatial_bias(d)

# %% [markdown]
# ## Spatial low-coverage map (ECR < 0.80)

# %%
def plot_lowcov(direction: str, threshold: float = ECR_LOW):
    if direction not in per_unit_dfs: return
    df = per_unit_dfs[direction]
    fig, axes = plt.subplots(2, 3, figsize=(16, 13))
    axes = axes.ravel()
    for i, (lbl, _) in enumerate(MODELS):
        ax = axes[i]
        if lbl not in df["model"].unique():
            ax.set_visible(False); continue
        sub_all = df[df.model == lbl].merge(geo_gdf, on="unit_id", how="inner")
        sub_low = sub_all[sub_all.ecr < threshold]
        sub_all_g = gpd.GeoDataFrame(sub_all, geometry="geometry", crs=4326)
        sub_low_g = gpd.GeoDataFrame(sub_low, geometry="geometry", crs=4326)
        _plot_geom(ax, geo_gdf, color="lightgrey", alpha=0.35)
        if len(sub_low):
            if cfg["geo_kind"] == "point":
                sizes = 30 + 800 * (threshold - sub_low_g.ecr).clip(lower=0)
                sub_low_g.plot(ax=ax, color="red", markersize=sizes, alpha=0.65,
                               edgecolor="darkred", linewidth=0.4)
            elif cfg["geo_kind"] == "line":
                sub_low_g.plot(ax=ax, color="red", linewidth=2.2, alpha=0.75)
            else:
                sub_low_g.plot(ax=ax, color="red", alpha=0.6, edgecolor="darkred", linewidth=0.2)
        n = len(sub_low); med = sub_low.ecr.median() if n else float("nan")
        ax.set_title(f"{lbl}\n{n} units below ECR=0.80  (med={med:.2f})", fontsize=10)
        ax.set_axis_off()
    fig.suptitle(f"{MODE} {direction}: low-coverage units (ECR < 0.80 on val)", fontsize=13, y=0.995)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.04, wspace=0.02, hspace=0.10)
    fig.savefig(fig_root / f"lowcov_map_{direction}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

for d in cfg["directions"]:
    plot_lowcov(d)

# %% [markdown]
# ## Worst-bias unit table (top 30 under raw chronos)

# %%
def worst_bias_table(direction: str, n: int = 30):
    if direction not in per_unit_dfs: return None
    df = per_unit_dfs[direction]
    raw_label = MODELS[0][0]   # "chronos (raw)"
    base = df[df.model == raw_label].copy()
    if base.empty: return None
    base["abs_rel_bias"] = base["rel_bias"].abs()
    worst = base.nlargest(n, "abs_rel_bias")[["unit_id", "level", "bias", "rel_bias"]]
    worst = worst.rename(columns={"bias": "bias_chronos_raw", "rel_bias": "rel_bias_chronos_raw"})
    for lbl, _ in MODELS[1:]:
        if lbl not in df["model"].unique(): continue
        sub = df[df.model == lbl].set_index("unit_id")[["bias", "rel_bias"]]
        short = lbl.replace(" (raw)", "_raw").replace(" ", "_")
        worst[f"bias_{short}"] = worst["unit_id"].map(sub["bias"])
        worst[f"rel_bias_{short}"] = worst["unit_id"].map(sub["rel_bias"])
    return worst

for d in cfg["directions"]:
    w = worst_bias_table(d)
    if w is not None:
        w.to_csv(fig_root / f"worst_bias_{d}.csv", index=False)

# %% [markdown]
# ## Test ATT summary (read from compute_effects outputs)

# %%
out_root = Path(paths["output_root"])
att_rows = []
for lbl, m in MODELS:
    for direction in cfg["directions"]:
        sub_path = "" if direction == "all" else f"/{direction}"
        suffix = f"_{direction}" if direction != "all" else ""
        f = out_root / MODE / m / sub_path.lstrip("/") / "effects" / f"{MODE}_{m}_test{suffix}_overall.csv"
        if not f.exists():
            print(f"missing test overall: {f}"); continue
        o = pd.read_csv(f).iloc[0]
        att_rows.append({
            "model": lbl, "direction": direction,
            "n_units": int(o.n_units),
            "total_att": float(o.total_att),
            "rel_eff": float(o.total_cum_relative_effect),
            "signif_days": int(o.total_signif_days),
            "att_signif": bool(o.total_att_signif),
        })
if att_rows:
    att_df = pd.DataFrame(att_rows)
    att_df["rel_eff_pct"] = att_df["rel_eff"].map(lambda x: f"{x*100:+.2f}%")
    att_df["ATT_M"] = (att_df["total_att"]/1e6).round(3)
    att_df.to_csv(fig_root / "att_summary.csv", index=False)
    print("\nTest ATT summary:")
    print(att_df[["model","direction","n_units","ATT_M","rel_eff_pct","signif_days","att_signif"]].to_string(index=False))

print(f"\n[06_residual_diagnostic] DONE — all outputs at {fig_root}")
