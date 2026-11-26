"""Align the student-excluded bus matrix with the existing 'final' bus matrix:
same 246 routes, same date range (2022-01-07 → 2025-04-30), no NaN.
Writes <data_root>/bus/bus_nostudent_2022_2025_daily_final.csv and prints a
reconciliation against the full-fare-class matrix."""
from pathlib import Path
import numpy as np, pandas as pd
from nyc_cp.config import load_paths

root = Path(load_paths()["data_root"]) / "bus"
full = pd.read_csv(root / "bus_data_2022_2025_daily_final.csv", index_col=0, parse_dates=True)
ns = pd.read_csv(root / "bus_nostudent_2022_2025_daily.csv", index_col=0, parse_dates=True)
ns = ns.reindex(index=full.index, columns=full.columns)
print("missing cells before fill:", int(ns.isna().sum().sum()), "of", ns.size)
# routes present in full but absent in API pull (naming differences) → report
missing_routes = [c for c in full.columns if ns[c].isna().all()]
print("routes entirely missing in API pull:", missing_routes)
ns = ns.ffill(limit=3).fillna(0.0)
ns.to_csv(root / "bus_nostudent_2022_2025_daily_final.csv")
# reconciliation: student share by month
share = 1 - ns.sum(axis=1).resample("MS").sum() / full.sum(axis=1).resample("MS").sum()
print("implied student share of counted boardings by month (%):")
print((share * 100).round(1).to_frame("student_%").T.to_string())
print("wrote", root / "bus_nostudent_2022_2025_daily_final.csv", ns.shape)
