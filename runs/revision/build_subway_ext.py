"""Extend the subway station-day panel back to 2022-01-01.

Inputs
  <data_root>/subway/ext2022/hourly_2022_plus_jan2023.csv   rows of the raw MTA hourly file for 2022 (+ Jan 2023 for a
                                                             consistency check), extracted with grep
  <data_root>/subway/patterns/station_mapping.csv           station_complex_id -> matrix index (424 stations)
  <data_root>/subway/patterns/pattern_2023_MM.npz           (7, n, n) day-of-week destination shares, 2023
  <data_root>/subway/subway_2023_2025_daily_{O,D,total}.csv existing panel (2023-01-01 .. 2025-04-30)

Method
  O_2022[d, s]  = Σ_hours ridership(s, d)                      (exact, same as the 2023+ pipeline)
  D_2022[d, :]  = Σ_s O_2022[d, s] · pattern_2023_MM[dow(d), s, :]   (2023 same-month pattern; no 2022 OD estimate)
  total = (O + D) / 2        # matches the existing subway_2023_2025_daily_total.csv definition
Output: <data_root>/subway/subway_2022_2025_daily_{O,D,total}.csv  (2022-01-01 .. 2025-04-30)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from nyc_cp.config import load_paths

ROOT = Path(load_paths()["data_root"]) / "subway"
smap = pd.read_csv(ROOT / "patterns" / "station_mapping.csv")
s_idx = dict(zip(smap["station_id"].astype(int), smap["station_index"].astype(int)))
n = len(s_idx)

print("reading hourly extract ...")
h = pd.read_csv(ROOT / "ext2022" / "hourly_2022_plus_jan2023.csv", usecols=["transit_timestamp", "station_complex_id", "ridership"])
h["ts"] = pd.to_datetime(h["transit_timestamp"], format="%m/%d/%Y %I:%M:%S %p")
h["date"] = h["ts"].dt.normalize()
h["station_complex_id"] = pd.to_numeric(h["station_complex_id"], errors="coerce")
h = h.dropna(subset=["station_complex_id"])
h["o"] = h["station_complex_id"].astype(int).map(s_idx)
unmapped = h["o"].isna().mean()
print(f"rows={len(h):,}  unmapped station share={unmapped:.4%}")
h = h.dropna(subset=["o"]); h["o"] = h["o"].astype(int)
O = h.pivot_table(index="date", columns="o", values="ridership", aggfunc="sum", fill_value=0.0).reindex(columns=range(n), fill_value=0.0)
O.columns = [str(c) for c in O.columns]

old_O = pd.read_csv(ROOT / "subway_2023_2025_daily_O.csv", index_col=0, parse_dates=True)
old_D = pd.read_csv(ROOT / "subway_2023_2025_daily_D.csv", index_col=0, parse_dates=True)
old_T = pd.read_csv(ROOT / "subway_2023_2025_daily_total.csv", index_col=0, parse_dates=True)
# consistency check on Jan 2023
jan = O.loc["2023-01-01":"2023-01-31"]
diff = (jan - old_O.loc[jan.index, jan.columns]).abs()
print(f"Jan-2023 overlap: max |diff| = {diff.to_numpy().max():.3f}, rel L1 = {diff.to_numpy().sum() / old_O.loc[jan.index].to_numpy().sum():.2e}")
assert diff.to_numpy().max() < 1.0, "2022 extraction does not reproduce the existing 2023 panel"

O22 = O.loc["2022-01-01":"2022-12-31"]
full = pd.date_range("2022-01-01", "2022-12-31", freq="D")
missing = full.difference(O22.index)
print(f"2022 days: {len(O22)}  missing: {list(missing.date)}")
O22 = O22.reindex(full).fillna(0.0)

pats = {m: np.load(ROOT / "patterns" / f"pattern_2023_{m:02d}.npz")["pattern"] for m in range(1, 13)}
D22 = np.zeros((len(full), n))
for i, d in enumerate(full):
    D22[i] = O22.iloc[i].to_numpy() @ pats[d.month][d.dayofweek]
D22 = pd.DataFrame(D22, index=full, columns=O22.columns)
# the inferred-OD pipeline stores float32 and sums; mirror that rounding
T22 = (O22 + D22) / 2

for name, new, old in (("O", O22, old_O), ("D", D22, old_D), ("total", T22, old_T)):
    out = pd.concat([new, old.loc[:"2025-04-30", new.columns]], axis=0)
    out.index.name = "date"
    assert out.index.is_monotonic_increasing and not out.index.duplicated().any()
    f = ROOT / f"subway_2022_2025_daily_{name}.csv"
    out.to_csv(f, float_format="%.3f")
    print(f"wrote {f}  shape={out.shape}  range={out.index.min().date()}..{out.index.max().date()}")
    print(f"   {name}: 2022-03 mean daily sum = {new.loc['2022-03'].sum(axis=1).mean():,.0f}   2023-03 = {old.loc['2023-03'].sum(axis=1).mean():,.0f}")
