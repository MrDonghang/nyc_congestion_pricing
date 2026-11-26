"""Daily system-wide subway ridership by fare class from the raw MTA hourly files (2023-01 → 2025-04).
Writes <data_root>/subway/subway_daily_by_fare_class.csv (date × fare_class_category).
"""
import pandas as pd
from pathlib import Path
RAW = Path("/public_dataset/donghang/nyc_congestion_pricing/raw_data/subway")
OUT = Path("/public_dataset/donghang/nyc_congestion_pricing/data_processed/subway/subway_daily_by_fare_class.csv")
parts = []
for f in ["MTA_Subway_Hourly_Ridership__2020-2024.csv", "MTA_Subway_Hourly_Ridership__Beginning_2025_250612.csv"]:
    for i, ch in enumerate(pd.read_csv(RAW / f, usecols=["transit_timestamp", "transit_mode", "fare_class_category", "ridership"], chunksize=4_000_000, dtype={"fare_class_category": "category", "transit_mode": "category"})):
        ch = ch[ch.transit_mode == "subway"]
        ch["date"] = ch.transit_timestamp.str.slice(0, 10)
        parts.append(ch.groupby(["date", "fare_class_category"], observed=True).ridership.sum())
        if i % 10 == 0: print(f, i, flush=True)
s = pd.concat(parts).groupby(level=[0, 1]).sum().unstack(fill_value=0)
s.index = pd.to_datetime(s.index, format="%m/%d/%Y"); s = s.sort_index().loc["2022-01-01":]
s.to_csv(OUT); print("wrote", OUT, s.shape, list(s.columns))
