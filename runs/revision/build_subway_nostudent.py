"""Station × day subway entries EXCLUDING student fare classes (Metrocard - Students, OMNY - Students),
same 424 stations / column order / date range as subway_2023_2025_daily_O.csv.
Writes <data_root>/subway/subway_nostudent_2023_2025_daily_O.csv and prints a consistency check.
"""
import pandas as pd
from pathlib import Path
RAW = Path("/public_dataset/donghang/nyc_congestion_pricing/raw_data/subway")
D = Path("/public_dataset/donghang/nyc_congestion_pricing/data_processed/subway")
ref = pd.read_csv(D / "subway_2023_2025_daily_O.csv", index_col=0, parse_dates=True)
mp = pd.read_csv(D / "patterns" / "station_mapping.csv")
m = dict(zip(mp.station_id.astype(int).astype(str), mp.station_index.astype(int)))
parts, allp = [], []
for f in ["MTA_Subway_Hourly_Ridership__2020-2024.csv", "MTA_Subway_Hourly_Ridership__Beginning_2025_250612.csv"]:
    for i, ch in enumerate(pd.read_csv(RAW / f, usecols=["transit_timestamp", "transit_mode", "station_complex_id", "fare_class_category", "ridership"], chunksize=4_000_000,
                                        dtype={"fare_class_category": "category", "transit_mode": "category", "station_complex_id": str})):
        ch = ch[ch.transit_mode == "subway"]; ch["date"] = ch.transit_timestamp.str.slice(0, 10)
        ch = ch[ch.date.str.slice(6, 10).isin(["2023", "2024", "2025"])]
        if ch.empty: continue
        ch["idx"] = ch.station_complex_id.map(m); ch = ch.dropna(subset=["idx"])
        allp.append(ch.groupby(["date", "idx"]).ridership.sum())
        parts.append(ch[~ch.fare_class_category.astype(str).str.contains("Students")].groupby(["date", "idx"]).ridership.sum())
        if i % 10 == 0: print(f, i, flush=True)
def wide(ps):
    w = pd.concat(ps).groupby(level=[0, 1]).sum().unstack(fill_value=0); w.index = pd.to_datetime(w.index, format="%m/%d/%Y")
    w.columns = [str(int(c)) for c in w.columns]; return w.sort_index().reindex(index=ref.index, columns=ref.columns).fillna(0)
ns, al = wide(parts), wide(allp)
ns.index.name = ref.index.name; ns.to_csv(D / "subway_nostudent_2023_2025_daily_O.csv")
print("wrote", ns.shape, "| rebuilt-all vs existing O panel: max abs diff", float((al - ref).abs().max().max()), " total ratio", float(al.sum().sum() / ref.sum().sum()))
print("student share of entries: %.2f%%" % (100 * (1 - ns.sum().sum() / al.sum().sum())))
