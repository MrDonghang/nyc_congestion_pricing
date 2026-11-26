"""Fetch post-April-2025 data, student fare classes excluded, aggregated server-side (data.ny.gov SODA API):
  bus    gxb3-akrn  → <data_root>/bus/ext2026/bus_nostudent_2025_2026_long.csv       (d, bus_route, r)
  subway 5wq4-mkjj  → <data_root>/subway/ext2026/subway_nostudent_2025_2026_long.csv (d, station_complex_id, r)
Usage: python runs/revision/fetch_extension.py bus|subway
"""
import io, sys, time
from pathlib import Path
import pandas as pd, requests
from nyc_cp.config import load_paths
which = sys.argv[1]; root = Path(load_paths()["data_root"]) / which / "ext2026"; root.mkdir(parents=True, exist_ok=True)
ds, key, extra = {"bus": ("gxb3-akrn", "bus_route", ""), "subway": ("5wq4-mkjj", "station_complex_id", " AND transit_mode = 'subway'")}[which]
frames = []
for start, end in [("2025-01-01", "2025-06-30"), ("2025-07-01", "2025-12-31"), ("2026-01-01", "2026-06-30"), ("2026-07-01", "2026-12-31")]:
    offset = 0
    while True:
        p = {"$select": f"date_trunc_ymd(transit_timestamp) as d, {key}, sum(ridership) as r",
             "$where": f"transit_timestamp between '{start}T00:00:00' and '{end}T23:59:59' AND fare_class_category not like '%Students%'{extra}",
             "$group": f"d, {key}", "$order": f"d, {key}", "$limit": 50000, "$offset": offset}
        for attempt in range(6):
            try:
                r = requests.get(f"https://data.ny.gov/resource/{ds}.csv", params=p, timeout=900); r.raise_for_status(); break
            except Exception as e:  # noqa: BLE001
                print(f"  retry {attempt} ({str(e)[:120]})", flush=True); time.sleep(30)
        else: raise SystemExit("API failed repeatedly")
        df = pd.read_csv(io.StringIO(r.text)); print(which, start, "offset", offset, len(df), flush=True); frames.append(df)
        if len(df) < 50000: break
        offset += 50000
out = pd.concat(frames, ignore_index=True); out["d"] = pd.to_datetime(out["d"]).dt.normalize()
out.to_csv(root / f"{which}_nostudent_2025_2026_long.csv", index=False); print("wrote", len(out), out.d.min().date(), out.d.max().date())
