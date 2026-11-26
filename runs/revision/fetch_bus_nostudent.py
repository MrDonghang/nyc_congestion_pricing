"""Rebuild the daily route × ridership bus matrix EXCLUDING student fare classes,
straight from the MTA open-data API (server-side aggregation; no multi-GB download).

Why: in Sept 2024 student fares moved from MetroCard to OMNY and counted student
boardings roughly doubled ("Metrocard - Students" ~2M/month → "OMNY - Students"
~3-4M/month). That is a measurement change, not a demand change, and it sits
inside the forecast horizon's training tail / the 2025 test window. Dropping the
two student classes gives a series that is consistent across the break.

Datasets: kv7t-n8in (2020-2024, static), gxb3-akrn (Beginning 2025).
Output: <data_root>/bus/bus_nostudent_2022_2025_daily.csv  (date × route)
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from nyc_cp.config import load_paths

DATASETS = {"kv7t-n8in": [("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31")],
            "gxb3-akrn": [("2025-01-01", "2025-06-30")]}
LIMIT = 50000
OUT = Path(load_paths()["data_root"]) / "bus" / "bus_nostudent_2022_2025_daily.csv"
RAW = OUT.with_suffix(".long.csv")


def fetch(ds: str, start: str, end: str) -> pd.DataFrame:
    frames, offset = [], 0
    while True:
        params = {
            "$select": "date_trunc_ymd(transit_timestamp) as d, bus_route, sum(ridership) as r",
            "$where": f"transit_timestamp between '{start}T00:00:00' and '{end}T23:59:59' AND fare_class_category not like '%Students%'",
            "$group": "d, bus_route", "$order": "d, bus_route", "$limit": LIMIT, "$offset": offset,
        }
        for attempt in range(4):
            try:
                r = requests.get(f"https://data.ny.gov/resource/{ds}.csv", params=params, timeout=600)
                r.raise_for_status(); break
            except Exception as e:  # noqa: BLE001
                print(f"  retry {attempt} ({e})", flush=True); time.sleep(20)
        else:
            raise SystemExit("API failed repeatedly")
        df = pd.read_csv(io.StringIO(r.text))
        print(f"  {ds} {start}..{end} offset={offset}: {len(df)} rows", flush=True)
        frames.append(df)
        if len(df) < LIMIT:
            break
        offset += LIMIT
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parts = [fetch(ds, s, e) for ds, rngs in DATASETS.items() for s, e in rngs]
    long = pd.concat(parts, ignore_index=True)
    long["d"] = pd.to_datetime(long["d"]).dt.normalize()
    long.to_csv(RAW, index=False)
    wide = long.pivot_table(index="d", columns="bus_route", values="r", aggfunc="sum")
    full_idx = pd.date_range("2022-01-01", wide.index.max(), freq="D")
    wide = wide.reindex(full_idx)
    wide.index.name = "date"
    wide.to_csv(OUT)
    print(f"wrote {OUT}  shape={wide.shape}  range={wide.index.min().date()}..{wide.index.max().date()}  NaN={int(wide.isna().sum().sum())}")


if __name__ == "__main__":
    main()
