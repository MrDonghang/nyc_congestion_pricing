"""Raw bus hourly ridership → daily route matrix.

Inputs
------
Two CSVs from the MTA open-data portal, both with columns ``transit_timestamp``,
``bus_route``, ``ridership``:
  * ``MTA_Bus_Hourly_Ridership__Beginning_2020-2024.csv``
  * ``MTA_Bus_Hourly_Ridership__Beginning_2025*.csv``

Output
------
``<output_root>/bus/bus_data_2022_2025_daily.csv`` — wide, date × route,
reindexed to a complete daily index from 2022-01-01 to the last observed day.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

START_DATE = pd.Timestamp("2022-01-01")


def _load_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])
    return df


def _build_daily_matrix(hourly: pd.DataFrame) -> pd.DataFrame:
    hourly["date"] = hourly["transit_timestamp"].dt.normalize()
    daily = hourly.groupby(["date", "bus_route"])["ridership"].sum().unstack()
    full_idx = pd.date_range(start=START_DATE, end=hourly["transit_timestamp"].max().normalize(), freq="D")
    daily = daily.reindex(full_idx)
    daily.index.name = "date"
    return daily


def process(
    hourly_2020_2024: Path,
    hourly_2025: Path,
    output_dir: Path,
) -> Path:
    """Run the full bus pipeline. Returns the written CSV path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading 2020-2024 hourly: %s", hourly_2020_2024)
    early = _load_hourly(hourly_2020_2024)
    early = early[early["transit_timestamp"] >= START_DATE]

    log.info("Loading 2025 hourly: %s", hourly_2025)
    late = _load_hourly(hourly_2025)

    hourly = pd.concat([early, late], ignore_index=True)
    log.info("Combined hourly rows: %d", len(hourly))

    daily = _build_daily_matrix(hourly)
    log.info("Daily matrix: %d days × %d routes", *daily.shape)

    out_csv = output_dir / "bus_data_2022_2025_daily.csv"
    daily.to_csv(out_csv)
    np.save(output_dir / "bus_data_2022_2025_daily.npy", daily.to_numpy())
    pd.Series(daily.columns, name="bus_route").to_csv(output_dir / "bus_routes.csv", index=False)

    log.info("Wrote %s", out_csv)
    return out_csv
