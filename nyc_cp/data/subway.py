"""Raw subway hourly + OD-estimate data → daily station-level OD matrices.

Pipeline (three steps)
----------------------
1. ``build_patterns`` — for each (year, month) bucket of the MTA OD-estimate
   file, build a ``(7, n_stations, n_stations)`` tensor giving the destination
   distribution by day-of-week. Hour-of-day is summed out before normalising.
   Patterns are saved per month: ``pattern_<year>_<month>.npz`` with key
   ``pattern``. This matches the schema produced by the original pipeline.
2. ``infer_daily_od`` — read hourly ridership, look up the matching per-month
   per-dow pattern slice for each row, and accumulate ``ridership * pattern``
   into a daily station-level OD tensor. Hour-of-day is not used at inference
   time (the dow pattern is reused across hours of the same day).
3. ``aggregate_od`` — sum daily station OD into census-tract or PUMA
   resolution using a station→region mapping.

Source-file schemas (verified from the actual MTA exports)
----------------------------------------------------------
* OD-estimate CSV: ``Year, Month, Day of Week, Hour of Day,
  Origin Station Complex ID, Destination Station Complex ID,
  Estimated Average Ridership``.
* Hourly ridership CSV: ``transit_timestamp, station_complex_id, ridership``.

Outputs (under ``<output_dir>/``)
---------------------------------
* ``patterns/station_mapping.csv``           — station_id ↔ matrix index
* ``patterns/pattern_<year>_<month>.npz``    — (7, n, n) per-month patterns
* ``patterns/metadata.json``                 — n_stations + month list
* ``subway_2023_2025_daily_O.csv``           — daily origin totals per station
* ``subway_2023_2025_daily_D.csv``           — daily destination totals per station
* ``subway_2023_2025_daily_OD.npz``          — full (T, O, D) tensor + dates
* ``subway_2023_2025_daily_<region>_O.csv`` / ``_D.csv`` — aggregates
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

DOW = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}


# ---------------------------------------------------------------------- step 1


def _build_station_index(od_estimate_csv: Path) -> tuple[list, dict]:
    """Take the union of origin+destination station IDs across the whole file."""
    log.info("Scanning station IDs from %s", od_estimate_csv)
    stations = set()
    for chunk in pd.read_csv(
        od_estimate_csv,
        usecols=["Origin Station Complex ID", "Destination Station Complex ID"],
        chunksize=500_000,
        low_memory=False,
    ):
        stations.update(chunk["Origin Station Complex ID"].unique())
        stations.update(chunk["Destination Station Complex ID"].unique())
    stations = sorted(stations)
    return stations, {s: i for i, s in enumerate(stations)}


def build_patterns(od_estimate_csv: Path, output_dir: Path) -> Path:
    """Build per-month ``(7, n, n)`` destination-share patterns.

    Hour-of-day is summed out within each (year, month, dow, origin, destination)
    bucket before destinations are normalised to a probability distribution.
    """
    output_dir = Path(output_dir)
    patterns_dir = output_dir / "patterns"
    patterns_dir.mkdir(parents=True, exist_ok=True)

    stations, s_idx = _build_station_index(od_estimate_csv)
    n = len(stations)
    log.info("Stations: %d", n)
    pd.DataFrame({"station_id": stations, "station_index": range(n)}).to_csv(
        patterns_dir / "station_mapping.csv", index=False
    )

    # Aggregate: counts[(year, month, dow)] is an (n, n) array.
    counts: dict[tuple[int, int, int], np.ndarray] = {}
    cols = [
        "Year",
        "Month",
        "Day of Week",
        "Origin Station Complex ID",
        "Destination Station Complex ID",
        "Estimated Average Ridership",
    ]
    for chunk in tqdm(
        pd.read_csv(od_estimate_csv, usecols=cols, chunksize=500_000, low_memory=False),
        desc="reading OD estimate",
    ):
        chunk["dow"] = chunk["Day of Week"].map(DOW)
        chunk["o"] = chunk["Origin Station Complex ID"].map(s_idx)
        chunk["d"] = chunk["Destination Station Complex ID"].map(s_idx)
        chunk = chunk.dropna(subset=["o", "d", "dow"])

        for (yr, mo, dw), sub in chunk.groupby(["Year", "Month", "dow"]):
            mat = counts.setdefault((int(yr), int(mo), int(dw)), np.zeros((n, n), dtype=np.float64))
            np.add.at(
                mat,
                (sub["o"].astype(int).to_numpy(), sub["d"].astype(int).to_numpy()),
                sub["Estimated Average Ridership"].to_numpy(),
            )

    # Group counts by (year, month) → (7, n, n) tensor, then row-normalise.
    months: dict[tuple[int, int], np.ndarray] = {}
    for (yr, mo, dw), mat in counts.items():
        months.setdefault((yr, mo), np.zeros((7, n, n), dtype=np.float64))[dw] = mat

    for (yr, mo), pat in months.items():
        row_sums = pat.sum(axis=2, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        pat_norm = (pat / row_sums).astype(np.float64)
        npz = patterns_dir / f"pattern_{yr}_{mo:02d}.npz"
        np.savez(npz, pattern=pat_norm)

    with (patterns_dir / "metadata.json").open("w") as f:
        json.dump(
            {
                "n_stations": n,
                "pattern_shape": [7, n, n],
                "year_month_combinations": sorted([list(k) for k in months.keys()]),
            },
            f,
            indent=2,
        )
    log.info("Wrote %d per-month patterns to %s", len(months), patterns_dir)
    return patterns_dir


# ---------------------------------------------------------------------- step 2


def _load_patterns(patterns_dir: Path) -> dict[tuple[int, int], np.ndarray]:
    """Load all ``pattern_<year>_<month>.npz`` into a (year, month) → array map."""
    pats: dict[tuple[int, int], np.ndarray] = {}
    for p in sorted(Path(patterns_dir).glob("pattern_*.npz")):
        # filename: pattern_<year>_<month>.npz
        parts = p.stem.split("_")
        yr, mo = int(parts[1]), int(parts[2])
        pats[(yr, mo)] = np.load(p)["pattern"]
    if not pats:
        raise FileNotFoundError(f"No pattern_*.npz files found under {patterns_dir}")
    return pats


def _iter_hourly(paths: Iterable[Path], chunksize: int = 1_000_000):
    cols = ["transit_timestamp", "station_complex_id", "ridership"]
    for path in paths:
        log.info("Reading hourly: %s", path)
        for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize, low_memory=False):
            chunk["transit_timestamp"] = pd.to_datetime(chunk["transit_timestamp"])
            yield chunk


def infer_daily_od(
    hourly_csvs: list[Path],
    patterns_dir: Path,
    station_map_csv: Path,
    output_dir: Path,
    start_date: pd.Timestamp = pd.Timestamp("2023-01-01"),
) -> Path:
    """Multiply hourly ridership by the per-month-dow destination pattern."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    smap = pd.read_csv(station_map_csv)
    s_idx = dict(zip(smap["station_id"], smap["station_index"].astype(int)))
    n = len(s_idx)

    pats = _load_patterns(patterns_dir)
    log.info("Loaded %d per-month patterns (n=%d)", len(pats), n)

    daily: dict[pd.Timestamp, np.ndarray] = {}

    for chunk in tqdm(_iter_hourly(hourly_csvs), desc="OD inference"):
        chunk["date"] = chunk["transit_timestamp"].dt.normalize()
        chunk = chunk[chunk["date"] >= start_date]
        if chunk.empty:
            continue
        chunk["dow"] = chunk["transit_timestamp"].dt.dayofweek
        chunk["o"] = chunk["station_complex_id"].map(s_idx)
        chunk = chunk.dropna(subset=["o"])
        chunk["o"] = chunk["o"].astype(int)
        chunk["year"] = chunk["transit_timestamp"].dt.year
        chunk["month"] = chunk["transit_timestamp"].dt.month

        for (date, yr, mo, dw), sub in chunk.groupby(["date", "year", "month", "dow"]):
            pat = pats.get((int(yr), int(mo)))
            if pat is None:
                continue
            slab = pat[int(dw)]  # (n, n)
            mat = daily.setdefault(pd.Timestamp(date), np.zeros((n, n), dtype=np.float32))
            ridership_per_origin = np.bincount(
                sub["o"].to_numpy(), weights=sub["ridership"].to_numpy(dtype=np.float64), minlength=n
            )
            mat += (ridership_per_origin[:, None] * slab).astype(np.float32)

    days = pd.DatetimeIndex(sorted(daily.keys()))
    od = np.stack([daily[d] for d in days])
    log.info("Output OD shape: %s", od.shape)

    cols = [str(i) for i in range(n)]
    pd.DataFrame(od.sum(axis=2), index=days, columns=cols).to_csv(
        output_dir / "subway_2023_2025_daily_O.csv", index_label="date"
    )
    pd.DataFrame(od.sum(axis=1), index=days, columns=cols).to_csv(
        output_dir / "subway_2023_2025_daily_D.csv", index_label="date"
    )
    npz = output_dir / "subway_2023_2025_daily_OD.npz"
    np.savez(npz, matrix=od, dates=days.astype(str).to_numpy())
    log.info("Wrote daily O/D CSVs and %s", npz)
    return npz


# ---------------------------------------------------------------------- step 3


def aggregate_od(
    od_npz: Path,
    station_to_region_csv: Path,
    output_dir: Path,
    region_name: Literal["census", "puma"] = "census",
) -> tuple[Path, Path]:
    """Aggregate station-level OD to a coarser geography.

    ``station_to_region_csv`` must have columns ``station_index`` (int) and
    ``region_id`` (string identifier).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(od_npz, allow_pickle=True)
    od = data["matrix"]  # (T, O, D)
    dates = pd.to_datetime(data["dates"])
    s2r = pd.read_csv(station_to_region_csv)
    region_ids = sorted(s2r["region_id"].astype(str).unique())
    r_idx = {r: i for i, r in enumerate(region_ids)}
    s_to_r = s2r.set_index("station_index")["region_id"].astype(str).map(r_idx).to_dict()

    R = len(region_ids)
    out = np.zeros((od.shape[0], R, R), dtype=np.float32)
    # Vectorised: build station→region row vectors then OD = M^T (od_per_station) M.
    M = np.zeros((od.shape[1], R), dtype=np.float32)
    for s, r in s_to_r.items():
        M[int(s), int(r)] = 1.0
    for t in range(od.shape[0]):
        out[t] = M.T @ od[t] @ M

    cols = region_ids
    o_csv = output_dir / f"subway_2023_2025_daily_{region_name}_O.csv"
    d_csv = output_dir / f"subway_2023_2025_daily_{region_name}_D.csv"
    pd.DataFrame(out.sum(axis=2), index=dates, columns=cols).to_csv(o_csv, index_label="date")
    pd.DataFrame(out.sum(axis=1), index=dates, columns=cols).to_csv(d_csv, index_label="date")
    log.info("Wrote %s and %s", o_csv, d_csv)
    return o_csv, d_csv


def process(
    od_estimate_csv: Path,
    hourly_csvs: list[Path],
    output_dir: Path,
    station_to_region_csv: Path | None = None,
    region_name: Literal["census", "puma"] = "census",
    start_date: pd.Timestamp = pd.Timestamp("2023-01-01"),
) -> Path:
    """Run all three steps end-to-end."""
    patterns_dir = build_patterns(od_estimate_csv, output_dir)
    smap = patterns_dir / "station_mapping.csv"
    npz = infer_daily_od(hourly_csvs, patterns_dir, smap, output_dir, start_date=start_date)
    if station_to_region_csv is not None:
        aggregate_od(npz, station_to_region_csv, output_dir, region_name=region_name)
    return npz
