"""Raw Replica weekly OD trip-count zips → census-tract / PUMA matrices.

Inputs
------
A directory of weekly trip-count zips matching
``trends-trip-count-od-origin-v2_from-week-of*.zip``. Each zip contains a CSV
with columns ``week_starting``, ``origin_geoid``, ``destination_geoid``,
``trip_count`` (and optionally ``mode``).

Outputs (under ``<output_root>/replica/od/``)
---------------------------------------------
* ``census_tract_origin_matrix.csv``      — week × tract origin totals
* ``census_tract_destination_matrix.csv`` — week × tract destination totals
* ``census_tract_od_<region>_<by_mode>.npz`` — full (T, O, D) tensor
"""

from __future__ import annotations

import glob
import logging
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _list_zips(raw_dir: Path) -> list[Path]:
    pattern = str(raw_dir / "trends-trip-count-od-origin-v2_from-week-of*.zip")
    return sorted(Path(p) for p in glob.glob(pattern))


def _load_geo_index(geo_csv: Path) -> dict[str, int]:
    """``geo_csv`` must have columns ``geoid`` (string) and ``index`` (int)."""
    df = pd.read_csv(geo_csv, dtype={"geoid": str})
    return dict(zip(df["geoid"], df["index"].astype(int)))


def _iter_chunks(zf: zipfile.ZipFile, chunk_size: int = 100_000):
    name = next(n for n in zf.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX"))
    yield from pd.read_csv(zf.open(name), chunksize=chunk_size)


def process(
    raw_dir: Path,
    geo_csv: Path,
    output_dir: Path,
    geo_level: Literal["census_tract", "puma"] = "census_tract",
    by_mode: bool = False,
) -> Path:
    """Build (T, O, D) trip-count tensors from Replica weekly zips."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir) / "od"
    output_dir.mkdir(parents=True, exist_ok=True)

    idx = _load_geo_index(geo_csv)
    n = len(idx)
    log.info("Loaded %s index: %d entries", geo_level, n)

    zips = _list_zips(raw_dir)
    if not zips:
        raise FileNotFoundError(f"No Replica weekly zips found in {raw_dir}")
    log.info("Found %d weekly zips", len(zips))

    weeks: list[pd.Timestamp] = []
    week_to_t: dict[pd.Timestamp, int] = {}
    od_by_mode: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((0, n, n), dtype=np.float32))

    for zp in zips:
        log.info("Reading %s", zp.name)
        with zipfile.ZipFile(zp) as zf:
            for chunk in _iter_chunks(zf):
                chunk["week_starting"] = pd.to_datetime(chunk["week_starting"])
                chunk["origin_geoid"] = chunk["origin_geoid"].astype(str)
                chunk["destination_geoid"] = chunk["destination_geoid"].astype(str)
                chunk["o"] = chunk["origin_geoid"].map(idx)
                chunk["d"] = chunk["destination_geoid"].map(idx)
                chunk = chunk.dropna(subset=["o", "d"])
                if chunk.empty:
                    continue
                chunk["o"] = chunk["o"].astype(int)
                chunk["d"] = chunk["d"].astype(int)

                modes = chunk["mode"].unique() if (by_mode and "mode" in chunk.columns) else ["all"]
                for m in modes:
                    sub = chunk if m == "all" else chunk[chunk["mode"] == m]
                    for week, sub_w in sub.groupby("week_starting"):
                        if week not in week_to_t:
                            week_to_t[week] = len(weeks)
                            weeks.append(week)
                            for k in od_by_mode:
                                od_by_mode[k] = np.concatenate(
                                    [od_by_mode[k], np.zeros((1, n, n), dtype=np.float32)], axis=0
                                )
                        if m not in od_by_mode:
                            od_by_mode[m] = np.zeros((len(weeks), n, n), dtype=np.float32)
                        elif od_by_mode[m].shape[0] < len(weeks):
                            od_by_mode[m] = np.concatenate(
                                [od_by_mode[m], np.zeros((len(weeks) - od_by_mode[m].shape[0], n, n), dtype=np.float32)],
                                axis=0,
                            )
                        t = week_to_t[week]
                        np.add.at(
                            od_by_mode[m],
                            (t, sub_w["o"].to_numpy(), sub_w["d"].to_numpy()),
                            sub_w["trip_count"].to_numpy(dtype=np.float32),
                        )

    weeks_idx = pd.DatetimeIndex(weeks)
    cols = [str(i) for i in range(n)]

    for mode_name, mat in od_by_mode.items():
        npz = output_dir / f"{geo_level}_od_{mode_name}.npz"
        np.savez(npz, matrix=mat, dates=weeks_idx.astype(str).to_numpy())
        log.info("Wrote %s shape=%s", npz, mat.shape)

    if "all" in od_by_mode:
        all_mat = od_by_mode["all"]
        pd.DataFrame(all_mat.sum(axis=2), index=weeks_idx, columns=cols).to_csv(
            output_dir / f"{geo_level}_origin_matrix.csv", index_label="date"
        )
        pd.DataFrame(all_mat.sum(axis=1), index=weeks_idx, columns=cols).to_csv(
            output_dir / f"{geo_level}_destination_matrix.csv", index_label="date"
        )

    return output_dir
