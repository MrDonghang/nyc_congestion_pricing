"""Raw Citibike trip data → census-tract origin/destination daily series.

Inputs
------
A directory of monthly trip-data zip files from the Citibike open-data S3 bucket
(``*citibike-tripdata*.zip``). Each zip contains a CSV with columns including
``started_at``, ``ended_at``, ``start_lat``, ``start_lng``, ``end_lat``,
``end_lng``. NYC census tract polygons (shapefile) are needed to spatially
assign each trip end to a tract.

Outputs (under ``<output_root>/citibike/census/``)
--------------------------------------------------
* ``citibike_censustract_od_<region>.npz`` — keys ``matrix`` (T, O, D),
  ``dates``, ``tract_index``.
* ``citibike_censustract_od_<region>_O.csv`` — daily origin-side totals per tract.
* ``citibike_censustract_od_<region>_D.csv`` — daily destination-side totals.
* ``censustract_idx_mapping.pkl``           — tract GEOID → matrix index.
"""

from __future__ import annotations

import glob
import logging
import pickle
import re
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

Region = Literal["NYC", "JC"]


def _list_zips(raw_dir: Path, region: Region) -> list[Path]:
    pattern = "JC-*citibike-tripdata*.zip" if region == "JC" else "*citibike-tripdata*.zip"
    files = [Path(p) for p in glob.glob(str(raw_dir / pattern))]
    if region == "NYC":
        files = [f for f in files if not f.name.startswith("JC-")]
    return sorted(files, key=lambda p: re.search(r"(\d{6})", p.name).group(1) if re.search(r"(\d{6})", p.name) else "0")


def _build_tract_index(census_geo: "geopandas.GeoDataFrame") -> dict[str, int]:
    geoids = sorted(census_geo["GEOID"].astype(str).unique())
    return {g: i for i, g in enumerate(geoids)}


def _read_trip_csv(zf: zipfile.ZipFile) -> pd.DataFrame:
    name = next(n for n in zf.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX"))
    return pd.read_csv(zf.open(name))


def _assign_tracts(trips: pd.DataFrame, tracts_gdf, idx_map: dict[str, int]) -> pd.DataFrame:
    """Spatially join trip start/end coordinates to census-tract indices."""
    import geopandas as gpd
    from shapely.geometry import Point

    trips = trips.dropna(subset=["start_lat", "start_lng", "end_lat", "end_lng"]).copy()
    trips["started_at"] = pd.to_datetime(trips["started_at"])
    trips["date"] = trips["started_at"].dt.normalize()

    starts = gpd.GeoDataFrame(
        trips[["date"]].copy(),
        geometry=[Point(x, y) for x, y in zip(trips["start_lng"], trips["start_lat"])],
        crs="EPSG:4326",
    ).to_crs(tracts_gdf.crs)
    ends = gpd.GeoDataFrame(
        trips[["date"]].copy(),
        geometry=[Point(x, y) for x, y in zip(trips["end_lng"], trips["end_lat"])],
        crs="EPSG:4326",
    ).to_crs(tracts_gdf.crs)

    starts = gpd.sjoin(starts, tracts_gdf[["GEOID", "geometry"]], how="left", predicate="within")
    ends = gpd.sjoin(ends, tracts_gdf[["GEOID", "geometry"]], how="left", predicate="within")

    out = pd.DataFrame({
        "date": starts["date"].to_numpy(),
        "o": starts["GEOID"].astype(str).map(idx_map).to_numpy(),
        "d": ends["GEOID"].astype(str).map(idx_map).to_numpy(),
    })
    return out.dropna().astype({"o": int, "d": int})


def process(
    raw_dir: Path,
    census_shp: Path,
    output_dir: Path,
    region: Region = "NYC",
) -> Path:
    """Run the citibike pipeline. Returns the .npz path."""
    import geopandas as gpd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading census tract polygons: %s", census_shp)
    tracts = gpd.read_file(census_shp)
    if "GEOID" not in tracts.columns:
        # Some shapefiles use lowercase
        for c in ("geoid", "BoroCT2020", "ct2020"):
            if c in tracts.columns:
                tracts = tracts.rename(columns={c: "GEOID"})
                break
    idx_map = _build_tract_index(tracts)
    n = len(idx_map)
    log.info("Tract index size: %d", n)

    zips = _list_zips(raw_dir, region)
    log.info("Found %d %s zips under %s", len(zips), region, raw_dir)

    daily_records: dict[pd.Timestamp, np.ndarray] = {}
    for zp in zips:
        log.info("Processing %s", zp.name)
        with zipfile.ZipFile(zp) as zf:
            trips = _read_trip_csv(zf)
        assigned = _assign_tracts(trips, tracts, idx_map)
        for date, sub in assigned.groupby("date"):
            mat = daily_records.setdefault(date, np.zeros((n, n), dtype=np.int32))
            np.add.at(mat, (sub["o"].to_numpy(), sub["d"].to_numpy()), 1)

    dates = pd.DatetimeIndex(sorted(daily_records.keys()))
    matrix = np.stack([daily_records[d] for d in dates])  # (T, O, D)
    log.info("Output matrix: T=%d O=%d D=%d", *matrix.shape)

    npz = output_dir / f"citibike_censustract_od_{region}.npz"
    np.savez(npz, matrix=matrix, dates=dates.astype(str).to_numpy(), tract_index=np.array(list(idx_map.keys())))
    with open(output_dir / "censustract_idx_mapping.pkl", "wb") as f:
        pickle.dump(idx_map, f)

    cols = [str(i) for i in range(n)]
    pd.DataFrame(matrix.sum(axis=2), index=dates, columns=cols).to_csv(
        output_dir / f"citibike_censustract_od_{region}_O.csv", index_label="date"
    )
    pd.DataFrame(matrix.sum(axis=1), index=dates, columns=cols).to_csv(
        output_dir / f"citibike_censustract_od_{region}_D.csv", index_label="date"
    )
    log.info("Wrote %s + O/D CSVs", npz)
    return npz
