"""Build the daily/weekly ridership matrices from raw open-data files.

Examples
--------
    python -m scripts.process_data --mode bus      \\
        --hourly-2020-2024 /raw/MTA_Bus_Hourly_2020_2024.csv \\
        --hourly-2025      /raw/MTA_Bus_Hourly_2025.csv

    python -m scripts.process_data --mode subway \\
        --od-estimate /raw/MTA_Subway_OD_Estimate.csv \\
        --hourly /raw/MTA_Subway_Hourly_2020_2024.csv  /raw/MTA_Subway_Hourly_2025.csv

    python -m scripts.process_data --mode replica \\
        --raw-dir /raw/replica/   --geo-csv configs/replica_geoid_to_tract_index.csv

The output directory is taken from ``configs/paths.yaml`` (``data_root``) unless
``--output-dir`` is given.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nyc_cp.config import load_paths
from nyc_cp.utils import setup_logging

log = logging.getLogger("process_data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["bus", "subway", "replica"])
    p.add_argument("--output-dir", type=Path, default=None)

    # Bus
    p.add_argument("--hourly-2020-2024", type=Path)
    p.add_argument("--hourly-2025", type=Path)

    # Subway
    p.add_argument("--od-estimate", type=Path)
    p.add_argument("--hourly", type=Path, nargs="+")
    p.add_argument("--station-to-region", type=Path)
    p.add_argument("--region-name", choices=["census", "puma"], default="census")

    # Replica
    p.add_argument("--raw-dir", type=Path)
    p.add_argument("--geo-csv", type=Path)
    p.add_argument("--geo-level", choices=["census_tract", "puma"], default="census_tract")
    p.add_argument("--by-mode", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths()
    setup_logging(f"process_{args.mode}", log_root=paths["log_root"])

    out = args.output_dir or Path(paths["data_root"]) / args.mode

    if args.mode == "bus":
        from nyc_cp.data.bus import process

        if not (args.hourly_2020_2024 and args.hourly_2025):
            raise SystemExit("--hourly-2020-2024 and --hourly-2025 are required for bus.")
        process(args.hourly_2020_2024, args.hourly_2025, out)

    elif args.mode == "subway":
        from nyc_cp.data.subway import process

        if not (args.od_estimate and args.hourly):
            raise SystemExit("--od-estimate and --hourly are required for subway.")
        process(
            args.od_estimate,
            list(args.hourly),
            out,
            station_to_region_csv=args.station_to_region,
            region_name=args.region_name,
        )

    elif args.mode == "replica":
        from nyc_cp.data.replica import process

        if not (args.raw_dir and args.geo_csv):
            raise SystemExit("--raw-dir and --geo-csv are required for replica.")
        out = args.output_dir or Path(paths["data_root"]) / "replica"
        process(args.raw_dir, args.geo_csv, out, geo_level=args.geo_level, by_mode=args.by_mode)


if __name__ == "__main__":
    main()
