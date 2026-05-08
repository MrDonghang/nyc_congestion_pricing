"""Read pre-processed daily/weekly ridership matrices for any mode.

Raw → matrix conversion lives in ``nyc_cp.data.{bus,subway,citibike,replica}``;
this module is *only* about loading the already-processed wide CSVs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nyc_cp.config import Direction, actual_csv, load_mode, load_paths


def load_actual(
    mode: str,
    direction: Direction = "all",
    mode_cfg: dict | None = None,
    paths: dict | None = None,
) -> pd.DataFrame:
    """Load the actual ridership matrix and apply mode-level filtering.

    Filters honoured (when present in the mode config):
      * ``exclude_columns``                 — drop named series
      * ``drop_all_zero_columns``           — drop series that are all 0
      * ``drop_columns_with_zero_share_above`` — drop sparse series
      * ``weekday_only``                    — keep Mon–Fri only
    """
    paths = paths or load_paths()
    mode_cfg = mode_cfg or load_mode(mode)
    path: Path = actual_csv(mode, direction=direction, paths=paths)

    df = pd.read_csv(path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"}).set_index("date").sort_index()

    if cols := mode_cfg.get("exclude_columns"):
        df = df.drop(columns=[c for c in cols if c in df.columns])

    if mode_cfg.get("drop_all_zero_columns"):
        df = df.loc[:, (df != 0).any(axis=0)]

    if (thr := mode_cfg.get("drop_columns_with_zero_share_above")) is not None:
        df = df.loc[:, df.apply(lambda s: (s == 0).mean() < thr)]

    if mode_cfg.get("weekday_only"):
        df = df[df.index.weekday < 5]

    df.columns = df.columns.astype(str)
    return df
