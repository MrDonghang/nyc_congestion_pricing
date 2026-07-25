"""Generate the simulated demo dataset (demo_data/bus_demo_daily.csv).

The dataset mimics the shape of the real processed bus panel — a date x route
matrix of daily ridership — so the demo exercises the exact same code paths as
the real pipeline without shipping any raw MTA data. 20 synthetic routes,
2022-01-01 through 2025-04-30, with weekly + annual seasonality, a mild trend,
noise, and a built-in "policy effect" after 2025-01-05 (some routes gain
riders, some lose, some are unaffected) so the effects step has something to
detect.

This file is committed for transparency; the CSV it produces is committed too,
so users never need to run it. Regenerate with:

    python demo_data/generate_demo_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROUTES = 20
START, END = "2022-01-01", "2025-04-30"
POLICY_START = "2025-01-05"


def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, END, freq="D")
    t = np.arange(len(dates))
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()

    post = (dates >= pd.Timestamp(POLICY_START)).astype(float)

    # Route-level parameters.
    base = rng.uniform(2_000, 20_000, N_ROUTES)             # mean daily riders
    trend = rng.uniform(-0.2, 0.6, N_ROUTES)                # riders/day drift
    weekly_amp = rng.uniform(0.15, 0.35, N_ROUTES)          # weekday/weekend swing
    annual_amp = rng.uniform(0.05, 0.15, N_ROUTES)          # seasonal swing
    noise_sd = rng.uniform(0.02, 0.05, N_ROUTES)            # relative noise
    # Policy effect: first 8 routes gain (+3..+10%), next 6 lose (-6..-2%),
    # last 6 unaffected — mirrors the mixed response seen in the real data.
    effect = np.concatenate([
        rng.uniform(0.03, 0.10, 8),
        rng.uniform(-0.06, -0.02, 6),
        np.zeros(6),
    ])

    cols = {}
    for i in range(N_ROUTES):
        weekly = 1.0 - weekly_amp[i] * np.isin(dow, [5, 6]).astype(float)
        annual = 1.0 + annual_amp[i] * np.sin(2 * np.pi * (doy - 60) / 365.25)
        level = (base[i] + trend[i] * t) * weekly * annual
        level *= 1.0 + effect[i] * post
        noise = rng.normal(0.0, noise_sd[i] * base[i], len(dates))
        cols[f"D{i + 1:02d}"] = np.clip(level + noise, 0, None).round(0)

    df = pd.DataFrame(cols, index=dates)
    df.index.name = "date"

    out = Path(__file__).parent / "bus_demo_daily.csv"
    df.to_csv(out)
    print(f"Wrote {out}  shape={df.shape}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
