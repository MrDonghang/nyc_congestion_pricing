"""ACS-based demographic, economic, and travel-mode variables per census tract.

The raw ACS table (one row per (GEOID, year)) is reduced to a single year
slice (default 2023) and joined to NYC tract polygons. Derived percentage
variables (race / ethnicity / education / commute mode / etc.) are computed
from the underlying counts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Standard variable groups used in causal regressions and choropleth panels.
GROUPS: dict[str, list[str]] = {
    "demographics": ["pop_total", "pct_male", "age_median"],
    "race_ethnicity": ["pct_white", "pct_black", "pct_asian", "pct_hispanic"],
    "economics": ["inc_median_ind", "inc_median_household", "unemployment_rate", "pct_no_inc"],
    "travel": [
        "pct_driving",
        "pct_public_transit",
        "pct_taxi",
        "pct_cycle",
        "pct_walk",
        "pct_wfh",
    ],
    "education": ["pct_bachelor", "pct_master", "pct_phd"],
    "housing": ["rent_median", "property_value_median", "vehicle_total_imputed", "vacancy_rate"],
}


def _safe_pct(num: pd.Series, denom: pd.Series) -> pd.Series:
    return num / denom.replace(0, np.nan) * 100


def derive_percentages(dem: pd.DataFrame) -> pd.DataFrame:
    """Add ``pct_*`` and rate columns to a raw ACS table.

    Negative count placeholders (the ACS uses negatives for null) are first
    masked to NaN so they don't produce nonsensical percentages.
    """
    dem = dem.copy()
    num_cols = dem.select_dtypes(include=[np.number]).columns
    dem[num_cols] = dem[num_cols].mask(dem[num_cols] < 0)

    dem["pct_male"] = _safe_pct(dem["sex_male"], dem["sex_total"])

    dem["pct_white"] = _safe_pct(dem["race_white"], dem["race_total"])
    dem["pct_black"] = _safe_pct(dem["race_black"], dem["race_total"])
    dem["pct_asian"] = _safe_pct(dem["race_asian"], dem["race_total"])
    dem["pct_hispanic"] = _safe_pct(dem["hispanic_latino"], dem["total_his_lat"])

    dem["pct_no_inc"] = _safe_pct(dem["inc_no_pop"], dem["inc_total_pop"])

    travel_total = dem["travel_total_to_work"]
    dem["pct_driving"] = _safe_pct(dem["travel_driving_to_work"], travel_total)
    dem["pct_public_transit"] = _safe_pct(dem["travel_pt_to_work"], travel_total)
    dem["pct_taxi"] = _safe_pct(dem["travel_taxi_to_work"], travel_total)
    dem["pct_cycle"] = _safe_pct(dem["travel_cycle_to_work"], travel_total)
    dem["pct_walk"] = _safe_pct(dem["travel_walk_to_work"], travel_total)
    dem["pct_wfh"] = _safe_pct(dem["travel_work_from_home"], travel_total)

    dem["pct_bachelor"] = _safe_pct(dem["edu_bachelor"], dem["edu_total"])
    dem["pct_master"] = _safe_pct(dem["edu_master"], dem["edu_total"])
    dem["pct_phd"] = _safe_pct(dem["edu_phd"], dem["edu_total"])

    dem["vacancy_rate"] = _safe_pct(dem["housing_units_vacant"], dem["housing_units_total"])
    dem["unemployment_rate"] = _safe_pct(dem["employment_unemployed"], dem["employment_total_labor"])
    return dem


def join_to_tracts(tracts_gdf, dem: pd.DataFrame, year: int = 2023, geoid_col: str = "geoid"):
    """Slice ACS to ``year`` and merge into ``tracts_gdf`` on GEOID."""
    dem_year = dem[dem["year"] == year].drop(columns="geometry", errors="ignore").reset_index(drop=True)
    tracts_gdf = tracts_gdf.copy()
    tracts_gdf[geoid_col] = tracts_gdf[geoid_col].astype(str)
    dem_year["GEOID"] = dem_year["GEOID"].astype(str)
    out = tracts_gdf.merge(dem_year, left_on=geoid_col, right_on="GEOID", how="left")
    return out


def build(tracts_gdf, acs_geojson: Path, year: int = 2023, geoid_col: str = "geoid"):
    """Convenience: load ACS GeoJSON, derive percentages, join to ``tracts_gdf``."""
    import geopandas as gpd

    dem = gpd.read_file(acs_geojson)
    valid_geoids = set(tracts_gdf[geoid_col].astype(str).unique())
    dem = dem[dem["GEOID"].astype(str).isin(valid_geoids)].reset_index(drop=True)
    dem = derive_percentages(dem)
    return join_to_tracts(tracts_gdf, dem, year=year, geoid_col=geoid_col)
