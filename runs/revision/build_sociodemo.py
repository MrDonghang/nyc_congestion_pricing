"""Socio-demographic profile of each analysis unit (ACS 5-year, 2023 vintage, census tracts; geo_data/nyc_census_tracts.geojson).
  subway station: tracts whose centroid lies within 800 m of the station complex (nearest tract if none)
  bus route:      tracts whose centroid lies within 400 m of the route alignment
  Replica:        the origin tract itself
Count-based shares are computed from summed counts; medians (income, rent, age) are population-weighted means of tract medians.
Writes uq/sociodemo_{subway,bus,replica}.csv
"""
import geopandas as gpd, numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT, load_paths
G = REPO_ROOT / "geo_data"; U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; paths = load_paths(); FT = 0.3048
acs = gpd.read_file(G / "nyc_census_tracts.geojson", ignore_geometry=True); acs = acs[acs.year == 2023].copy(); acs["GEOID"] = acs.GEOID.astype(str)
num = acs.select_dtypes("number").columns; acs[num] = acs[num].mask(acs[num] < 0)
tr = gpd.read_file(G / "NYC_Census_Tracts_2020" / "NYC_Census_Tracts_2020.shp").to_crs(2263); idc = [c for c in tr.columns if c.lower() == "geoid"][0]; tr["GEOID"] = tr[idc].astype(str)
tr["land_km2"] = tr.geometry.area * FT ** 2 / 1e6; T = tr[["GEOID", "land_km2", "geometry"]].merge(acs, on="GEOID", how="inner"); T = T[T.pop_total > 0].reset_index(drop=True); print("tracts with ACS and population:", len(T))
crz = gpd.read_file(G / "nyc_cp_boundary_poly_json.geojson").to_crs(2263).geometry.unary_union; T["dist_km"] = T.geometry.centroid.distance(crz) * FT / 1000
SUMS = ["pop_total", "households", "land_km2", "vehicle_total_imputed", "race_total", "race_white", "race_black", "race_asian", "total_his_lat", "hispanic_latino", "travel_total_to_work", "travel_driving_to_work", "travel_pt_to_work",
        "travel_work_from_home", "travel_walk_to_work", "edu_total", "edu_bachelor", "edu_master", "edu_phd", "employment_total_labor", "employment_unemployed"]
MEDS = ["inc_median_household", "rent_median", "age_median"]
def div(a, b): return a / b if b else np.nan
def profile(g):
    """g: DataFrame of tracts (rows of T) forming one catchment"""
    s = g[SUMS].sum(); o = {"n_tracts": len(g), "pop": s.pop_total}
    for m in MEDS: ok = g[m].notna(); o[m] = np.average(g.loc[ok, m], weights=g.loc[ok, "pop_total"]) if ok.any() else np.nan
    o.update(income_k=o["inc_median_household"] / 1e3, veh_per_hh=div(s.vehicle_total_imputed, s.households), pop_density_k=div(s.pop_total, s.land_km2) / 1e3,
             pct_car_commute=100 * div(s.travel_driving_to_work, s.travel_total_to_work), pct_transit_commute=100 * div(s.travel_pt_to_work, s.travel_total_to_work), pct_wfh=100 * div(s.travel_work_from_home, s.travel_total_to_work),
             pct_bachelor_plus=100 * div(s.edu_bachelor + s.edu_master + s.edu_phd, s.edu_total), pct_white=100 * div(s.race_white, s.race_total), pct_black=100 * div(s.race_black, s.race_total), pct_asian=100 * div(s.race_asian, s.race_total),
             pct_hispanic=100 * div(s.hispanic_latino, s.total_his_lat), unemployment=100 * div(s.employment_unemployed, s.employment_total_labor))
    return pd.Series(o)
cen = gpd.GeoDataFrame(T.drop(columns="geometry"), geometry=T.geometry.centroid, crs=2263)
def catch(units, radius_m):
    """units: GeoDataFrame indexed by unit id; returns one profile per unit from tract centroids within radius of the geometry"""
    j = gpd.sjoin(cen, gpd.GeoDataFrame({"u": units.index}, geometry=units.geometry.buffer(radius_m / FT).to_numpy(), crs=2263), predicate="within"); out = {}
    for u in units.index:
        g = j[j.u == u]
        if len(g) == 0: g = cen.iloc[[cen.geometry.distance(units.geometry[u]).to_numpy().argmin()]]
        out[u] = profile(g)
    return pd.DataFrame(out).T
# subway
st = gpd.read_file(G / "MTA_Subway_Stations_20251029.geojson").to_crs(2263); st["complex_id"] = st.complex_id.astype(int); cx = st.dissolve("complex_id"); cx["geometry"] = cx.geometry.centroid
mp = pd.read_csv(paths["data_root"] + "/subway/patterns/station_mapping.csv"); mp["unit"] = mp.station_index.astype(str); s = gpd.GeoDataFrame(mp.merge(cx[["geometry"]], left_on="station_id", right_index=True), geometry="geometry", crs=2263).set_index("unit")
S = catch(s, 800); S["dist_km"] = s.geometry.distance(crz) * FT / 1000; S.index.name = "unit"; S.to_csv(U / "sociodemo_subway.csv"); print("subway", S.shape, "median tracts per station", S.n_tracts.median())
# bus
br = gpd.read_file(G / "bus_routes" / "bus_routes_nyc_dec2019.shp").to_crs(2263).dissolve("route_id"); geo = pd.read_csv(U / "unit_geo_bus_ns.csv"); geo["key"] = geo.unit.str.replace("+", "-SBS", regex=False)
geo["rid"] = np.where(geo.unit.isin(br.index), geo.unit, np.where(geo.key.isin(br.index), geo.key, None)); g2 = geo.dropna(subset=["rid"]); b = gpd.GeoDataFrame({"unit": g2.unit.to_numpy()}, geometry=br.geometry.reindex(g2.rid).to_numpy(), crs=2263).set_index("unit")
B = catch(b, 400); B["dist_km"] = b.geometry.centroid.distance(crz) * FT / 1000; B.index.name = "unit"; B.to_csv(U / "sociodemo_bus.csv"); print("bus", B.shape, "of", len(geo), "median tracts per route", B.n_tracts.median())
# Replica
R = pd.DataFrame({u: profile(T.iloc[[i]]) for i, u in enumerate(T.GEOID)}).T; R["dist_km"] = T.dist_km.to_numpy(); R.index.name = "unit"; R.to_csv(U / "sociodemo_replica.csv"); print("replica", R.shape)
V = ["income_k", "veh_per_hh", "pct_car_commute", "pct_transit_commute", "pct_wfh", "pct_bachelor_plus", "pct_white", "pct_black", "pct_hispanic", "pct_asian", "age_median", "pop_density_k", "unemployment", "rent_median", "dist_km"]
pd.set_option("display.width", 220)
for n, d in (("subway", S), ("bus", B), ("replica", R)): print(n); print(d[V].astype(float).describe().loc[["count", "mean", "25%", "50%", "75%"]].round(1).to_string())
print(R[V].astype(float).corr(method="spearman").round(2).to_string())
