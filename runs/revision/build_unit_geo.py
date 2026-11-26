"""Unit geography for the spatial contrast: relation of each subway station / bus route to the Congestion Relief Zone (CRZ).
Writes outputs/_summary/revision/uq/unit_geo_{subway_ns,bus_ns}.csv
  subway: unit (station_index), complex_id, name, borough, in_crz, dist_km (to CRZ boundary, 0 inside)
  bus:    unit (route_id), share_in_crz (route length inside CRZ), touches_crz, express (route to/from Manhattan core), borough prefix
"""
import geopandas as gpd, numpy as np, pandas as pd
from nyc_cp.config import REPO_ROOT, load_mode, load_paths
from nyc_cp.data import load_actual
G = REPO_ROOT / "geo_data"; U = REPO_ROOT / "outputs" / "_summary" / "revision" / "uq"; paths = load_paths()
crz = gpd.read_file(G / "nyc_cp_boundary_poly_json.geojson").to_crs(2263); cg = crz.geometry.unary_union
# ---- subway
st = gpd.read_file(G / "MTA_Subway_Stations_20251029.geojson").to_crs(2263); st["complex_id"] = st.complex_id.astype(int)
cx = st.dissolve("complex_id", aggfunc={"stop_name": "first", "borough": "first"}); cx["geometry"] = cx.geometry.centroid
mp = pd.read_csv(paths["data_root"] + "/subway/patterns/station_mapping.csv")
s = mp.merge(cx.reset_index(), left_on="station_id", right_on="complex_id", how="left"); s = gpd.GeoDataFrame(s, geometry="geometry", crs=2263)
s["in_crz"] = s.geometry.within(cg); s["dist_km"] = s.geometry.distance(cg) * 0.3048 / 1000
a = load_actual("subway_ns", direction="O", mode_cfg=load_mode("subway_ns"), paths=paths)
s["unit"] = s.station_index.astype(str); s["vol_2024"] = s.unit.map(a.loc["2024"].mean()); s["yoy_closed"] = s.unit.map(a.loc["2025-01-05":"2025-04-30"].sum() / a.loc["2024-01-05":"2024-04-30"].sum()) < 0.5
s[["unit", "complex_id", "stop_name", "borough", "in_crz", "dist_km", "vol_2024", "yoy_closed"]].to_csv(U / "unit_geo_subway_ns.csv", index=False)
print("subway: matched", s.complex_id.notna().sum(), "/", len(s), "| in CRZ", int(s.in_crz.sum()), "| closed in policy period", int(s.yoy_closed.sum()))
print(s.assign(band=pd.cut(s.dist_km, [-1, 0, 2, 5, 10, 100], labels=["inside", "0-2km", "2-5km", "5-10km", ">10km"])).groupby("band", observed=True).agg(n=("unit", "size"), vol=("vol_2024", "sum")).assign(share=lambda d: (d.vol / d.vol.sum()).round(3)).to_string())
# ---- bus
br = gpd.read_file(G / "bus_routes" / "bus_routes_nyc_dec2019.shp").to_crs(2263); br = br.dissolve("route_id")
br["len"] = br.geometry.length; br["share_in_crz"] = br.geometry.intersection(cg).length / br.len; br["touches_crz"] = br.share_in_crz > 0
b = load_actual("bus_ns", direction="all", mode_cfg=load_mode("bus_ns"), paths=paths)
g = pd.DataFrame({"unit": b.columns.astype(str)}); g["key"] = g.unit.str.replace("+", "-SBS", regex=False)
m = g.merge(br[["share_in_crz", "touches_crz"]].reset_index(), left_on="unit", right_on="route_id", how="left")
miss = m.share_in_crz.isna(); m.loc[miss, ["share_in_crz", "touches_crz"]] = m[miss].merge(br[["share_in_crz", "touches_crz"]].reset_index(), left_on="key", right_on="route_id", how="left")[["share_in_crz_y", "touches_crz_y"]].to_numpy()
m["prefix"] = m.unit.str.extract(r"^([A-Za-z]+)")[0].str.upper(); m["express"] = m.prefix.isin(["BM", "BXM", "QM", "X", "SIM"])
m["vol_2024"] = m.unit.map(b.loc["2024"].mean())
m[["unit", "prefix", "express", "share_in_crz", "touches_crz", "vol_2024"]].to_csv(U / "unit_geo_bus_ns.csv", index=False)
print("bus: matched", int(m.share_in_crz.notna().sum()), "/", len(m), "| unmatched:", m[m.share_in_crz.isna()].unit.tolist()[:40])
print(m.assign(grp=np.select([m.express, m.share_in_crz.fillna(0) > 0.5, m.share_in_crz.fillna(0) > 0, m.prefix == "M"], ["express", "mostly in CRZ", "touches CRZ", "Manhattan, outside"], "outer boroughs")).groupby("grp").agg(n=("unit", "size"), vol=("vol_2024", "sum")).assign(share=lambda d: (d.vol / d.vol.sum()).round(3)).to_string())
