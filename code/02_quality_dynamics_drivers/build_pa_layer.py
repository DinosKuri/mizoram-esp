"""Validated protected-area layer (10 areas, 971 km2).
Nine boundaries from OpenStreetMap (Miz_Protectedarea.shp); Thorangtlang WLS, missing from OSM, from the WDPA (wdpa_mizoram.geojson).
A WDPA polygon with no matching notification ('Buhvum WLS') is excluded."""
import geopandas as gpd
o=gpd.read_file('pa_osm/Miz_Protectedarea.shp').to_crs(32646)
w=gpd.read_file('wdpa_mizoram.geojson').to_crs(32646)
mz=gpd.read_file('districts_2019.geojson').to_crs(32646).union_all()
name={'Phawngpui Blue Mountain NP':'Phawngpui NP'}
des={'Murlen NP':'National Park','Phawngpui NP':'National Park','Dampa Tiger Reserve':'Tiger Reserve'}
rows=[]
for _,r in o.iterrows():
    n=name.get(r['name'],r['name']); rows.append(dict(NAME=n,DESIG=des.get(n,'Wildlife Sanctuary'),SOURCE='OpenStreetMap',geometry=r.geometry))
t=w[w.NAME=='Thorang WLS'].iloc[0]; rows.append(dict(NAME='Thorang WLS',DESIG='Wildlife Sanctuary',SOURCE='WDPA',geometry=t.geometry))
g=gpd.GeoDataFrame(rows,crs=32646); g['geometry']=g.geometry.buffer(0).intersection(mz); g['area_km2']=g.area/1e6
# agreement with WDPA (intersection over union)
for i,r in g.iterrows():
    m=w[w.intersects(r.geometry)]
    if len(m): a=m.geometry.intersection(r.geometry).area; j=a.idxmax(); g.loc[i,'IoU_WDPA']=a[j]/m.loc[j].geometry.union(r.geometry).area
print(g.drop(columns='geometry').round(2).to_string(), '\ntotal km2', round(g.area_km2.sum(),1))
g.to_crs(4326).to_file('pa_validated.geojson',driver='GeoJSON')
