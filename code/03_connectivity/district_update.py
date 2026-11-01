"""Rasterise the 11 post-2019 districts (LGD boundaries) onto the 100 m grid; recompute district tables."""
import os; os.environ['SHAPE_RESTORE_SHX']='YES'
import numpy as np, json, pandas as pd, geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
G=json.load(open('grid.json')); SH=(G['H'],G['W']); TR=from_origin(G['x0'],G['y1'],100,100)
d=gpd.read_file('district_new/MIZORAM_DISTRICT_BDY.shp').to_crs(32646).sort_values('DISTRICT').reset_index(drop=True)
names=[n.title() for n in d.DISTRICT]
D=rasterize([(g,i+1) for i,g in enumerate(d.geometry)],out_shape=SH,transform=TR,dtype='uint8')
np.save('district100.npy',D); json.dump(names,open('district_names.json','w'))
d.assign(name=names)[['name','DIST_LGD','geometry']].to_crs(4326).to_file('districts_2019.geojson',driver='GeoJSON')
valid=np.isfinite(np.load('rsei_stack100.npz')['2025'])
print('valid cells without district:',int((valid&(D==0)).sum()),'of',int(valid.sum()))
# trend by district
t=np.load('trend_mmk100.npy'); cut=float(np.load('fdr_cut.npy')[0]); sen,p=t[0],t[1]
rows=[]
for i,n in enumerate(names,1):
    m=(D==i)&valid&np.isfinite(sen)
    inc=m&(p<=cut)&(sen>0); dec=m&(p<=cut)&(sen<0)
    rows.append(dict(district=n,area_km2=m.sum()/100,inc_km2=inc.sum()/100,inc_pct=inc.sum()/m.sum()*100,dec_km2=dec.sum()/100,dec_pct=dec.sum()/m.sum()*100,median_sen=float(np.nanmedian(sen[m]))))
pd.DataFrame(rows).sort_values('inc_pct',ascending=False).to_csv('trend_by_district.csv',index=False)
# mautam by district (digitised flowering map; anomalies from the 2002-2004 baseline)
S=np.load('rsei_stack100_obs.npz'); base=np.nanmean(np.stack([S[str(y)] for y in (2002,2003,2004)]),0)
cls=np.load('bamboo_class100.npy'); rows=[]
for i,n in enumerate(names,1):
    m=(D==i)&valid&(cls!=255)
    bam=np.isin(cls[m],[1,2,3,4]).mean()*100; fl=np.isin(cls[m],[1,2]).sum()
    rows.append(dict(district=n,bamboo_pct=bam,flowered_pct_of_bamboo_0608=fl/max(np.isin(cls[m],[1,2,3,4]).sum(),1)*100,flowered_pct_of_district=fl/m.sum()*100,
                     dRSEI2007=float(np.nanmean((S['2007']-base)[m])),dRSEI2008=float(np.nanmean((S['2008']-base)[m]))))
pd.DataFrame(rows).to_csv('mautam_districts.csv',index=False)
print(pd.read_csv('trend_by_district.csv').round(3).to_string()); print(pd.read_csv('mautam_districts.csv').round(3).to_string())
