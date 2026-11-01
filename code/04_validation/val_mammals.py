"""Mammal checks: camera-trap stations (digitised from Gogoi et al. 2025, Fig. 1) and arboreal GBIF records vs model layers."""
import re, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from pyproj import Transformer
G=json.load(open('grid.json')); WG=json.load(open('corridor_W.json'))
txt=open('Mizoram_Reliable_Data_Inventory.txt').read()
st=pd.DataFrame(re.findall(r'((?:DTR|MNP|NGP)_\d\d)\s+(Dampa TR|Murlen NP|Ngengpui WLS)\s+(S\d)\s+([\d.]+)\s+([\d.]+)\s+(\d+)',txt),columns=['id','pa','season','lat','lon','acc'])
st[['lat','lon']]=st[['lat','lon']].astype(float); st['acc']=st.acc.astype(int)
T=Transformer.from_crs(4326,32646,always_xy=True)
def rc(lon,lat):
    x,y=T.transform(lon,lat); return ((G['y1']-y)//100).astype(int),((x-G['x0'])//100).astype(int)
L={}
for g in ('birds','arboreal','terrestrial'):
    b=np.load(f'step3_{g}.npz'); L[g]=dict(h=b['h'],R=b['R'],corr=b['mosaic']<=WG[g])
src=np.load('sources100.npy')>0
r,c=rc(st.lon.values,st.lat.values)
def win(a,r,c,k=3):   # mean in a 700 m window (station accuracy ~±350 m)
    return np.array([np.nanmean(a[max(0,i-k):i+k+1,max(0,j-k):j+k+1]) for i,j in zip(r,c)])
for g in ('arboreal','terrestrial'):
    st[f'h_{g}']=win(L[g]['h'],r,c); st[f'R_{g}']=win(L[g]['R'],r,c)
st['in_source']=win(src.astype(float),r,c)
pa=st.groupby('pa').agg(stations=('id','size'),h_terr=('h_terrestrial','mean'),R_terr=('R_terrestrial','mean'),h_arb=('h_arboreal','mean'),src=('in_source','mean')).round(3)
# published detections (Gogoi et al. 2025): species recorded and selected focal-species detections
pub=pd.DataFrame({'pa':['Dampa TR','Ngengpui WLS','Murlen NP'],'species_recorded':[26,17,14],'clouded_leopard':['15 + 23','not reported','1'],'sun_bear':['9 + 26','1','not reported'],'serow':['12 + 6','not reported','1']})
pa=pa.merge(pub,on='pa'); pa.to_csv('val_camera_pa.csv',index=False); st.to_csv('val_camera_stations.csv',index=False)
# state-wide percentiles of station suitability
v=np.isfinite(L['terrestrial']['h'])
pct=lambda a,x: float((a[v]<x).mean()*100)
pa['h_terr_state_pct']=[pct(L['terrestrial']['h'],x) for x in pa.h_terr]
# arboreal GBIF records
F='gbif_mizoram_2000_2025.csv'
d=pd.read_csv(F,low_memory=False)
ARB=['Hoolock hoolock','Trachypithecus pileatus','Trachypithecus phayrei','Nycticebus bengalensis','Ratufa bicolor']
a=d[d.species.isin(ARB)&((d.coordinateUncertaintyInMeters.isna())|(d.coordinateUncertaintyInMeters<=1000))].copy()
ar,ac=rc(a.decimalLongitude.values,a.decimalLatitude.values); ok=(ar>=0)&(ar<G['H'])&(ac>=0)&(ac<G['W']); a=a[ok]; ar,ac=ar[ok],ac[ok]
a['h_arb']=L['arboreal']['h'][ar,ac]; a=a[np.isfinite(a.h_arb)]
bs=pd.read_csv('val_bird_sites.csv')   # background: sampled 1 km sites (target group: all GBIF observers)
bg=[np.nanmean(L['arboreal']['h'][r*10:(r+1)*10,c*10:(c+1)*10]) for r,c in zip(bs.r,bs.c)]
out=dict(stations=int(len(st)),by_pa=pa.to_dict('records'),arboreal_records=int(len(a)),arboreal_species=a.species.value_counts().to_dict(),
         arboreal_h_median=float(a.h_arb.median()),background_h_median=float(np.nanmedian(bg)),
         arboreal_records_pct_of_background=float(np.mean([np.mean(np.array(bg)<x) for x in a.h_arb])*100))
json.dump(out,open('val_mammals.json','w'),indent=1); print(json.dumps(out,indent=1))
