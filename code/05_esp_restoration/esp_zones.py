"""ESP zoning (100 m) from sources, robust corridors, pinch points and barriers of the three guilds; overlap with PAs and ESZ envelopes."""
import numpy as np, json, pandas as pd, geopandas as gpd, warnings; warnings.filterwarnings('ignore')
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage as ndi
G=json.load(open('grid.json')); WG=json.load(open('corridor_W.json')); TR=from_origin(G['x0'],G['y1'],100,100); SH=(G['H'],G['W'])
GU=['birds','arboreal','terrestrial']
valid=np.isfinite(np.load('rsei_stack100.npz')['2025']); src=np.load('sources100.npy')>0
age=np.load('fallow_age100.npz')['age']; CH=np.nan_to_num(np.load('step3_layers100.npz')['canopy_h'])
up=lambda a: np.kron(a,np.ones((2,2),a.dtype))[:SH[0],:SH[1]] if a.shape[0]*2>=SH[0] else np.pad(np.kron(a,np.ones((2,2),a.dtype)),((0,SH[0]-a.shape[0]*2),(0,SH[1]-a.shape[1]*2)))
def pad(a):
    o=np.zeros(SH,a.dtype); o[:a.shape[0],:a.shape[1]]=a[:SH[0],:SH[1]]; return o
core_n=np.zeros(SH,np.int8); zone_any=np.zeros(SH,bool); robust=np.zeros(SH,bool); pinch=np.zeros(SH,bool); barrier=np.zeros(SH,bool)
for g in GU:
    b=np.load(f'step3_{g}.npz'); p=np.load(f'step3_post_{g}.npz')
    core_n+=(b['mosaic']<=WG[g]/5); zone_any|=(b['mosaic']<=WG[g])
    f=np.load(f'mc_lcp_freq_{g}.npy'); robust|=pad(np.kron(f>=0.5,np.ones((2,2),bool)))
    pinch|=pad(np.kron(p['pinch_robust']>0,np.ones((2,2),bool))); barrier|=p['barriers']>0
degraded=(np.nan_to_num(age,nan=99)<5)|(CH<10)
Z=np.zeros(SH,np.int8)                                  # 5 matrix
Z[valid]=5
Z[valid&zone_any]=4                                     # corridor buffer / managed mosaic
prio=valid&~src&((core_n>=2)|robust|pinch)
Z[prio]=2                                               # priority corridor
Z[valid&~src&(barrier|(prio&degraded))]=3               # restoration priority
Z[valid&src]=1                                          # ecological source
np.save('esp_zones.npy',Z)
# protected areas and ESZ envelopes
pa=gpd.read_file('pa_validated.geojson').to_crs(32646)
ESZ={'Pualreng WLS':(0.8,2.0),'Tawi WLS':(0.1,0.8),'Thorang WLS':(0.5,0.8),'Murlen NP':(0.1,0.8),'Lengteng WLS':(0.1,0.8),'Dampa Tiger Reserve':(0.0,11.44),'Khawnglung WLS':(0.1,0.6),
     'Tokalo WLS':(0.0,1.5),'Ngengpui WLS':(0.0,1.0),'Phawngpui NP':(0.0,1.0)}
pa=pa[pa.NAME.isin(ESZ)]
PA=rasterize([(g,1) for g in pa.geometry],out_shape=SH,transform=TR).astype(bool)&valid
def env(k):
    geoms=[(r.geometry.buffer(ESZ[r.NAME][k]*1000).difference(r.geometry),1) for r in pa.itertuples() if ESZ[r.NAME][k]>0]
    return rasterize(geoms,out_shape=SH,transform=TR).astype(bool)&valid&~PA
ESZmin,ESZmax=env(0),env(1)
np.savez_compressed('pa_esz100.npz',PA=PA,ESZmin=ESZmin,ESZmax=ESZmax)
names=['Ecological source','Priority corridor','Restoration priority','Corridor buffer','Matrix']
D=np.load('district100.npy'); dn=json.load(open('district_names.json'))
rows=[]
for k,nm in enumerate(names,1):
    m=Z==k
    rows.append(dict(zone=nm,area_km2=m.sum()/100,share_state=m.sum()/valid.sum(),in_PA=float(PA[m].mean()),in_ESZ_max=float(ESZmax[m].mean()),in_PA_or_ESZmax=float((PA|ESZmax)[m].mean()),
                     young_fallow=float((np.nan_to_num(age,nan=99)[m]<5).mean()),**{dn[i]:(m&(D==i+1)).sum()/100 for i in range(len(dn))}))
T=pd.DataFrame(rows); T.to_csv('esp_zone_table.csv',index=False)
st=dict(PA_km2=float(PA.sum()/100),ESZmin_km2=float(ESZmin.sum()/100),ESZmax_km2=float(ESZmax.sum()/100),
        pinch_cells_in_PA_or_ESZmax=float((PA|ESZmax)[pinch&valid].mean()),barrier_cells_in_PA_or_ESZmax=float((PA|ESZmax)[barrier&valid].mean()),
        priority_corridor_in_PA_or_ESZmax=float((PA|ESZmax)[Z==2].mean()),robust_km2=float((robust&valid).sum()/100),pinch_km2=float((pinch&valid).sum()/100),barrier_km2=float((barrier&valid).sum()/100))
json.dump(st,open('esp_zone_stats.json','w'),indent=1)
print(T[['zone','area_km2','share_state','in_PA','in_ESZ_max','in_PA_or_ESZmax','young_fallow']].round(3).to_string()); print(st)
