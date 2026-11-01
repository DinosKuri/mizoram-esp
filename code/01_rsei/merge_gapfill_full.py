from common import *
import numpy as np, json, shutil, os
A=f'projects/{PROJECT}/assets/mizoram_rsei/RSEI_gapfill_v3'
g=json.load(open('grid.json'))
grid={'dimensions':{'width':g['W'],'height':g['H']},'affineTransform':{'scaleX':g['res'],'shearX':0,'translateX':g['x0'],'shearY':0,'scaleY':-g['res'],'translateY':g['y1']},'crsCode':'EPSG:32646'}
img=ee.Image(A); GAP=[1990,1992,1993,1994,1995,1996,1997,1998,2003,2013]
pass
S=dict(np.load('rsei_stack100_obs.npz')); V=~np.isnan(S['2025'])
fillfrac={}; src={}
for Y in GAP:
    a=ee.data.computePixels({'expression':img.select([f'RSEI_{Y}']).unmask(-9999),'fileFormat':'NUMPY_NDARRAY','grid':grid})[f'RSEI_{Y}'].astype('float32')
    a[a<=-9999]=np.nan; a/=10000
    f1=ee.data.computePixels({'expression':img.select([f'fill_{Y}']).eq(1).unmask(0).toFloat(),'fileFormat':'NUMPY_NDARRAY','grid':grid})[f'fill_{Y}'].astype('float32')
    f2=ee.data.computePixels({'expression':img.select([f'fill_{Y}']).eq(2).unmask(0).toFloat(),'fileFormat':'NUMPY_NDARRAY','grid':grid})[f'fill_{Y}'].astype('float32')
    hole=np.isnan(S[str(Y)])&V
    new=S[str(Y)].copy(); new[hole]=a[hole]; S[str(Y)]=new
    left=np.isnan(new)&V
    src[Y]=dict(gap_km2=round(hole.sum()*0.01,1),filled_window_km2=round(float(np.nansum(np.where(hole,f1,0)))*0.01/ max(1,1),1),
               filled_neighbour_km2=round(float(np.nansum(np.where(hole,f2,0)))*0.01,1),still_missing_km2=round(left.sum()*0.01,1))
    print(Y,src[Y],flush=True)
np.savez_compressed('rsei_stack100.npz',**S); json.dump(src,open('gapfill_summary.json','w'),indent=1); print('SAVED')

# ---- final pass: cells still empty (neighbouring seasons also cloudy) take the mean of the nearest
#      observed seasons within +-3, +-5, then +-8 years
import warnings; warnings.filterwarnings('ignore')
S=dict(np.load('rsei_stack100.npz')); O=np.load('rsei_stack100_obs.npz'); V=~np.isnan(S['2025'])
yrs=sorted(int(k) for k in S); GF=json.load(open('gapfill_summary.json'))
for Y in yrs:
    left0=np.isnan(S[str(Y)])&V
    if not left0.any(): continue
    new=S[str(Y)].copy()
    for w in (3,5,8):
        left=np.isnan(new)&V
        if not left.any(): break
        m=np.nanmean(np.stack([O[str(y)] for y in yrs if y!=Y and abs(y-Y)<=w]),0); new[left]=m[left]
    S[str(Y)]=new
    g=GF.get(str(Y),{'gap_km2':round(left0.sum()*0.01,1),'filled_window_km2':0.0,'filled_neighbour_km2':0.0})
    g['filled_neighbour_km2']=(g['filled_neighbour_km2'] if str(Y) in GF else 0)+left0.sum()*0.01
    g['still_missing_km2']=float((np.isnan(new)&V).sum()*0.01); GF[str(Y)]={k:round(float(v),1) for k,v in g.items()}
np.savez_compressed('rsei_stack100.npz',**S); json.dump(GF,open('gapfill_summary.json','w'),indent=1); print('final pass done')
