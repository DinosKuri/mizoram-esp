"""Pinch points: corridor cells (W cut-off) with the highest cumulative current (Circuitscape, 200 m)."""
import numpy as np, pandas as pd, json, sys, warnings; warnings.filterwarnings('ignore')
from scipy import ndimage as ndi
W=5.0
def read_asc(fn):
    a=np.loadtxt(fn,skiprows=6); a[a==-9999]=np.nan; return a
def agg(a,f=2,how='min'):
    H,Wd=a.shape; a=a[:H//f*f,:Wd//f*f].reshape(H//f,f,Wd//f,f)
    return {'min':np.nanmin,'max':np.nanmax,'mean':np.nanmean}[how](a,axis=(1,3))
src=agg(np.load('sources100.npy').astype(float),how='max')>0
out={}
for g in sys.argv[1:]:
    b=np.load(f'step3_{g}.npz'); cur=read_asc(f'cs/out/cs_{g}_cum_curmap.asc')
    corr=(agg(b['mosaic'])<=W)&~src
    res={}
    for q in (0.80,0.90,0.95):
        thr=np.nanquantile(cur[corr],q); pp=corr&(cur>=thr)
        lab,n=ndi.label(pp,structure=np.ones((3,3))); sz=np.bincount(lab.ravel()); keep=np.where(sz>=2)[0]; keep=keep[keep>0]
        pp=np.isin(lab,keep); lab,n=ndi.label(pp,structure=np.ones((3,3)))
        res[f'top{int(round((1-q)*100))}']=dict(n_clusters=int(n),area_km2=float(pp.sum()*0.04),threshold=float(thr))
        if q==0.90: np.save(f'pinch_{g}.npy',lab.astype(np.int16))
    E=pd.read_csv(f'cs/out/cs_{g}_resistances_3columns.out',sep=r'\s+',header=None,names=['a','b','R_eff']); E=E[E.R_eff>=0]
    E.to_csv(f'effres_{g}.csv',index=False)
    res['n_pairs']=len(E); res['R_eff_median']=float(E.R_eff.median()); res['current_total_in_corridor_share']=float(np.nansum(cur[corr])/np.nansum(cur))
    out[g]=res; print(g,json.dumps(res))
json.dump(out,open('pinch_stats.json','w'),indent=1)
