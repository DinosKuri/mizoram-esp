"""Ecological sources: mature-forest core patches that lie in temporally persistent RSEI hotspots.
Mature forest = no detected disturbance for >= AGE seasons and mean RSEI 2021-2025 >= QMIN.
Core = mature forest farther than EDGE m from non-mature cells. Source = core patch >= AMIN km2 with
>= HHSHARE of its area inside 2 km cells that were LISA High-High in >= 18 of 35 seasons."""
import numpy as np, json, itertools, pandas as pd
from scipy import ndimage as ndi
S=np.load('rsei_stack100.npz'); cur=np.nanmean(np.stack([S[str(y)] for y in range(2021,2026)]),0)
F=np.load('fallow_age100.npz'); age=F['age']
L3=np.load('step3_layers100.npz'); P=np.load('predictors100.npz')
valid=np.isfinite(cur)
lisa=np.load('lisa2km.npz'); keep=lisa['keep']; pc=np.load('lisa_persist_class.npy'); B=int(lisa['B'])
pg=np.zeros(keep.shape,np.int8); pg[keep]=pc
PG=np.zeros(cur.shape,np.int8); h,w=keep.shape; PG[:h*B,:w*B]=np.kron(pg,np.ones((B,B),np.int8))
def make(AGE=15,QMIN=0.6,EDGE=200,AMIN=5,HH=(1,),HHSHARE=0.5,MMU=3):
    M=valid&(age>=AGE)&(cur>=QMIN)&(L3['built_frac']<0.05)&(L3['water']<0.5)
    # minimum mapping unit (MMU cells = 3 ha, about one jhum plot): absorb smaller gaps, drop smaller islands
    gl,gn=ndi.label(valid&~M,structure=np.ones((3,3))); gs=np.bincount(gl.ravel()); small=(gs<MMU); small[0]=False; M=M|small[gl]
    il,inn=ndi.label(M,structure=np.ones((3,3))); is_=np.bincount(il.ravel()); tiny=(is_<MMU); tiny[0]=False; M=M&~tiny[il]
    d=ndi.distance_transform_edt(M)*100
    core=M&(d>EDGE)
    lab,n=ndi.label(core,structure=np.ones((3,3)))
    area=ndi.sum(np.ones_like(lab),lab,np.arange(1,n+1))/100.0
    hh=ndi.mean(np.isin(PG,HH).astype(float),lab,np.arange(1,n+1))
    ok=np.where((area>=AMIN)&(hh>=HHSHARE))[0]+1
    src=np.zeros(cur.shape,np.int32); src[np.isin(lab,ok)]=np.isin(lab,ok)[np.isin(lab,ok)]
    rl,k=ndi.label(np.isin(lab,ok),structure=np.ones((3,3)))
    return M,core,rl,k
if __name__=='__main__':
    rows=[]
    for EDGE,AMIN,HH in itertools.product([100,200,300],[1,2,5,10,20,50],[(1,),(1,2)]):
        M,core,rl,k=make(EDGE=EDGE,AMIN=AMIN,HH=HH)
        a=(rl>0).sum()/100; pa=((rl>0)&(P['pa_inside']>0)).sum()/100
        rows.append(dict(edge_m=EDGE,amin_km2=AMIN,hh='persistent' if HH==(1,) else 'persistent+recurrent',n_sources=k,area_km2=a,in_PA_km2=pa,
                         mature_km2=M.sum()/100,core_km2=core.sum()/100))
        print(rows[-1],flush=True)
    pd.DataFrame(rows).to_csv('source_sensitivity.csv',index=False)
