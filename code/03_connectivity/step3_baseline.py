"""Baseline Step 3 run per guild: resistance, least-cost corridors, barrier improvement scores, Circuitscape inputs."""
import numpy as np, pandas as pd, json, sys, warnings, time; warnings.filterwarnings('ignore')
from scipy import ndimage as ndi
from resistance import resistance, suitability, pressure
from corridors import run
from cs_prep import prep
src=np.load('sources100.npy')
def disk(r):
    y,x=np.ogrid[-r:r+1,-r:r+1]; return (x*x+y*y)<=r*r
for g in sys.argv[1:]:
    t=time.time()
    R=resistance(g); h=suitability(g)
    o=run(R,src,keep_cwd=True); T=o['table']; T.to_csv(f'links_{g}.csv',index=False)
    good=T[~T.through_source]
    # barrier improvement (McRae et al. 2012): restore a circle of radius r to resistance 1
    bar={}
    idx={v:i for i,v in enumerate(o['ids'])}
    for r in (1,3,5):            # 100, 300, 500 m radius
        best=np.zeros(R.shape,'float32'); fp=disk(r); L=2*r*0.1   # restored length in km (cost 1 per km)
        for a,b,lcd in zip(good.a,good.b,good.lcd_ckm):
            ma=ndi.minimum_filter(o['cwd'][idx[a]],footprint=fp,mode='nearest'); mb=ndi.minimum_filter(o['cwd'][idx[b]],footprint=fp,mode='nearest')
            IS=lcd-(ma+mb+L); best=np.maximum(best,np.where(np.isfinite(IS),IS,0)/L)
        best[~np.isfinite(R)]=np.nan; bar[f'r{r*100}']=best
    np.savez_compressed(f'step3_{g}.npz',R=R,h=h,mosaic=o['mosaic'],lcp=o['lcp'],alloc=o['alloc'].astype(np.int16),**{'barrier_'+k:v for k,v in bar.items()})
    prep(g,R,src,good,h)
    print(g,len(T),len(good),round(time.time()-t),flush=True)
