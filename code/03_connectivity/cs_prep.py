"""Write Circuitscape/Omniscape inputs at 200 m (2x2 mean of 100 m resistance)."""
import numpy as np, json, sys, warnings; warnings.filterwarnings('ignore')
from resistance import resistance, suitability
G=json.load(open('grid.json'))
def agg(a,f=2,how='mean'):
    H,W=a.shape; a=a[:H//f*f,:W//f*f].reshape(H//f,f,W//f,f)
    return np.nanmean(a,axis=(1,3)) if how=='mean' else np.nanmax(a,axis=(1,3))
def write_asc(fn,a,res=200,nodata=-9999,fmt='%.4f'):
    a=np.where(np.isfinite(a),a,nodata)
    with open(fn,'w') as f:
        f.write(f"ncols {a.shape[1]}\nnrows {a.shape[0]}\nxllcorner {G['x0']}\nyllcorner {G['y1']-a.shape[0]*res}\ncellsize {res}\nNODATA_value {nodata}\n")
        np.savetxt(f,a,fmt=fmt)
def prep(tag,R,src,links,h=None):
    R2=agg(R); write_asc(f'cs/R_{tag}.asc',R2)
    s2=agg(src.astype('float32'),how='max'); s2[~np.isfinite(R2)]=0; write_asc(f'cs/src_{tag}.asc',np.where(s2>0,s2,-9999),fmt='%d')
    ids=sorted(set(links.a)|set(links.b)); M=np.zeros((len(ids)+1,len(ids)+1)); M[0,1:]=ids; M[1:,0]=ids
    ix={v:i+1 for i,v in enumerate(ids)}
    for a,b in zip(links.a,links.b): M[ix[a],ix[b]]=M[ix[b],ix[a]]=1
    with open(f'cs/pairs_{tag}.txt','w') as f:
        f.write('min 1\nmax 1\n'); np.savetxt(f,M,fmt='%d')
    if h is not None:
        h2=agg(h); thr=np.nanquantile(h2,0.6); write_asc(f'cs/omsrc_{tag}.asc',np.where(h2>=thr,h2,0))
    return R2
