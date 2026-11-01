"""Least-cost corridors between cost-adjacent sources (Linkage Mapper logic, McRae & Kavanagh 2011).
Costs in cost-weighted km (resistance x km). Links whose least-cost path crosses a third source are dropped."""
import numpy as np, pandas as pd
from skimage.graph import MCP_Geometric
from scipy import ndimage as ndi
NB=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
def descend(c,start,stop):
    """Steepest descent on an accumulated-cost surface from start to the source region."""
    H,W=c.shape; p=[start]; y,x=start; seen=0
    while not stop[y,x] and seen<200000:
        best=None; bv=c[y,x]
        for dy,dx in NB:
            yy,xx=y+dy,x+dx
            if 0<=yy<H and 0<=xx<W and c[yy,xx]<bv: bv=c[yy,xx]; best=(yy,xx)
        if best is None: break
        y,x=best; p.append(best); seen+=1
    return np.array(p)
def run(R, src, res=100.0, keep_cwd=False):
    cost=np.where(np.isfinite(R),R,np.inf).astype('float64')
    ids=[i for i in np.unique(src) if i>0]
    k=len(ids); H,W=R.shape
    cwd=np.empty((k,H,W),'float32')
    for n,i in enumerate(ids):
        m=MCP_Geometric(cost); st=np.argwhere(src==i)
        c,_=m.find_costs(st); cwd[n]=c*res/1000.0; del m
    alloc=np.argmin(cwd,0); alloc=np.where(np.isfinite(R),alloc,-1)
    pairs=set()
    for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
        a=alloc[max(0,-dy):H-max(0,dy), max(0,-dx):W-max(0,dx)]
        b=alloc[max(0,dy):, max(0,dx):][:a.shape[0],:a.shape[1]] if dx>=0 else alloc[dy:, :W+dx][:a.shape[0],:a.shape[1]]
        if dx<0: a=alloc[:H-dy, -dx:]; b=alloc[dy:, :W+dx]
        m=(a>=0)&(b>=0)&(a!=b)
        for u,v in set(zip(a[m].tolist(),b[m].tolist())): pairs.add((min(u,v),max(u,v)))
    rows=[]; mosaic=np.full((H,W),np.inf,'float32'); lcp=np.zeros((H,W),np.int16)
    for u,v in sorted(pairs):
        s=cwd[u]+cwd[v]; lcd=float(np.nanmin(np.where(np.isfinite(s),s,np.inf)))
        # path: trace from best cell in source v back to source u
        tgt=np.argwhere(src==ids[v]); cu=cwd[u][src==ids[v]]; t=tuple(tgt[np.argmin(cu)])
        path=descend(cwd[u],t,src==ids[u])
        through=set(src[path[:,0],path[:,1]].tolist())-{0,ids[u],ids[v]}
        steps=np.sqrt((np.diff(path,axis=0)**2).sum(1)).sum()*res/1000.0
        eu=float(np.min(ndi.distance_transform_edt(src!=ids[u])[src==ids[v]])*res/1000.0)
        rows.append(dict(a=ids[u],b=ids[v],lcd_ckm=lcd,path_km=steps,euclid_km=eu,cwd_pl=lcd/max(steps,1e-6),through_source=bool(through)))
        if not through:
            mosaic=np.minimum(mosaic,(s-lcd).astype('float32'))
            lcp[path[:,0],path[:,1]]=len(rows)
    T=pd.DataFrame(rows); T['link_id']=np.arange(1,len(T)+1)
    mosaic[~np.isfinite(R)]=np.nan
    out=dict(table=T,mosaic=mosaic,lcp=lcp,ids=ids)
    if keep_cwd: out['cwd']=cwd; out['alloc']=alloc
    return out
