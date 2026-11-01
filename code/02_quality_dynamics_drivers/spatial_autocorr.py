"""Global Moran's I (35 seasons), LISA (2 km grid), LISA persistence and a correlogram (100 m sample)."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from libpysal.weights import lat2W, WSP
from esda.moran import Moran, Moran_Local
S=np.load('rsei_stack100.npz'); yrs=sorted(int(k) for k in S.files); V=~np.isnan(S['2025'])
B=20; H,W=V.shape; h,w=H//B,W//B
def block(a):
    a=a[:h*B,:w*B].reshape(h,B,w,B); return np.nanmean(a,axis=(1,3)), (~np.isnan(a)).mean(axis=(1,3))
_,frac=block(np.where(V,1.0,np.nan)); keep=frac>=0.5
idx=-np.ones((h,w),int); idx[keep]=np.arange(keep.sum())
# queen contiguity among kept cells
from libpysal.weights import W as PW
nb={};
for i,j in zip(*np.where(keep)):
    k=idx[i,j]; nb[k]=[idx[a,b] for a in range(i-1,i+2) for b in range(j-1,j+2) if (a,b)!=(i,j) and 0<=a<h and 0<=b<w and keep[a,b]]
isl=[k for k,v in nb.items() if not v]
wq=PW(nb,silence_warnings=True); wq.transform='r'
np.random.seed(42)
rows=[]; lisa={}
for y in yrs:
    m,_=block(S[str(y)]); x=m[keep]
    mi=Moran(x,wq,permutations=999); rows.append(dict(season=y,I=mi.I,p_sim=mi.p_sim,z=mi.z_sim))
    lm=Moran_Local(x,wq,permutations=999,seed=42)
    q=lm.q.copy(); q[lm.p_sim>=0.05]=0   # 1 HH, 2 LH, 3 LL, 4 HL
    lisa[y]=q
    print(y,round(mi.I,3),mi.p_sim,flush=True)
T=pd.DataFrame(rows); T.to_csv('moran_global.csv',index=False)
Q=np.stack([lisa[y] for y in yrs])
hh=(Q==1).sum(0); ll=(Q==3).sum(0)
np.savez_compressed('lisa2km.npz',Q=Q,yrs=np.array(yrs),keep=keep,hh=hh,ll=ll,B=B)
# correlogram on 100 m cells (2025 and 1990, and the 2021-2025 mean)
rng=np.random.default_rng(1); ii,jj=np.where(V); sel=rng.choice(len(ii),6000,replace=False)
X=jj[sel]*0.1; Y=ii[sel]*0.1
D=np.sqrt((X[:,None]-X[None,:])**2+(Y[:,None]-Y[None,:])**2)
bands=np.arange(0,30.5,1.0); cor={}
for y in [1990,2008,2025]:
    z=S[str(y)][ii[sel],jj[sel]]; z=(z-z.mean())/z.std(); Zp=z[:,None]*z[None,:]
    cor[y]=[float(Zp[(D>a)&(D<=b)].mean()) for a,b in zip(bands[:-1],bands[1:])]
C=pd.DataFrame(cor,index=bands[1:]); C.index.name='distance_km'; C.to_csv('correlogram.csv')
print(T.describe().round(3)); print(C.round(3).head(15))
print('persistent HH cells (>=50% seasons):',(hh>=len(yrs)/2).sum(),' LL:',(ll>=len(yrs)/2).sum(),' of',keep.sum())
