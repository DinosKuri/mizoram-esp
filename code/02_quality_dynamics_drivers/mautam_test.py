import numpy as np, pandas as pd, json
from pyproj import Transformer
G=json.load(open('grid.json')); R=json.load(open('bamboo_georef.json')); cls_img=np.load('bamboo_cls_img.npy'); Hi,Wi=cls_img.shape
S=np.load('rsei_stack100_obs.npz'); V=~np.isnan(S['2025']); H,W=V.shape
ii,jj=np.where(V); E=G['x0']+(jj+0.5)*G['res']; N=G['y1']-(ii+0.5)*G['res']
lon,lat=Transformer.from_crs(32646,4326,always_xy=True).transform(E,N)
sx,tx,sy,ty,r=R['p']; X=sx*lon+tx; Y=-sy*lat+ty
# rotation about the boundary-centroid used in the fit is ~0 (r=-8e-4 rad): ignore
xi=np.clip(np.round(X).astype(int),0,Wi-1); yi=np.clip(np.round(Y).astype(int),0,Hi-1)
# majority class within 1 map pixel (3x3) to reduce anti-aliasing noise
from scipy import ndimage
def frac(k): return ndimage.uniform_filter((cls_img==k).astype(float),3)
F={k:frac(k)[yi,xi] for k in range(1,5)}
fall=np.stack([F[k] for k in range(1,5)],1); anyb=fall.sum(1)
cls=np.where(anyb<0.2,0,np.argmax(fall,1)+1)
bam=np.full((H,W),255,np.uint8); bam[ii,jj]=cls; np.save('bamboo_class100.npy',bam)
names={0:'Non-bamboo',1:'Flowered 2006',2:'Flowered 2007–08',3:'Flowered 2009',4:'Non-flowering bamboo'}
print('cells per class',{names[k]:int((cls==k).sum()) for k in names})
yrs=list(range(2000,2015)); A=np.stack([S[str(y)][ii,jj] for y in yrs],1)
base=np.nanmean(A[:,yrs.index(2002):yrs.index(2004)+1],1)
anom=A-base[:,None]
blk=((E-G['x0'])//15000).astype(int)*100+((G['y1']-N)//15000).astype(int)
rng=np.random.default_rng(3); ub=np.unique(blk)
def boot(vals,b,n=500):
    m=[]; d=pd.Series(vals).groupby(b).agg(['sum','count'])
    for _ in range(n):
        s=rng.choice(d.index,len(d),replace=True); m.append(d.loc[s,'sum'].sum()/d.loc[s,'count'].sum())
    return np.percentile(m,[2.5,97.5])
rows=[]; traj={}
for k in names:
    m=cls==k; traj[names[k]]=np.nanmean(anom[m],0)
    d08=anom[m,yrs.index(2008)]; ok=~np.isnan(d08)
    lo,hi=boot(d08[ok],blk[m][ok])
    rows.append(dict(cls=names[k],n=int(m.sum()),d2006=float(np.nanmean(anom[m,yrs.index(2006)])),d2007=float(np.nanmean(anom[m,yrs.index(2007)])),
                     d2008=float(np.nanmean(d08)),ci_lo=lo,ci_hi=hi,d2009=float(np.nanmean(anom[m,yrs.index(2009)])),d2011=float(np.nanmean(anom[m,yrs.index(2011)]))))
T=pd.DataFrame(rows); print(T.round(3).to_string())
pd.DataFrame(traj,index=yrs).to_csv('mautam_trajectories.csv'); T.to_csv('mautam_class_effects.csv',index=False)
# 2 km cells: flowered fraction vs dRSEI 2008
B=20; cell=(ii//B)*1000+(jj//B)
df=pd.DataFrame({'cell':cell,'flow':np.isin(cls,[1,2,3]).astype(float),'flow0708':(cls==2).astype(float),'bam':(cls>0).astype(float),'d08':anom[:,yrs.index(2008)],'blk':blk})
g=df.groupby('cell').agg(flow=('flow','mean'),flow0708=('flow0708','mean'),bam=('bam','mean'),d08=('d08','mean'),blk=('blk','first'),n=('d08','size'))
g=g[g.n>=200]
from scipy.stats import spearmanr
rho,p=spearmanr(g.flow,g.d08); rho2,p2=spearmanr(g.flow0708,g.d08)
# block bootstrap of rho
bs=[]; gb=g.groupby('blk')
keys=list(gb.groups)
for _ in range(500):
    s=rng.choice(keys,len(keys),replace=True); sub=pd.concat([gb.get_group(k) for k in s]); bs.append(spearmanr(sub.flow,sub.d08)[0])
g.to_csv('mautam_cells.csv'); q=g.groupby(pd.cut(g.flow,[-0.01,0.0,0.05,0.10,0.20,1.0])).d08.agg(['mean','size'])
out=dict(rho_flow=rho,p_flow=p,rho_flow0708=rho2,rho_ci=list(np.percentile(bs,[2.5,97.5])),n_cells=len(g),bin_d08=q['mean'].round(4).tolist(),bin_n=q['size'].tolist())
json.dump(out,open('mautam_cells.json','w'),indent=1,default=float); print(out)
