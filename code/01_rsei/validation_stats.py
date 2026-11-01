import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr, kruskal
S=np.load('rsei_stack100.npz'); Vd=np.load('validation100.npz'); V=~np.isnan(S['2025'])
lvl=np.nanmean(np.stack([S[str(y)] for y in range(2021,2026)]),0)
wc=Vd['worldcover']; names={10:'Tree cover',20:'Shrubland',30:'Grassland',40:'Cropland',50:'Built-up',60:'Bare/sparse',80:'Water'}
rows=[]
for k,n in names.items():
    m=V&(wc==k)&~np.isnan(lvl)
    if m.sum()>=50: rows.append(dict(cls=n,code=k,n=int(m.sum()),area_km2=m.sum()*0.01,mean=float(lvl[m].mean()),sd=float(lvl[m].std()),median=float(np.median(lvl[m]))))
W=pd.DataFrame(rows).sort_values('mean',ascending=False); print(W.round(3).to_string()); W.to_csv('rsei_by_worldcover.csv',index=False)
groups=[lvl[V&(wc==k)&~np.isnan(lvl)] for k in W.code]; H,pk=kruskal(*[np.random.default_rng(1).choice(g,min(len(g),5000),replace=False) for g in groups])
# GEDI
g=np.nanmean(np.stack([S[str(y)] for y in [2020,2021,2022,2023]]),0)
rh=Vd['gedi_rh98']; m=V&np.isfinite(rh)&(rh<=60)&np.isfinite(g)
rho,p=spearmanr(g[m],rh[m])
bins=[0,5,10,15,20,25,30,40,60]; b=pd.cut(rh[m],bins); gb=pd.Series(g[m]).groupby(b).agg(['mean','size'])
mt=m&(wc==10); rho_t,_=spearmanr(g[mt],rh[mt])
out=dict(kruskal_H=float(H),kruskal_p=float(pk),gedi_n=int(m.sum()),gedi_spearman=float(rho),gedi_spearman_treecover=float(rho_t),gedi_bins=gb['mean'].round(3).tolist(),gedi_bin_n=gb['size'].tolist(),gedi_bin_edges=bins)
json.dump(out,open('validation_stats.json','w'),indent=1); print(out)
np.savez_compressed('val_points_gedi.npz',rsei=g[m][::5],rh98=rh[m][::5])
