"""Corridor cut-off W per guild: largest W at which < 5% of corridor cells fall in the guild's top-quartile resistance."""
import numpy as np, pandas as pd, json, sys, warnings; warnings.filterwarnings('ignore')
from resistance import pressure, VALID
M=np.load('mature_forest100.npy'); src=np.load('sources100.npy')>0
G=sys.argv[1:] or ['birds','arboreal','terrestrial','hornbills']
rows=[]; pick={}
for g in G:
    b=np.load(f'step3_{g}.npz'); hp=pressure(g); R=b['R']; p75=np.nanpercentile(R[VALID],75)
    for W in [0.5,1,2,5,10,20,50,100]:
        c=(b['mosaic']<=W)&~src
        if c.sum()==0: continue
        rows.append(dict(guild=g,W_ckm=W,area_km2=c.sum()/100,mean_suit=float(np.nanmean(b['h'][c])),mean_pressure=float(np.nanmean(hp[c])),
                         share_mature=float(M[c].mean()),share_R_top25=float((R[c]>=p75).mean()),R_p75=float(p75)))
    T=pd.DataFrame([r for r in rows if r['guild']==g]); ok=T[T.share_R_top25<0.05]
    pick[g]=float(ok.W_ckm.max()) if len(ok) else float(T.W_ckm.min())
T=pd.DataFrame(rows); T.to_csv('corridor_width_test.csv',index=False); json.dump(pick,open('corridor_W.json','w'),indent=1)
print(T.round(3).to_string()); print(pick)
