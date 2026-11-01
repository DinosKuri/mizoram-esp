import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from scipy import ndimage as ndi
P=np.load('predictors100.npz'); F=np.load('fallow_age100.npz')['age']; L3=np.load('step3_layers100.npz')
valid=np.isfinite(np.load('rsei_stack100.npz')['2025']); tpi=P['tpi']
D=np.load('district100.npy'); names=json.load(open('district_names.json')); nm=lambda k: names[k-1] if 1<=k<=len(names) else 'outside'
out=dict(tpi_state_ridge_share=float((tpi[valid]>0).mean()),young_fallow_state=float((F[valid]<5).mean()),
         ridge_share_built=float((tpi[(np.nan_to_num(L3['built_frac'])>0.05)&valid]>0).mean()),ridge_share_roads=float((tpi[(P['dist_road']<100)&valid]>0).mean()))
for g in ('birds','arboreal','terrestrial'):
    b=np.load(f'step3_{g}.npz'); p=np.load(f'step3_post_{g}.npz'); l=b['lcp']>0; bl=p['barriers']>0
    pr=p['pinch_robust']; ids=np.arange(1,pr.max()+1); c=np.array(ndi.center_of_mass(np.ones_like(pr),pr,ids)).astype(int)
    pd_=pd.Series([nm(D[min(y*2,D.shape[0]-1),min(x*2,D.shape[1]-1)]) for y,x in c]).value_counts().to_dict()
    o=p['ocls']; dd=D[:o.shape[0]*2:2,:o.shape[1]*2:2]
    imp=pd.DataFrame(dict(d=[nm(k) for k in dd[o>0]],imp=(o[o>0]==1))).groupby('d').imp.mean().round(3).to_dict()
    out[g]=dict(ridge_share_lcp=float((tpi[l]>0).mean()),elev_lcp=float(np.nanmedian(P['elev'][l])),
                barrier_young_fallow=float((F[bl]<5).mean()),barrier_valley=float((tpi[bl]<0).mean()),pinch_by_district=pd_,impeded_by_district=imp)
json.dump(out,open('step3_extra.json','w'),indent=1); print(json.dumps(out,indent=1)[:2500])
