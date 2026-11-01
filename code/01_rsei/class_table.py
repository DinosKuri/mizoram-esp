import numpy as np, pandas as pd
S=np.load('rsei_stack100.npz'); V=~np.isnan(S['2025']); CLS=['Poor','Fair','Moderate','Good','Excellent']
rows=[]
for y in sorted(int(k) for k in S.files):
    r=S[str(y)][V]; ok=~np.isnan(r); c=np.floor(np.clip(r[ok],0,0.9999)*5).astype(int); cnt=np.bincount(c,minlength=5)
    rows.append(dict(season=y,valid_km2=round(ok.sum()*0.01),**{f'{k}_km2':round(cnt[i]*0.01) for i,k in enumerate(CLS)},**{f'{k}_pct':round(cnt[i]/ok.sum()*100,2) for i,k in enumerate(CLS)},mean=round(float(r[ok].mean()),4)))
T=pd.DataFrame(rows); T.to_csv('class_areas_all.csv',index=False); print(T[T.season.isin([1990,1995,2000,2005,2008,2010,2015,2020,2025])].to_string())
