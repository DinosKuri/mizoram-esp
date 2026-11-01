import pandas as pd, numpy as np, sys
MODE=sys.argv[3] if len(sys.argv)>3 else 'pct'
d=pd.read_csv(sys.argv[1]).dropna(subset=['NDVI','WET','NDBSI','LST'])
rows=[]
for s,g in d.groupby('season'):
    X=g[['NDVI','WET','NDBSI','LST']].values; lo,hi=(np.percentile(X,[2,98],0) if MODE=='pct' else (X.min(0),X.max(0))); Z=np.clip((X-lo)/(hi-lo),0,1)
    w,V=np.linalg.eigh(np.cov(Z.T)); v=V[:,-1]; v*=np.sign(v[0]); rows.append([s,*v.round(3),round(w[-1]/w.sum()*100,1)])
P=pd.DataFrame(rows,columns=['season','NDVI','WET','NDBSI','LST','var_pct']); 
if len(sys.argv)>2 and sys.argv[2]: P.to_csv(sys.argv[2],index=False)
print(P.head(3).to_string()); print(P.describe().loc[['min','max']].round(3).to_string())
