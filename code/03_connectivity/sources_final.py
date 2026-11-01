import numpy as np, json
from sources import make, PG, cur
from scipy import ndimage as ndi
P=np.load('predictors100.npz')
M,core,rl,k=make(EDGE=100,AMIN=5,HH=(1,))
np.save('sources100.npy',rl.astype(np.int16)); np.save('mature_forest100.npy',M); np.save('core_forest100.npy',core)
ids=np.arange(1,k+1)
area=ndi.sum(np.ones_like(rl),rl,ids)/100; pa=ndi.mean((P['pa_inside']>0).astype(float),rl,ids)
q=ndi.mean(np.nan_to_num(cur),rl,ids); el=ndi.mean(np.nan_to_num(P['elev']),rl,ids)
import pandas as pd
T=pd.DataFrame(dict(source=ids,area_km2=area,share_in_PA=pa,mean_rsei=q,mean_elev=el)); T.to_csv('sources_table.csv',index=False)
st=dict(n=int(k),area_km2=float(area.sum()),share_state=float(area.sum()/(np.isfinite(cur).sum()/100)),in_PA_km2=float((pa*area).sum()),
        n_with_PA=int((pa>0.1).sum()),median_area=float(np.median(area)),max_area=float(area.max()),mature_km2=float(M.sum()/100),core_km2=float(core.sum()/100),
        mean_rsei_sources=float(np.average(q,weights=area)))
json.dump(st,open('sources_stats.json','w'),indent=1); print(st)
