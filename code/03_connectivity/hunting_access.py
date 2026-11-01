"""Hunting accessibility: walking time (hours) from villages and roads over terrain (Tobler's hiking function), 100 m."""
import numpy as np, json
from skimage.graph import MCP_Geometric
P=np.load('predictors100.npz'); L3=np.load('step3_layers100.npz')
valid=np.isfinite(np.load('rsei_stack100.npz')['2025'])
slope=np.deg2rad(np.nan_to_num(P['slope']))
speed=6*np.exp(-3.5*np.abs(np.tan(slope)+0.05))          # km/h on the steepest gradient (conservative)
tcell=np.where(valid,0.1/speed,np.inf)                    # hours to cross a 100 m cell
access=valid&((np.nan_to_num(L3['built_frac'])>0.05)|(np.nan_to_num(P['dist_road'],nan=1e5)<100))
m=MCP_Geometric(tcell); t,_=m.find_costs(np.argwhere(access))
t=np.where(valid,t,np.nan).astype('float32')
np.save('walk_time_h.npy',t)
st=dict(access_cells_share=float(access[valid].mean()),median_h=float(np.nanmedian(t)),p90_h=float(np.nanpercentile(t,90)),max_h=float(np.nanmax(t)),
        share_within_1h=float((t[valid]<=1).mean()),share_within_2h=float((t[valid]<=2).mean()),share_within_4h=float((t[valid]<=4).mean()))
json.dump(st,open('walk_time_stats.json','w'),indent=1); print(st)
