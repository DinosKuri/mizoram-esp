import rsei_lib as R
from common import *
import numpy as np, pandas as pd
dem=ee.Image('NASA/NASADEM_HGT/001').select('elevation'); north=ee.Terrain.aspect(dem).multiply(3.14159265/180).cos().rename('north'); sl=ee.Terrain.slope(dem).rename('slope')
pts=ee.Image.random(1).clip(miz).sample(region=miz,scale=300,numPixels=4000,seed=5,geometries=True).limit(3000)
rows=[]
for Y in [2025]:
    for flag in [False,True]:
        R.TOPO=flag
        im=R.calibrated(Y).addBands(north).addBands(sl)
        d=pd.DataFrame([f['properties'] for f in im.sampleRegions(collection=pts,scale=300,tileScale=8).getInfo()['features']])
        d=d[d.slope>10]
        rows.append(dict(season=Y,topo=flag,**{k:round(d[k].corr(d.north),3) for k in ['NDVI','WET','NDBSI','LST']}))
        print(rows[-1],flush=True)
