from rsei_lib import *
import numpy as np, pandas as pd, json
pts=ee.Image.random(1).clip(miz).sample(region=miz, scale=300, numPixels=3000, seed=21, geometries=True).limit(2500)
imgs={Y:calibrated(Y) for Y in SEASONS}
fc=ee.FeatureCollection([imgs[Y].sampleRegions(collection=pts, scale=300, tileScale=8).map(lambda f, Y=Y: f.set('season',Y)) for Y in SEASONS]).flatten()
# pull samples locally (small) for PCA + per-year comparison
rows=[]
for Y in SEASONS:
    for att in range(4):
        try: sub=imgs[Y].sampleRegions(collection=pts, scale=300, tileScale=8).getInfo()['features']; break
        except Exception as e: print('retry',Y,str(e)[:60],flush=True); import time; time.sleep(20)
    rows+= [dict(f['properties'],season=Y) for f in sub]
    print('sampled',Y,len(sub),flush=True)
df=pd.DataFrame(rows); df.to_csv('pooled_samples.csv',index=False)
print('SAVED')
