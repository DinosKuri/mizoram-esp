from rsei_lib import *
import rsei_lib; _C=rsei_lib._C
import pandas as pd
ids={'L5':'LANDSAT/LT05/C02/T1_L2','L7':'LANDSAT/LE07/C02/T1_L2','L8':'LANDSAT/LC08/C02/T1_L2','L9':'LANDSAT/LC09/C02/T1_L2'}
def used(Y):
    s=[]
    if Y>=2014: s=['L8','L9']
    else:
        if Y<=2011 and Y not in (2002,2003): s.append('L5')
        if Y>=2000: s.append('L7')
    return s
rows=[]
for Y in SEASONS:
    d={'season':Y,'window':f'Nov {Y-1} - Feb {Y}'}
    for k,cid in ids.items():
        d[k]=ee.ImageCollection(cid).filterBounds(miz).filterDate(f'{Y-1}-11-01',f'{Y}-03-01').size() if k in used(Y) else 0
    sens=used(Y); col=None
    for k in sens: col=_C[k] if col is None else col.merge(_C[k])
    nobs=col.filterDate(f'{Y-1}-11-01',f'{Y}-03-01').select('N').count().updateMask(water.Not())
    st=nobs.unmask(0).gt(0).addBands(nobs).reduceRegion(ee.Reducer.mean().combine(ee.Reducer.median(),'',True),miz,300,maxPixels=1e13,bestEffort=True)
    d['stats']=st; rows.append(d)
out=ee.List([ee.Dictionary({k:v for k,v in r.items()}) for r in rows]).getInfo()
df=pd.DataFrame(out); print(df.head())
st=pd.json_normalize(df.pop('stats')); df=pd.concat([df,st],axis=1); df.to_csv('scenes_per_season.csv',index=False); print(df.to_string())
