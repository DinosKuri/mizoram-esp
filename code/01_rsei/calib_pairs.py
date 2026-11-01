from rsei_lib import *
import pandas as pd, sys
L5=landsat(('L5',)); L7=landsat(('L7',)); L8=landsat(('L8',)); L9=landsat(('L9',))
pts=ee.Image.random(1).clip(miz).sample(region=miz, scale=300, numPixels=3500, seed=11, geometries=True).limit(3000)
def pair(Y, A, B, la, lb):
    a=indicators(season(Y,A)).select(IND).rename([f'{i}_a' for i in IND])
    b=indicators(season(Y,B)).select(IND).rename([f'{i}_b' for i in IND])
    fc=a.addBands(b).sampleRegions(collection=pts, scale=300, tileScale=8)
    return pd.DataFrame([f['properties'] for f in fc.getInfo()['features']]).assign(season=Y, pair=f'{la}-{lb}')
jobs=[(Y,L5,L7,'L5','L7') for Y in [2000,2001,2004,2005,2006,2007,2009,2010,2011]]+\
     [(Y,L7,L8,'L7','L8') for Y in [2014,2015,2016,2017,2018,2019,2020]]+\
     [(Y,L8,L9,'L8','L9') for Y in [2022,2023,2024,2025]]
out=[]
for j in jobs:
    for attempt in range(3):
        try: out.append(pair(*j)); print('done',j[0],j[3],j[4],flush=True); break
        except Exception as e: print('retry',j[0],e,flush=True)
pd.concat(out).to_csv('calib_pairs.csv',index=False); print('SAVED')
