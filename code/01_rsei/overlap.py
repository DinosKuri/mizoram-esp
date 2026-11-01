from rsei_lib import *
import numpy as np, pandas as pd
L7=landsat(('L7',)); L8=landsat(('L8',)); L5=landsat(('L5',))
pts=ee.Image.random(1).clip(miz).sample(region=miz, scale=300, numPixels=3500, seed=4, geometries=True).limit(3000)
def pair(Y, A, B, la, lb):
    a=indicators(season(Y,A)).select(IND).rename([f'{i}_{la}' for i in IND])
    b=indicators(season(Y,B)).select(IND).rename([f'{i}_{lb}' for i in IND])
    fc=a.addBands(b).sampleRegions(collection=pts, scale=300, tileScale=8)
    return pd.DataFrame([f['properties'] for f in fc.getInfo()['features']]).assign(season=Y)
d78=pd.concat([pair(Y,L7,L8,'L7','L8') for Y in [2015,2017,2019,2021]])
d57=pd.concat([pair(Y,L5,L7,'L5','L7') for Y in [2005,2007,2009,2011]])
for name,d,a,b in [('L7 vs L8 (Roy-harmonized)',d78,'L7','L8'),('L5 vs L7',d57,'L5','L7')]:
    print('\n'+name, 'n=',len(d))
    for i in IND:
        x=d[f'{i}_{a}']; y=d[f'{i}_{b}']; ok=x.notna()&y.notna(); x,y=x[ok],y[ok]
        sl,ic=np.polyfit(x,y,1)
        print(f'  {i:6s} mean {a}={x.mean():.4f} {b}={y.mean():.4f} diff={(y-x).mean():+.4f}  sd(diff)={(y-x).std():.4f}  r={np.corrcoef(x,y)[0,1]:.3f}  fit {b}={sl:.3f}*{a}{ic:+.4f}')
d78.to_csv('overlap_L7_L8.csv',index=False); d57.to_csv('overlap_L5_L7.csv',index=False)
