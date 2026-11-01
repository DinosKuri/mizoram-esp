from rsei_lib import *
GAP=[1990,1992,1993,1994,1995,1996,1997,1998,2003,2013]
A=f'projects/{PROJECT}/assets/mizoram_rsei'
ims=[rsei_filled(Y) for Y in GAP]
r=ee.Image.cat([im.select('RSEI').multiply(10000).round().toInt16().rename(f'RSEI_{Y}') for im,Y in zip(ims,GAP)])
f=ee.Image.cat([im.select('fill').toInt16().rename(f'fill_{Y}') for im,Y in zip(ims,GAP)])
t=ee.batch.Export.image.toAsset(image=r.addBands(f), description='RSEI_gapfill_v3', assetId=f'{A}/RSEI_gapfill_v3',
    region=miz, scale=30, crs='EPSG:32646', maxPixels=1e13, pyramidingPolicy={'.default':'mean'})
t.start(); print('started',t.id,flush=True)
import time
while True:
    s=t.status()['state']; open('gap_status.txt','w').write(s)
    if s in ('COMPLETED','FAILED','CANCELLED'): print(s,t.status().get('error_message',''),flush=True); break
    time.sleep(45)
