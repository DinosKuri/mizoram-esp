from common import *
A=f'projects/{PROJECT}/assets/mizoram_rsei'
wc=ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
def gq(im): return im.updateMask(im.select('quality_flag').eq(1).And(im.select('degrade_flag').eq(0)).And(im.select('sensitivity').gt(0.95)))
gedi=ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY').filterBounds(miz).filterDate('2019-04-01','2023-03-31').map(gq).select('rh98')
gm=gedi.mean().rename('rh98'); gn=gedi.count().rename('n')
img=wc.rename('wc').toFloat().addBands(gm.toFloat()).addBands(gn.toFloat())
t=ee.batch.Export.image.toAsset(image=img.clip(miz.buffer(1000)),description='val_layers',assetId=f'{A}/validation_layers_30m',region=miz.buffer(1000),scale=30,crs='EPSG:32646',maxPixels=1e13,
    pyramidingPolicy={'wc':'mode','rh98':'mean','n':'mean'})
t.start(); print('started',t.id,flush=True)
import time
while True:
    s=t.status()['state']; open('val_status.txt','w').write(s)
    if s in ('COMPLETED','FAILED','CANCELLED'): print(s,t.status().get('error_message',''),flush=True); break
    time.sleep(45)
