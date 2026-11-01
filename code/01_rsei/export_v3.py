from rsei_lib import *
A=f'projects/{PROJECT}/assets/mizoram_rsei'
col=ee.ImageCollection([rsei_image(Y) for Y in SEASONS])
stack=col.toBands().rename([f'RSEI_{Y}' for Y in SEASONS]).multiply(10000).round().toInt16()
t=ee.batch.Export.image.toAsset(image=stack, description='RSEI_annual_v3', assetId=f'{A}/RSEI_annual_1990_2025_v3',
    region=miz, scale=30, crs='EPSG:32646', maxPixels=1e13, pyramidingPolicy={'.default':'mean'})
t.start(); print('started',t.id,flush=True)
import time
while True:
    s=t.status()['state']; open('stack_status.txt','w').write(s)
    if s in ('COMPLETED','FAILED','CANCELLED'): print(s,t.status().get('error_message','')); break
    time.sleep(60)
