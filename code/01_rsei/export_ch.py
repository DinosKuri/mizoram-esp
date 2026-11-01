from common import *
import json
g=json.load(open('grid.json'))
ch=ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').select(0).rename('ch').toFloat().reduceResolution(ee.Reducer.mean(),maxPixels=1024)
reg=ee.Geometry.Rectangle([g['x0'],g['y1']-g['H']*100,g['x0']+g['W']*100,g['y1']],'EPSG:32646',False)
t=ee.batch.Export.image.toAsset(image=ch.clip(miz.buffer(2000)),description='canopy_h_100m',assetId=f'projects/{PROJECT}/assets/mizoram_rsei/canopy_h_100m',
  region=reg,crs='EPSG:32646',crsTransform=[100,0,g['x0'],0,-100,g['y1']],maxPixels=1e10,pyramidingPolicy={'ch':'mean'})
t.start(); print(t.id)
