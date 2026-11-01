"""Independent layers for RSEI validation on the 100 m grid: WorldCover 2021 (mode), GEDI rh98 (2019-2023 mean),
Hansen loss year (dominant) and loss fraction, tree cover 2000."""
from common import *
import numpy as np, json
g=json.load(open('grid.json'))
grid={'dimensions':{'width':g['W'],'height':g['H']},'affineTransform':{'scaleX':g['res'],'shearX':0,'translateX':g['x0'],'shearY':0,'scaleY':-g['res'],'translateY':g['y1']},'crsCode':'EPSG:32646'}
prj=ee.Projection('EPSG:32646').atScale(100)
wc=ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').clip(miz.buffer(2000)).reduceResolution(ee.Reducer.mode(),maxPixels=200)
def gq(im): return im.updateMask(im.select('quality_flag').eq(1).And(im.select('degrade_flag').eq(0)).And(im.select('sensitivity').gt(0.95)))
gedi=ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY').filterBounds(miz).filterDate('2019-04-01','2023-03-31').map(gq).select('rh98').mean().setDefaultProjection('EPSG:4326',None,25)
gedi100=gedi.reduceResolution(ee.Reducer.mean(),maxPixels=400)
gcount=gedi.gt(0).reduceResolution(ee.Reducer.count(),maxPixels=400)
h=ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
lossfrac=h.select('loss').reduceResolution(ee.Reducer.mean(),maxPixels=200).reproject(prj)
lossyr=h.select('lossyear').selfMask().clip(miz.buffer(2000)).reduceResolution(ee.Reducer.max(),maxPixels=200)
tc=h.select('treecover2000').reduceResolution(ee.Reducer.mean(),maxPixels=200).reproject(prj)
L={'worldcover':wc,'gedi_rh98':gedi100,'gedi_n':gcount,'loss_year':lossyr}
import os
old=dict(np.load('validation100.npz')) if os.path.exists('validation100.npz') else {}
out={}
for k,im in L.items():
    for att in range(3):
        try:
            a=ee.data.computePixels({'expression':ee.Image(im).rename('b').toFloat().unmask(-9999),'fileFormat':'NUMPY_NDARRAY','grid':grid})['b'].astype('float32')
            a[a<=-9999]=np.nan; out[k]=a; print(k,np.nanmin(a),np.nanmax(a),np.isfinite(a).sum(),flush=True); break
        except Exception as e: print('retry',k,str(e)[:150],flush=True)
old.update(out); np.savez_compressed('validation100.npz',**old); print('SAVED')
