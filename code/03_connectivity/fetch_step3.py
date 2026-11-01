"""Extra layers for Step 3 on the 100 m grid: VIIRS night lights, GHS built-up fraction, ETH canopy height, JRC water."""
from common import *
import numpy as np, json
g=json.load(open('grid.json'))
grid={'dimensions':{'width':g['W'],'height':g['H']},'affineTransform':{'scaleX':g['res'],'shearX':0,'translateX':g['x0'],'shearY':0,'scaleY':-g['res'],'translateY':g['y1']},'crsCode':'EPSG:32646'}
ntl=ee.ImageCollection('NOAA/VIIRS/DNB/ANNUAL_V22').filterDate('2022-01-01','2025-01-01').select('average').median()
built=ee.ImageCollection('JRC/GHSL/P2023A/GHS_BUILT_S').filter(ee.Filter.eq('system:index','2020')).first().select('built_surface').divide(10000)
ch=ee.ImageCollection('projects/sat-io/open-datasets/GLOBAL-CANOPY-HEIGHT/ETH').mosaic() if False else ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1')
water=ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(0).gt(50)
L={'ntl':ntl,'built_frac':built,'canopy_h':ch.select(0),'water':water}
out={}
for k,im in L.items():
    for att in range(3):
        try:
            img=ee.Image(im).rename('b').toFloat()
            if k in ('canopy_h','water'): img=img.reduceResolution(ee.Reducer.mean(),maxPixels=1024).reproject('EPSG:32646',None,100)
            a=ee.data.computePixels({'expression':img.unmask(-9999),'fileFormat':'NUMPY_NDARRAY','grid':grid})['b'].astype('float32')
            a[a<=-9999]=np.nan; out[k]=a; print(k,np.nanmin(a),np.nanmax(a),flush=True); break
        except Exception as e: print('retry',k,str(e)[:200],flush=True)
np.savez_compressed('step3_layers100.npz',**out); print('SAVED',list(out))
