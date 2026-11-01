from common import *
import numpy as np, json
A=f'projects/{PROJECT}/assets/mizoram_rsei/RSEI_annual_1990_2025_v3'
g=json.load(open('grid.json'))
grid={'dimensions':{'width':g['W'],'height':g['H']},'affineTransform':{'scaleX':g['res'],'shearX':0,'translateX':g['x0'],'shearY':0,'scaleY':-g['res'],'translateY':g['y1']},'crsCode':'EPSG:32646'}
img=ee.Image(A); O=np.load('v2_backup/rsei_stack100_obs.npz'); out={}
for Y in [y for y in range(1990,2026) if y!=1991]:
    b=f'RSEI_{Y}'
    for att in range(3):
        try:
            a=ee.data.computePixels({'expression':img.select([b]).unmask(-9999),'fileFormat':'NUMPY_NDARRAY','grid':grid})[b].astype('float32'); break
        except Exception as e: print('retry',Y,e,flush=True)
    a[a<=-9999]=np.nan; a/=10000; a[np.isnan(O[str(Y)])]=np.nan; out[str(Y)]=a; print('got',Y,flush=True)
np.savez_compressed('rsei_stack100_obs.npz',**out); np.savez_compressed('rsei_stack100.npz',**out); print('SAVED')
