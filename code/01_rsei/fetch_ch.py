from common import *
import numpy as np, json
g=json.load(open('grid.json'))
grid={'dimensions':{'width':g['W'],'height':g['H']},'affineTransform':{'scaleX':100,'shearX':0,'translateX':g['x0'],'shearY':0,'scaleY':-100,'translateY':g['y1']},'crsCode':'EPSG:32646'}
im=ee.Image(f'projects/{PROJECT}/assets/mizoram_rsei/canopy_h_100m').rename('b').toFloat().unmask(-9999)
a=ee.data.computePixels({'expression':im,'fileFormat':'NUMPY_NDARRAY','grid':grid})['b'].astype('float32'); a[a<=-9999]=np.nan
d=dict(np.load('step3_layers100.npz')); d['canopy_h']=a; np.savez_compressed('step3_layers100.npz',**d); print(np.nanpercentile(a,[5,50,95]))
