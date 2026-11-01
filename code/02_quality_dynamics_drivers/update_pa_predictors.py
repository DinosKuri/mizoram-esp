"""Replace the protected-area predictors (inside, distance) on the 100 m grid with the validated layer."""
import geopandas as gpd, numpy as np, json
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage as ndi
g=json.load(open('grid.json')); TR=from_origin(g['x0'],g['y1'],g['res'],g['res']); SH=(g['H'],g['W'])
pa=gpd.read_file('pa_validated.geojson').to_crs(32646)
inside=rasterize([(x,1) for x in pa.geometry],out_shape=SH,transform=TR).astype('float32')
d=(ndi.distance_transform_edt(inside==0)*g['res']).astype('float32')
P=dict(np.load('predictors100.npz')); nan=np.isnan(P['elev'])
inside[nan]=np.nan; d[nan]=np.nan; d[d>80000]=np.nan
P['pa_inside']=inside; P['dist_pa']=d; np.savez_compressed('predictors100.npz',**P); print('ok')
