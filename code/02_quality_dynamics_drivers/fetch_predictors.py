"""Non-circular explanatory variables on the 100 m analysis grid (EPSG:32646). None is derived from Landsat reflectance."""
from common import *
import numpy as np, json
g=json.load(open('grid.json'))
grid={'dimensions':{'width':g['W'],'height':g['H']},'affineTransform':{'scaleX':g['res'],'shearX':0,'translateX':g['x0'],'shearY':0,'scaleY':-g['res'],'translateY':g['y1']},'crsCode':'EPSG:32646'}
dem=ee.Image('NASA/NASADEM_HGT/001').select('elevation')
slope=ee.Terrain.slope(dem); aspect=ee.Terrain.aspect(dem)
tpi=dem.subtract(dem.focalMean(1000,'circle','meters'))
north=aspect.multiply(3.14159265/180).cos()
yrs=ee.List.sequence(1991,2020)
chirps=ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD')
prec=ee.ImageCollection(yrs.map(lambda y: chirps.filter(ee.Filter.calendarRange(y,y,'year')).sum())).mean()
roads=ee.FeatureCollection('projects/sat-io/open-datasets/GRIP4/South-East-Asia').filterBounds(miz.buffer(30000))
droad=roads.distance(60000)
access=ee.Image('Oxford/MAP/accessibility_to_cities_2015_v1_0').select('accessibility')
pop=ee.ImageCollection('JRC/GHSL/P2023A/GHS_POP').filter(ee.Filter.eq('system:index','2020')).first()
built=ee.ImageCollection('JRC/GHSL/P2023A/GHS_BUILT_S').filter(ee.Filter.eq('system:index','2020')).first().select('built_surface')
dbuilt=built.gt(1000).selfMask().fastDistanceTransform(600,'pixels','squared_euclidean').sqrt().multiply(ee.Image.pixelArea().sqrt())
fires=ee.ImageCollection('FIRMS').filterDate('2001-01-01','2025-01-01').filterBounds(miz).select('T21').map(lambda i: i.gt(0).unmask(0))
firecount=fires.sum().reproject('EPSG:4326',None,1000)
burn=ee.ImageCollection('MODIS/061/MCD64A1').filterDate('2001-01-01','2025-01-01').select('BurnDate').map(lambda i: i.gt(0).unmask(0)).sum()
pa=ee.FeatureCollection('WCMC/WDPA/current/polygons').filterBounds(miz.buffer(20000))
pa_in=ee.Image(0).paint(pa,1)
dpa=pa.distance(80000)
L={'elev':dem,'slope':slope,'tpi':tpi,'northness':north,'precip':prec,'dist_road':droad,'travel_time':access,
   'pop':pop,'dist_built':dbuilt,'fire_count':firecount,'burn_count':burn,'pa_inside':pa_in,'dist_pa':dpa}
out={}
for k,im in L.items():
    for att in range(3):
        try:
            a=ee.data.computePixels({'expression':ee.Image(im).rename('b').toFloat().unmask(-9999),'fileFormat':'NUMPY_NDARRAY','grid':grid})['b'].astype('float32')
            a[a<=-9999]=np.nan; out[k]=a; print(k,np.nanmin(a),np.nanmax(a),flush=True); break
        except Exception as e: print('retry',k,str(e)[:150],flush=True)
np.savez_compressed('predictors100.npz',**out); print('SAVED')
