# %% [init]
import ee, math, time
PROJECT = 'YOUR-GEE-PROJECT-ID'       # <-- the only thing to edit
ee.Authenticate()
ee.Initialize(project=PROJECT)
DRIVE_FOLDER = 'mndvi_vs_ndvi'
miz = ee.FeatureCollection('FAO/GAUL/2015/level1').filter(ee.Filter.eq('ADM1_NAME', 'Mizoram')).geometry()
# If you prefer your own boundary asset: miz = ee.FeatureCollection('projects/.../Mizoram').geometry()
print('Mizoram area km2:', miz.area(100).divide(1e6).getInfo())

# %% [landsat]
def prep(img):
    qa = img.select('QA_PIXEL')
    clear = (qa.bitwiseAnd(1 << 1).eq(0)          # dilated cloud
             .And(qa.bitwiseAnd(1 << 3).eq(0))   # cloud
             .And(qa.bitwiseAnd(1 << 4).eq(0))   # cloud shadow
             .And(qa.bitwiseAnd(1 << 5).eq(0)))  # snow
    sr = (img.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
             .multiply(0.0000275).add(-0.2).rename(['B', 'G', 'R', 'N', 'S1', 'S2']))
    lst = img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')
    return (sr.addBands(lst).updateMask(clear).updateMask(img.select('QA_RADSAT').eq(0))
              .copyProperties(img, ['system:time_start', 'SUN_ELEVATION', 'SUN_AZIMUTH']))

L89 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
       .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
       .filterBounds(miz).map(prep))
water = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(0).gt(50)

def season_composite(y0, y1):
    """Median of Nov-Feb (post-monsoon, leaf-on, before jhum burning) for seasons y0..y1."""
    f = ee.Filter.Or(*[ee.Filter.date(f'{y}-11-01', f'{y + 1}-03-01') for y in range(y0, y1 + 1)])
    col = L89.filter(f)
    comp = col.median().updateMask(water.Not()).clip(miz)
    return comp, col.aggregate_mean('SUN_ELEVATION'), col.aggregate_mean('SUN_AZIMUTH')

def s1_bounds(comp):
    p = comp.select('S1').reduceRegion(ee.Reducer.percentile([2, 98]), miz, 90, maxPixels=1e13, bestEffort=True, tileScale=4).getInfo()
    return p['S1_p2'], p['S1_p98']

# %% [indices]
dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
slope = ee.Terrain.slope(dem); aspect = ee.Terrain.aspect(dem)

def cos_i(sun_el, sun_az):
    d2r = math.pi / 180
    zen = ee.Image(ee.Number(90).subtract(sun_el)).multiply(d2r)
    az = ee.Image(ee.Number(sun_az)).multiply(d2r)
    s = slope.multiply(d2r); a = aspect.multiply(d2r)
    return zen.cos().multiply(s.cos()).add(zen.sin().multiply(s.sin()).multiply(az.subtract(a).cos())).rename('cos_i')

def mndvi(ndvi, s1, lo, hi, name):
    # Nemani et al. (1993): NDVI * (SWIRmax - SWIR)/(SWIRmax - SWIRmin), bounds = 2nd/98th percentile, factor clamped to [0,1]
    return ndvi.multiply(ee.Image.constant(hi).subtract(s1).divide(hi - lo).clamp(0, 1)).rename(name)

def indices(comp, scene_bounds, fixed_bounds):
    B, G, R, N, S1, S2 = [comp.select(b) for b in ['B', 'G', 'R', 'N', 'S1', 'S2']]
    ndvi = N.subtract(R).divide(N.add(R)).rename('NDVI')
    out = [ndvi,
           mndvi(ndvi, S1, *scene_bounds, 'MNDVI_scene'),
           mndvi(ndvi, S1, *fixed_bounds, 'MNDVI_fixed'),
           ndvi.pow(2).tanh().rename('kNDVI'),                                   # Camps-Valls et al. 2021
           comp.expression('2.5*(N-R)/(N+6*R-7.5*B+1)', {'N': N, 'R': R, 'B': B}).rename('EVI'),
           ndvi.subtract(0.08).multiply(N).rename('NIRv'),                        # Badgley et al. 2017
           N.subtract(S1).divide(N.add(S1)).rename('NDMI'),
           comp.expression('0.1511*B+0.1973*G+0.3283*R+0.3407*N-0.7117*S1-0.4559*S2',
                           {'B': B, 'G': G, 'R': R, 'N': N, 'S1': S1, 'S2': S2}).rename('TCW')]
    si = comp.expression('((S1+R)-(N+B))/((S1+R)+(N+B))', {'S1': S1, 'R': R, 'N': N, 'B': B})
    ibi = comp.expression('(2*S1/(S1+N)-(N/(N+R)+G/(G+S1)))/(2*S1/(S1+N)+(N/(N+R)+G/(G+S1)))', {'S1': S1, 'N': N, 'R': R, 'G': G})
    out.append(si.add(ibi).divide(2).rename('NDBSI'))
    out.append(comp.select('LST'))
    return ee.Image.cat(out)

# %% [bounds]
# Scene-dependent bounds per season (as in the submitted manuscript) and one FIXED pair (median across seasons)
SEASONS = list(range(2013, 2024))      # season y = Nov y .. Feb y+1 (Landsat 8 from 2013)
seasonal = {}
for y in SEASONS:
    comp, el, az = season_composite(y, y)
    seasonal[y] = (comp, el, az, s1_bounds(comp))
    print(y, 'SWIR1 p2/p98:', [round(v, 4) for v in seasonal[y][3]])
FIXED = (sorted(v[3][0] for v in seasonal.values())[len(SEASONS) // 2],
         sorted(v[3][1] for v in seasonal.values())[len(SEASONS) // 2])
print('FIXED bounds:', FIXED)

# %% [ancillary]
hansen = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
tc2000 = hansen.select('treecover2000').rename('tc2000')
lossyear = hansen.select('lossyear')
wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').rename('wc')
lonlat = ee.Image.pixelLonLat().rename(['lon', 'lat'])
anc = ee.Image.cat([tc2000, lossyear.rename('lossyear'), wc, dem.rename('elev'), slope.rename('slope'), lonlat])

# GEDI-epoch composite (seasons 2019-2022 = Nov 2019 - Feb 2023)
gcomp, gel, gaz = season_composite(2019, 2022)
G_IDX = indices(gcomp, s1_bounds(gcomp), FIXED).addBands(cos_i(gel, gaz)).addBands(anc)
recent_loss = lossyear.gte(18)   # exclude pixels cleared 2018-2023 (canopy may differ between lidar and image dates)

# %% [gedi]
def gedi(cid, qflag, bands):
    col = ee.ImageCollection(cid).filterBounds(miz).filterDate('2019-04-01', '2023-03-17')
    bn = ee.Image(col.first()).bandNames().getInfo()
    print(cid, 'images:', col.size().getInfo(), '| bands:', bn)
    assert qflag in bn and all(b in bn for b in bands), f'Band names differ in this GEDI version - check {bn}'
    extra = [b for b in ['beam', 'solar_elevation', 'sensitivity'] if b in bn]
    def q(im):
        ok = (im.select(qflag).eq(1).And(im.select('degrade_flag').eq(0))
                .And(im.select('sensitivity').gt(0.98)))         # 0.98 recommended for dense tropical canopy
        return im.updateMask(ok)
    mos = col.map(q).select(bands + extra).mosaic()
    return mos, ee.Image(col.first()).projection()

def export_gedi(cid, qflag, bands, name):
    mos, proj = gedi(cid, qflag, bands)
    img = mos.addBands(G_IDX).updateMask(recent_loss.Not()).updateMask(mos.select(bands[0]).mask())
    fc = img.sample(region=miz, projection=proj, dropNulls=True, tileScale=16, geometries=False)
    t = ee.batch.Export.table.toDrive(collection=fc, description=name, folder=DRIVE_FOLDER, fileFormat='CSV')
    t.start(); return t

tasks = []
tasks.append(export_gedi('LARSE/GEDI/GEDI02_A_002_MONTHLY', 'quality_flag', ['rh98'], 'gedi_l2a'))
tasks.append(export_gedi('LARSE/GEDI/GEDI02_B_002_MONTHLY', 'l2b_quality_flag', ['cover', 'pai'], 'gedi_l2b'))
tasks.append(export_gedi('LARSE/GEDI/GEDI04_A_002_MONTHLY', 'l4_quality_flag', ['agbd'], 'gedi_l4a'))

# %% [chronosequence]
ccomp, cel, caz = seasonal[2023][0], seasonal[2023][1], seasonal[2023][2]     # Nov 2023 - Feb 2024
C_IDX = indices(ccomp, seasonal[2023][3], FIXED).addBands(anc)
was_forest = tc2000.gte(50)
mature = tc2000.gte(80).And(lossyear.eq(0)).And(hansen.select('gain').eq(0)).And(wc.eq(10))
cls = ee.Image(0).where(lossyear.gt(0).And(was_forest), lossyear).updateMask(mature.Or(lossyear.gt(0).And(was_forest))).rename('cls')
chrono = C_IDX.addBands(cls).stratifiedSample(numPoints=1500, classBand='cls', region=miz, scale=30, seed=7, dropNulls=True, tileScale=16, geometries=False)
t = ee.batch.Export.table.toDrive(collection=chrono, description='chronosequence', folder=DRIVE_FOLDER, fileFormat='CSV'); t.start(); tasks.append(t)

# %% [temporal]
pts = mature.selfMask().addBands(lonlat).sample(region=miz, scale=30, numPixels=60000, seed=3, geometries=True, tileScale=16).limit(4000)
def per_season(y):
    comp, el, az, b = seasonal[y]
    return indices(comp, b, FIXED).sampleRegions(collection=pts, scale=30, tileScale=16).map(lambda f: f.set('year', y))
temporal = ee.FeatureCollection([per_season(y) for y in SEASONS]).flatten()
t = ee.batch.Export.table.toDrive(collection=temporal, description='temporal_stability', folder=DRIVE_FOLDER, fileFormat='CSV'); t.start(); tasks.append(t)

# %% [random]
rand = G_IDX.sample(region=miz, scale=30, numPixels=40000, seed=11, dropNulls=True, tileScale=16, geometries=False)
t = ee.batch.Export.table.toDrive(collection=rand, description='random_state', folder=DRIVE_FOLDER, fileFormat='CSV'); t.start(); tasks.append(t)

# %% [wait]
while any(t.status()['state'] in ('READY', 'RUNNING') for t in tasks):
    print({t.status()['description']: t.status()['state'] for t in tasks}); time.sleep(60)
print({t.status()['description']: (t.status()['state'], t.status().get('error_message', '')) for t in tasks})
