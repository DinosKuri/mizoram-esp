from common import *
NAMES=['B','G','R','N','S1','S2']
# Roy et al. (2016) ETM+/TM -> OLI surface reflectance (OLS)
ROY_S=ee.Image.constant([0.8474,0.8483,0.9047,0.8462,0.8937,0.9071])
ROY_I=ee.Image.constant([0.0003,0.0088,0.0061,0.0412,0.0254,0.0172])
water=ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(0).gt(50)

def _mask(img):
    qa=img.select('QA_PIXEL')
    ok=(qa.bitwiseAnd(1<<1).eq(0).And(qa.bitwiseAnd(1<<3).eq(0))
        .And(qa.bitwiseAnd(1<<4).eq(0)).And(qa.bitwiseAnd(1<<5).eq(0)))
    return img.updateMask(ok).updateMask(img.select('QA_RADSAT').eq(0))

def _lst(img, band):
    return img.select(band).multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')

def prep_tm(img, harmonize=True):
    m=_mask(img)
    sr=m.select(['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7']).multiply(0.0000275).add(-0.2)
    if harmonize: sr=sr.multiply(ROY_S).add(ROY_I)
    return sr.rename(NAMES).addBands(_lst(m,'ST_B6')).toFloat().copyProperties(img,['system:time_start','SPACECRAFT_ID','SUN_ELEVATION','SUN_AZIMUTH'])

def prep_oli(img, harmonize=True):
    m=_mask(img)
    sr=m.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']).multiply(0.0000275).add(-0.2)
    return sr.rename(NAMES).addBands(_lst(m,'ST_B10')).toFloat().copyProperties(img,['system:time_start','SPACECRAFT_ID','SUN_ELEVATION','SUN_AZIMUTH'])

def landsat(sensors=('L5','L7','L8','L9'), harmonize=True):
    ids={'L5':'LANDSAT/LT05/C02/T1_L2','L7':'LANDSAT/LE07/C02/T1_L2','L8':'LANDSAT/LC08/C02/T1_L2','L9':'LANDSAT/LC09/C02/T1_L2'}
    out=None
    for s in sensors:
        f=prep_tm if s in ('L5','L7') else prep_oli
        c=ee.ImageCollection(ids[s]).filterBounds(miz).map(lambda im, f=f: f(im, harmonize))
        out=c if out is None else out.merge(c)
    return out

import math, hashlib
TOPO=True
_dem=ee.Image('NASA/NASADEM_HGT/001').select('elevation')
_slope=ee.Terrain.slope(_dem).multiply(3.141592653589793/180)
_aspect=ee.Terrain.aspect(_dem).multiply(3.141592653589793/180)
_TOPO_CACHE={}
def topo_coefs(comp, col, key):
    """Client-side SCS+C coefficients for one composite (cached): season-mean sun geometry, C per band, LST slope on cos(i)."""
    if key in _TOPO_CACHE: return _TOPO_CACHE[key]
    for att in range(4):
        try:
            g=ee.Dictionary({'el':col.aggregate_mean('SUN_ELEVATION'),'az':col.aggregate_mean('SUN_AZIMUTH')}).getInfo()
            zen=math.radians(90-g['el']); azr=math.radians(g['az'])
            cosi=_slope.cos().multiply(math.cos(zen)).add(_slope.sin().multiply(math.sin(zen)).multiply(_aspect.subtract(azr).cos())).rename('cosi')
            ndvi=comp.normalizedDifference(['N','R'])
            smp=ee.Image.constant(1).rename('c').addBands(cosi).addBands(comp.select(NAMES+['LST'])).updateMask(ndvi.gt(0.5).And(_slope.gt(0.0873)))
            r=smp.reduceRegion(ee.Reducer.linearRegression(2,7),miz,300,maxPixels=1e13,bestEffort=True,tileScale=8)
            cm=smp.select('cosi').reduceRegion(ee.Reducer.mean(),miz,300,maxPixels=1e13,bestEffort=True,tileScale=8)
            out=ee.Dictionary({'coef':r.get('coefficients'),'cm':cm.get('cosi')}).getInfo()
            co=out['coef']; C=[co[0][i]/co[1][i] for i in range(6)]
            res=dict(zen=zen,azr=azr,C=C,lst_slope=co[1][6],cosi_mean=out['cm'])
            _TOPO_CACHE[key]=res; return res
        except Exception as e:
            print('topo retry',key,str(e)[:80],flush=True); import time; time.sleep(10)
    raise RuntimeError('topographic coefficients failed for '+str(key))

def topo_correct(comp, col, key):
    """SCS+C topographic correction (Soenen et al. 2005) of the six reflectance bands of a seasonal composite with the
    season-mean solar geometry; C per band from vegetated pixels (NDVI > 0.5, slope > 5 deg). The illumination-driven part of
    LST (its regression slope on cos(i) over the same pixels) is removed."""
    k=topo_coefs(comp,col,key)
    cosi=_slope.cos().multiply(math.cos(k['zen'])).add(_slope.sin().multiply(math.sin(k['zen'])).multiply(_aspect.subtract(k['azr']).cos()))
    scs=_slope.cos().multiply(math.cos(k['zen']))
    out=[comp.select(b).multiply(scs.add(c)).divide(cosi.add(c)).rename(b) for b,c in zip(NAMES,k['C'])]
    lst=comp.select('LST').subtract(cosi.subtract(k['cosi_mean']).multiply(k['lst_slope'])).rename('LST')
    return ee.Image.cat(out).addBands(lst)

def season(Y, col=None, ext=0):
    """Season Y = 1 Nov (Y-1) .. end Feb Y (post-monsoon, leaf-on, pre-burning).
    ext=1 widens the window by one month on each side (1 Oct .. end Mar); used only to fill cloud gaps."""
    col=col if col is not None else landsat()
    c=col.filterDate(f'{Y-1}-{11-ext:02d}-01', f'{Y}-{3+ext:02d}-01')
    comp=c.median().updateMask(water.Not()).clip(miz)
    if TOPO: comp=topo_correct(comp, c, (Y, ext, hashlib.md5(col.serialize().encode()).hexdigest()))
    nobs=c.select('N').count().rename('nobs')
    return comp.addBands(nobs).set('season', Y)

def indicators(comp):
    B,G,R,N,S1,S2=[comp.select(b) for b in NAMES]
    ndvi=N.subtract(R).divide(N.add(R)).rename('NDVI')
    # Tasseled-cap wetness, Crist (1985) reflectance-factor coefficients, one set for all years
    wet=comp.expression('0.0315*B+0.2021*G+0.3102*R+0.1594*N-0.6806*S1-0.6109*S2',
                        {'B':B,'G':G,'R':R,'N':N,'S1':S1,'S2':S2}).rename('WET')
    si=comp.expression('((S1+R)-(N+B))/((S1+R)+(N+B))',{'S1':S1,'R':R,'N':N,'B':B})
    ibi=comp.expression('(2*S1/(S1+N)-(N/(N+R)+G/(G+S1)))/(2*S1/(S1+N)+(N/(N+R)+G/(G+S1)))',{'S1':S1,'N':N,'R':R,'G':G})
    ndbsi=si.add(ibi).divide(2).rename('NDBSI')
    return ee.Image.cat([ndvi,wet,ndbsi,comp.select('LST'),comp.select('nobs')]).toFloat()

IND=['NDVI','WET','NDBSI','LST']

import json as _json, os as _os
OFF=_json.load(open(_os.path.join(_os.path.dirname(__file__),'calib_offsets.json')))
_C={s:landsat((s,)) for s in ('L5','L7','L8','L9')}
_OLI=_C['L8'].merge(_C['L9'])
SEASONS=[y for y in range(1990,2026) if y!=1991]

def _shift(img, pairs):
    o=[sum(OFF[p][i] for p in pairs) for i in IND]
    return img.select(IND).add(ee.Image.constant(o).rename(IND))

def calibrated(Y, ext=0):
    """Season-Y indicators on the Landsat-8 OLI scale (empirical offsets, Mizoram overlap seasons)."""
    if Y>=2014:
        ind=indicators(season(Y,_OLI,ext)).select(IND)
    else:
        parts=[]
        if Y<=2011 and Y not in (2002,2003): parts.append(_shift(indicators(season(Y,_C['L5'],ext)),['L5-L7','L7-L8']))
        if Y>=2000: parts.append(_shift(indicators(season(Y,_C['L7'],ext)),['L7-L8']))
        ind=ee.ImageCollection(parts).mean() if len(parts)>1 else parts[0]
    return ind.rename(IND).toFloat().set('season',Y)

_M=_json.load(open(_os.path.join(_os.path.dirname(__file__),'rsei_model.json')))
def rsei_image(Y, ext=0):
    ind=calibrated(Y, ext)
    med=ee.Number(ind.select('LST').reduceRegion(ee.Reducer.median(),miz,300,maxPixels=1e13,tileScale=4).get('LST'))
    lstc=ind.select('LST').subtract(med).rename('LSTc')
    X=ind.select(['NDVI','WET','NDBSI']).addBands(lstc)
    Xn=X.subtract(ee.Image.constant(_M['lo'])).divide(ee.Image.constant(_M['hi']).subtract(ee.Image.constant(_M['lo']))).clamp(0,1)
    r0=Xn.multiply(ee.Image.constant(_M['v'])).reduce(ee.Reducer.sum())
    r=r0.subtract(_M['r0_p2']).divide(_M['r0_p98']-_M['r0_p2']).clamp(0,1).rename('RSEI')
    return r.toFloat().set('season',Y,'system:time_start',ee.Date.fromYMD(Y,1,1).millis())


def rsei_filled(Y):
    """RSEI for season Y with cloud gaps filled: (1) same season, Oct-Mar window; (2) mean of the
    neighbouring seasons. Band 'fill': 0 observed, 1 widened window, 2 neighbour-season mean."""
    main=rsei_image(Y); ext=rsei_image(Y,1)
    i=SEASONS.index(Y); nb=[SEASONS[k] for k in (i-1,i+1) if 0<=k<len(SEASONS)]
    neigh=ee.ImageCollection([rsei_image(n) for n in nb]).mean().rename('RSEI')
    r=main.unmask(ext).unmask(neigh).clip(miz)
    fill=ee.Image(0).where(main.mask().Not().And(ext.mask()),1).where(main.mask().Not().And(ext.mask().Not()).And(neigh.mask()),2)
    return r.addBands(fill.rename('fill').updateMask(r.mask())).set('season',Y)
