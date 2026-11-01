/*******************************************************************************************
 * NDVI and MNDVI generation for Mizoram, 2000-2025 (Google Earth Engine Code Editor)
 * ----------------------------------------------------------------------------------------
 * Landsat Collection 2 Level-2 surface reflectance:
 *   TM (L5), ETM+ (L7), OLI (L8), OLI-2 (L9)
 * Composite window: Nov (Y-1) - Feb (Y). Post-monsoon, leaf-on, before jhum burning (Mar-Apr).
 *
 * NDVI  = (NIR - Red) / (NIR + Red)                                   Rouse et al. 1974
 * MNDVI = NDVI x (SWIRmax - SWIR) / (SWIRmax - SWIRmin)               Nemani et al. 1993
 *         SWIR = SWIR1 (~1.6 um); SWIRmin/max = 2nd/98th percentile; scaling factor clamped to [0,1]
 *   MNDVI_scene : bounds recomputed for each epoch (as in the submitted manuscript)
 *   MNDVI_fixed : one pair of bounds for all epochs (median across epochs), so values are
 *                 comparable over time
 *
 * Outputs: map layers, printed SWIR bounds, a per-epoch summary chart, and GeoTIFF exports
 * (3 bands: NDVI, MNDVI_scene, MNDVI_fixed; 30 m; EPSG:32646) to Google Drive.
 *******************************************************************************************/

// ------------------------------ 1. Settings ---------------------------------------------
var EPOCHS        = [2000, 2005, 2010, 2015, 2020, 2025];  // epoch Y = Nov (Y-1) to Feb (Y)
var POOL_SEASONS  = 1;      // 1 = that season only; 2 = also add the previous season (fills cloud gaps)
var HARMONIZE     = true;   // put TM/ETM+ reflectance on the OLI scale (Roy et al. 2016)
var DRIVE_FOLDER  = 'Mizoram_NDVI_MNDVI';
var EXPORT_CRS    = 'EPSG:32646';
var EXPORT_SCALE  = 30;

var miz = ee.FeatureCollection('FAO/GAUL/2015/level1')
            .filter(ee.Filter.eq('ADM1_NAME', 'Mizoram')).geometry();
// To use your own boundary asset instead:
// var miz = ee.FeatureCollection('projects/YOUR_PROJECT/assets/Mizoram').geometry();
Map.centerObject(miz, 8);

// ------------------------------ 2. Landsat preparation ----------------------------------
function cloudMask(img) {
  var qa = img.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 1).eq(0)       // dilated cloud
      .and(qa.bitwiseAnd(1 << 3).eq(0))         // cloud
      .and(qa.bitwiseAnd(1 << 4).eq(0))         // cloud shadow
      .and(qa.bitwiseAnd(1 << 5).eq(0));        // snow
  return img.updateMask(clear).updateMask(img.select('QA_RADSAT').eq(0));
}

// Surface reflectance scaling for Collection 2 Level-2
function scaleSR(img, bands) {
  return img.select(bands).multiply(0.0000275).add(-0.2)
            .rename(['B', 'G', 'R', 'N', 'S1', 'S2']);
}

// TM / ETM+  ->  OLI (Roy et al. 2016, Remote Sens. Environ. 185:57-70, Table 2, OLS)
var ROY_SLOPE = ee.Image.constant([0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071]);
var ROY_INTER = ee.Image.constant([0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172]);

function prepTM(img) {   // Landsat 5 TM and Landsat 7 ETM+
  var sr = scaleSR(cloudMask(img), ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']);
  sr = HARMONIZE ? sr.multiply(ROY_SLOPE).add(ROY_INTER)
                     .rename(['B', 'G', 'R', 'N', 'S1', 'S2']) : sr;
  return sr.toFloat()   // cast: TM/ETM+ and OLI must share one band type to merge
           .copyProperties(img, ['system:time_start', 'SPACECRAFT_ID']);
}
function prepOLI(img) {  // Landsat 8 and 9 OLI
  return scaleSR(cloudMask(img), ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).toFloat()
           .copyProperties(img, ['system:time_start', 'SPACECRAFT_ID']);
}

var LS = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').map(prepTM)
  .merge(ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').map(prepTM))
  .merge(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').map(prepOLI))
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').map(prepOLI))
  .filterBounds(miz);

// Exclude permanent water (would distort SWIR percentiles and NDVI)
var water = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(0).gt(50);

// ------------------------------ 3. Seasonal composite -----------------------------------
function composite(Y) {
  var filters = [];
  for (var k = 0; k < POOL_SEASONS; k++) {
    var y = Y - k;
    filters.push(ee.Filter.date(ee.Date.fromYMD(y - 1, 11, 1), ee.Date.fromYMD(y, 3, 1)));
  }
  var col = LS.filter(ee.Filter.or.apply(null, filters));
  return col.median().updateMask(water.not()).clip(miz)
            .set({epoch: Y, n_scenes: col.size(),
                  sensors: col.aggregate_array('SPACECRAFT_ID').distinct()});
}

// ------------------------------ 4. Index functions --------------------------------------
function ndviOf(c) {
  return c.normalizedDifference(['N', 'R']).rename('NDVI');   // (N-R)/(N+R)
}

function swirBounds(c) {
  var p = c.select('S1').reduceRegion({
    reducer: ee.Reducer.percentile([2, 98]), geometry: miz, scale: 90,
    maxPixels: 1e13, bestEffort: true, tileScale: 4});
  return ee.Dictionary({min: p.get('S1_p2'), max: p.get('S1_p98')});
}

function mndviOf(c, ndvi, bounds, name) {
  var mn = ee.Number(bounds.get('min')), mx = ee.Number(bounds.get('max'));
  var factor = ee.Image(mx).subtract(c.select('S1'))
                 .divide(mx.subtract(mn)).clamp(0, 1);
  return ndvi.multiply(factor).rename(name);
}

// ------------------------------ 5. Build all epochs -------------------------------------
var comps  = EPOCHS.map(composite);
var bounds = comps.map(swirBounds);

// Fixed bounds = median of the per-epoch bounds (server side)
var fixedBounds = ee.Dictionary({
  min: ee.List(bounds.map(function (b) { return b.get('min'); })).reduce(ee.Reducer.median()),
  max: ee.List(bounds.map(function (b) { return b.get('max'); })).reduce(ee.Reducer.median())
});
print('Fixed SWIR1 bounds (median of epochs):', fixedBounds);

var outputs = EPOCHS.map(function (Y, i) {
  var c = comps[i], ndvi = ndviOf(c);
  var img = ee.Image.cat([
      ndvi,
      mndviOf(c, ndvi, bounds[i], 'MNDVI_scene'),
      mndviOf(c, ndvi, fixedBounds, 'MNDVI_fixed')
    ]).toFloat()
    .set({epoch: Y, swir_min_scene: bounds[i].get('min'), swir_max_scene: bounds[i].get('max'),
          n_scenes: c.get('n_scenes')});
  print('Epoch ' + Y, {scenes: c.get('n_scenes'), sensors: c.get('sensors'),
                       swir_bounds_scene: bounds[i]});
  return img;
});
var stack = ee.ImageCollection.fromImages(outputs);

// ------------------------------ 6. Display ----------------------------------------------
var pal = ['#a50026', '#f46d43', '#fee08b', '#d9ef8b', '#66bd63', '#006837'];
var last = outputs[outputs.length - 1], first = outputs[0];
Map.addLayer(first.select('NDVI'),        {min: 0, max: 0.9, palette: pal}, 'NDVI ' + EPOCHS[0], false);
Map.addLayer(first.select('MNDVI_fixed'), {min: 0, max: 0.8, palette: pal}, 'MNDVI fixed ' + EPOCHS[0], false);
Map.addLayer(last.select('NDVI'),         {min: 0, max: 0.9, palette: pal}, 'NDVI ' + EPOCHS[EPOCHS.length - 1]);
Map.addLayer(last.select('MNDVI_scene'),  {min: 0, max: 0.8, palette: pal}, 'MNDVI scene ' + EPOCHS[EPOCHS.length - 1], false);
Map.addLayer(last.select('MNDVI_fixed'),  {min: 0, max: 0.8, palette: pal}, 'MNDVI fixed ' + EPOCHS[EPOCHS.length - 1], false);

// Statewide mean of each index per epoch (MNDVI_scene vs MNDVI_fixed shows the effect of re-scaling)
var means = stack.map(function (im) {
  return ee.Feature(null, im.reduceRegion({reducer: ee.Reducer.mean(), geometry: miz, scale: 300,
                                           maxPixels: 1e13, bestEffort: true}))
           .set('epoch', im.get('epoch'));
});
print(ui.Chart.feature.byFeature(means, 'epoch', ['NDVI', 'MNDVI_scene', 'MNDVI_fixed'])
        .setOptions({title: 'Mizoram mean index by epoch', hAxis: {title: 'Epoch'},
                     vAxis: {title: 'Mean value'}, pointSize: 4}));

// ------------------------------ 7. Export GeoTIFFs --------------------------------------
EPOCHS.forEach(function (Y, i) {
  Export.image.toDrive({
    image: outputs[i], description: 'Mizoram_NDVI_MNDVI_' + Y, folder: DRIVE_FOLDER,
    fileNamePrefix: 'Mizoram_NDVI_MNDVI_' + Y, region: miz, scale: EXPORT_SCALE,
    crs: EXPORT_CRS, maxPixels: 1e13, fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true}});
});
// Run the export tasks from the "Tasks" tab (top right of the Code Editor).
