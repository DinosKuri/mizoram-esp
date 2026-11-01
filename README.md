# Code: testing an ecological security pattern in Mizoram, India

This archive accompanies the article:

> Khiangte, L., Sati, V.P., Khuma, Z. Testing an ecological security pattern against independent species records in a shifting-cultivation mountain of the Indo-Burma hotspot. Manuscript under review.

It holds the code that produces the results reported in the article: the pooled, terrain-corrected Remote Sensing Ecological Index (RSEI) for 35 dry seasons, trend and driver analyses, guild-specific connectivity models, validation with independent species records, and the ecological security pattern (ESP) with restoration scenarios. The tables behind the results are published with the article as Supplementary Data S1–S5; the derived raster layers are available from the corresponding author on reasonable request. All input data are openly available and are cited, not redistributed.

**Corresponding author:** Laldinchhuaha Khiangte, Department of Geography and Resource Management, Mizoram University, Aizawl 796004, India (khiangtedinchhuaha@gmail.com; ORCID 0009-0003-3258-7299).

## Contents

```
code/
  shared/                         Earth Engine set-up (common.py), RSEI library (rsei_lib.py), analysis grid (grid.json)
  01_rsei/                        compositing, cross-sensor correction, terrain correction, pooled PCA, gap filling, validation
  02_quality_dynamics_drivers/    trends, spatial autocorrelation, drivers, mautam test, protected-area layer
  03_connectivity/                sources, resistance, corridors, pinch points, barriers, sensitivity, Monte Carlo
    circuitscape_omniscape/       Julia runner for Circuitscape and Omniscape
  04_validation/                  bird (GBIF/eBird) and camera-trap validation
  05_esp_restoration/             ESP zoning and restoration scenarios
```

## Grid and coordinate system

All rasters produced by the code use EPSG:32646 (WGS 84 / UTM zone 46N) and one grid of 100 m cells: upper-left corner x = 422,500 m, y = 2,714,000 m; 1,261 columns × 2,910 rows (`code/shared/grid.json`).

## Running the code

**Software.** Python 3.13 with earthengine-api 1.7, NumPy 2.5, pandas 3.0, SciPy 1.18, scikit-image 0.26, rasterio 1.5, GeoPandas 1.2, Shapely 2.2, pyproj 3.8, libpysal 4.15, esda 2.10, LightGBM 4.7, SHAP 0.54, scikit-learn 1.9 and statsmodels 0.15; Circuitscape 5.17 and Omniscape 0.6 in Julia 1.11.

**Credentials (not included).** `shared/common.py` reads a Google Earth Engine service-account key from the file named in the environment variable `GEE_KEY`. `01_rsei/planet_sample.py` reads a Planet key from `PLANET_API_KEY`; Planet imagery is licensed and not redistributed (the interpreted points are Supplementary Data S3).

**Order.** Copy `shared/` (including `grid.json`) into one working directory and run the scripts there; each step writes the intermediate files the next step reads.

1. **RSEI (01_rsei).** `fetch_grids.py` → `scenes.py`, `calib_pairs.py`, `overlap.py`, `calib_cv.py`, `set_offsets.py` (cross-sensor offsets) → `topo_test.py` (terrain-correction check) → `pooled_pca.py`, `make_model.py` (pooled normalisation and PCA; `peryear.py` for the per-image comparison) → `export_v3.py`, `gapfill_export_v3.py` → `fetch_v3.py`, `merge_gapfill.py`, `merge_gapfill_full.py` → `class_areas.py`, `class_table.py` → validation: `export_val.py`, `fetch_validation.py`, `export_ch.py`, `fetch_ch.py`, `validation_stats.py`, `planet_sample.py`. `NDVI_MNDVI_Mizoram_GEE.js` and `mndvi_test_gee_cells.py` reproduce the NDVI–MNDVI comparison.
2. **Dynamics and drivers (02_quality_dynamics_drivers).** `trend_mmk.py`, `mk_sens.py` → `spatial_autocorr.py` → `fetch_predictors.py`, `build_pa_layer.py`, `update_pa_predictors.py` → `drivers.py`, `lulc_sensitivity.py` → `bamboo_georef.py`, `mautam_test.py`, `hansen_event_study.py`.
3. **Connectivity (03_connectivity).** `fetch_step3.py`, `district_update.py` → `fallow_age.py`, `hunting_access.py` → `sources.py`, `sources_final.py` → `resistance.py` → `step3_baseline.py` (uses `corridors.py`, `corridor_width.py`) → `cs_prep.py`, then `julia circuitscape_omniscape/cs_run.jl <guild> <radius_cells> <block_size>` → `pinch.py`, `step3_post.py` → `omni_sens_prep.py`, `omni_radius.py`, `step3_sensitivity.py`, `mc_stats.py`, `step3_extra.py`, `hornbill_post.py`.
4. **Validation (04_validation).** `val_prep.py` → `val_birds.py` → `val_mammals.py`, using the GBIF download https://doi.org/10.15468/dl.vg4anj.
5. **ESP and restoration (05_esp_restoration).** `esp_zones.py` → `restore.py`.

## Input data

All input data are openly available from their providers (Table S1 of the article): Landsat Collection 2 (USGS), NASADEM, CHIRPS, GHSL, VIIRS night lights, ESA WorldCover, GEDI, ETH canopy height, Hansen Global Forest Change, MODIS fire products, the accessibility surface and GBIF. Protected-area boundaries come from OpenStreetMap (© OpenStreetMap contributors, ODbL) and, for Thorangtlang Wildlife Sanctuary, from the World Database on Protected Areas, whose terms do not allow redistribution; `build_pa_layer.py` rebuilds the layer. District boundaries (2019) are from the Local Government Directory, Government of India. The bamboo-flowering map was digitised from Lallianthanga and Sailo (2012).

## Licence

MIT License (`LICENSE`). Please cite the article and this archive.
