#!/usr/bin/env python3
'''
Per-window gross offsets for the offsets product from an ice velocity map.

For each point P of the offsets product grid of the reference RSLC:
  1. rdr2geo  -> ground position P (x, y, h) in the velocity map projection
  2. P' = P + v(P) * dt, dt = secondary - reference acquisition time,
     h' = h + S(P') - S(P), S = smoothed DEM (surface features keep their
     height, only the large-scale surface slope changes it)
  3. geo2rdr of P' with the reference orbit -> azimuth/range offset of P'
     from P in reference RSLC pixels, i.e. the offset ampcor measures between
     the reference and the geometry-coregistered secondary RSLC.

Outputs in <scratch>/velocity_offsets/freq<freq> (offsets grid geometry):
  velocity_offsets  ENVI float32, band 1 azimuth, band 2 range [RSLC pixels]
  velocity          ENVI float32, band 1 vx, band 2 vy [map units/yr]
  gross_offset.bin  int32 (azimuth, range) per window, the format of
                    offsets_product `gross_offset_filepath`
  flow_direction.bin  float32 (azimuth, range, band half-width) per window
                    for the DLC peak search: unit flow direction, (0, 0)
                    where the predicted offset is below `dlc_min_offset`
                    pixels or, with `dlc_shear_dominant_only`, outside
                    shear-dominant flow; band half-width [pixels] of the
                    band search
'''
import pathlib
import time

import isce3
import journal
import numpy as np
from isce3.core import crop_external_orbit
from nisar.products.readers import SLC
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.workflows.helpers import get_cfg_freq_pols, get_offset_radar_grid
from nisar.workflows.rdr2geo import get_raster_obj
from nisar.workflows.yaml_argparse import YamlArgparse
from osgeo import gdal, osr
from scipy.ndimage import gaussian_filter, map_coordinates

SECONDS_PER_YEAR = 365.25 * 86400.0


def gross_offset_path(scratch_path, freq):
    '''Path of the gross offset file computed for frequency `freq`'''
    return pathlib.Path(scratch_path) / 'velocity_offsets' / \
        f'freq{freq}' / 'gross_offset.bin'


def flow_direction_path(scratch_path, freq):
    '''Path of the flow direction file computed for frequency `freq`'''
    return gross_offset_path(scratch_path, freq).with_name(
        'flow_direction.bin')


def flow_direction(az_off, rg_off, min_offset, use=True, band_angle=0.0,
                   band_min_half_width=0.0):
    '''
    Unit (azimuth, range) direction of predicted offsets, (0, 0) where their
    magnitude is below `min_offset` pixels, undefined or `use` is False, and
    the band half-width max(|offset| * tan(band_angle [deg]),
    band_min_half_width) in pixels
    '''
    mag = np.hypot(az_off, rg_off)
    valid = np.isfinite(mag) & (mag >= min_offset) & use
    scale = np.where(valid, 1.0 / np.where(valid, mag, 1.0), 0.0)
    width = np.maximum(np.nan_to_num(mag) * np.tan(np.radians(band_angle)),
                       band_min_half_width)
    return np.stack([np.nan_to_num(az_off) * scale,
                     np.nan_to_num(rg_off) * scale, width], axis=-1)


def shear_dominant(vx, vy, geotransform, x, y, sigma, min_rate):
    '''
    Whether the lateral shear strain rate of the velocity grid vx, vy
    (Gaussian smoothed by `sigma` grid pixels) exceeds both normal strain
    rates in the flow frame and `min_rate` [1/yr] at map points x, y
    '''
    _, dx, _, _, _, dy = geotransform
    vx, vy = gaussian_filter(vx, sigma), gaussian_filter(vy, sigma)
    exx = np.gradient(vx, axis=1) / dx
    eyy = np.gradient(vy, axis=0) / dy
    exy = 0.5 * (np.gradient(vx, axis=0) / dy + np.gradient(vy, axis=1) / dx)
    th = np.arctan2(vy, vx)
    c, s = np.cos(th), np.sin(th)
    strain = [exx * c * c + eyy * s * s + 2 * exy * s * c,    # longitudinal
              exx * s * s + eyy * c * c - 2 * exy * s * c,    # transverse
              (eyy - exx) * s * c + exy * (c * c - s * s)]    # lateral shear
    e_long, e_trans, e_shear = (sample(np.abs(e), geotransform, x, y)
                                for e in strain)
    return (e_shear > np.maximum(e_long, e_trans)) & (e_shear > min_rate)


def time_interval_years(ref_grid, sec_grid):
    '''Secondary minus reference mid-scene time in years'''
    dt = (sec_grid.ref_epoch - ref_grid.ref_epoch).total_seconds() \
        + sec_grid.sensing_mid - ref_grid.sensing_mid
    return dt / SECONDS_PER_YEAR


def read_window(path, xmin, xmax, ymin, ymax, nodata=None):
    '''
    Read the part of a north-up raster covering [xmin, xmax] x [ymin, ymax].
    Returns float64 array (nodata -> NaN) and its geotransform.
    '''
    ds = gdal.Open(path)
    x0, dx, _, y0, _, dy = ds.GetGeoTransform()
    c0 = max(int(np.floor((xmin - x0) / dx)), 0)
    c1 = min(int(np.ceil((xmax - x0) / dx)) + 1, ds.RasterXSize)
    r0 = max(int(np.floor((ymax - y0) / dy)), 0)
    r1 = min(int(np.ceil((ymin - y0) / dy)) + 1, ds.RasterYSize)
    data = ds.GetRasterBand(1).ReadAsArray(c0, r0, c1 - c0, r1 - r0)
    data = data.astype(np.float64)
    if nodata is None:
        nodata = ds.GetRasterBand(1).GetNoDataValue()
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, (x0 + c0 * dx, dx, 0.0, y0 + r0 * dy, 0.0, dy)


def read_smoothed(path, xmin, xmax, ymin, ymax, spacing):
    '''
    Raster covering [xmin, xmax] x [ymin, ymax] block-averaged to `spacing`
    (map units) to keep only its large-scale trend
    '''
    ds = gdal.Warp('', path, format='MEM', outputBounds=(
        xmin - spacing, ymin - spacing, xmax + spacing, ymax + spacing),
                   xRes=spacing, yRes=spacing, resampleAlg='average')
    data = ds.ReadAsArray().astype(np.float64)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, ds.GetGeoTransform()


def fill_gaps(data, max_distance, smoothing_iterations):
    '''
    Fill NaN gaps by inverse distance interpolation (gdal.FillNodata) up to
    max_distance pixels from valid data; farther gaps are set to 0.
    '''
    mem = gdal.GetDriverByName('MEM').Create('', data.shape[1], data.shape[0],
                                             1, gdal.GDT_Float64)
    band = mem.GetRasterBand(1)
    band.SetNoDataValue(np.nan)
    band.WriteArray(data)
    gdal.FillNodata(band, None, max_distance, smoothing_iterations)
    return np.nan_to_num(band.ReadAsArray(), nan=0.0)


def sample(data, geotransform, x, y):
    '''Bilinear sampling of a north-up raster at map coordinates x, y'''
    x0, dx, _, y0, _, dy = geotransform
    # pixel-is-area: centers at x0 + (col + 0.5) * dx
    cols = (x - x0) / dx - 0.5
    rows = (y - y0) / dy - 0.5
    return map_coordinates(data, [rows, cols], order=1, mode='nearest')


def transform_xy(x, y, epsg_in, epsg_out):
    '''Transform map coordinates between EPSG codes'''
    if epsg_in == epsg_out:
        return x, y
    srs_in, srs_out = osr.SpatialReference(), osr.SpatialReference()
    for srs, epsg in ((srs_in, epsg_in), (srs_out, epsg_out)):
        srs.ImportFromEPSG(epsg)
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    pts = np.array(osr.CoordinateTransformation(srs_in, srs_out)
                   .TransformPoints(np.c_[x.ravel(), y.ravel()]))
    return pts[:, 0].reshape(x.shape), pts[:, 1].reshape(y.shape)


def write_envi(path, bands, dtype=gdal.GDT_Float32):
    '''Write a list of equally shaped 2D arrays as a multiband ENVI file'''
    length, width = bands[0].shape
    ds = gdal.GetDriverByName('ENVI').Create(str(path), width, length,
                                             len(bands), dtype)
    for i, band in enumerate(bands):
        ds.GetRasterBand(i + 1).WriteArray(band)
    ds.FlushCache()


def run_rdr2geo(radar_grid, orbit, ellipsoid, dem_raster, epsg, rdr2geo_cfg,
                outdir):
    '''Ground x, y (in epsg) and height of each radar grid point'''
    rdr2geo_obj = isce3.geometry.Rdr2Geo(
        radar_grid, orbit, ellipsoid, isce3.core.LUT2d(),
        threshold=rdr2geo_cfg['threshold'], numiter=rdr2geo_cfg['numiter'],
        extraiter=rdr2geo_cfg['extraiter'], epsg_out=epsg,
        lines_per_block=rdr2geo_cfg['lines_per_block'])
    rasters = [get_raster_obj(str(outdir / f'{name}.rdr'), radar_grid, True,
                              gdal.GDT_Float64) for name in 'xyz']
    rdr2geo_obj.topo(dem_raster, *rasters)
    del rasters
    return [gdal.Open(str(outdir / f'{name}.rdr')).ReadAsArray()
            for name in 'xyz']


def run_geo2rdr(radar_grid, orbit, ellipsoid, x, y, z, epsg, geo2rdr_cfg,
                outdir):
    '''
    Azimuth/range offsets (radar grid pixels) of points x, y, z (in epsg)
    from the radar grid point they are stored at; NaN where geo2rdr fails.
    '''
    for name, data in zip('xyz', (x, y, z)):
        write_envi(outdir / f'{name}_displaced.rdr', [data], gdal.GDT_Float64)
    rasters = [isce3.io.Raster(str(outdir / f'{name}_displaced.rdr'))
               for name in 'xyz']
    topo = isce3.io.Raster(str(outdir / 'topo_displaced.vrt'), rasters)
    topo.set_epsg(epsg)
    geo2rdr_obj = isce3.geometry.Geo2Rdr(
        radar_grid, orbit, ellipsoid, isce3.core.LUT2d(),
        geo2rdr_cfg['threshold'], geo2rdr_cfg['maxiter'],
        geo2rdr_cfg['lines_per_block'])
    geo2rdr_obj.geo2rdr(topo, str(outdir))
    del topo, rasters
    az, rg = [gdal.Open(str(outdir / f'{name}.off')).ReadAsArray()
              for name in ('azimuth', 'range')]
    # geo2rdr marks failures with a large negative NULL_VALUE
    invalid = (az < -1e5) | (rg < -1e5)
    az[invalid] = np.nan
    rg[invalid] = np.nan
    return az, rg


def velocity_offsets(cfg, freq, outdir):
    '''
    Predicted azimuth/range offsets [reference RSLC pixels] of the offsets
    product grid of frequency `freq`, and the velocity sampled on that grid
    '''
    info = journal.info('velocity_offsets.velocity_offsets')
    vel_cfg = cfg['processing']['velocity_gross_offset']
    proc_cfg = cfg['processing']

    ref_slc = SLC(hdf5file=cfg['input_file_group']['reference_rslc_file'])
    sec_slc = SLC(hdf5file=cfg['input_file_group']['secondary_rslc_file'])
    ref_grid = ref_slc.getRadarGrid(freq)
    dt = time_interval_years(ref_grid, sec_slc.getRadarGrid(freq))
    info.log(f'time interval: {dt * 365.25:.4f} days')

    orbit = ref_slc.getOrbit()
    orbit_file = cfg['dynamic_ancillary_file_group']['orbit_files'][
        'reference_orbit_file']
    if orbit_file is not None:
        orbit = crop_external_orbit(
            load_orbit_from_xml(orbit_file, ref_grid.ref_epoch), orbit)

    dem_file = cfg['dynamic_ancillary_file_group']['dem_file']
    dem_raster = isce3.io.Raster(dem_file)
    dem_epsg = dem_raster.get_epsg()
    ellipsoid = isce3.core.make_projection(dem_epsg).ellipsoid

    off_grid = get_offset_radar_grid(cfg, ref_grid)
    info.log(f'offsets grid: {off_grid.length} x {off_grid.width}')

    # 1. ground positions in the velocity map projection
    vel_epsg = vel_cfg['epsg']
    x, y, z = run_rdr2geo(off_grid, orbit, ellipsoid, dem_raster, vel_epsg,
                          proc_cfg['rdr2geo'], outdir)

    # 2. velocity at P (gap filled) and displaced position P'
    m = vel_cfg['read_margin']
    bbox = (np.nanmin(x) - m, np.nanmax(x) + m, np.nanmin(y) - m,
            np.nanmax(y) + m)
    grids = []
    for key in ('vx', 'vy'):
        data, gt = read_window(vel_cfg[key], *bbox, nodata=vel_cfg['nodata'])
        info.log(f'{key}: {np.isnan(data).mean() * 100:.1f}% gaps')
        grids.append(fill_gaps(data, vel_cfg['fill_max_distance'],
                               vel_cfg['fill_smoothing_iterations']))
    vx, vy = (sample(data, gt, x, y) for data in grids)
    shear = shear_dominant(*grids, gt, x, y, vel_cfg['dlc_strain_smoothing'],
                           vel_cfg['dlc_min_shear_strain_rate'])
    info.log(f'shear-dominant flow: {shear.mean() * 100:.1f}% of windows')
    x_disp, y_disp = x + vx * dt, y + vy * dt

    # h' = h + S(P') - S(P), S smoothed DEM in the DEM projection
    xs, ys = transform_xy(np.stack([x, x_disp]), np.stack([y, y_disp]),
                          vel_epsg, dem_epsg)
    dem, dem_gt = read_smoothed(dem_file, xs.min(), xs.max(), ys.min(),
                                ys.max(), vel_cfg['dem_smoothing'])
    s, s_disp = sample(dem, dem_gt, xs, ys)
    z_disp = z + np.nan_to_num(s_disp - s)

    # 3. radar offsets of P' w.r.t. P, converted to RSLC pixels
    az_off, rg_off = run_geo2rdr(off_grid, orbit, ellipsoid, x_disp, y_disp,
                                 z_disp, vel_epsg, proc_cfg['geo2rdr'],
                                 outdir)
    az_off *= ref_grid.prf / off_grid.prf
    rg_off *= off_grid.range_pixel_spacing / ref_grid.range_pixel_spacing
    return az_off, rg_off, vx, vy, shear


def run(cfg: dict):
    '''Compute velocity-based gross offsets for each processed frequency'''
    info = journal.info('velocity_offsets.run')
    info.log('Start velocity-based gross offsets')
    t_all = time.time()
    scratch_path = cfg['product_path_group']['scratch_path']

    for freq in dict.fromkeys(f for f, _, _ in get_cfg_freq_pols(cfg)):
        out_path = gross_offset_path(scratch_path, freq)
        outdir = out_path.parent
        outdir.mkdir(parents=True, exist_ok=True)

        az_off, rg_off, vx, vy, shear = velocity_offsets(cfg, freq, outdir)
        write_envi(outdir / 'velocity_offsets', [az_off, rg_off])
        write_envi(outdir / 'velocity', [vx, vy])
        gross = np.stack([np.nan_to_num(az_off), np.nan_to_num(rg_off)],
                         axis=-1)
        np.rint(gross).astype(np.int32).tofile(out_path)
        vel_cfg = cfg['processing']['velocity_gross_offset']
        use = shear if vel_cfg['dlc_shear_dominant_only'] else True
        flow_direction(az_off, rg_off, vel_cfg['dlc_min_offset'], use,
                       vel_cfg['dlc_band_angle'],
                       vel_cfg['dlc_band_min_half_width']).astype(
            np.float32).tofile(flow_direction_path(scratch_path, freq))

        info.log(f'freq{freq} azimuth offsets [px] min/max: '
                 f'{np.nanmin(az_off):.1f} {np.nanmax(az_off):.1f}; range: '
                 f'{np.nanmin(rg_off):.1f} {np.nanmax(rg_off):.1f}')

    # message names the step for Persistence restarts
    info.log(f'Successfully ran velocity_offsets in '
             f'{time.time() - t_all:.3f} seconds')


if __name__ == '__main__':
    from nisar.workflows.insar_runconfig import InsarRunConfig
    run(InsarRunConfig(YamlArgparse().parse()).cfg)
