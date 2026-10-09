#!/usr/bin/env python3
'''
unit tests for CPU pybind Crossmul
'''

import os

import numpy as np
import numpy.testing as npt

from osgeo import gdal

import iscetest
import isce3.ext.isce3 as isce3
from nisar.products.readers import SLC


def common_crossmul_obj():
    '''
    instantiate and return common crossmul object for both run tests
    '''
    # make SLC object and extract parameters
    slc_obj = SLC(hdf5file=os.path.join(iscetest.data, 'envisat.h5'))
    dopp = isce3.core.avg_lut2d_to_lut1d(slc_obj.getDopplerCentroid())
    prf = slc_obj.getRadarGrid().prf

    crossmul = isce3.signal.Crossmul()
    crossmul.set_dopplers(dopp, dopp)

    return crossmul


def test_run_no_filter():
    '''
    run pybind CPU crossmul module without azimuth filtering
    '''
    ref_slc_raster = isce3.io.Raster(os.path.join(iscetest.data, 'warped_envisat.slc.vrt'))

    crossmul = common_crossmul_obj()

    # prepare output rasters
    width = ref_slc_raster.width
    length = ref_slc_raster.length
    igram = isce3.io.Raster(
        'igram.int', width, length, 1, gdal.GDT_CFloat32, "ISCE")
    coherence = isce3.io.Raster(
        'coherence.bin', width, length, 1, gdal.GDT_Float32, "ISCE")

    crossmul.crossmul(ref_slc_raster, ref_slc_raster, igram, coherence)


def test_validate_no_filter():
    '''
    make sure pybind CPU crossmul results have zero phase
    '''
    # convert complex test data to angle
    data = np.angle(np.fromfile('igram.int', dtype=np.complex64))

    # check if interferometric phase is very small (should be zero)
    npt.assert_array_less(data, 1.0e-6)


def test_two_looks():
    '''
    outputs of crossmul_two_looks match two crossmul runs with each number
    of looks (with oversampling and coprime numbers of azimuth looks)
    '''
    ref = isce3.io.Raster(os.path.join(iscetest.data, 'warped_envisat.slc.vrt'))
    sec = ref
    looks = [(3, 5), (2, 7)]

    def run(tag, *looks2):
        crossmul = common_crossmul_obj()
        crossmul.range_looks, crossmul.az_looks = looks[0]
        crossmul.oversample_factor = 2
        crossmul.lines_per_block = 64
        paths, rasters = [], []
        for k, (rg, az) in enumerate(looks[:1 + bool(looks2)]):
            for kind, dtype in (('igram', gdal.GDT_CFloat32),
                                ('coherence', gdal.GDT_Float32)):
                paths.append(f'{kind}_{tag}{k}.bin')
                rasters.append(isce3.io.Raster(paths[-1], ref.width // rg,
                    ref.length // az, 1, dtype, 'ENVI'))
        if looks2:
            crossmul.crossmul_two_looks(ref, sec, *rasters, *looks2)
        else:
            crossmul.crossmul(ref, sec, *rasters)
        del rasters
        return [gdal.Open(p).ReadAsArray() for p in paths]

    fused = run('fused', *looks[1])
    single = run('single')
    looks[0] = looks[1]
    single += run('single2')
    for a, b in zip(fused, single):
        npt.assert_array_equal(a, b)


if __name__ == '__main__':
    test_run_no_filter()
    test_validate_no_filter()
    test_two_looks()
