'''
A priori unwrapped phase from slant range offsets, used to deramp the
wrapped interferogram before phase unwrapping (see unwrap.run apriori_phase)

Fast motion (e.g. glacier flow) produces fringes too dense to unwrap. The
interferometric phase predicted from slant range offsets, 4 pi f / c * offset
(interferogram = reference * conj(secondary), offset = secondary - reference),
is removed before unwrapping and added back afterwards.
'''
import pathlib

import isce3
import numpy as np
from isce3.io import HDF5OptimizedReader
from nisar.products.insar.product_paths import RIFGGroupsPaths, RUNWGroupsPaths
from nisar.products.utils import interpret_subswath_mask
from nisar.workflows.helpers import get_cfg_freq_pols
from osgeo import gdal
from scipy import ndimage


def _product_paths(h5):
    '''RIFG or RUNW group paths of an open L1 InSAR product'''
    for paths in (RUNWGroupsPaths(), RIFGGroupsPaths()):
        if paths.SwathsPath in h5:
            return paths
    raise ValueError(f'{h5.filename} is not an RIFG or RUNW product')


def read_slant_range_offsets(h5, freq, pol, velocity_offsets_dir=None):
    '''
    Slant range offsets of an RIFG or RUNW product, or the velocity-map
    prediction of the velocity_offsets step on the same offsets grid

    Parameters
    ----------
    h5: h5py.File
        Open RIFG or RUNW product
    freq: str
        Frequency; frequency A offsets are used if the product has no pixel
        offsets for `freq`
    pol: str
        Polarization; the first available offset polarization is used if
        `pol` has no offsets
    velocity_offsets_dir: str or pathlib.Path, optional
        velocity_offsets scratch directory (containing freq{A,B}); if given,
        its predicted range offsets [reference RSLC pixels] are used instead
        of the product offsets

    Returns
    -------
    offsets: np.ndarray
        Slant range offsets (meters); zero, non-finite and subswath-invalid
        samples set to NaN
    zero_doppler_time: np.ndarray
        Zero-Doppler times of the offsets grid rows
    slant_range: np.ndarray
        Slant ranges of the offsets grid columns
    '''
    paths = _product_paths(h5)
    off_freq = freq if f'{paths.SwathsPath}/frequency{freq}/pixelOffsets' in h5 else 'A'
    group = h5[f'{paths.SwathsPath}/frequency{off_freq}/pixelOffsets']
    off_pols = [p for p in group if f'{p}/slantRangeOffset' in group]
    if not off_pols:
        raise ValueError(f'No slant range offsets in {group.name}')
    off_pol = pol if pol in off_pols else off_pols[0]

    if velocity_offsets_dir is None:
        offsets = group[f'{off_pol}/slantRangeOffset'][()].astype(np.float64)
        invalid = (offsets == 0) | ~np.isfinite(offsets)
        if 'mask' in group:
            reference_valid, secondary_valid, _ = \
                interpret_subswath_mask(group['mask'][()])
            invalid |= ~reference_valid | ~secondary_valid
    else:
        path = pathlib.Path(velocity_offsets_dir) / f'freq{off_freq}' / 'velocity_offsets'
        ds = gdal.Open(str(path))
        offsets = ds.GetRasterBand(2).ReadAsArray().astype(np.float64)
        del ds
        if offsets.shape != group[f'{off_pol}/slantRangeOffset'].shape:
            raise ValueError(f'{path} does not match the offsets grid of {group.name}')
        offsets *= h5[f'{paths.ProcessingInformationPath}/parameters/reference/'
                      f'frequency{off_freq}/slantRangeSpacing'][()]
        invalid = ~np.isfinite(offsets)
    if invalid.all():
        raise ValueError(f'No valid slant range offsets in {group.name}')
    offsets[invalid] = np.nan
    return offsets, group['zeroDopplerTime'][()], group['slantRange'][()]


def offsets_to_phase(offsets, zero_doppler_time, slant_range, out_time,
                     out_range, center_frequency, smoothing_sigma=0.0):
    '''
    Interferometric phase of slant range offsets on another radar grid

    The offsets are smoothed with a NaN-aware Gaussian filter, gap-filled
    with the nearest valid value and bilinearly interpolated to
    (out_time, out_range).

    Parameters
    ----------
    offsets: np.ndarray
        Slant range offsets (meters), NaN where invalid
    zero_doppler_time, slant_range: np.ndarray
        Row times and column ranges of `offsets`
    out_time, out_range: np.ndarray
        Row times and column ranges of the output grid
    center_frequency: float
        Center frequency (Hz); the difference of the center frequencies for
        a differential (first * conj(second)) interferogram
    smoothing_sigma: float
        Gaussian standard deviation in offset-grid pixels (0: no smoothing)

    Returns
    -------
    phase: np.ndarray
        4 pi f / c * offset (radians), float32, shape (out_time, out_range)
    '''
    offsets = offsets.copy()
    if smoothing_sigma > 0:
        valid = np.isfinite(offsets)
        weights = ndimage.gaussian_filter(valid.astype(np.float64), smoothing_sigma)
        smoothed = ndimage.gaussian_filter(np.where(valid, offsets, 0.0), smoothing_sigma)
        with np.errstate(invalid='ignore', divide='ignore'):
            offsets = np.where(weights > 1e-3, smoothed / weights, np.nan)
    nearest = ndimage.distance_transform_edt(np.isnan(offsets), return_distances=False,
                                             return_indices=True)
    offsets = offsets[tuple(nearest)]

    cols = (out_range - slant_range[0]) / (slant_range[1] - slant_range[0])
    phase = np.empty((out_time.size, out_range.size), dtype=np.float32)
    scale = 4.0 * np.pi * center_frequency / isce3.core.speed_of_light
    lines_per_block = 1024
    for row_start in range(0, out_time.size, lines_per_block):
        block_time = out_time[row_start:row_start + lines_per_block]
        rows = (block_time - zero_doppler_time[0]) / (zero_doppler_time[1] - zero_doppler_time[0])
        row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
        phase[row_start:row_start + block_time.size] = scale * ndimage.map_coordinates(
            offsets, [row_grid, col_grid], order=1, mode='nearest')
    return phase


def deramp_phase_function(cfg, offsets_hdf5, runw_hdf5, center_frequency=None,
                          velocity_offsets_dir=None):
    '''
    A priori phase function for unwrap.run when
    processing.phase_unwrap.deramp is enabled

    Offsets, grids and center frequencies are read here, before unwrap.run
    opens the products.

    Parameters
    ----------
    cfg: dict
        InSAR runconfig
    offsets_hdf5: str
        RIFG or RUNW product with the (rubbersheet) slant range offsets
    runw_hdf5: str
        RUNW product to write; defines the unwrapping grid and the center
        frequency
    center_frequency: float, optional
        Center frequency (Hz) overriding that of `runw_hdf5`; the difference
        of the center frequencies for differential interferograms
    velocity_offsets_dir: str, optional
        velocity_offsets scratch directory for offset_source 'velocity';
        defaults to <scratch_path>/velocity_offsets

    Returns
    -------
    apriori_phase: callable or None
        Function (freq, pol) -> phase (radians) on the unwrapping grid, or
        None when deramping is disabled
    '''
    deramp_cfg = cfg['processing']['phase_unwrap']['deramp']
    if not deramp_cfg['enabled']:
        return None
    if deramp_cfg['offset_source'] == 'velocity':
        if velocity_offsets_dir is None:
            velocity_offsets_dir = pathlib.Path(
                cfg['product_path_group']['scratch_path']) / 'velocity_offsets'
    else:
        velocity_offsets_dir = None

    freq_pols = [(freq, pol) for freq, pols, _ in get_cfg_freq_pols(cfg) for pol in pols]
    grids, offsets = {}, {}
    with HDF5OptimizedReader(name=runw_hdf5, mode='r', libver='latest', swmr=True) as h5:
        swaths = RUNWGroupsPaths().SwathsPath
        for freq, _ in freq_pols:
            group = h5[f'{swaths}/frequency{freq}']
            fc = group['centerFrequency'][()] if center_frequency is None else center_frequency
            grids[freq] = (group['interferogram/zeroDopplerTime'][()],
                           group['interferogram/slantRange'][()], fc)
    with HDF5OptimizedReader(name=offsets_hdf5, mode='r', libver='latest', swmr=True) as h5:
        for freq, pol in freq_pols:
            offsets[freq, pol] = read_slant_range_offsets(h5, freq, pol, velocity_offsets_dir)

    def apriori_phase(freq, pol):
        out_time, out_range, fc = grids[freq]
        return offsets_to_phase(*offsets[freq, pol], out_time, out_range, fc,
                                deramp_cfg['offset_smoothing_sigma'])

    return apriori_phase
