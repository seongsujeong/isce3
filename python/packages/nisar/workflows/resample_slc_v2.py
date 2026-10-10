from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

import journal
import numpy as np
from journal.Channel import Channel

from isce3.core.gpu_check import use_gpu
from isce3.core.types import ComplexFloat16Decoder, is_complex32
from isce3.image.v2.resample_slc import resample_slc_blocks
from isce3.io import HDF5OptimizedReader
from isce3.io.gdal.gdal_raster import GDALRaster

from nisar.products.readers import RSLC
from nisar.workflows.helpers import ParallelChunkReader
from nisar.workflows.rubbersheet import open_resample_offsets
from nisar.workflows.resample_slc_runconfig import ResampleSlcRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse


def run(cfg: dict, resample_type: str) -> None:
    """
    Resample a secondary RSLC product onto the reference RSLC grid.

    Parameters
    ----------
    cfg : dict
        The Resample SLC Runconfig dict for this run.
    resample_type : "coarse" or "fine"
        The type of offsets file to use.
        "coarse" uses ".off" files in the geo2rdr subdirectory of the offsets directory
        of the runconfig.
        "fine" uses ".off.vrt" files under the rubbersheet_offsets subdirectory of the
        offsets directory of the runconfig.
    """
    sec_file_path = cfg["input_file_group"]["secondary_rslc_file"]
    ref_file_path = cfg["input_file_group"]["reference_rslc_file"]
    scratch_path = Path(cfg["product_path_group"]["scratch_path"])
    freq_pols: dict[str, list[str]] = \
        cfg["processing"]["input_subset"]["list_of_frequencies"]

    # According to the type of resampling, choose proper resample cfg
    resamp_args = cfg["processing"][f"{resample_type}_resample"]

    # Determine if the user requested to use the GPU.
    with_gpu = use_gpu(cfg['worker']['gpu_enabled'], cfg['worker']['gpu_id'])
    # or a Metal GPU (Apple), if available
    with_metal = not with_gpu and cfg['worker'].get('metal_enabled', False)

    # Python sees journal.info as returning Any.
    # It is a subclasse of Channel and being used as such, so this type hint
    # helps with proper type checking despite being a statement of the obvious.
    info_channel: Channel = journal.info("resample_slc_v2.run")
    info_channel.log("starting resampling SLC")

    t_all = perf_counter()

    for freq in freq_pols.keys():

        # Open offsets
        offsets_dir = Path(resamp_args["offsets_dir"])

        if resample_type == "coarse":
            offsets_path = offsets_dir / f"geo2rdr/freq{freq}"
        elif resample_type == "fine":
            # We checked the existence of HH/VV offsets in resample_slc_runconfig.py
            # Select the first offsets available between HH and VV
            freq_offsets_path = \
                offsets_dir / f"rubbersheet_offsets/freq{freq}"
            if os.path.isdir(freq_offsets_path / "HH"):
                offsets_path = freq_offsets_path / "HH"
            else:
                offsets_path = freq_offsets_path / "VV"
        else:
            raise ValueError(
                "resample_type must be 'coarse' or 'fine', instead got "
                f"{resample_type!r}")

        az_off_path = offsets_path / "azimuth.off"
        rg_off_path = offsets_path / "range.off"

        # Create separate directories for coarse and fine resample
        # Open corresponding range/azimuth offsets
        resample_slc_scratch_path = (
            scratch_path / f"{resample_type}_resample_slc" / f"freq{freq}"
        )

        # Create resample slc directory
        resample_slc_scratch_path.mkdir(parents=True, exist_ok=True)

        block_length = int(resamp_args["lines_per_tile"])
        block_width = int(resamp_args["columns_per_tile"])

        # Get polarization list for which resample SLCs
        pol_list = freq_pols[freq]

        info_channel.log(f"Resampling SLC for frequency {freq}.")
        t_freq_elapsed = -perf_counter()

        resample_secondary_rslc_onto_reference(
            ref_file_path=ref_file_path,
            sec_file_path=sec_file_path,
            out_path=resample_slc_scratch_path,
            az_off_file=az_off_path,
            rg_off_file=rg_off_path,
            freq=freq,
            pols=pol_list,
            block_size_az=block_length,
            block_size_rg=block_width,
            with_gpu=with_gpu,
            with_metal=with_metal,
        )

        t_freq_elapsed += perf_counter()
        info_channel.log(f"successfully ran resample for frequency {freq} in "
                         f"{t_freq_elapsed:.3f} seconds")

    t_all_elapsed = perf_counter() - t_all
    info_channel.log(f"successfully ran resample in {t_all_elapsed:.3f} seconds")


def align_tile_to_chunks(lines, columns, chunks, width):
    """
    Round a tile size to a nonzero multiple of the HDF5 chunk shape of the
    RSLC it is read from; columns equal to the full width stay so.

    Parameters
    ----------
    lines, columns : int
        Requested tile size
    chunks : tuple of int or None
        Chunk shape (lines, columns) of the RSLC dataset; None (contiguous
        dataset) keeps the requested size
    width : int
        Number of columns of the output (full-width tiles)

    Returns
    -------
    tuple of int
        Tile (lines, columns)
    """
    if chunks is None:
        return lines, columns
    lines = max(1, round(lines / chunks[0])) * chunks[0]
    if columns < width:
        columns = min(width, max(1, round(columns / chunks[1])) * chunks[1])
    return lines, columns


def resample_secondary_rslc_onto_reference(
    ref_file_path: str | os.PathLike,
    sec_file_path: str | os.PathLike,
    out_path: str | os.PathLike,
    az_off_file: str | os.PathLike,
    rg_off_file: str | os.PathLike,
    freq: str,
    pols: Iterable[str],
    block_size_az: int,
    block_size_rg: int,
    with_gpu: bool = False,
    with_metal: bool = False,
) -> None:
    """
    Resample a secondary RSLC product onto a reference one using NISAR HDF5 datasets.

    This function outputs several files named `coregistered_secondary.slc` into
    subdirectories at `out_path`, one for each polarization on this frequency.

    Parameters
    ----------
    ref_file : path-like
        The reference HDF5 RSLC file path.
    sec_file : path-like
        The secondary HDF5 RSLC file path.
    out_path : path-like
        The path to the root directory into which the output SLC files will be placed.
    az_off_file : path-like
        The azimuth offsets GDAL-readable file path.
    rg_off_file : path-like
        The range offsets GDAL-readable file path.
    freq : str
        The frequency letter.
    pols : Iterable[str]
        The set of polarizations to be resampled.
    block_size_az : int
        The block size along the azimuth axis. Must be greater than 0.
    block_size_rg : int
        The block size along the range axis. Must be 0 or greater. If 0, the block size
        in the range dimension will be the entire line of range values.
    with_gpu : bool, optional
        If True, run the GPU resample workflow. If False, run the CPU resample workflow.
        Defaults to False.
    with_metal : bool, optional
        If True and `with_gpu` is False, interpolate on a Metal GPU (Apple) where
        available. Defaults to False.
    """
    sec_slc_obj = RSLC(hdf5file=os.fspath(sec_file_path))
    sec_grid = sec_slc_obj.getRadarGrid(freq)
    doppler = sec_slc_obj.getDopplerCentroid(frequency=freq)

    ref_radar_grid = RSLC(hdf5file=os.fspath(ref_file_path)).getRadarGrid(freq)

    # Get dimensions of sec grid
    out_length = ref_radar_grid.length
    out_width = ref_radar_grid.width

    out_shape = (out_length, out_width)

    if block_size_rg == 0:
        block_size_rg = out_width

    # Initialize the data reader objects.
    # First, initialize the azimuth and range offset readers: memory-mapped
    # azimuth.off/range.off, or rubbersheet offsets computed on read from the
    # offsets grid when rubbersheet did not write them at full resolution
    if Path(az_off_file).parent != Path(rg_off_file).parent:
        raise ValueError("azimuth and range offsets must share a directory")
    az_off_reader, rg_off_reader = open_resample_offsets(
        Path(az_off_file).parent, out_shape)

    # For each polarization being output, create a GDALRaster to write to it.
    out_writers: list[GDALRaster] = []
    for pol in pols:
        writer_dir = Path(out_path) / pol
        writer_dir.mkdir(parents=True, exist_ok=True)

        writer_path = writer_dir / "coregistered_secondary.slc"
        out_writers.append(
            GDALRaster.create_dataset_file(
                filepath=Path(writer_path),
                dtype=np.complex64,
                shape=(out_length, out_width),
                num_bands=1,
                driver_name="ENVI",
            )
        )

    # ComplexFloat16Decoders and h5py Datasets implement the ISCE3 DatasetReader
    # protocol. Get a list of these for all polarizations in this frequency.
    #
    # Open directly with a cache sized for this dataset's chunking, rather
    # than via RSLC.getSlcDatasetAsNativeComplex()'s default h5py cache.
    with HDF5OptimizedReader(
        name=os.fspath(sec_file_path), mode='r', libver='latest', swmr=True
    ) as sec_h5:
        sec_readers = []
        for pol in pols:
            ds_path = sec_slc_obj.imageDatasetPath(freq, pol)
            dataset = sec_h5[ds_path]
            chunks = dataset.chunks
            if is_complex32(dataset):
                dataset = ComplexFloat16Decoder(dataset)
            else:
                # chunks decoded in parallel threads (h5py decodes serially)
                dataset = ParallelChunkReader(dataset)
            sec_readers.append(dataset)

        # Tiles of whole secondary RSLC chunks: each chunk is then
        # decompressed once, and full-width tiles also read the offsets and
        # write the output sequentially
        block_size_az, block_size_rg = align_tile_to_chunks(
            block_size_az, block_size_rg, chunks, out_width)
        journal.info("resample_slc_v2.resample_secondary_rslc_onto_reference"
                     ).log(f"tile size (lines x columns) aligned to the RSLC "
                           f"chunks {chunks}: {block_size_az} x {block_size_rg}")

        # Resample the secondary RSLC onto the reference coordinate system.
        # Because this function receives GDALRasters in its output, it will write
        # automatically to these rasters.
        resample_slc_blocks(
            output_resampled_slcs=out_writers,
            input_slcs=sec_readers,
            az_offsets_dataset=az_off_reader,
            rg_offsets_dataset=rg_off_reader,
            input_radar_grid=sec_grid,
            doppler=doppler,
            block_size_az=block_size_az,
            block_size_rg=block_size_rg,
            fill_value=0.0 + 0.0j,
            with_gpu=with_gpu,
            with_metal=with_metal,
        )


if __name__ == "__main__":
    """
    run resample_slc from command line
    """

    # load command line args
    resample_slc_parser = YamlArgparse(resample_type=True)
    args = resample_slc_parser.parse()

    # Extract resample_type
    resample_type = args.resample_type

    # Get a runconfig dictionary from command line args
    resample_slc_runconfig = ResampleSlcRunConfig(args, resample_type)

    # Run resample_slc
    run(resample_slc_runconfig.cfg, resample_type)
