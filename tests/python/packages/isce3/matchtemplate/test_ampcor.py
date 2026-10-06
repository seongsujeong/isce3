from osgeo import gdal
import isce3
import itertools
import iscetest
import numpy
import os


def create_empty_dataset(
    filename, width, length, bands, dtype, interleave="bip", file_type="ENVI"
):
    """
    Create empty dataset with user-defined options
    """
    driver = gdal.GetDriverByName(file_type)
    driver.Create(
        filename,
        xsize=width,
        ysize=length,
        bands=bands,
        eType=dtype,
        options=[f"INTERLEAVE={interleave}"],
    )


def test_ampcor():
    try:
        impls = (
            isce3.cuda.matchtemplate.PyCuAmpcor,
            isce3.matchtemplate.PyCPUAmpcor,
        )
    except AttributeError:
        # Fall back to CPU only if not compiled with CUDA support
        impls = (isce3.matchtemplate.PyCPUAmpcor,)
    for impl in impls:
        # DLC peak search and Metal steps (CPU ampcor only) must find the
        # same unique peaks
        cpu = impl is isce3.matchtemplate.PyCPUAmpcor
        dlcs = (False, True) if cpu else (False,)
        metals = (False, True) if cpu and \
            isce3.matchtemplate.metal_available() else (False,)
        # test FFT and sinc oversamplers
        for ovs, dlc, metal in itertools.product((0, 1), dlcs, metals):
            ampcor = impl()

            ampcor.useMmap = 1
            if metal:
                ampcor.useMetal = 1

            datadir = os.path.join(
                iscetest.data, "ampcor", "accuracy-testdata", "ovs128-rho0.8"
            )

            ref = os.path.join(datadir, "img1_WN_512x512_1x1_128")
            ref_raster = isce3.io.Raster(ref)
            width = ref_raster.width
            length = ref_raster.length
            ampcor.referenceImageName = ref
            ampcor.referenceImageWidth = width
            ampcor.referenceImageHeight = length

            sec = os.path.join(datadir, "img2_WN_512x512_1x1_128")
            sec_raster = isce3.io.Raster(sec)
            ampcor.secondaryImageName = sec
            assert width == sec_raster.width
            assert length == sec_raster.length
            ampcor.secondaryImageWidth = width
            ampcor.secondaryImageHeight = length

            ampcor.windowSizeWidth = 64
            ampcor.windowSizeHeight = 32
            ampcor.halfSearchRangeAcross = 20
            ampcor.halfSearchRangeDown = 20
            ampcor.skipSampleAcross = 32
            ampcor.skipSampleDown = 32

            margin = 0
            margin_rg = (
                2 * margin + 2 * ampcor.halfSearchRangeAcross + ampcor.windowSizeWidth
            )
            margin_az = (
                2 * margin + 2 * ampcor.halfSearchRangeDown + ampcor.windowSizeHeight
            )

            offset_width = (width - margin_rg) // ampcor.skipSampleAcross
            ampcor.numberWindowAcross = offset_width
            offset_length = (length - margin_az) // ampcor.skipSampleDown
            ampcor.numberWindowDown = offset_length

            ampcor.referenceStartPixelAcrossStatic = margin + ampcor.halfSearchRangeAcross
            ampcor.referenceStartPixelDownStatic = margin + ampcor.halfSearchRangeDown

            ampcor.algorithm = 0  # frequency
            ampcor.corrSurfaceOverSamplingMethod = ovs
            ampcor.derampMethod = 1
            ampcor.derampAxis = 0

            ampcor.corrStatWindowSize = 21
            ampcor.corrSurfaceZoomInWindow = 8

            ampcor.offsetImageName = "dense_offsets"
            ampcor.grossOffsetImageName = "gross_offset"
            ampcor.snrImageName = "snr"
            ampcor.covImageName = "covariance"
            ampcor.corrImageName = "correlation_peak"

            ampcor.rawDataOversamplingFactor = 2
            ampcor.corrSurfaceOverSamplingFactor = 64

            ampcor.numberWindowAcrossInChunk = 2
            ampcor.numberWindowDownInChunk = 1

            ampcor.setupParams()
            ampcor.setConstantGrossOffset(0, 0)
            if dlc:
                n = ampcor.numberWindowDown * ampcor.numberWindowAcross
                ampcor.setFlowDirection([0.6] * n, [0.8] * n)

            ampcor.checkPixelInImageRange()
            create_empty_dataset(
                "dense_offsets",
                ampcor.numberWindowAcross,
                ampcor.numberWindowDown,
                2,
                gdal.GDT_Float32,
            )
            create_empty_dataset(
                "gross_offsets",
                ampcor.numberWindowAcross,
                ampcor.numberWindowDown,
                2,
                gdal.GDT_Float32,
            )
            create_empty_dataset(
                "snr",
                ampcor.numberWindowAcross,
                ampcor.numberWindowDown,
                1,
                gdal.GDT_Float32,
            )
            create_empty_dataset(
                "covariance",
                ampcor.numberWindowAcross,
                ampcor.numberWindowDown,
                3,
                gdal.GDT_Float32,
            )
            create_empty_dataset(
                "correlation_peak",
                ampcor.numberWindowAcross,
                ampcor.numberWindowDown,
                1,
                gdal.GDT_Float32,
            )
            ampcor.runAmpcor()

            # Compare results to golden output
            for fname in (
                "covariance",
                "dense_offsets",
                "gross_offsets",
                "snr",
                "correlation_peak",
            ):
                print("comparing", fname)
                golden_path = os.path.join(datadir, "golden", fname)
                expected = numpy.fromfile(golden_path, dtype=numpy.float32)
                got = numpy.fromfile(fname, dtype=numpy.float32)

                assert len(got) == len(expected)

                if fname == "dense_offsets":
                    meantol = 2e-2
                    tol = 1e-1
                elif fname == "correlation_peak":
                    meantol = 2e-2
                    tol = 5e-2
                else:
                    meantol = 1 / 64 / 5
                    tol = 1 / 64

                for i in range(len(got)):
                    if abs(got[i] - expected[i]) > tol:
                        print(
                            "got",
                            got[i],
                            "but expected",
                            expected[i],
                            "diff is",
                            abs(got[i] - expected[i]),
                        )
                        print("at index", i)
                        assert False

                meandiff = numpy.mean(abs(got - expected))
                print("meandiff", meandiff)
                assert meandiff < meantol


def run_cpu_ampcor(ref, sec, size, direction=None):
    '''CPU ampcor of two square complex rasters; returns (down, across) offsets'''
    ampcor = isce3.matchtemplate.PyCPUAmpcor()
    ampcor.useMmap = 1
    ampcor.referenceImageName, ampcor.secondaryImageName = ref, sec
    ampcor.referenceImageWidth = ampcor.referenceImageHeight = size
    ampcor.secondaryImageWidth = ampcor.secondaryImageHeight = size
    ampcor.windowSizeWidth = ampcor.windowSizeHeight = 32
    ampcor.halfSearchRangeAcross = ampcor.halfSearchRangeDown = 16
    ampcor.skipSampleAcross = ampcor.skipSampleDown = 32
    ampcor.numberWindowAcross = ampcor.numberWindowDown = (size - 64) // 32
    ampcor.referenceStartPixelAcrossStatic = 16
    ampcor.referenceStartPixelDownStatic = 16
    ampcor.offsetImageName = "dlc_offsets"
    ampcor.grossOffsetImageName = "dlc_gross_offset"
    ampcor.snrImageName = "dlc_snr"
    ampcor.covImageName = "dlc_covariance"
    ampcor.corrImageName = "dlc_correlation_peak"
    ampcor.setupParams()
    ampcor.setConstantGrossOffset(0, 0)
    n = ampcor.numberWindowDown * ampcor.numberWindowAcross
    if direction is not None:
        ampcor.setFlowDirection([direction[0]] * n, [direction[1]] * n)
    for name, bands in (("dlc_offsets", 2), ("dlc_gross_offset", 2),
                        ("dlc_snr", 1), ("dlc_covariance", 3),
                        ("dlc_correlation_peak", 1)):
        create_empty_dataset(name, ampcor.numberWindowAcross,
                             ampcor.numberWindowDown, bands, gdal.GDT_Float32)
    ampcor.runAmpcor()
    return numpy.fromfile("dlc_offsets", dtype=numpy.float32).reshape(n, 2)


def test_ampcor_dlc():
    '''
    DLC peak search keeps the peak along the flow direction: the secondary
    holds a weaker copy of the reference shifted along the flow direction
    (6 lines down) and a stronger decoy shifted across it (10 columns).
    '''
    size = 256
    rng = numpy.random.default_rng(0)
    ref = (rng.normal(size=(size, size)) +
           1j * rng.normal(size=(size, size))).astype(numpy.complex64)
    sec = 0.7 * numpy.roll(ref, 6, axis=0) + numpy.roll(ref, 10, axis=1)
    for name, data in (("dlc_ref", ref), ("dlc_sec", sec)):
        ds = gdal.GetDriverByName("ENVI").Create(name, size, size, 1,
                                                 gdal.GDT_CFloat32)
        ds.GetRasterBand(1).WriteArray(data.astype(numpy.complex64))
        ds = None

    # global maximum: the stronger decoy across the flow
    off = run_cpu_ampcor("dlc_ref", "dlc_sec", size)
    assert numpy.allclose(numpy.median(off, axis=0), [0, 10], atol=0.1)
    # DLC along the down direction: the true shift
    off = run_cpu_ampcor("dlc_ref", "dlc_sec", size, direction=(1.0, 0.0))
    assert numpy.allclose(numpy.median(off, axis=0), [6, 0], atol=0.1)
