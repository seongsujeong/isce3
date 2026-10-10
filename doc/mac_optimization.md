# InSAR workflow on macOS (CPU + Metal)

Changes of the `optimization_for_mac` branch for running the NISAR InSAR
workflow on Apple silicon: multithreaded CPU paths, Metal (Apple GPU) kernels
for the heavy steps and I/O that follows the chunking of the HDF5 RSLC
(512 x 512, gzip + shuffle). CUDA paths are unchanged except where noted.

## Configuration

| Key | Default | Effect |
|---|---|---|
| `worker.metal_enabled` | `False` | Run ampcor (dense offsets, offsets product), rdr2geo height iterations and SLC resampling on a Metal GPU when one exists. Off, or without a Metal device: CPU FP64 paths as before. |
| `worker.ampcor_memory_fraction` | `0.25` | Fraction of the physical memory for the row caches through which the CPU ampcor reads both images; `0` memory-maps the images instead. |
| `processing.rdr2geo.threshold` | `1.0e-4` (was `1.0e-7`) | Slant range convergence threshold (m). Targets change by < 0.1 mm (p99.99). |
| `ISCE3_GEOCODE_GEOMETRY_CACHE_MB` (env) | `4096` | Size limit of the geocode geometry reuse (`0` disables). |
| `ISCE3_GEOCODE_GEOMETRY_CACHE_DIR` (env) | scratch | Memory-mapped radar positions for geogrids above the limit. |
| `ISCE3_METAL_PROFILE` (env) | unset | Per-kernel GPU times of the Metal ampcor (runs each kernel alone; slow). |

Metal support is built with `cmake -DISCE3_WITH_METAL=ON` (Apple only;
defines `ISCE3_METAL`). Kernels are compiled at run time from sources that CMake embeds
as strings, with safe math (the double-float arithmetic needs exact IEEE
rounding).

## Code map

| Area | Files |
|---|---|
| Shared Metal device, queue, pipelines, buffers | `cxx/isce3/core/detail/MetalContext.{h,mm}` |
| Ampcor chunk pipeline on Metal (FFT, correlation, oversampling) | `cxx/isce3/matchtemplate/pycuampcor/cuMetal.mm`, `cuAmpcor.metal` |
| Ampcor row cache, all layers in one pass | `GDALImage::enableRowCache`, `cuAmpcorController::runAmpcorLayers` |
| rdr2geo mixed precision (FP32 iterations on GPU, FP64 finish) | `cxx/isce3/geometry/detail/Rdr2GeoMixed.h`, `Rdr2GeoMixed.metal`, `Rdr2GeoMetal.{h,mm}` |
| rdr2geo in double-float on GPU (x, y, z only, constant Doppler) | `Rdr2GeoDD.metal`, `Rdr2GeoDDMetal.{h,mm}` |
| SLC resampling on GPU | `cxx/isce3/image/Resample.metal`, `ResampleMetal.{h,mm}` |
| Parallel HDF5 chunk read/write | `python/packages/isce3/io/hdf5_chunks.py` |
| Reference RSLC ENVI copy shared by offsets and crossmul | `nisar.workflows.helpers.reference_slc_copy` |

## Stages

**rdr2geo.** With `metal_enabled`, `Topo` picks one of two GPU paths per run:

- only x, y, z written and a constant Doppler: the whole FP64 iteration in
  double-float arithmetic on the GPU (ellipsoidal height and normal by
  Vermeille, map projections as per-tile quadratic models around FP64
  anchors);
- otherwise: FP64 setup on the CPU, FP32 residual iterations on the GPU, FP64
  finish on the CPU (~1.05 iterations); pixels that do not converge in FP32
  use the FP64 iteration.

**geo2rdr.** CPU; the next block is read and the previous one written while
the current one is computed; the initial azimuth time is the previous pixel's
solution (was uninitialized).

**Ampcor (dense offsets, offsets product).** CPU threads and Metal feeding
threads share the chunks. Images are read through row-block caches aligned
to the chunk height; all offset layers run in one pass over the images. The
Metal pipeline fuses copies, padding, magnitudes, packing and deramping into
the FFT load/store passes (`FFTLoadMode`, `FFTStoreMode` in `cuAmpcor.metal`).

**Rubbersheet.** Saves only the offsets grid; fine resampling and the
ionosphere decimation compute the full-resolution offsets on read.

**Resample (coarse, fine).** Full-width 512-line tiles, HDF5 chunks decoded
in parallel threads, sinc interpolation on the GPU.

**Crossmul.** Upsampling without per-call copies, parallel normalization,
background reads; the phase unwrapping looks are made in the same pass.

**Geocode.** Radar positions reused across rasters of the same geometry;
HDF5 chunks compressed in parallel threads.

## Agreement with the FP64 / develop results

| Stage | Level |
|---|---|
| Crossmul, unwrap looks, rubbersheet, coarse resample tiling, geocode reuse, HDF5 chunk I/O | bit-identical |
| Fine resample tiling | 1 ulp in ~1e-6 of the pixels |
| geo2rdr | offsets differ by up to 3.5e-5 lines (within the threshold) |
| rdr2geo threshold 1e-4 | < 0.1 mm (p99.99) vs 1e-7 |
| rdr2geo mixed / double-float | p99.99 ~0.1 mm; > 1 m in ~1e-5 of the pixels, where several solutions exist (layover, ice front) |
| Metal resample | relative difference median 2e-7, p99.99 8.4e-6, same NaN mask |
| Ampcor | 0.03-0.05% of the windows differ run to run (CPU/GPU chunk split, FFTW plans); Metal-only runs are deterministic |

Product differences at the end of the workflow are dominated by whole-cycle
unwrapping differences between regions and the low-coherence islands in the
ionosphere screen; compare products region by region.

## Benchmark (PIG, Mac CPU + Metal, 32 GB)

| Revision | InSAR total |
|---|---|
| develop | 11,637 s |
| a23f3cef | 2721 s |
| 7d2c7db1 | 1843 s |
| 21bf437d | 1309 s |
| 3597478d | 1183 s |
| 8230f4b5 | 1092 s |

Largest steps at 8230f4b5: offsets product 243-265 s, crossmul ~90 s, unwrap
77 s, rdr2geo 74 s, fine resample 69 s, coarse resample 56 s.

## Verifying changes

Metal-only ampcor runs (`OMP_NUM_THREADS=1`, all chunks on the GPU) and the
rdr2geo/resample kernels are deterministic: refactors are checked
bit-for-bit against outputs of the previous build. Changes to arithmetic are
checked against the CPU FP64 results with the tolerances above.
