"""
Parallel I/O of 2-D chunked HDF5 datasets compressed with gzip (optionally
shuffled), such as the NISAR RSLC images: the HDF5 library (de)compresses
chunks serially, so these helpers read or write the raw chunks and run
zlib, which releases the GIL, in threads. The stored data are the same as
with h5py; other datasets fall back to h5py.
"""
import os
import zlib
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np


def chunk_aligned_lines(lines, chunks):
    """
    Lines per block rounded to a nonzero multiple of the HDF5 chunk height,
    so that full-width line blocks read (and write) whole chunk rows: each
    compressed chunk is then decompressed (compressed) once instead of once
    per block it straddles.

    Parameters
    ----------
    lines : int
        Requested lines per block
    chunks : tuple of int or None
        Chunk shape of the dataset (None: not chunked, lines unchanged)
    """
    if not chunks:
        return lines
    return max(1, round(lines / chunks[0])) * chunks[0]


def write_hdf5_dataset_parallel(dset, data, num_threads=None, chunk_rows=8):
    """
    Write a 2-D array into an HDF5 dataset, compressing its chunks in
    parallel threads (the HDF5 library deflates serially, which dominates the
    time of writing large geocoded layers). The chunks are written as
    pre-filtered chunks (shuffle, then deflate, as the dataset's filter
    pipeline defines), so the stored data are the same as with dset[...] =
    data. Datasets that are not chunked, or use other filters, are written
    with dset[...] = data.

    Parameters
    ----------
    dset: h5py.Dataset
        2-D output dataset
    data: numpy.ndarray
        Array (e.g. numpy.memmap) of the dataset's shape
    num_threads: int, optional
        Compression threads (default: CPU count)
    chunk_rows: int
        Rows of chunks compressed per batch (bounds the memory held)
    """

    plist = dset.id.get_create_plist()
    filters = [plist.get_filter(i) for i in range(plist.get_nfilters())]
    ids = [f[0] for f in filters]
    if dset.chunks is None or dset.ndim != 2 or ids not in (
            [h5py.h5z.FILTER_DEFLATE],
            [h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FILTER_DEFLATE]):
        dset[...] = data
        return
    level = filters[-1][2][0] if filters[-1][2] else 6
    shuffle = ids[0] == h5py.h5z.FILTER_SHUFFLE
    dtype = dset.dtype
    cy, cx = dset.chunks
    ny, nx = dset.shape

    def chunk_bytes(origin):
        i, j = origin
        # Edge chunks are stored at full chunk size
        block = np.zeros((cy, cx), dtype=dtype)
        part = np.asarray(data[i:i + cy, j:j + cx], dtype=dtype)
        block[:part.shape[0], :part.shape[1]] = part
        raw = block.tobytes()
        if shuffle:
            raw = np.frombuffer(raw, np.uint8).reshape(
                -1, dtype.itemsize).T.tobytes()
        return origin, zlib.compress(raw, level)

    with ThreadPoolExecutor(num_threads or os.cpu_count()) as executor:
        for row0 in range(0, ny, cy * chunk_rows):
            origins = [(i, j) for i in range(row0, min(row0 + cy * chunk_rows, ny), cy)
                       for j in range(0, nx, cx)]
            for origin, raw in executor.map(chunk_bytes, origins):
                dset.id.write_direct_chunk(origin, raw)


def _gzip_chunked(dset):
    '''Whether a dataset is chunked and compressed with gzip only (optionally
    shuffled), so that its chunks can be decoded without h5py'''
    return dset.chunks is not None and len(dset.chunks) == 2 and \
        dset.compression == 'gzip' and not dset.fletcher32 and \
        dset.scaleoffset is None


def _decode_chunk_row(dset, executor, i):
    '''Rows [i, i + chunk height) of a 2D gzip-chunked dataset (within the
    dataset), its chunks decoded by the executor's threads'''
    cy, cx = dset.chunks
    ny, nx = dset.shape
    item = dset.dtype.itemsize
    out = np.empty((min(cy, ny - i), nx), dtype=dset.dtype)

    def decode(j):
        mask, raw = dset.id.read_direct_chunk((i, j))
        if mask:  # a filter was skipped when writing: let h5py decode it
            chunk = dset[i:i + cy, j:j + cx]
        else:
            data = np.frombuffer(zlib.decompress(raw), np.uint8)
            if dset.shuffle:
                data = data.reshape(item, -1).T
            chunk = np.ascontiguousarray(data).view(dset.dtype).reshape(cy, cx)
        c1 = min(j + cx, nx)
        out[:, j:c1] = chunk[:out.shape[0], :c1 - j]

    list(executor.map(decode, range(0, nx, cx)))
    return out


def read_hdf5_rows_parallel(dset, row0, rows, num_threads=None):
    '''
    Rows [row0, row0 + rows) of a 2D HDF5 dataset. Chunked datasets
    compressed with gzip only (optionally shuffled) are read chunk by chunk
    without the filters and decoded in parallel threads (zlib releases the
    GIL; h5py decodes serially); other datasets are read by h5py.
    '''
    return ParallelChunkReader(dset, num_threads, cache_rows=0)[
        row0:row0 + rows, :]


class ParallelChunkReader:
    '''
    DatasetReader of a 2D HDF5 dataset whose rows are decoded chunk row by
    chunk row in parallel threads when it is gzip-chunked (else read by
    h5py); the last cache_rows decoded chunk rows are kept, as h5py's chunk
    cache does, for reads overlapping the previous ones.
    '''
    def __init__(self, dataset, num_threads=None, cache_rows=2):
        self.dataset = dataset
        self.shape = dataset.shape
        self.dtype = dataset.dtype
        self.ndim = dataset.ndim
        self.chunks = dataset.chunks
        self._threads = num_threads or os.cpu_count()
        self._cache_rows = cache_rows
        self._cache = {}  # chunk row start -> rows, oldest first

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self[:, :], dtype=dtype)

    def __getitem__(self, key):
        rows, cols = key
        r0, r1, step = rows.indices(self.shape[0])
        if step != 1 or not _gzip_chunked(self.dataset):
            return self.dataset[key]
        cy = self.chunks[0]
        out = np.empty((max(r1 - r0, 0), self.shape[1]), dtype=self.dtype)
        with ThreadPoolExecutor(self._threads) as executor:
            for i in range(r0 // cy * cy, r1, cy):
                band = self._cache.pop(i, None)
                if band is None:
                    band = _decode_chunk_row(self.dataset, executor, i)
                if self._cache_rows:
                    self._cache[i] = band
                    while len(self._cache) > self._cache_rows:
                        del self._cache[next(iter(self._cache))]
                a, b = max(i, r0), min(i + band.shape[0], r1)
                out[a - r0:b - r0] = band[a - i:b - i]
        return out[:, cols]
