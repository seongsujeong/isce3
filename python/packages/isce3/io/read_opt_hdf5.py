import copy
import h5py
import numpy as np


def _next_prime(n):
    '''Smallest prime >= n (n >= 2)'''
    n = max(int(n), 2)
    while any(n % d == 0 for d in range(2, int(n ** 0.5) + 1)):
        n += 1
    return n


class HDF5OptimizedReader(h5py.File):
    """
    The HDF5 optimizer reader class inheriting from h5py.File
    to avoid passing h5py.File parameter
    """

    def __init__(self,name, **kwds):
        """
        Constructor of the HDF5 optimizer reader inheriting
        from h5py.File to avoid passing h5py.File parameter.

        Parameters
        ----------
        name : str
            HDF5 file name
        num_rows : int, optional
            Minimal number of rows you want to be able to cache (default: 2048)
        kwds
            Keyword arguments forwarded to h5py.File
        """

        # To avoid the change of the kwds in-place
        new_kwds = copy.deepcopy(kwds)
        hdf5_file = name
        num_rows = new_kwds.pop('num_rows', 2048)

        # The minimum chunk cache size is set to 1 Mb
        largest_chunk_cache_size = 1024 ** 2
        # Number of chunks that fit in that cache
        largest_cache_chunks = 1

        # Get the largest chunk cache size
        def _get_largest_chunk_cache_size(ds_name, ds):
            """
            Get the largest chunk cache size

            Parameters
            ----------
            ds_name : str
                Dataset name
            ds : h5py.Dataset
                h5py Dataset object
            """

            # nonlocal so largest_chunk_cache_size declared
            # above can be altered when this helper function
            # is iteratively applied to h5py datasets below
            nonlocal largest_chunk_cache_size, largest_cache_chunks

            if isinstance(ds, h5py.Dataset):
                ds_ndims = len(ds.shape)
                if ds_ndims in [2,3] and ds.chunks is not None:
                    i_width_dim = ds_ndims - 1
                    i_length_dim = ds_ndims - 2
                    # Ensure that the number of chunk blocks is large
                    # enough to cover the width of the image
                    num_of_blocks = int(
                        float(ds.shape[i_width_dim] +
                              ds.chunks[i_width_dim] - 1.0)/
                         ds.chunks[i_width_dim])

                    chunk_cache_size = \
                        num_of_blocks * np.prod(ds.chunks)\
                            * ds.dtype.itemsize

                    chunk_cache_size *= int(
                        float(num_rows + ds.chunks[i_length_dim] - 1.0)/
                         ds.chunks[i_length_dim])

                    if chunk_cache_size > largest_chunk_cache_size:
                        largest_chunk_cache_size = chunk_cache_size
                        largest_cache_chunks = chunk_cache_size // (
                            np.prod(ds.chunks) * ds.dtype.itemsize)

        with h5py.File(hdf5_file, **new_kwds) as h5:
            h5.visititems(_get_largest_chunk_cache_size)

        new_kwds['rdcc_nbytes'] = largest_chunk_cache_size
        # Hash table size of the chunk cache: HDF5 recommends a prime about
        # 100 times the number of cached chunks; h5py's default (521) is
        # too small for a cache of hundreds of chunks, whose collisions
        # evict chunks that are then decompressed again
        new_kwds.setdefault('rdcc_nslots', _next_prime(100 * largest_cache_chunks))

        # Initialize the h5py File object
        super().__init__(hdf5_file,**new_kwds)