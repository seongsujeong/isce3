#include "Resample.h"

#include <thrust/complex.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#include <isce3/core/Constants.h>

#include <isce3/cuda/core/gpuLUT2d.h>
#include <isce3/cuda/core/gpuInterpolator.h>
#include <isce3/cuda/except/Error.h>

namespace isce3::cuda::image::v2 {

using isce3::cuda::core::gpuLUT2d;

using isce3::core::SINC_HALF;
using isce3::core::SINC_LEN;
using isce3::core::SINC_SUB;

__global__
void _resampleToCoordsGlobal(
    thrust::complex<float>* resampled_data_block,
    const size_t resampled_block_width,
    const size_t resampled_block_length,
    const thrust::complex<float>* input_data_block,
    const size_t input_block_width,
    const size_t input_block_length,
    const double* range_input_indices,
    const double* azimuth_input_indices,
    const double startingRange,
    const double rangePixelSpacing,
    const double sensingStart,
    const double pri,                       // Pulse repetition interval, inverse of prf
    const gpuLUT2d<double> native_doppler_lut,
    const float* sinc_filter,               // SINC_SUB phases x SINC_LEN taps
    const thrust::complex<float> fill_value
)
{
    // NOTE: This function uses PRI instead of PRF since operations rely on dividing
    // by PRF. Division is more expensive than multiplication on the device, so passing
    // the reciprocal of PRF and multiplying instead is preferable.

    const auto pixel_index = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    // Prior to any data operations, return if the pixel is outside of the output grid.
    // This check is necessary because a function call from host to device must be
    // done with a multiple of the thrd_per_block pixels, but the output data size will
    // typically be smaller than this multiple. So, some calls to this function on
    // the device will be for non-existent pixels which must be discarded.
    if (pixel_index >= resampled_block_width * resampled_block_length) return;

    // The indices on the resampled data block. Assumes that range/azimuth indices
    // vectors are the same shape as the resampled data vector.
    // unit: column pixels on input array (double)
    const auto range_input_ind = range_input_indices[pixel_index];
    // unit: row pixels on input array (double)
    const auto azimuth_input_ind = azimuth_input_indices[pixel_index];

    // Skip if either the azimuth or range input index are NaN.
    if (std::isnan(azimuth_input_ind) || std::isnan(range_input_ind)) {
        resampled_data_block[pixel_index] = fill_value;
        return;
    }

    // unit: range column indices (int)
    const auto range_input_ind_int = __double2int_rd(range_input_ind);
    // unit: azimuth row indices (int)
    const auto azimuth_input_ind_int = __double2int_rd(azimuth_input_ind);
    
    // Check if chip indices could be outside radar grid minus margin to
    // account for sinc chip. Fill with fill_value and skip if chip indices
    // out of bounds.
    if (
        (range_input_ind_int < SINC_HALF) ||
        (range_input_ind_int >= (input_block_width - SINC_HALF)) ||
        (azimuth_input_ind_int < SINC_HALF) ||
        (azimuth_input_ind_int >= (input_block_length - SINC_HALF))
    ) {
        resampled_data_block[pixel_index] = fill_value;
        return;
    }

    // unit: range column indices (double)
    const auto range_input_index_remainder =
        range_input_ind - __int2double_rn(range_input_ind_int);
    // unit: azimuth row indices (double)
    const auto azimuth_input_index_remainder =
        azimuth_input_ind - __int2double_rn(azimuth_input_ind_int);

    // Slant Range at the current output pixel
    // unit: distance (meters)
    const double rg_distance = startingRange + range_input_ind * rangePixelSpacing;

    // Azimuth time at the current output pixel
    // unit: time (seconds)
    const double az_time = sensingStart + azimuth_input_ind * pri;

    // If the doppler LUT doesn't contain this coordinate, fill this pixel
    // with the given fill_value and skip it.
    if (not native_doppler_lut.contains(az_time, rg_distance)) {
        resampled_data_block[pixel_index] = fill_value;
        return;
    }
    
    // Evaluate doppler at current range and azimuth time
    // unit: frequency (radians per sample)
    const auto doppler_freq =
        native_doppler_lut.eval(az_time, rg_distance) * 2.0 * M_PI * pri;
    // float suffices for the doppler phasors: |doppler_freq| <= pi and the
    // phases below are at most pi * SINC_HALF
    const float doppler_freq_f = static_cast<float>(doppler_freq);

    // Sinc-interpolate the doppler-stripped data directly from the input block.
    // Same arithmetic as gpuSinc2dInterpolator::interpolate on a SINC_ONE x
    // SINC_ONE chip centered on the integer indices, without storing the chip
    // (a per-pixel chip buffer took 648 B of device memory per output pixel).
    //
    // Chip coordinates and nearest filter phases, as the interpolator computes
    // them from x/y = SINC_HALF + remainder
    const double x = SINC_HALF + range_input_index_remainder;
    const double y = SINC_HALF + azimuth_input_index_remainder;
    const int ix = __double2int_rd(x);
    const int iy = __double2int_rd(y);
    const int ifracx = min(max(0, int((x - ix) * SINC_SUB)), SINC_SUB - 1);
    const int ifracy = min(max(0, int((y - iy) * SINC_SUB)), SINC_SUB - 1);
    const float* kx = sinc_filter + ifracx * SINC_LEN;
    const float* ky = sinc_filter + ifracy * SINC_LEN;

    thrust::complex<float> interpolated_complex_val(0.0f);
    // The interpolator's edge check on the chip (a remainder rounding x or y
    // up to SINC_HALF + 1 falls outside)
    const int half = SINC_LEN / 2;
    if (ix >= half - 1 && ix <= SINC_HALF && iy >= half - 1 && iy <= SINC_HALF) {
        for (int i = 0; i < SINC_LEN; ++i) {
            // Chip row (and its offset from the chip center)
            const int chip_az = iy + half - i;

            // Compute doppler phase to be removed from radar data.
            // (i.e. as a unit vector on the complex plane.)
            float doppler_sin, doppler_cos;
            sincosf(doppler_freq_f * (chip_az - SINC_HALF), &doppler_sin,
                    &doppler_cos);
            const thrust::complex<float> doppler_phase_conj(doppler_cos,
                                                            -doppler_sin);

            // Input sample of chip column ix + half, the first tap of the row
            const thrust::complex<float>* row = input_data_block +
                input_block_width * (azimuth_input_ind_int + chip_az - SINC_HALF) +
                (range_input_ind_int + ix + half - SINC_HALF);

            thrust::complex<float> row_sum(0.0f);
            for (int j = 0; j < SINC_LEN; ++j) {
                row_sum += (row[-j] * doppler_phase_conj) * kx[j];
            }
            interpolated_complex_val += row_sum * ky[i];
        }
    }

    // Interpolation performed on data stripped of doppler.
    // Calculate the doppler phase shift to be reintroduced.
    float doppler_sin, doppler_cos;
    sincosf(doppler_freq_f * static_cast<float>(azimuth_input_index_remainder),
            &doppler_sin, &doppler_cos);
    const thrust::complex<float> doppler_resampled_phasor(doppler_cos, doppler_sin);

    // Add doppler to interpolated value
    resampled_data_block[pixel_index] = 
        interpolated_complex_val * doppler_resampled_phasor;

} // end _resampleToCoordsGlobal


/** Copy the contents of `arr` to the GPU and convert the elements to type `T`. */
template<class T, class U>
auto _copyToDeviceAs(const ConstArrayRef2D<U>& arr)
{
    thrust::device_vector<T> d_vec(arr.size());
    thrust::copy(arr.data(), arr.data() + arr.size(), d_vec.begin());
    return d_vec;
}

/** Copy the contents of `arr` to the GPU, preserving the element type. */
template<class T>
auto _copyToDevice(const ConstArrayRef2D<T>& arr)
{
    return _copyToDeviceAs<T>(arr);
}


// Interpolate tile to perform transformation
void
gpuResampleToCoords(
    ArrayRef2D<std::complex<float>> resampled_data_block,
    const ConstArrayRef2D<std::complex<float>> input_data_block,
    const ConstArrayRef2D<double> range_input_indices,
    const ConstArrayRef2D<double> azimuth_input_indices,
    const isce3::product::RadarGridParameters& radar_grid,
    const isce3::core::LUT2d<double>& native_doppler_lut,
    const std::complex<float> fill_value
) {
    // number of columns on input array
    const auto in_width = static_cast<size_t>(input_data_block.cols());
    // number of rows on input array
    const auto in_length = static_cast<size_t>(input_data_block.rows());
    // number of columns on output array
    const auto out_width = static_cast<size_t>(resampled_data_block.cols());
    // number of rows on output array
    const auto out_length = static_cast<size_t>(resampled_data_block.rows());

    // Number of threads per block (should always %32==0)
    const int thrd_per_block = 256;

    // Determine the number of pixels.
    const size_t num_resampled_pixels = out_width * out_length;

    // Sinc interpolation filter, as gpuSinc2dInterpolator builds it (computed in
    // double, used in float for complex<float> data).
    // In order to add support for different interpolators, a Python binding needs
    // to be made for these interpolator objects.
    thrust::host_vector<double> h_sinc_filter(SINC_SUB * SINC_LEN, 0.0);
    isce3::cuda::core::compute_normalized_coefficients(
        1.0, SINC_LEN, SINC_SUB, 0.0, h_sinc_filter);
    const thrust::host_vector<float> h_sinc_filter_f(h_sinc_filter);
    const thrust::device_vector<float> d_sinc_filter(h_sinc_filter_f);

    // Declare device vectors for all input data and copy the input data to them.
    auto d_input_data = _copyToDeviceAs<thrust::complex<float>>(input_data_block);
    auto d_range_indices = _copyToDevice(range_input_indices);
    auto d_azimuth_indices = _copyToDevice(azimuth_input_indices);

    gpuLUT2d<double> d_doppler(native_doppler_lut);

    // Convert std::complex to thrust::complex for invalid value.
    const thrust::complex d_fill_value(fill_value);

    // Declare the output vector.
    thrust::device_vector<thrust::complex<float>> d_resampled_data(
        num_resampled_pixels
    );

    // Determine the grid of blocks needed to run this algorithm on the device.
    dim3 block(thrd_per_block);
    dim3 grid((num_resampled_pixels + (thrd_per_block - 1)) / thrd_per_block);

    // Launch the kernel on the GPU.
    _resampleToCoordsGlobal<<<grid, block>>>(
        d_resampled_data.data().get(),
        out_width,
        out_length,
        d_input_data.data().get(),
        in_width,
        in_length,
        d_range_indices.data().get(),
        d_azimuth_indices.data().get(),
        radar_grid.startingRange(),
        radar_grid.rangePixelSpacing(),
        radar_grid.sensingStart(),
        1 / radar_grid.prf(),
        d_doppler,
        d_sinc_filter.data().get(),
        d_fill_value
    );

    // Check for any kernel errors.
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Write the output data from the device to the host.
    thrust::copy(
        d_resampled_data.begin(),
        d_resampled_data.end(),
        resampled_data_block.data()
    );
}

} // end namespace isce3::cuda::image::v2