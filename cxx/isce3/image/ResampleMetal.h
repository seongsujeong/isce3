#pragma once

#include "Resample.h"

namespace isce3::image::v2 {

/** Whether resampleToCoordsMetal runs on a Metal GPU (ISCE3_METAL builds
 * with a Metal device) */
bool metalResampleAvailable();

/** resampleToCoords on the Metal GPU (FP32; see Resample.metal). Falls back
 * to the CPU resampleToCoords without a Metal GPU or for a native Doppler
 * LUT with data and an interpolation method other than bilinear. */
void resampleToCoordsMetal(
    ArrayRef2D<std::complex<float>> resampled_data_block,
    const ConstArrayRef2D<std::complex<float>> input_data_block,
    const ConstArrayRef2D<double> range_input_indices,
    const ConstArrayRef2D<double> azimuth_input_indices,
    const isce3::product::RadarGridParameters& radar_grid,
    const isce3::core::LUT2d<double>& native_doppler_lut,
    const std::complex<float> fill_value = std::complex<float>(
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN()
    )
);

} // namespace isce3::image::v2
