// Host side of the Metal SLC resampling (ResampleMetal.h, Resample.metal)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ResampleMetal.h"

#include <isce3/core/Constants.h>
#include <isce3/core/detail/MetalContext.h>
#include <isce3/image/ResampleMetalSource.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace isce3::image::v2 {

using isce3::core::SINC_LEN;
using isce3::core::SINC_SUB;
using namespace isce3::core::detail;

namespace {
// layout matches Params in Resample.metal
struct Params {
    uint32_t count;
    int inWidth, inLength;
    float fill[2];
    int lutMode, lutWidth, lutLength;
    float lutValue, cx0, cx1, cy0, cy1;
    float dopplerScale;
};

// Normalized sinc filter of Sinc2dInterpolator(SINC_LEN, SINC_SUB) in float,
// SINC_SUB phases x SINC_LEN taps
id<MTLBuffer> sincFilter()
{
    static id<MTLBuffer> buffer = [] {
        const int n = SINC_SUB * SINC_LEN;
        std::vector<double> filter(n);
        // Sinc2dInterpolator::_sinc_coef(1, SINC_LEN, SINC_SUB, 0, 1, filter)
        const double wgthgt = 0.5, soff = (n - 1.) / 2.;
        for (int i = 0; i < n; ++i) {
            const double wgt = (1. - wgthgt) + wgthgt * std::cos(M_PI * (i - soff) / soff);
            const double s = std::floor(i - soff) / SINC_SUB;
            const double fct = s != 0. ? std::sin(M_PI * s) / (M_PI * s) : 1.;
            filter[i] = fct * wgt;
        }
        std::vector<float> table(n);
        for (int i = 0; i < SINC_SUB; ++i) {
            double ssum = 0.;
            for (int j = 0; j < SINC_LEN; ++j)
                ssum += filter[i + SINC_SUB * j];
            for (int j = 0; j < SINC_LEN; ++j)
                table[i * SINC_LEN + j] =
                        static_cast<float>(filter[i + SINC_SUB * j] / ssum);
        }
        return metalBuffer(table.data(), table.size() * sizeof(float));
    }();
    return buffer;
}
}

bool metalResampleAvailable() { return metalDevice() != nil; }

void resampleToCoordsMetal(
    ArrayRef2D<std::complex<float>> resampled_data_block,
    const ConstArrayRef2D<std::complex<float>> input_data_block,
    const ConstArrayRef2D<double> range_input_indices,
    const ConstArrayRef2D<double> azimuth_input_indices,
    const isce3::product::RadarGridParameters& radar_grid,
    const isce3::core::LUT2d<double>& native_doppler_lut,
    const std::complex<float> fill_value)
{
    const auto& lut = native_doppler_lut;
    const bool bilinear = lut.haveData();
    if (!metalDevice() ||
        (bilinear && lut.interpMethod() != isce3::core::BILINEAR_METHOD)) {
        resampleToCoords(resampled_data_block, input_data_block,
                         range_input_indices, azimuth_input_indices,
                         radar_grid, native_doppler_lut, fill_value);
        return;
    }
    const size_t count = resampled_data_block.size();
    if (range_input_indices.size() != count || azimuth_input_indices.size() != count)
        throw std::invalid_argument("resample indices and output sizes differ");
    // the GPU reads and writes the arrays as contiguous rows
    auto contiguous = [](const auto& a) { return a.outerStride() == a.cols(); };
    if (!contiguous(resampled_data_block) || !contiguous(input_data_block) ||
        !contiguous(range_input_indices) || !contiguous(azimuth_input_indices)) {
        resampleToCoords(resampled_data_block, input_data_block,
                         range_input_indices, azimuth_input_indices,
                         radar_grid, native_doppler_lut, fill_value);
        return;
    }

    @autoreleasepool {
        Params p {};
        p.count = static_cast<uint32_t>(count);
        p.inWidth = static_cast<int>(input_data_block.cols());
        p.inLength = static_cast<int>(input_data_block.rows());
        p.fill[0] = fill_value.real();
        p.fill[1] = fill_value.imag();
        p.dopplerScale = static_cast<float>(2. * M_PI / radar_grid.prf());
        std::vector<float> lutData(1, 0.f);
        if (bilinear) {
            // LUT indices of the input indices: x = (slant range - x0) / dx
            // with slant range = r0 + range index * spacing; same for time
            p.lutMode = 1;
            p.lutWidth = static_cast<int>(lut.width());
            p.lutLength = static_cast<int>(lut.length());
            p.cx0 = static_cast<float>((radar_grid.startingRange() - lut.xStart()) / lut.xSpacing());
            p.cx1 = static_cast<float>(radar_grid.rangePixelSpacing() / lut.xSpacing());
            p.cy0 = static_cast<float>((radar_grid.sensingStart() - lut.yStart()) / lut.ySpacing());
            p.cy1 = static_cast<float>(1. / (radar_grid.prf() * lut.ySpacing()));
            lutData.resize(lut.width() * lut.length());
            for (size_t i = 0; i < lutData.size(); ++i)
                lutData[i] = static_cast<float>(lut.data().data()[i]);
        } else {
            p.lutValue = static_cast<float>(lut.refValue());
        }

        id<MTLBuffer> input = metalBuffer(input_data_block.data(),
                                     input_data_block.size() * sizeof(std::complex<float>));
        id<MTLBuffer> rg = metalBuffer(range_input_indices.data(), count * sizeof(double));
        id<MTLBuffer> az = metalBuffer(azimuth_input_indices.data(), count * sizeof(double));
        id<MTLBuffer> lutBuffer = metalBuffer(lutData.data(), lutData.size() * sizeof(float));
        id<MTLBuffer> out = metalBuffer(count * sizeof(std::complex<float>));

        id<MTLComputePipelineState> pso = metalPipeline(resampleMetalSource, "resampleToCoords");
        id<MTLCommandBuffer> cmd = [metalQueue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:input offset:0 atIndex:0];
        [enc setBuffer:rg offset:0 atIndex:1];
        [enc setBuffer:az offset:0 atIndex:2];
        [enc setBuffer:sincFilter() offset:0 atIndex:3];
        [enc setBuffer:lutBuffer offset:0 atIndex:4];
        [enc setBuffer:out offset:0 atIndex:5];
        [enc setBytes:&p length:sizeof(p) atIndex:6];
        [enc dispatchThreads:MTLSizeMake(std::max<size_t>(count, 1), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(pso.maxTotalThreadsPerThreadgroup, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        metalWait(cmd, "Metal resample");
        std::memcpy(resampled_data_block.data(), out.contents,
                    count * sizeof(std::complex<float>));
    }
}

} // namespace isce3::image::v2
