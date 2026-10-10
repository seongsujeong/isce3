// Host side of the Metal FP32 rdr2geo height iteration (Rdr2GeoMetal.h)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "Rdr2GeoMetal.h"

#include <isce3/core/detail/MetalContext.h>
#include <isce3/geometry/detail/Rdr2GeoMixedMetalSource.h>

#include <algorithm>

namespace isce3 { namespace geometry { namespace detail {

using namespace isce3::core::detail;

namespace {
// layout matches Params in Rdr2GeoMixed.metal
struct Params { int count, width, nx, ny, maxiter; float tol, refHeight; };

// buffer b with room for `bytes`, reallocated when too small
void reserve(id<MTLBuffer> __strong &b, size_t bytes)
{
    if (!b || b.length < bytes)
        b = metalBuffer(bytes);
}
}

struct Rdr2GeoMetal::Impl {
    id<MTLComputePipelineState> pso = nil;
    id<MTLBuffer> dem = nil;
    int nx = 0, ny = 0;
    float refHeight = 0.f;
    struct Slot {
        id<MTLBuffer> lines = nil, pixels = nil, out = nil;
        id<MTLCommandBuffer> cmd = nil;
    } slots[2];
};

Rdr2GeoMetal::Rdr2GeoMetal() : _impl(new Impl) {}
Rdr2GeoMetal::~Rdr2GeoMetal()
{
    for (auto &s : _impl->slots)
        if (s.cmd) [s.cmd waitUntilCompleted];
}

std::unique_ptr<Rdr2GeoMetal> Rdr2GeoMetal::create()
{
    if (!metalDevice())
        return nullptr;
    std::unique_ptr<Rdr2GeoMetal> g(new Rdr2GeoMetal);
    g->_impl->pso = metalPipeline(rdr2geoMixedMetalSource, "rdr2geoResidualIterate");
    return g;
}

void Rdr2GeoMetal::dem(const float* z, int nx, int ny, float refHeight)
{
    for (auto &s : _impl->slots)
        if (s.cmd) [s.cmd waitUntilCompleted];
    _impl->dem = metalBuffer(z, sizeof(float) * nx * ny);
    _impl->nx = nx;
    _impl->ny = ny;
    _impl->refHeight = refHeight;
}

Rdr2GeoMetalLine* Rdr2GeoMetal::lines(int slot, size_t n)
{
    auto &s = _impl->slots[slot];
    reserve(s.lines, n * sizeof(Rdr2GeoMetalLine));
    return static_cast<Rdr2GeoMetalLine*>(s.lines.contents);
}

Rdr2GeoMetalPixel* Rdr2GeoMetal::pixels(int slot, size_t n)
{
    auto &s = _impl->slots[slot];
    reserve(s.pixels, n * sizeof(Rdr2GeoMetalPixel));
    reserve(s.out, n * sizeof(float));
    return static_cast<Rdr2GeoMetalPixel*>(s.pixels.contents);
}

void Rdr2GeoMetal::run(int slot, size_t nPixels, int width, int maxiter, float tol)
{
    @autoreleasepool {
        auto &s = _impl->slots[slot];
        const Params p {static_cast<int>(nPixels), width, _impl->nx, _impl->ny,
                        maxiter, tol, _impl->refHeight};
        s.cmd = [metalQueue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [s.cmd computeCommandEncoder];
        [enc setComputePipelineState:_impl->pso];
        [enc setBuffer:s.lines offset:0 atIndex:0];
        [enc setBuffer:s.pixels offset:0 atIndex:1];
        [enc setBuffer:_impl->dem offset:0 atIndex:2];
        [enc setBuffer:s.out offset:0 atIndex:3];
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        const NSUInteger tw = _impl->pso.maxTotalThreadsPerThreadgroup;
        [enc dispatchThreads:MTLSizeMake(std::max<size_t>(nPixels, 1), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [s.cmd commit];
    }
}

const float* Rdr2GeoMetal::wait(int slot)
{
    auto &s = _impl->slots[slot];
    metalWait(s.cmd, "Metal rdr2geo");
    s.cmd = nil;
    return static_cast<const float*>(s.out.contents);
}

}}} // namespace isce3::geometry::detail
