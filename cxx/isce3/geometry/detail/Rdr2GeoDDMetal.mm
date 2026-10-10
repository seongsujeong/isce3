// Host side of the double-float Metal rdr2geo (Rdr2GeoDDMetal.h)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "Rdr2GeoDDMetal.h"

#include <isce3/core/detail/MetalContext.h>
#include <isce3/geometry/detail/Rdr2GeoDDMetalSource.h>

#include <algorithm>

namespace isce3 { namespace geometry { namespace detail {

using namespace isce3::core::detail;

struct Rdr2GeoDDMetal::Impl {
    id<MTLComputePipelineState> pso = nil;
    id<MTLBuffer> dem = nil, lines = nil, ranges = nil, prior = nil, tiles = nil, out = nil;
};

namespace {
template<class T>
T* reserve(id<MTLBuffer> __strong &b, size_t n)
{
    if (!b || b.length < n * sizeof(T))
        b = metalBuffer(n * sizeof(T));
    return static_cast<T*>(b.contents);
}
}

Rdr2GeoDDMetal::Rdr2GeoDDMetal() : _impl(new Impl) {}
Rdr2GeoDDMetal::~Rdr2GeoDDMetal() = default;

std::unique_ptr<Rdr2GeoDDMetal> Rdr2GeoDDMetal::create()
{
    if (!metalDevice())
        return nullptr;
    std::unique_ptr<Rdr2GeoDDMetal> g(new Rdr2GeoDDMetal);
    g->_impl->pso = metalPipeline(rdr2geoDDMetalSource, "rdr2geoDD");
    return g;
}

void Rdr2GeoDDMetal::dem(const float* z, int nx, int ny)
{
    _impl->dem = metalBuffer(z, sizeof(float) * nx * ny);
}

Rdr2GeoDDLine* Rdr2GeoDDMetal::lines(size_t n) { return reserve<Rdr2GeoDDLine>(_impl->lines, n); }
DD* Rdr2GeoDDMetal::ranges(size_t n) { return reserve<DD>(_impl->ranges, n); }
float* Rdr2GeoDDMetal::prior(size_t n) { return reserve<float>(_impl->prior, n); }
Rdr2GeoDDTile* Rdr2GeoDDMetal::tiles(size_t n) { return reserve<Rdr2GeoDDTile>(_impl->tiles, n); }
Rdr2GeoDDOut* Rdr2GeoDDMetal::out(size_t n) { return reserve<Rdr2GeoDDOut>(_impl->out, n); }

void Rdr2GeoDDMetal::run(const Rdr2GeoDDParams& p)
{
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [metalQueue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:_impl->pso];
        [enc setBuffer:_impl->lines offset:0 atIndex:0];
        [enc setBuffer:_impl->ranges offset:0 atIndex:1];
        [enc setBuffer:_impl->prior offset:0 atIndex:2];
        [enc setBuffer:_impl->tiles offset:0 atIndex:3];
        [enc setBuffer:_impl->dem offset:0 atIndex:4];
        [enc setBuffer:_impl->out offset:0 atIndex:5];
        [enc setBytes:&p length:sizeof(p) atIndex:6];
        [enc dispatchThreads:MTLSizeMake(std::max<size_t>(p.count, 1), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(_impl->pso.maxTotalThreadsPerThreadgroup, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        metalWait(cmd, "Metal rdr2geo");
    }
}

}}} // namespace isce3::geometry::detail
