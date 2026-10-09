// Host side of the Metal FP32 rdr2geo height iteration (Rdr2GeoMetal.h)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "Rdr2GeoMetal.h"

#include <isce3/geometry/detail/Rdr2GeoMixedMetalSource.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace isce3 { namespace geometry { namespace detail {

namespace {
// layout matches Params in Rdr2GeoMixed.metal
struct Params { int count, width, nx, ny, maxiter; float tol, refHeight; };

// shared-storage buffer of at least `bytes`, reallocated when too small
void reserve(id<MTLDevice> device, id<MTLBuffer> __strong &b, size_t bytes)
{
    if (b && b.length >= bytes) return;
    b = [device newBufferWithLength:std::max<size_t>(bytes, 4)
                            options:MTLResourceStorageModeShared];
    if (!b) throw std::runtime_error("Metal rdr2geo buffer allocation failed");
}
}

struct Rdr2GeoMetal::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
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
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return nullptr;
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;  // follow the CPU arithmetic
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@(rdr2geoMixedMetalSource)
                                                      options:options error:&error];
        id<MTLFunction> f = [library newFunctionWithName:@"rdr2geoResidualIterate"];
        id<MTLComputePipelineState> pso = f ?
            [device newComputePipelineStateWithFunction:f error:&error] : nil;
        if (!pso)
            throw std::runtime_error(std::string("Metal rdr2geo kernel: ") +
                    (error ? error.localizedDescription.UTF8String : "not found"));
        std::unique_ptr<Rdr2GeoMetal> g(new Rdr2GeoMetal);
        g->_impl->device = device;
        g->_impl->queue = [device newCommandQueue];
        g->_impl->pso = pso;
        return g;
    }
}

void Rdr2GeoMetal::dem(const float* z, int nx, int ny, float refHeight)
{
    for (auto &s : _impl->slots)
        if (s.cmd) [s.cmd waitUntilCompleted];
    _impl->dem = [_impl->device newBufferWithBytes:z length:sizeof(float) * nx * ny
                                           options:MTLResourceStorageModeShared];
    if (!_impl->dem) throw std::runtime_error("Metal rdr2geo DEM upload failed");
    _impl->nx = nx;
    _impl->ny = ny;
    _impl->refHeight = refHeight;
}

Rdr2GeoMetalLine* Rdr2GeoMetal::lines(int slot, size_t n)
{
    auto &s = _impl->slots[slot];
    reserve(_impl->device, s.lines, n * sizeof(Rdr2GeoMetalLine));
    return static_cast<Rdr2GeoMetalLine*>(s.lines.contents);
}

Rdr2GeoMetalPixel* Rdr2GeoMetal::pixels(int slot, size_t n)
{
    auto &s = _impl->slots[slot];
    reserve(_impl->device, s.pixels, n * sizeof(Rdr2GeoMetalPixel));
    reserve(_impl->device, s.out, n * sizeof(float));
    return static_cast<Rdr2GeoMetalPixel*>(s.pixels.contents);
}

void Rdr2GeoMetal::run(int slot, size_t nPixels, int width, int maxiter, float tol)
{
    @autoreleasepool {
        auto &s = _impl->slots[slot];
        const Params p {static_cast<int>(nPixels), width, _impl->nx, _impl->ny,
                        maxiter, tol, _impl->refHeight};
        s.cmd = [_impl->queue commandBuffer];
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
    [s.cmd waitUntilCompleted];
    if (s.cmd.status == MTLCommandBufferStatusError)
        throw std::runtime_error(std::string("Metal rdr2geo kernel failed: ") +
                                 s.cmd.error.localizedDescription.UTF8String);
    s.cmd = nil;
    return static_cast<const float*>(s.out.contents);
}

}}} // namespace isce3::geometry::detail
