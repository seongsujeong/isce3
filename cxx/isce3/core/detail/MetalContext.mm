// Process-wide Metal context (MetalContext.h)
#import <Foundation/Foundation.h>

#include "MetalContext.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace isce3 { namespace core { namespace detail {

namespace {
struct Context {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = device ? [device newCommandQueue] : nil;
    std::mutex mutex;
    std::map<const char*, id<MTLLibrary>> libraries;
    std::map<std::pair<const char*, std::string>, id<MTLComputePipelineState>> pipelines;
};

Context& context()
{
    static Context ctx;
    return ctx;
}
}

id<MTLDevice> metalDevice() { return context().device; }

id<MTLCommandQueue> metalQueue() { return context().queue; }

id<MTLComputePipelineState> metalPipeline(const char* source, const char* name)
{
    auto& ctx = context();
    std::lock_guard<std::mutex> lock(ctx.mutex);
    auto& pso = ctx.pipelines[{source, name}];
    if (pso)
        return pso;
    @autoreleasepool {
        NSError* error = nil;
        auto& library = ctx.libraries[source];
        if (!library) {
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.mathMode = MTLMathModeSafe;  // follow the CPU arithmetic
            library = [ctx.device newLibraryWithSource:@(source) options:options
                                                 error:&error];
        }
        id<MTLFunction> f = [library newFunctionWithName:@(name)];
        if (f)
            pso = [ctx.device newComputePipelineStateWithFunction:f error:&error];
        if (!pso)
            throw std::runtime_error(std::string("Metal kernel ") + name + ": " +
                    (error ? error.localizedDescription.UTF8String : "not found"));
    }
    return pso;
}

id<MTLBuffer> metalBuffer(size_t bytes)
{
    id<MTLBuffer> b = [metalDevice() newBufferWithLength:std::max<size_t>(bytes, 4)
                                                 options:MTLResourceStorageModeShared];
    if (!b)
        throw std::runtime_error("Metal buffer allocation failed");
    return b;
}

id<MTLBuffer> metalBuffer(const void* data, size_t bytes)
{
    id<MTLBuffer> b = bytes ? [metalDevice() newBufferWithBytes:data length:bytes
                                                         options:MTLResourceStorageModeShared]
                            : metalBuffer(0);
    if (!b)
        throw std::runtime_error("Metal buffer allocation failed");
    return b;
}

id<MTLBuffer> metalWrap(const void* data, size_t bytes)
{
    const size_t page = getpagesize();
    const size_t length = std::max<size_t>((bytes + page - 1) / page * page, page);
    id<MTLBuffer> b = [metalDevice() newBufferWithBytesNoCopy:const_cast<void*>(data)
                                                       length:length
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    if (!b)
        throw std::runtime_error("Metal buffer over host memory failed");
    return b;
}

void metalWait(id<MTLCommandBuffer> cmd, const char* what)
{
    [cmd waitUntilCompleted];
    if (cmd.status == MTLCommandBufferStatusError)
        throw std::runtime_error(std::string(what) + ": " +
                                 cmd.error.localizedDescription.UTF8String);
}

}}} // namespace isce3::core::detail
