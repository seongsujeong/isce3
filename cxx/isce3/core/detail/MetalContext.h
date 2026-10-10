// -*- C++ -*-
// Process-wide Metal (Apple GPU) device, command queue and compute
// pipelines compiled at run time from kernel sources embedded by CMake.
// For Objective-C++ (.mm) sources built with ARC.
#pragma once

#import <Metal/Metal.h>

namespace isce3 { namespace core { namespace detail {

/** Default Metal device, nil if none */
id<MTLDevice> metalDevice();

/** Command queue of the default device (nil if none) */
id<MTLCommandQueue> metalQueue();

/** Compute pipeline of kernel `name` in the Metal source `source` (a string
 * with static storage: pipelines are cached by its address), compiled with
 * safe math on first use. Throws std::runtime_error on compile errors. */
id<MTLComputePipelineState> metalPipeline(const char* source, const char* name);

/** New shared-storage buffer (CPU and GPU see the same memory) of at least
 * `bytes` (4 at least: Metal rejects empty buffers); throws on failure */
id<MTLBuffer> metalBuffer(size_t bytes);

/** New shared-storage buffer holding a copy of `bytes` bytes at `data` */
id<MTLBuffer> metalBuffer(const void* data, size_t bytes);

/** Shared-storage buffer over host memory without copy: `data` must be
 * page-aligned (e.g. from posix_memalign), the length is rounded up to
 * whole pages (allocated by the host as well) and the host keeps ownership
 * (outliving the buffer). Wrapping maps the pages for the GPU: wrap large
 * memory once, not per use. */
id<MTLBuffer> metalWrap(const void* data, size_t bytes);

/** Wait for a committed command buffer; throws (prefixed by `what`) if it
 * failed */
void metalWait(id<MTLCommandBuffer> cmd, const char* what);

}}} // namespace isce3::core::detail
