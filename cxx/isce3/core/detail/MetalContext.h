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

}}} // namespace isce3::core::detail
