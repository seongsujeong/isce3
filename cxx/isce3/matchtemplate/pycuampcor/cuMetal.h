/**
 * @file  cuMetal.h
 * @brief Metal (Apple GPU) implementation of the CPU ampcor chunk pipeline
 */
#ifndef __CUMETAL_H
#define __CUMETAL_H

#include "cuArrays.h"
#include "float2.h"

#include <functional>
#include <utility>
#include <vector>

namespace isce3::matchtemplate::pycuampcor {

class cuAmpcorParameter;
class GDALImage;

#ifdef ISCE3_METAL
/// whether a Metal device is available for ampcor
bool metalAvailable();

/// whether ampcor with these parameters can run on the Metal GPU (device and
/// FFT lengths with prime factors the Metal FFT supports)
bool metalSupported(const cuAmpcorParameter *param);

/// Parameters and run images of one controller (offset layer)
struct MetalLayer {
    cuAmpcorParameter *param;
    cuArrays<float2> *offsetImageRun;
    cuArrays<float> *snrImageRun;
    cuArrays<float3> *covImageRun;
    cuArrays<float> *corrImageRun;
};

/// Process chunks of the layers on the Metal GPU into their run images:
/// (layer, chunk) indices come from nextChunk (layer negative when none is
/// left), chunkDone is called after each chunk. Returns the number of
/// chunks processed.
int runAmpcorMetal(const std::vector<MetalLayer> &layers, GDALImage *reference,
    GDALImage *secondary, const std::function<std::pair<int, int>()> &nextChunk,
    const std::function<void()> &chunkDone);
#else
inline bool metalAvailable() { return false; }
#endif

} // namespace

#endif
