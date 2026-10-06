/**
 * @file  cuMetal.h
 * @brief Metal (Apple GPU) implementation of the CPU ampcor chunk pipeline
 */
#ifndef __CUMETAL_H
#define __CUMETAL_H

#include "cuArrays.h"
#include "float2.h"

#include <functional>

namespace isce3::matchtemplate::pycuampcor {

class cuAmpcorParameter;
class GDALImage;

#ifdef ISCE3_METAL
/// whether a Metal device is available for ampcor
bool metalAvailable();

/// whether ampcor with these parameters can run on the Metal GPU (device and
/// FFT lengths with prime factors the Metal FFT supports)
bool metalSupported(const cuAmpcorParameter *param);

/// Process chunks on the Metal GPU into the run images: chunk indices come
/// from nextChunk (negative when none is left), chunkDone is called after
/// each chunk. Returns the number of chunks processed.
int runAmpcorMetal(cuAmpcorParameter *param, GDALImage *reference, GDALImage *secondary,
    cuArrays<float2> *offsetImageRun, cuArrays<float> *snrImageRun,
    cuArrays<float3> *covImageRun, cuArrays<float> *corrImageRun,
    const std::function<int()> &nextChunk, const std::function<void()> &chunkDone);
#else
inline bool metalAvailable() { return false; }
#endif

} // namespace

#endif
