/*
 * @file cuOffset.cu
 * @brief Utilities used to determine the offset field
 *
 */

// my module dependencies
#include "cuAmpcorUtil.h"

// for FLT_MAX
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include "float2.h"

namespace isce3::matchtemplate::pycuampcor {

// kernel for 2D array(image), find max value and location
void  cudaKernel_maxloc2D(const float* const images, int2* maxloc, float* maxval,
    const size_t imageNX, const size_t imageNY, const size_t nImages)
{
    const int imageSize = imageNX * imageNY;

    for (int bid = 0; bid < nImages; bid++) {
        float my_maxval = std::numeric_limits<float>::lowest();
        int2 my_maxloc;
        const float* image = &images[bid * imageSize];
        for (int i = 0; i < imageSize; i++) {
            if (image[i] > my_maxval) {
                my_maxval = image[i];
                my_maxloc = make_int2(i / imageNY, i % imageNY);
            }
        }
        maxval[bid] = my_maxval;
        maxloc[bid] = my_maxloc;
    }
}

/**
 * Find both the maximum value and the location for a batch of 2D images
 * @param[in] images input batch of images
 * @param[out] maxval arrays to hold the max values
 * @param[out] maxloc arrays to hold the max locations
 * @note This routine is overloaded with the routine without maxval
 */
void cuArraysMaxloc2D(cuArrays<float> *images, cuArrays<int2> *maxloc,
                      cuArrays<float> *maxval)
{
    cudaKernel_maxloc2D(images->devData, maxloc->devData, maxval->devData,
            images->height, images->width, images->count);
}

/**
 * Find the correlation peak of each 2D image with the directional line
 * constrained (DLC) search of Jeong et al. (2017, IEEE TGRS,
 * doi:10.1109/TGRS.2016.2643699): pivots are placed on the line through the
 * image center (the gross offset) along the flow direction, both forward and
 * backward, and each pivot climbs to a local maximum by steepest ascent over
 * its 8 neighbors. The highest of these local maxima is the peak.
 * @param[in] images batch of correlation surfaces
 * @param[in] direction flow direction (x: down, y: across) of each image;
 *   (0, 0) falls back to the global maximum
 * @param[out] maxloc peak locations (x: down, y: across)
 * @param[out] maxval peak values
 */
void cuArraysMaxlocDLC(cuArrays<float> *images, const float2 *direction,
                       cuArrays<int2> *maxloc, cuArrays<float> *maxval)
{
    const int nx = images->height, ny = images->width;
    for (int bid = 0; bid < images->count; bid++) {
        const float* image = &images->devData[(size_t)bid * nx * ny];
        auto val = [&](int i, int j) { return image[i * ny + j]; };
        const float2 d = direction[bid];
        float best = std::numeric_limits<float>::lowest();
        int2 loc = make_int2(nx / 2, ny / 2);

        const float dmax = std::max(std::abs(d.x), std::abs(d.y));
        if (dmax == 0.0f) {
            for (int i = 0; i < nx * ny; i++)
                if (image[i] > best) { best = image[i]; loc = make_int2(i / ny, i % ny); }
        } else {
            // one pixel step along the dominant axis avoids duplicate pivots
            const float sx = d.x / dmax, sy = d.y / dmax;
            const int nstep = std::max(nx, ny) / 2;
            for (int k = -nstep; k <= nstep; k++) {
                int i = (int)std::lround(nx / 2 + k * sx);
                int j = (int)std::lround(ny / 2 + k * sy);
                if (i < 0 || i >= nx || j < 0 || j >= ny) continue;
                // steepest ascent to a local maximum
                while (true) {
                    int bi = i, bj = j;
                    for (int di = -1; di <= 1; di++)
                        for (int dj = -1; dj <= 1; dj++) {
                            int ii = i + di, jj = j + dj;
                            if (ii >= 0 && ii < nx && jj >= 0 && jj < ny
                                    && val(ii, jj) > val(bi, bj)) { bi = ii; bj = jj; }
                        }
                    if (bi == i && bj == j) break;
                    i = bi; j = bj;
                }
                if (val(i, j) > best) { best = val(i, j); loc = make_int2(i, j); }
            }
        }
        maxval->devData[bid] = best;
        maxloc->devData[bid] = loc;
    }
}

/**
 * Find the correlation peak of each 2D image within a band around the line
 * through the image center (the gross offset) along the flow direction
 * @param[in] images batch of correlation surfaces
 * @param[in] direction flow direction (x: down, y: across) of each image,
 *   its length the band half-width in pixels; (0, 0) falls back to the
 *   global maximum
 * @param[out] maxloc peak locations (x: down, y: across)
 * @param[out] maxval peak values
 */
void cuArraysMaxlocBand(cuArrays<float> *images, const float2 *direction,
                        cuArrays<int2> *maxloc, cuArrays<float> *maxval)
{
    const int nx = images->height, ny = images->width;
    for (int bid = 0; bid < images->count; bid++) {
        const float* image = &images->devData[(size_t)bid * nx * ny];
        const float2 d = direction[bid];
        const float w = std::hypot(d.x, d.y);
        const float ux = w > 0.0f ? d.x / w : 0.0f, uy = w > 0.0f ? d.y / w : 0.0f;
        float best = std::numeric_limits<float>::lowest();
        int2 loc = make_int2(nx / 2, ny / 2);
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++) {
                // distance from the flow line
                if (w > 0.0f && std::abs((i - nx / 2) * uy - (j - ny / 2) * ux) > w) continue;
                if (image[i * ny + j] > best) { best = image[i * ny + j]; loc = make_int2(i, j); }
            }
        maxval->devData[bid] = best;
        maxloc->devData[bid] = loc;
    }
}

/**
 * Determine the final offset value
 * @param[in] offsetInit max location (adjusted to the starting location for extraction) determined from
 *   the cross-correlation before oversampling, in dimensions of pixel
 * @param[in] offsetZoomIn max location from the oversampled cross-correlation surface
 * @param[out] offsetFinal the combined offset value
 * @param[in] OversampleRatioZoomIn the correlation surface oversampling factor
 * @param[in] OversampleRatioRaw the oversampling factor of reference/secondary windows before cross-correlation
 * @param[in] xHalfRangInit the original half search range along x, to be subtracted
 * @param[in] yHalfRangInit the original half search range along y, to be subtracted
 *
 * 1. Cross-correlation is performed at first for the un-oversampled data with a larger search range.
 *   The secondary window is then extracted to a smaller size (a smaller search range) around the max location.
 *   The extraction starting location (offsetInit) - original half search range (xHalfRangeInit, yHalfRangeInit)
 *        = pixel size offset
 * 2. Reference/secondary windows are then oversampled by OversampleRatioRaw, and cross-correlated.
 * 3. The correlation surface is further oversampled by OversampleRatioZoomIn.
 *    The overall oversampling factor is OversampleRatioZoomIn*OversampleRatioRaw.
 *    The max location in oversampled correlation surface (offsetZoomIn) / overall oversampling factor
 *        = subpixel offset
 *    Final offset =  pixel size offset +  subpixel offset
 */
void cuSubPixelOffset(cuArrays<int2> *offsetInit, cuArrays<int2> *offsetZoomIn,
    cuArrays<float2> *offsetFinal,
    int OverSampleRatioZoomin, int OverSampleRatioRaw,
    int xHalfRangeInit,  int yHalfRangeInit)
{
    int size = offsetInit->getSize();
    float OSratio = 1.0f/(float)(OverSampleRatioZoomin*OverSampleRatioRaw);
    float xoffset = xHalfRangeInit ;
    float yoffset = yHalfRangeInit ;

    float2* final = offsetFinal->devData;
    const int2* zoomin = offsetZoomIn->devData;
    const int2* init = offsetInit->devData;

    for (int idx = 0; idx < size; idx++) {
        final[idx].x = OSratio*(zoomin[idx].x ) + init[idx].x  - xoffset;
        final[idx].y = OSratio*(zoomin[idx].y ) + init[idx].y - yoffset;
    }
}

// function to compute the shift of center
static inline int2 adjustOffset(
    const int oldRange, const int newRange, const int maxloc)
{
    // determine the starting point around the maxloc
    // oldRange is the half search window size, e.g., = 32
    // newRange is the half extract size, e.g., = 4
    // maxloc is in range [0, 64]
    // we want to extract \pm 4 centered at maxloc
    // Examples:
    // 1. maxloc = 40: we set start=maxloc-newRange=36, and extract [36,44), shift=0
    // 2. maxloc = 2, start=-2: we set start=0, shift=-2,
    //   (shift means the max is -2 from the extracted center 4)
    // 3. maxloc =64, start=60: set start=56, shift = 4
    //   (shift means the max is 4 from the extracted center 60).

    // shift the max location by -newRange to find the start
    int start = maxloc - newRange;
    // if start is within the range, the max location will be in the center
    int shift = 0;
    // right boundary
    int rbound = 2*(oldRange-newRange);
    if(start<0)     // if exceeding the limit on the left
    {
        // set start at 0 and record the shift of center
        shift = -start;
        start = 0;
    }
    else if(start > rbound ) // if exceeding the limit on the right
    {
        //
        shift = start-rbound;
        start = rbound;
    }
    return make_int2(start, shift);
}

// kernel for cuDetermineSecondaryExtractOffset
void cudaKernel_determineSecondaryExtractOffset(int2 * maxLoc, int2 *shift,
    const int imageIndex, int xOldRange, int yOldRange, int xNewRange, int yNewRange)
{
    // get the starting pixel (stored back to maxloc) and shift
    int2 result = adjustOffset(xOldRange, xNewRange, maxLoc[imageIndex].x);
    maxLoc[imageIndex].x = result.x;
    shift[imageIndex].x = result.y;
    result = adjustOffset(yOldRange, yNewRange, maxLoc[imageIndex].y);
    maxLoc[imageIndex].y = result.x;
    shift[imageIndex].y = result.y;
}

/**
 * Determine the secondary window extract offset from the max location
 * @param[in] xOldRange, yOldRange are (half) search ranges in first step
 * @param[in] xNewRange, yNewRange are (half) search range
 *
 * After the first run of cross-correlation, with a larger search range,
 *  We now choose a smaller search range around the max location for oversampling.
 *  This procedure is used to determine the starting pixel locations for extraction.
 */
void cuDetermineSecondaryExtractOffset(cuArrays<int2> *maxLoc, cuArrays<int2> *maxLocShift,
    int xOldRange, int yOldRange, int xNewRange, int yNewRange)
{
    for (int i = 0; i < maxLoc->size; i++)
        cudaKernel_determineSecondaryExtractOffset(
                maxLoc->devData, maxLocShift->devData,
                i, xOldRange, yOldRange, xNewRange, yNewRange);
}

} // namespace
