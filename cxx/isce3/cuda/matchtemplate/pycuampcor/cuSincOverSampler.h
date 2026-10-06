/*
 * @file cuSincOverSampler.h
 * @brief A class performs sinc interpolation/oversampling
 *
 * Oversample a given 2d signal by i_covs factor.
 * Only signals within(-i_sincwindow, i_sincwindow) are oversampled
 * The interpolation zone may also be shifted, if the max location is not at the center.
 *
 * The sinc interpolation is based on the formula
 *   $$x(t) = \sum_{n=-\infty}^{\infty} x_n f( \Omega_c t-n )$$
 *   with $f(x) = \text{sinc}(x)$, or a complex filter
 *   such as the sinc(x) convoluted with Hamming Window used here.
 *   In practice, a finite length of n (i_intplength) is used for interpolation.
 *
 * @note most parameters are currently hardwired; you need to change
 *    the source code below if you need to adjust the parameters.
 */

// code guard
#ifndef __CUSINCOVERSAMPLER_H
#define __CUSINCOVERSAMPLER_H

// dependencites
#include "cuArrays.h"
#include "cudaUtil.h"

#ifndef PI
#define PI 3.14159265359f
#endif

class cuSincOverSamplerR2R
{
 private:
    static const int i_sincwindow = 2;
    ///< the oversampling is only performed within \pm i_sincwindow*i_covs around the peak
    static const int i_weight = 1;       ///< weight for cos() pedestal
    const float r_pedestal = 0.0f;       ///< height of pedestal
    const float r_beta = 0.75f;          ///< a low-band pass
    const float r_relfiltlen = 6.0f;     ///< relative filter length

    static const int i_decfactor = 4096; ///< max decimals between original grid to set up the sinc kernel

    int i_covs;         ///< oversampling factor
    int i_intplength;   ///< actual filter length = r_relfiltlen/r_beta
    int i_filtercoef;   //< length of the sinc kernel i_intplength*i_decfactor+1

    float * r_filter;   // sinc kernel with size i_filtercoef

    cudaStream_t stream;

    // work arrays of the separable interpolation, grown on demand
    int * d_tapIndex = nullptr;    // input index of each tap, per window coordinate
    float * d_tapCoef = nullptr;   // coefficient of each tap
    float * d_tapSum = nullptr;    // sum of the taps of each window coordinate
    float * d_rows = nullptr;      // input rows interpolated along y
    float * d_window = nullptr;    // oversampled window around the peak
    size_t workCount = 0, workRows = 0;
    void allocateWork(size_t nImages, size_t inNX);
    void freeWork();

 public:
    // constructor
    cuSincOverSamplerR2R(const int i_covs_, cudaStream_t stream_);
    // set up sinc interpolation coefficients
    void cuSetupSincKernel();
    // oversample the images around their peaks and find the max value and
    // location of the (zero elsewhere) outNX x outNY oversampled surfaces
    void executeMaxloc(cuArrays<float> *imagesIn, int outNX, int outNY, cuArrays<int2> *center,
        int oversamplingFactor, cuArrays<int2> *maxloc, cuArrays<float> *maxval);
    // destructor
    ~cuSincOverSamplerR2R();
};

#endif // _CUSINCOVERSAMPLER_H
// end of file