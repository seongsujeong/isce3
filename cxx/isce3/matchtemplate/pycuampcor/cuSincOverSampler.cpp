/**
 * @file cuSincOverSampler.cu
 * @brief Implementation for cuSinOversampler class
 *
 */

// my declaration
#include "cuSincOverSampler.h"

// dependencies
#include "cuArrays.h"
#include "cuAmpcorUtil.h"

#include <vector>

namespace isce3::matchtemplate::pycuampcor {

/**
 * cuSincOverSamplerR2R constructor
 * @param i_covs oversampling factor
 */
cuSincOverSamplerR2R::cuSincOverSamplerR2R(const int i_covs_)
 : i_covs(i_covs_)
{
    i_intplength = int(r_relfiltlen/r_beta+0.5f);
    i_filtercoef = i_intplength*i_decfactor;
    r_filter = (float*) malloc((i_filtercoef+1)*sizeof(float));
    cuSetupSincKernel();
}

/// destructor
cuSincOverSamplerR2R::~cuSincOverSamplerR2R()
{
    free(r_filter);
}

// cuda kernel for cuSetupSincKernel
void cuSetupSincKernel_kernel(float *r_filter_, const int i_filtercoef_,
    const float r_soff_, const float r_wgthgt_, const int i_weight_,
    const float r_soff_inverse_, const float r_beta_, const float r_decfactor_inverse_, int i)
{
    if(i > i_filtercoef_) return;
    float r_wa = i - r_soff_;
    float r_wgt = (1.0f - r_wgthgt_) + r_wgthgt_*cos(M_PI*r_wa*r_soff_inverse_);
    float r_s = r_wa*r_beta_*r_decfactor_inverse_*M_PI;
    float r_fct;
    if(r_s != 0.0f) {
        r_fct = sin(r_s)/r_s;
    }
    else {
        r_fct = 1.0f;
    }
    if(i_weight_ == 1) {
        r_filter_[i] = r_fct*r_wgt;
    }
    else {
        r_filter_[i] = r_fct;
    }
}

/**
 * Set up the sinc interpolation kernel (coefficient)
 */
void cuSincOverSamplerR2R::cuSetupSincKernel()
{
    // compute some commonly used constants at first
    float r_wgthgt =  (1.0f - r_pedestal)/2.0f;
    float r_soff = (i_filtercoef-1.0f)/2.0f;
    float r_soff_inverse = 1.0f/r_soff;
    float r_decfactor_inverse = 1.0f/i_decfactor;

    for (int i = 0; i <= i_filtercoef; i++) {

        cuSetupSincKernel_kernel(
            r_filter, i_filtercoef, r_soff, r_wgthgt, i_weight,
            r_soff_inverse, r_beta, r_decfactor_inverse, i);
    }
}


/**
 * Sinc interpolation taps of one output coordinate along one axis
 * @param[in] out output (oversampled) coordinate
 * @param[in] inN input size along the axis (taps wrap around)
 * @param[out] index input indices of the i_intplength taps
 * @param[out] coef filter coefficients of the taps
 * @return sum of the coefficients
 */
static float sincTaps(int out, int inN, const float *r_filter, int i_covs,
    int i_decfactor, int i_intplength, int *index, float *coef)
{
    // index in input grid: integer part and fraction in kernel grid units
    float r_out = (float)out/i_covs;
    int i_out = int(r_out);
    int i_frac = int((r_out - i_out)*i_decfactor);
    float sum = 0.0f;
    for(int i = 0; i < i_intplength; i++) {
        int in = i_out - i + i_intplength/2;
        if(in < 0) in += inN;
        if(in >= inN) in -= inN;
        index[i] = in;
        coef[i] = r_filter[i*i_decfactor + i_frac];
        sum += coef[i];
    }
    return sum;
}

/**
 * Execute sinc interpolation
 * @param[in] imagesIn input images
 * @param[out] imagesOut output images
 * @param[in] centerShift the shift of interpolation center
 * @param[in] rawOversamplingFactor the multiplier of the centerShift
 * @note rawOversamplingFactor is for the centerShift, not the signal oversampling factor
 *
 * The 2D sinc kernel is the product of 1D kernels along x and y, so it is
 * applied separably: first along y for every input row, then along x.
 */
void cuSincOverSamplerR2R::execute(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut,
    cuArrays<int2> *centerShift, int rawOversamplingFactor)
{
    const int nImages = imagesIn->count;
    const int inNX = imagesIn->height;
    const int inNY = imagesIn->width;
    const int outNX = imagesOut->height;
    const int outNY = imagesOut->width;

    // only compute the overampled signals within a window
    const int i_int_range = i_sincwindow * i_covs;
    // set the start pixel, will be shifted by centerShift*oversamplingFactor (from raw image)
    const int i_int_startX = outNX/2 - i_int_range;
    const int i_int_startY = outNY/2 - i_int_range;
    const int i_int_size = 2*i_int_range + 1;
    const int L = i_intplength;
    // preset all pixels in out image to 0
    imagesOut->setZero();

    std::vector<int> outx(i_int_size), outy(i_int_size);
    std::vector<int> xIndex(i_int_size*L), yIndex(i_int_size*L);
    std::vector<float> xCoef(i_int_size*L), yCoef(i_int_size*L);
    std::vector<float> xSum(i_int_size), ySum(i_int_size);
    std::vector<float> rows(inNX*i_int_size);  // y-interpolated input rows

    for (int idxImage = 0; idxImage < nImages; idxImage++) {
        const int2 shift = centerShift->devData[idxImage];
        const float *in = imagesIn->devData + (size_t)idxImage*inNX*inNY;
        float *out = imagesOut->devData + (size_t)idxImage*outNX*outNY;

        // output coordinates and taps along each axis
        for (int k = 0; k < i_int_size; k++) {
            outx[k] = k + i_int_startX + shift.x*rawOversamplingFactor;
            if (outx[k] >= outNX) outx[k] -= outNX;
            outy[k] = k + i_int_startY + shift.y*rawOversamplingFactor;
            if (outy[k] >= outNY) outy[k] -= outNY;
            xSum[k] = sincTaps(outx[k], inNX, r_filter, i_covs, i_decfactor, L,
                &xIndex[k*L], &xCoef[k*L]);
            ySum[k] = sincTaps(outy[k], inNY, r_filter, i_covs, i_decfactor, L,
                &yIndex[k*L], &yCoef[k*L]);
        }

        // interpolate every input row along y
        for (int r = 0; r < inNX; r++) {
            for (int k = 0; k < i_int_size; k++) {
                float v = 0.0f;
                for (int j = 0; j < L; j++)
                    v += in[r*inNY + yIndex[k*L+j]]*yCoef[k*L+j];
                rows[r*i_int_size + k] = v;
            }
        }

        // interpolate along x and normalize by the total filter weight
        for (int kx = 0; kx < i_int_size; kx++) {
            for (int ky = 0; ky < i_int_size; ky++) {
                float v = 0.0f;
                for (int i = 0; i < L; i++)
                    v += rows[xIndex[kx*L+i]*i_int_size + ky]*xCoef[kx*L+i];
                out[outx[kx]*outNY + outy[ky]] = v/(xSum[kx]*ySum[ky]);
            }
        }
    }
}

} // namespace
