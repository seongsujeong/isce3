/**
 * @file cuSincOverSampler.cu
 * @brief Implementation for cuSinOversampler class
 *
 */

// my declaration
#include "cuSincOverSampler.h"

// dependencies
#include "cuArrays.h"
#include "cudaUtil.h"
#include "cudaError.h"
#include "cuAmpcorUtil.h"

#include <cfloat>

/**
 * cuSincOverSamplerR2R constructor
 * @param i_covs oversampling factor
 * @param stream cuda stream
 */
cuSincOverSamplerR2R::cuSincOverSamplerR2R(const int i_covs_, cudaStream_t stream_)
 : i_covs(i_covs_)
{
    stream = stream_;
    i_intplength = int(r_relfiltlen/r_beta+0.5f);
    i_filtercoef = i_intplength*i_decfactor;
    checkCudaErrors(cudaMalloc((void **)&r_filter, (i_filtercoef+1)*sizeof(float)));
    cuSetupSincKernel();
}

/// destructor
cuSincOverSamplerR2R::~cuSincOverSamplerR2R()
{
    checkCudaErrors(cudaFree(r_filter));
    freeWork();
}

// cuda kernel for cuSetupSincKernel
__global__ void cuSetupSincKernel_kernel(float *r_filter_, const int i_filtercoef_,
    const float r_soff_, const float r_wgthgt_, const int i_weight_,
    const float r_soff_inverse_, const float r_beta_, const float r_decfactor_inverse_)
{
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i > i_filtercoef_) return;
    float r_wa = i - r_soff_;
    float r_wgt = (1.0f - r_wgthgt_) + r_wgthgt_*cos(PI*r_wa*r_soff_inverse_);
    float r_s = r_wa*r_beta_*r_decfactor_inverse_*PI;
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
    const int nthreads = 128;
    const int nblocks = IDIVUP(i_filtercoef+1, nthreads);

    // compute some commonly used constants at first
    float r_wgthgt =  (1.0f - r_pedestal)/2.0f;
    float r_soff = (i_filtercoef-1.0f)/2.0f;
    float r_soff_inverse = 1.0f/r_soff;
    float r_decfactor_inverse = 1.0f/i_decfactor;

    cuSetupSincKernel_kernel<<<nblocks, nthreads, 0, stream>>> (
        r_filter, i_filtercoef, r_soff, r_wgthgt, i_weight,
        r_soff_inverse, r_beta, r_decfactor_inverse);
    getLastCudaError("cuSetupSincKernel_kernel");
}


// output coordinate (wrapped as the original kernel) of window index k
__device__ inline int sincOut(int k, int start, int shift, int factor, int outN)
{
    int o = k + start + shift * factor;
    if (o >= outN) o -= outN;
    return o;
}

// i-th tap of output coordinate `out` along an axis: input index and coefficient
__device__ inline float sincTap(int out, int i, int inN, const float *filter,
    int covs, int decfactor, int intplength, int &in)
{
    const float r_out = (float)out / covs;
    const int i_out = int(r_out);
    const int i_frac = int((r_out - i_out) * decfactor);
    in = i_out - i + intplength / 2;
    if (in < 0) in += inN;
    if (in >= inN) in -= inN;
    return filter[i * decfactor + i_frac];
}

// whether surface coordinate o lies in the oversampled window along an axis
__device__ inline bool sincInWindow(int o, int start, int shift, int factor, int outN, int size)
{
    int k = o - (start + shift * factor);
    if (k < 0) k += outN;
    return k < size;
}

struct SincGeometry {
    int inNX, inNY, outNX, outNY;
    int factor, covs, decfactor, intplength;
    int startX, startY, size;
};

// taps of every window coordinate along x (axis 0) and y (axis 1):
// tap arrays [((image * 2 + axis) * size + k) * intplength + i]
__global__ void cuSincTaps_kernel(const int2 *centerShift, const float *filter,
    int *tapIndex, float *tapCoef, float *tapSum, const SincGeometry g, const int nImages)
{
    const int k = threadIdx.x + blockDim.x * blockIdx.x;
    const int axis = blockIdx.y, img = blockIdx.z;
    if (k >= g.size || img >= nImages) return;
    const int2 shift = centerShift[img];
    const int out = axis == 0 ? sincOut(k, g.startX, shift.x, g.factor, g.outNX)
                              : sincOut(k, g.startY, shift.y, g.factor, g.outNY);
    const int inN = axis == 0 ? g.inNX : g.inNY;
    const size_t base = ((size_t)(img * 2 + axis) * g.size + k) * g.intplength;
    float sum = 0.0f;
    for (int i = 0; i < g.intplength; i++) {
        int in;
        const float c = sincTap(out, i, inN, filter, g.covs, g.decfactor, g.intplength, in);
        tapIndex[base + i] = in;
        tapCoef[base + i] = c;
        sum += c;
    }
    tapSum[(size_t)(img * 2 + axis) * g.size + k] = sum;
}

// pass 1: every input row interpolated along y; rows[image][row][ky]
__global__ void cuSincRows_kernel(const float *in, float *rows, const int *tapIndex,
    const float *tapCoef, const SincGeometry g, const int nImages)
{
    const int ky = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;
    const int img = blockIdx.z;
    if (ky >= g.size || row >= g.inNX || img >= nImages) return;
    const size_t base = ((size_t)(img * 2 + 1) * g.size + ky) * g.intplength;
    const float *line = in + ((size_t)img * g.inNX + row) * g.inNY;
    float v = 0.0f;
    for (int j = 0; j < g.intplength; j++) v += line[tapIndex[base + j]] * tapCoef[base + j];
    rows[((size_t)img * g.inNX + row) * g.size + ky] = v;
}

// pass 2: interpolation along x, normalized by the product of the tap sums;
// window[image][kx][ky]
__global__ void cuSincCols_kernel(const float *rows, float *window, const int *tapIndex,
    const float *tapCoef, const float *tapSum, const SincGeometry g, const int nImages)
{
    const int ky = threadIdx.x + blockDim.x * blockIdx.x;
    const int kx = threadIdx.y + blockDim.y * blockIdx.y;
    const int img = blockIdx.z;
    if (ky >= g.size || kx >= g.size || img >= nImages) return;
    const size_t base = ((size_t)(img * 2) * g.size + kx) * g.intplength;
    const float *r = rows + (size_t)img * g.inNX * g.size + ky;
    float v = 0.0f;
    for (int i = 0; i < g.intplength; i++) v += r[tapIndex[base + i] * g.size] * tapCoef[base + i];
    const float norm = tapSum[(size_t)(img * 2) * g.size + kx] * tapSum[(size_t)(img * 2 + 1) * g.size + ky];
    window[((size_t)img * g.size + kx) * g.size + ky] = v / norm;
}

// Max value and location of the outNX x outNY surface that is the window
// and 0 elsewhere (ties: first row-major index). One block per image.
template <const int BLOCKSIZE>
__global__ void cuSincMaxloc_kernel(const float *window, const int2 *centerShift,
    int2 *maxloc, float *maxval, const SincGeometry g, const int nImages)
{
    __shared__ float vals[BLOCKSIZE];
    __shared__ int idxs[BLOCKSIZE];
    const int img = blockIdx.x, tid = threadIdx.x;
    if (img >= nImages) return;
    const int2 shift = centerShift[img];
    const int n = g.size * g.size;
    const float *w = window + (size_t)img * n;
    float best = -FLT_MAX;
    int bestIdx = g.outNX * g.outNY;
    for (int i = tid; i < n; i += BLOCKSIZE) {
        const int kx = i / g.size, ky = i - kx * g.size;
        const int idx = sincOut(kx, g.startX, shift.x, g.factor, g.outNX) * g.outNY +
                        sincOut(ky, g.startY, shift.y, g.factor, g.outNY);
        const float v = w[i];
        if (v > best || (v == best && idx < bestIdx)) { best = v; bestIdx = idx; }
    }
    vals[tid] = best;
    idxs[tid] = bestIdx;
    __syncthreads();
    for (int s = BLOCKSIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float v = vals[tid + s];
            const int k = idxs[tid + s];
            if (v > vals[tid] || (v == vals[tid] && k < idxs[tid])) { vals[tid] = v; idxs[tid] = k; }
        }
        __syncthreads();
    }
    if (tid == 0) {
        best = vals[0];
        bestIdx = idxs[0];
        if (best <= 0.0f) {
            // first surface index outside the window: (0, 0) if row 0 is
            // outside, else the first column of row 0 outside the window
            int zero = 0;
            if (sincInWindow(0, g.startX, shift.x, g.factor, g.outNX, g.size))
                while (sincInWindow(zero, g.startY, shift.y, g.factor, g.outNY, g.size)) zero++;
            if (best < 0.0f || zero < bestIdx) { best = 0.0f; bestIdx = zero; }
        }
        maxval[img] = best;
        maxloc[img] = make_int2(bestIdx / g.outNY, bestIdx % g.outNY);
    }
}

/// (re)allocate the work arrays for nImages images of inNX rows
void cuSincOverSamplerR2R::allocateWork(size_t nImages, size_t inNX)
{
    if (nImages <= workCount && inNX <= workRows) return;
    freeWork();
    const size_t size = 2 * i_sincwindow * i_covs + 1;
    const size_t taps = nImages * 2 * size * i_intplength;
    checkCudaErrors(cudaMalloc((void **)&d_tapIndex, taps * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_tapCoef, taps * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_tapSum, nImages * 2 * size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_rows, nImages * inNX * size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_window, nImages * size * size * sizeof(float)));
    workCount = nImages;
    workRows = inNX;
}

/// free the work arrays
void cuSincOverSamplerR2R::freeWork()
{
    if (d_tapIndex) checkCudaErrors(cudaFree(d_tapIndex));
    if (d_tapCoef) checkCudaErrors(cudaFree(d_tapCoef));
    if (d_tapSum) checkCudaErrors(cudaFree(d_tapSum));
    if (d_rows) checkCudaErrors(cudaFree(d_rows));
    if (d_window) checkCudaErrors(cudaFree(d_window));
    d_tapIndex = nullptr;
    d_tapCoef = d_tapSum = d_rows = d_window = nullptr;
    workCount = workRows = 0;
}

/**
 * Sinc oversampling around the peaks and the max of the oversampled surfaces
 * @param[in] imagesIn input images
 * @param[in] outNX, outNY size of the oversampled surfaces, which are 0
 *   outside the window of \pm i_sincwindow*i_covs around the shifted center
 * @param[in] centerShift the shift of interpolation center
 * @param[in] rawOversamplingFactor the multiplier of the centerShift
 * @param[out] maxloc, maxval max location and value of each surface
 * @note The 2D sinc kernel is the product of 1D kernels along x and y, so it
 *   is applied separably. Only the window is computed and searched; the
 *   zeros elsewhere enter the max explicitly.
 */
void cuSincOverSamplerR2R::executeMaxloc(cuArrays<float> *imagesIn, int outNX, int outNY,
    cuArrays<int2> *centerShift, int rawOversamplingFactor,
    cuArrays<int2> *maxloc, cuArrays<float> *maxval)
{
    const int nImages = imagesIn->count;
    const int i_int_range = i_sincwindow * i_covs;
    const SincGeometry g{imagesIn->height, imagesIn->width, outNX, outNY,
        rawOversamplingFactor, i_covs, i_decfactor, i_intplength,
        outNX / 2 - i_int_range, outNY / 2 - i_int_range, 2 * i_int_range + 1};
    allocateWork(nImages, g.inNX);

    cuSincTaps_kernel<<<dim3(IDIVUP(g.size, 128), 2, nImages), 128, 0, stream>>>(
        centerShift->devData, r_filter, d_tapIndex, d_tapCoef, d_tapSum, g, nImages);
    getLastCudaError("cuSincTaps_kernel");

    const dim3 threads(NTHREADS2D, NTHREADS2D, 1);
    cuSincRows_kernel<<<dim3(IDIVUP(g.size, NTHREADS2D), IDIVUP(g.inNX, NTHREADS2D), nImages),
        threads, 0, stream>>>(imagesIn->devData, d_rows, d_tapIndex, d_tapCoef, g, nImages);
    getLastCudaError("cuSincRows_kernel");

    cuSincCols_kernel<<<dim3(IDIVUP(g.size, NTHREADS2D), IDIVUP(g.size, NTHREADS2D), nImages),
        threads, 0, stream>>>(d_rows, d_window, d_tapIndex, d_tapCoef, d_tapSum, g, nImages);
    getLastCudaError("cuSincCols_kernel");

    cuSincMaxloc_kernel<256><<<nImages, 256, 0, stream>>>(
        d_window, centerShift->devData, maxloc->devData, maxval->devData, g, nImages);
    getLastCudaError("cuSincMaxloc_kernel");
}

// end of file
