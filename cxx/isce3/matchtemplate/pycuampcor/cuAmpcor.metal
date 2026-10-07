// Metal kernels of the ampcor chunk pipeline (see cuMetal.mm).
// Each kernel mirrors the CPU function named in its comment; arrays are
// batches of row-major images, image index = last grid dimension.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint REDUCE_THREADS = 256;

// threadgroup sum of `v` (all REDUCE_THREADS threads must call it)
template <typename T>
T threadgroupSum(T v, threadgroup T *scratch, uint tid)
{
    scratch[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    T total = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

inline float2 cmul(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// complexMulConj of float2.h: a * conj(b)
inline float2 mulConj(float2 a, float2 b)
{
    return float2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}

kernel void copyWords(device const uint *in [[buffer(0)]],
                      device uint *out [[buffer(1)]],
                      uint i [[thread_position_in_grid]])
{
    out[i] = in[i];
}

// ----------------------------------------------------------------- copies

struct GatherParams { int inNX, inNY, outNX, outNY, absolute; };

// cuArraysCopyToBatch(Abs)WithOffset: windows of a chunk into a batch
kernel void gatherBatch(device const float2 *in [[buffer(0)]],
                        device float2 *out [[buffer(1)]],
                        device const int *offsetX [[buffer(2)]],
                        device const int *offsetY [[buffer(3)]],
                        constant GatherParams &p [[buffer(4)]],
                        uint3 gid [[thread_position_in_grid]])
{
    const int outy = gid.x, outx = gid.y, img = gid.z;
    if (outx >= p.outNX || outy >= p.outNY) return;
    const int inx = offsetX[img] + outx, iny = offsetY[img] + outy;
    float2 v = float2(0.0f);
    if (inx >= 0 && inx < p.inNX && iny >= 0 && iny < p.inNY) {
        v = in[inx * p.inNY + iny];
        if (p.absolute) v = float2(sqrt(v.x * v.x + v.y * v.y), 0.0f);
    }
    out[(img * p.outNX + outx) * p.outNY + outy] = v;
}

// cuArraysAbs
kernel void complexAbs(device const float2 *in [[buffer(0)]],
                       device float *out [[buffer(1)]],
                       uint i [[thread_position_in_grid]])
{
    const float2 v = in[i];
    out[i] = sqrt(v.x * v.x + v.y * v.y);
}

struct Shape2 { int inNX, inNY, outNX, outNY, offsetX, offsetY; };

// cuArraysCopyPadded (real -> complex, zero padded)
kernel void padRealToComplex(device const float *in [[buffer(0)]],
                             device float2 *out [[buffer(1)]],
                             constant Shape2 &p [[buffer(2)]],
                             uint3 gid [[thread_position_in_grid]])
{
    const int j = gid.x, i = gid.y, img = gid.z;
    if (i >= p.outNX || j >= p.outNY) return;
    float v = 0.0f;
    if (i < p.inNX && j < p.inNY) v = in[(img * p.inNX + i) * p.inNY + j];
    out[(img * p.outNX + i) * p.outNY + j] = float2(v, 0.0f);
}

// cuArraysCopyExtract (complex -> real part, fixed offset)
kernel void extractReal(device const float2 *in [[buffer(0)]],
                        device float *out [[buffer(1)]],
                        constant Shape2 &p [[buffer(2)]],
                        uint3 gid [[thread_position_in_grid]])
{
    const int y = gid.x, x = gid.y, img = gid.z;
    if (x >= p.outNX || y >= p.outNY) return;
    out[(img * p.outNX + x) * p.outNY + y] =
        in[(img * p.inNX + x + p.offsetX) * p.inNY + y + p.offsetY].x;
}

// cuArraysCopyExtract (real, fixed offset)
kernel void extractFloat(device const float *in [[buffer(0)]],
                         device float *out [[buffer(1)]],
                         constant Shape2 &p [[buffer(2)]],
                         uint3 gid [[thread_position_in_grid]])
{
    const int y = gid.x, x = gid.y, img = gid.z;
    if (x >= p.outNX || y >= p.outNY) return;
    out[(img * p.outNX + x) * p.outNY + y] =
        in[(img * p.inNX + x + p.offsetX) * p.inNY + y + p.offsetY];
}

// cuArraysCopyExtract (complex, per-image offsets)
kernel void extractComplexOffsets(device const float2 *in [[buffer(0)]],
                                  device float2 *out [[buffer(1)]],
                                  device const int2 *offsets [[buffer(2)]],
                                  constant Shape2 &p [[buffer(3)]],
                                  uint3 gid [[thread_position_in_grid]])
{
    const int y = gid.x, x = gid.y, img = gid.z;
    if (x >= p.outNX || y >= p.outNY) return;
    const int2 o = offsets[img];
    out[(img * p.outNX + x) * p.outNY + y] =
        in[(img * p.inNX + x + o.x) * p.inNY + y + o.y];
}

// cuArraysCopyExtractCorr: correlation around the peak with valid flags
kernel void extractCorr(device const float *in [[buffer(0)]],
                        device float *out [[buffer(1)]],
                        device int *valid [[buffer(2)]],
                        device const int2 *maxloc [[buffer(3)]],
                        constant Shape2 &p [[buffer(4)]],
                        uint3 gid [[thread_position_in_grid]])
{
    const int outy = gid.x, outx = gid.y, img = gid.z;
    if (outx >= p.outNX || outy >= p.outNY) return;
    const int inx = outx + maxloc[img].x - p.outNX / 2;
    const int iny = outy + maxloc[img].y - p.outNY / 2;
    const int idxOut = (img * p.outNX + outx) * p.outNY + outy;
    if (inx >= 0 && iny >= 0 && inx < p.inNX && iny < p.inNY) {
        out[idxOut] = in[(img * p.inNX + inx) * p.inNY + iny];
        valid[idxOut] = 1;
    } else {
        out[idxOut] = 0.0f;
        valid[idxOut] = 0;
    }
}

// source index along an axis of output index o of cuArraysPaddingMany:
// the first and last inN/2 outputs take the first and last inN/2 inputs
inline int padSource(int o, int inN, int outN)
{
    const int h = inN / 2;
    if (o < h) return o;
    if (o >= outN - h) return inN - outN + o;
    return -1;
}

// cuArraysPaddingMany: spectrum of in (inNX x inNY) into the corners of out,
// zeros elsewhere (one thread per output element)
kernel void padSpectrum(device const float2 *in [[buffer(0)]],
                        device float2 *out [[buffer(1)]],
                        constant Shape2 &p [[buffer(2)]],
                        uint3 gid [[thread_position_in_grid]])
{
    const int j = gid.x, i = gid.y, img = gid.z;
    if (i >= p.outNX || j >= p.outNY) return;
    const int si = padSource(i, p.inNX, p.outNX), sj = padSource(j, p.inNY, p.outNY);
    float2 v = float2(0.0f);
    if (si >= 0 && sj >= 0)
        v = in[((size_t)img * p.inNX + si) * p.inNY + sj] * (1.0f / (p.inNX * p.inNY));
    out[((size_t)img * p.outNX + i) * p.outNY + j] = v;
}

struct InsertParams { int inNX, inNY, outNY, offsetX, offsetY, elemWords; };

// cuArraysCopyInsert of a chunk result into the full run image;
// elements are copied as elemWords 32-bit words
kernel void insertChunk(device const uint *in [[buffer(0)]],
                        device uint *out [[buffer(1)]],
                        constant InsertParams &p [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]])
{
    const int y = gid.x, x = gid.y;
    if (x >= p.inNX || y >= p.inNY) return;
    const int idxIn = x * p.inNY + y;
    const int idxOut = (x + p.offsetX) * p.outNY + y + p.offsetY;
    for (int w = 0; w < p.elemWords; w++)
        out[idxOut * p.elemWords + w] = in[idxIn * p.elemWords + w];
}

// -------------------------------------------------------------- statistics

// cuArraysSubtractMean (one threadgroup per image)
kernel void subtractMean(device float *images [[buffer(0)]],
                         constant int &imageSize [[buffer(1)]],
                         uint img [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[REDUCE_THREADS];
    device float *image = images + (size_t)img * imageSize;
    float sum = 0.0f;
    for (int i = tid; i < imageSize; i += REDUCE_THREADS) sum += image[i];
    const float mean = threadgroupSum(sum, scratch, tid) * (1.0f / imageSize);
    for (int i = tid; i < imageSize; i += REDUCE_THREADS) image[i] -= mean;
}

// sum_square_kernel (one threadgroup per image)
kernel void sumSquare(device const float *images [[buffer(0)]],
                      device float *sum2 [[buffer(1)]],
                      constant int &imageSize [[buffer(2)]],
                      uint img [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[REDUCE_THREADS];
    device const float *image = images + (size_t)img * imageSize;
    float s = 0.0f;
    for (int i = tid; i < imageSize; i += REDUCE_THREADS) s += image[i] * image[i];
    s = threadgroupSum(s, scratch, tid);
    if (tid == 0) sum2[img] = s;
}

// cuArraysSumCorr (one threadgroup per image)
kernel void sumCorr(device const float *images [[buffer(0)]],
                    device const int *valid [[buffer(1)]],
                    device float *sum [[buffer(2)]],
                    device int *count [[buffer(3)]],
                    constant int &imageSize [[buffer(4)]],
                    uint img [[threadgroup_position_in_grid]],
                    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float fscratch[REDUCE_THREADS];
    threadgroup int iscratch[REDUCE_THREADS];
    device const float *image = images + (size_t)img * imageSize;
    device const int *v = valid + (size_t)img * imageSize;
    float s = 0.0f;
    int c = 0;
    for (int i = tid; i < imageSize; i += REDUCE_THREADS) {
        s += image[i] * image[i];
        c += v[i];
    }
    s = threadgroupSum(s, fscratch, tid);
    c = threadgroupSum(c, iscratch, tid);
    if (tid == 0) { sum[img] = s; count[img] = c; }
}

// cuEstimateSnr
kernel void estimateSnr(device const float *corrSum [[buffer(0)]],
                        device const int *validCount [[buffer(1)]],
                        device const float *maxval [[buffer(2)]],
                        device float *snr [[buffer(3)]],
                        uint i [[thread_position_in_grid]])
{
    const float maxvalsq = maxval[i] * maxval[i];
    const float mean = (corrSum[i] - maxvalsq) / (validCount[i] - 1);
    snr[i] = maxvalsq / mean;
}

struct VarParams { int NX, NY, templateSize; };

// cudaKernel_estimateVar
kernel void estimateVariance(device const float *corr [[buffer(0)]],
                             device const int2 *maxloc [[buffer(1)]],
                             device const float *maxval [[buffer(2)]],
                             device packed_float3 *cov [[buffer(3)]],
                             constant VarParams &p [[buffer(4)]],
                             uint img [[thread_position_in_grid]])
{
    const int px = maxloc[img].x, py = maxloc[img].y;
    const float peak = maxval[img];
    if (px - 1 < 0 || py - 1 < 0 || px + 1 >= p.NX || py + 1 >= p.NY) {
        cov[img] = packed_float3(99.0f, 99.0f, 0.0f);
        return;
    }
    device const float *c = corr + (size_t)p.NX * p.NY * img;
#define AT(x, y) c[(x) * p.NY + (y)]
    float dxx = -(AT(px + 1, py) + AT(px - 1, py) - 2.0f * AT(px, py));
    float dyy = -(AT(px, py + 1) + AT(px, py - 1) - 2.0f * AT(px, py));
    float dxy = (AT(px + 1, py + 1) + AT(px - 1, py - 1) - AT(px + 1, py - 1) -
                 AT(px - 1, py + 1)) * 0.25f;
#undef AT
    float n2 = fmax(1.0f - peak, 0.0f);
    dxx *= p.templateSize; dyy *= p.templateSize; dxy *= p.templateSize;
    float n4 = n2 * n2;
    n2 = n2 * 2;
    n4 = n4 * 0.5f * p.templateSize;
    const float u = dxy * dxy - dxx * dyy;
    const float u2 = u * u;
    if (fabs(u) < 1e-2f) {
        cov[img] = packed_float3(99.0f, 99.0f, 0.0f);
    } else {
        cov[img] = packed_float3(
            (-n2 * u * dyy + n4 * (dyy * dyy + dxy * dxy)) / u2,
            (-n2 * u * dxx + n4 * (dxx * dxx + dxy * dxy)) / u2,
            (n2 * u * dxy - n4 * (dxx + dyy) * dxy) / u2);
    }
}

// ---------------------------------------------------------- normalization

struct SatParams { int nx, ny; };

// sat2d_kernel, first pass: running sums of value and value^2 along rows;
// one SIMD group (32 threads) per row, prefix sums over 32-element blocks
kernel void satRows(device const float *data [[buffer(0)]],
                    device float *sat [[buffer(1)]],
                    device float *sat2 [[buffer(2)]],
                    constant SatParams &p [[buffer(3)]],
                    constant int &count [[buffer(4)]],
                    uint row [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]])
{
    if ((int)row >= p.nx * count) return;
    const size_t base = (size_t)row * p.ny;
    float carry = 0.0f, carry2 = 0.0f;
    for (int i0 = 0; i0 < p.ny; i0 += 32) {
        const int i = i0 + lane;
        const float val = i < p.ny ? data[base + i] : 0.0f;
        const float sum = carry + simd_prefix_inclusive_sum(val);
        const float sum2 = carry2 + simd_prefix_inclusive_sum(val * val);
        if (i < p.ny) { sat[base + i] = sum; sat2[base + i] = sum2; }
        carry = simd_broadcast(sum, 31);
        carry2 = simd_broadcast(sum2, 31);
    }
}

// sat2d_kernel, second pass: running sums along columns
kernel void satCols(device float *sat [[buffer(0)]],
                    device float *sat2 [[buffer(1)]],
                    constant SatParams &p [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]])
{
    const int col = gid.x, img = gid.y;
    if (col >= p.ny) return;
    int index = col + img * p.nx * p.ny;
    float sum = sat[index], sum2 = sat2[index];
    for (int i = 1; i < p.nx; i++) {
        index += p.ny;
        sum += sat[index]; sat[index] = sum;
        sum2 += sat2[index]; sat2[index] = sum2;
    }
}

struct NormParams { int corNX, corNY, refNX, refNY, secNX, secNY; };

// sum of a summed-area table s over the reference-sized box at (tx, ty)
inline float boxSum(device const float *s, int tx, int ty, constant NormParams &p)
{
    const float topleft = (tx > 0 && ty > 0) ? s[(tx - 1) * p.secNY + (ty - 1)] : 0.0f;
    const float topright = (tx > 0) ? s[(tx - 1) * p.secNY + (ty + p.refNY - 1)] : 0.0f;
    const float bottomleft = (ty > 0) ? s[(tx + p.refNX - 1) * p.secNY + (ty - 1)] : 0.0f;
    const float bottomright = s[(tx + p.refNX - 1) * p.secNY + (ty + p.refNY - 1)];
    return bottomright + topleft - topright - bottomleft;
}

// cuCorrNormalizeSAT_kernel
kernel void normalizeSat(device float *corr [[buffer(0)]],
                         device const float *refSum2 [[buffer(1)]],
                         device const float *satBuf [[buffer(2)]],
                         device const float *sat2Buf [[buffer(3)]],
                         constant NormParams &p [[buffer(4)]],
                         uint3 gid [[thread_position_in_grid]])
{
    const int ty = gid.x, tx = gid.y, img = gid.z;
    if (tx >= p.corNX || ty >= p.corNY) return;
    device const float *sat = satBuf + img * p.secNX * p.secNY;
    device const float *sat2 = sat2Buf + img * p.secNX * p.secNY;
    const float secondarySum = boxSum(sat, tx, ty, p);
    const float secondarySum2 = boxSum(sat2, tx, ty, p);
    const float norm2 = (secondarySum2 - secondarySum * secondarySum / (p.refNX * p.refNY)) * refSum2[img];
    corr[(img * p.corNX + tx) * p.corNY + ty] *= 1 / sqrt(norm2 + FLT_EPSILON);
}

// ------------------------------------------------------------ correlation

struct PackParams { int tNX, tNY, iNX, iNY, outNX, outNY; };

// template + i * image, both zero padded to the FFT size, for one FFT of
// both real inputs
kernel void packRealPair(device const float *templates [[buffer(0)]],
                         device const float *images [[buffer(1)]],
                         device float2 *out [[buffer(2)]],
                         constant PackParams &p [[buffer(3)]],
                         uint3 gid [[thread_position_in_grid]])
{
    const int j = gid.x, i = gid.y, img = gid.z;
    if (i >= p.outNX || j >= p.outNY) return;
    const float t = (i < p.tNX && j < p.tNY) ? templates[(img * p.tNX + i) * p.tNY + j] : 0.0f;
    const float v = (i < p.iNX && j < p.iNY) ? images[(img * p.iNX + i) * p.iNY + j] : 0.0f;
    out[((size_t)img * p.outNX + i) * p.outNY + j] = float2(t, v);
}

// From Z = FFT(t + i s): T = (Z_k + conj Z_-k) / 2, S = (Z_k - conj Z_-k) / 2i;
// writes conj(T) S * coef (cuMulConj) for the inverse transform
kernel void mulConjPacked(device const float2 *z [[buffer(0)]],
                          device float2 *out [[buffer(1)]],
                          constant Shape2 &p [[buffer(2)]],
                          constant float &coef [[buffer(3)]],
                          uint3 gid [[thread_position_in_grid]])
{
    const int j = gid.x, i = gid.y, img = gid.z;
    if (i >= p.outNX || j >= p.outNY) return;
    device const float2 *zi = z + (size_t)img * p.outNX * p.outNY;
    const float2 a = zi[i * p.outNY + j];
    const int ni = i == 0 ? 0 : p.outNX - i, nj = j == 0 ? 0 : p.outNY - j;
    const float2 b = zi[ni * p.outNY + nj];  // Z_-k; conj below
    const float2 t = float2(a.x + b.x, a.y - b.y) * 0.5f;
    const float2 s = float2(a.y + b.y, b.x - a.x) * 0.5f;
    out[(size_t)img * p.outNX * p.outNY + i * p.outNY + j] =
        float2(t.x * s.x + t.y * s.y, -t.y * s.x + t.x * s.y) * coef;
}

// cuMulConj (template spectrum a, image spectrum b) scaled
kernel void mulConjScale(device float2 *a [[buffer(0)]],
                         device const float2 *b [[buffer(1)]],
                         constant float &coef [[buffer(2)]],
                         uint i [[thread_position_in_grid]])
{
    const float2 x = a[i], y = b[i];
    a[i] = float2(x.x * y.x + x.y * y.y, -x.y * y.x + x.x * y.y) * coef;
}

struct TimeCorrParams { int tNX, tNY, iNX, iNY, rNX, rNY; };

// cuCorrTimeDomain
kernel void corrTimeDomain(device const float *templates [[buffer(0)]],
                           device const float *images [[buffer(1)]],
                           device float *results [[buffer(2)]],
                           constant TimeCorrParams &p [[buffer(3)]],
                           uint3 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y, i = gid.z;
    if (y >= p.rNX || x >= p.rNY) return;
    device const float *image = images + i * p.iNX * p.iNY;
    device const float *templ = templates + i * p.tNX * p.tNY;
    float pixel = 0;
    for (int y0 = 0; y0 < p.tNX; y0++)
        for (int x0 = 0; x0 < p.tNY; x0++)
            pixel += templ[y0 * p.tNY + x0] * image[(y + y0) * p.iNY + (x + x0)];
    results[(i * p.rNX + y) * p.rNY + x] = pixel;
}

// --------------------------------------------------------------------- FFT

struct FFTParams {
    int n;              // transform length
    int nradix;         // number of Stockham stages
    int sign;           // -1 forward (FFTW_FORWARD), +1 backward
    int linesPerImage;  // lines of one image along the transform axis
    int imageSize;      // elements per image
    int lineStride;     // element distance between lines
    int elemStride;     // element distance within a line
    int lines;          // total lines
    int group;          // lines per threadgroup
};

// v * w^{sign}, w = (cos, sin)
inline float2 twiddle(float2 v, float2 w, int sign)
{
    return cmul(v, float2(w.x, sign * w.y));
}

// v * (i * sign)
inline float2 mulI(float2 v, int sign)
{
    return float2(-sign * v.y, sign * v.x);
}

// Radix-R butterfly (R odd prime): stage twiddles, then an R-point DFT;
// R is a template parameter so that v stays in registers
template <int R>
inline void butterfly(threadgroup const float2 *x, threadgroup float2 *y, int j, int m,
                      int k, int step, int out, int ns, int n, device const float2 *tw,
                      int sign)
{
    float2 v[R];
    v[0] = x[j];
    for (int rr = 1; rr < R; rr++)
        v[rr] = twiddle(x[j + rr * m], tw[rr * k * step], sign);  // < n since k < ns
    const int rootStep = n / R;  // tw index of exp(2 pi i / R)
    for (int q = 0; q < R; q++) {
        float2 acc = v[0];
        int e = 0;
        for (int rr = 1; rr < R; rr++) {
            e += q;
            if (e >= R) e -= R;
            acc += twiddle(v[rr], tw[e * rootStep], sign);
        }
        y[out + q * ns] = acc;
    }
}

// Unnormalized 1D DFTs of p.group lines per threadgroup, mixed-radix
// Stockham autosort in threadgroup memory (2 * n * group complex).
// tw[k] = (cos, sin)(2 pi k / n). Lines adjacent in memory (columns) are
// loaded together for coalesced access.
kernel void fft1d(device float2 *data [[buffer(0)]],
                  device const float2 *tw [[buffer(1)]],
                  device const int *radix [[buffer(2)]],
                  constant FFTParams &p [[buffer(3)]],
                  threadgroup float2 *shared [[threadgroup(0)]],
                  uint groupIdx [[threadgroup_position_in_grid]],
                  uint tid [[thread_position_in_threadgroup]],
                  uint nt [[threads_per_threadgroup]])
{
    const int n = p.n, C = min(p.group, p.lines - (int)groupIdx * p.group);
    const int line0 = groupIdx * p.group;
    threadgroup float2 *a = shared, *b = shared + n * p.group;
    // element i of local line c is at a[c * n + i]
    const bool contiguous = p.elemStride == 1;
    for (int idx = tid; idx < n * C; idx += nt) {
        const int c = contiguous ? idx / n : idx % C;
        const int i = contiguous ? idx % n : idx / C;
        const int line = line0 + c;
        a[c * n + i] = data[(size_t)(line / p.linesPerImage) * p.imageSize +
                            (line % p.linesPerImage) * p.lineStride + i * p.elemStride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int ns = 1;
    for (int s = 0; s < p.nradix; s++) {
        const int r = radix[s], m = n / r, step = n / (ns * r);
        for (int t = tid; t < C * m; t += nt) {
            // two integer divisions per butterfly (they are slow on the GPU)
            const int c = t / m, j = t - c * m;
            const int jq = j / ns, k = j - jq * ns;
            threadgroup float2 *x = a + c * n, *y = b + c * n;
            const int out = jq * ns * r + k;
            if (r == 4) {
                const float2 v0 = x[j];
                const float2 v1 = twiddle(x[j + m], tw[k * step], p.sign);
                const float2 v2 = twiddle(x[j + 2 * m], tw[2 * k * step], p.sign);
                const float2 v3 = twiddle(x[j + 3 * m], tw[3 * k * step], p.sign);
                const float2 s02 = v0 + v2, d02 = v0 - v2, s13 = v1 + v3;
                const float2 d13 = mulI(v1 - v3, p.sign);
                y[out] = s02 + s13;
                y[out + ns] = d02 + d13;
                y[out + 2 * ns] = s02 - s13;
                y[out + 3 * ns] = d02 - d13;
            } else if (r == 2) {
                const float2 v0 = x[j];
                const float2 v1 = twiddle(x[j + m], tw[k * step], p.sign);
                y[out] = v0 + v1;
                y[out + ns] = v0 - v1;
            } else {
#define BUTTERFLY(R) case R: butterfly<R>(x, y, j, m, k, step, out, ns, n, tw, p.sign); break;
                switch (r) {
                    BUTTERFLY(3) BUTTERFLY(5) BUTTERFLY(7) BUTTERFLY(11) BUTTERFLY(13)
                    BUTTERFLY(17) BUTTERFLY(19) BUTTERFLY(23) BUTTERFLY(29) BUTTERFLY(31)
                }
#undef BUTTERFLY
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float2 *t = a; a = b; b = t;
        ns *= r;
    }
    for (int idx = tid; idx < n * C; idx += nt) {
        const int c = contiguous ? idx / n : idx % C;
        const int i = contiguous ? idx % n : idx / C;
        const int line = line0 + c;
        data[(size_t)(line / p.linesPerImage) * p.imageSize +
             (line % p.linesPerImage) * p.lineStride + i * p.elemStride] = a[c * n + i];
    }
}

// ---------------------------------------------------------------- deramp

struct DerampParams { int nx, ny, axis; };

// cuLinearDeramp_kernel (one threadgroup per image)
kernel void deramp(device float2 *images [[buffer(0)]],
                   constant DerampParams &p [[buffer(1)]],
                   uint img [[threadgroup_position_in_grid]],
                   uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float2 scratch[REDUCE_THREADS];
    device float2 *image = images + (size_t)img * p.nx * p.ny;
    float phaseY = 0.0f;
    if (p.axis != 0) {
        float2 d = float2(0.0f);
        for (int i = tid; i < p.nx * (p.ny - 1); i += REDUCE_THREADS) {
            const int row = i / (p.ny - 1), col = i % (p.ny - 1);
            const int idx = row * p.ny + col;
            d += mulConj(image[idx], image[idx + 1]);
        }
        d = threadgroupSum(d, scratch, tid);
        phaseY = atan2(d.y, d.x);
    }
    float phaseX = 0.0f;
    if (p.axis != 1) {
        float2 d = float2(0.0f);
        for (int i = tid; i < (p.nx - 1) * p.ny; i += REDUCE_THREADS)
            d += mulConj(image[i], image[i + p.ny]);
        d = threadgroupSum(d, scratch, tid);
        phaseX = atan2(d.y, d.x);
    }
    for (int i = tid; i < p.nx * p.ny; i += REDUCE_THREADS) {
        const float phase = (i / p.ny) * phaseX + (i % p.ny) * phaseY;
        const float c = cos(phase), s = sin(phase);
        const float2 v = image[i];
        image[i] = float2(v.x * c - v.y * s, v.x * s + v.y * c);
    }
}

// ------------------------------------------------------------------ peaks

// cuArraysMaxloc2D (one threadgroup per image; ties -> first row-major index)
kernel void maxloc2D(device const float *images [[buffer(0)]],
                     device int2 *maxloc [[buffer(1)]],
                     device float *maxval [[buffer(2)]],
                     constant int2 &shape [[buffer(3)]],
                     uint img [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float vals[REDUCE_THREADS];
    threadgroup int idxs[REDUCE_THREADS];
    const int n = shape.x * shape.y;
    device const float *data = images + (size_t)img * n;
    float best = -INFINITY;
    int bestIdx = n;
    for (int i = tid; i < n; i += REDUCE_THREADS) {
        const float v = data[i];
        if (v > best) { best = v; bestIdx = i; }
    }
    vals[tid] = best; idxs[tid] = bestIdx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float v = vals[tid + s];
            const int k = idxs[tid + s];
            if (v > vals[tid] || (v == vals[tid] && k < idxs[tid])) {
                vals[tid] = v; idxs[tid] = k;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        maxval[img] = vals[0];
        maxloc[img] = int2(idxs[0] / shape.y, idxs[0] % shape.y);
    }
}

// cuArraysMaxlocDLC (one thread per image): steepest ascent from pivots on
// the flow line through the image center; (0, 0) direction -> global max
kernel void maxlocDLC(device const float *images [[buffer(0)]],
                      device const float2 *direction [[buffer(1)]],
                      device int2 *maxloc [[buffer(2)]],
                      device float *maxval [[buffer(3)]],
                      constant int2 &shape [[buffer(4)]],
                      constant int &count [[buffer(5)]],
                      uint img [[thread_position_in_grid]])
{
    if ((int)img >= count) return;
    const int nx = shape.x, ny = shape.y;
    device const float *image = images + (size_t)img * nx * ny;
    const float2 d = direction[img];
    float best = -INFINITY;
    int2 loc = int2(nx / 2, ny / 2);
    const float dmax = max(fabs(d.x), fabs(d.y));
    if (dmax == 0.0f) {
        for (int i = 0; i < nx * ny; i++)
            if (image[i] > best) { best = image[i]; loc = int2(i / ny, i % ny); }
    } else {
        const float sx = d.x / dmax, sy = d.y / dmax;
        const int nstep = max(nx, ny) / 2;
        for (int k = -nstep; k <= nstep; k++) {
            int i = (int)rint(nx / 2 + k * sx);
            int j = (int)rint(ny / 2 + k * sy);
            if (i < 0 || i >= nx || j < 0 || j >= ny) continue;
            while (true) {
                int bi = i, bj = j;
                for (int di = -1; di <= 1; di++)
                    for (int dj = -1; dj <= 1; dj++) {
                        const int ii = i + di, jj = j + dj;
                        if (ii >= 0 && ii < nx && jj >= 0 && jj < ny &&
                            image[ii * ny + jj] > image[bi * ny + bj]) { bi = ii; bj = jj; }
                    }
                if (bi == i && bj == j) break;
                i = bi; j = bj;
            }
            if (image[i * ny + j] > best) { best = image[i * ny + j]; loc = int2(i, j); }
        }
    }
    maxval[img] = best;
    maxloc[img] = loc;
}

// cuArraysMaxlocBand (one thread per image): max within |direction| pixels
// of the flow line through the image center; (0, 0) direction -> global max
kernel void maxlocBand(device const float *images [[buffer(0)]],
                       device const float2 *direction [[buffer(1)]],
                       device int2 *maxloc [[buffer(2)]],
                       device float *maxval [[buffer(3)]],
                       constant int2 &shape [[buffer(4)]],
                       constant int &count [[buffer(5)]],
                       uint img [[thread_position_in_grid]])
{
    if ((int)img >= count) return;
    const int nx = shape.x, ny = shape.y;
    device const float *image = images + (size_t)img * nx * ny;
    const float2 d = direction[img];
    const float w = length(d);
    const float2 u = w > 0.0f ? d / w : float2(0.0f);
    float best = -INFINITY;
    int2 loc = int2(nx / 2, ny / 2);
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            if (w > 0.0f && fabs((i - nx / 2) * u.y - (j - ny / 2) * u.x) > w) continue;
            if (image[i * ny + j] > best) { best = image[i * ny + j]; loc = int2(i, j); }
        }
    maxval[img] = best;
    maxloc[img] = loc;
}

struct ExtractOffsetParams { int xOldRange, yOldRange, xNewRange, yNewRange, count; };

// adjustOffset of cuOffset.cpp
inline int2 adjustOffset(int oldRange, int newRange, int maxloc)
{
    int start = maxloc - newRange;
    int shift = 0;
    const int rbound = 2 * (oldRange - newRange);
    if (start < 0) { shift = -start; start = 0; }
    else if (start > rbound) { shift = start - rbound; start = rbound; }
    return int2(start, shift);
}

// cuDetermineSecondaryExtractOffset
kernel void secondaryExtractOffset(device int2 *maxLoc [[buffer(0)]],
                                   device int2 *shift [[buffer(1)]],
                                   constant ExtractOffsetParams &p [[buffer(2)]],
                                   uint i [[thread_position_in_grid]])
{
    if ((int)i >= p.count) return;
    const int2 rx = adjustOffset(p.xOldRange, p.xNewRange, maxLoc[i].x);
    const int2 ry = adjustOffset(p.yOldRange, p.yNewRange, maxLoc[i].y);
    maxLoc[i] = int2(rx.x, ry.x);
    shift[i] = int2(rx.y, ry.y);
}

struct SubPixelParams { int ovsZoomIn, ovsRaw, xHalfRange, yHalfRange, count; };

// cuSubPixelOffset
kernel void subPixelOffset(device const int2 *offsetInit [[buffer(0)]],
                           device const int2 *offsetZoomIn [[buffer(1)]],
                           device float2 *offsetFinal [[buffer(2)]],
                           constant SubPixelParams &p [[buffer(3)]],
                           uint i [[thread_position_in_grid]])
{
    if ((int)i >= p.count) return;
    const float ratio = 1.0f / (float)(p.ovsZoomIn * p.ovsRaw);
    offsetFinal[i] = float2(ratio * offsetZoomIn[i].x + offsetInit[i].x - (float)p.xHalfRange,
                            ratio * offsetZoomIn[i].y + offsetInit[i].y - (float)p.yHalfRange);
}

// ------------------------------------------------------- sinc oversampler

struct SincParams {
    int inNX, inNY, outNX, outNY;
    int factor, covs, decfactor, intplength;
    int startX, startY, size;
};

// Output coordinate (wrapped as the CPU) of oversampled index k along an axis
inline int sincOut(int k, int start, int shift, int factor, int outN)
{
    int o = k + start + shift * factor;
    if (o >= outN) o -= outN;
    return o;
}

// i-th sinc tap of output coordinate `out`: input index and coefficient
inline float sincTap(int out, int i, int inN, device const float *filter,
                     constant SincParams &p, thread int &in)
{
    const float r_out = (float)out / p.covs;
    const int i_out = int(r_out);
    const int i_frac = int((r_out - i_out) * p.decfactor);
    in = i_out - i + p.intplength / 2;
    if (in < 0) in += inN;
    if (in >= inN) in -= inN;
    return filter[i * p.decfactor + i_frac];
}

// Sinc taps of every oversampled index along x (axis 0) and y (axis 1):
// input index and coefficient of each of the intplength taps, and their sum.
// taps[((img * 2 + axis) * size + k) * intplength + i]
kernel void sincTaps(device const int2 *centerShift [[buffer(0)]],
                     device const float *filter [[buffer(1)]],
                     device int *tapIndex [[buffer(2)]],
                     device float *tapCoef [[buffer(3)]],
                     device float *tapSum [[buffer(4)]],
                     constant SincParams &p [[buffer(5)]],
                     uint3 gid [[thread_position_in_grid]])
{
    const int k = gid.x, axis = gid.y, img = gid.z;
    if (k >= p.size) return;
    const int2 shift = centerShift[img];
    const int out = axis == 0 ? sincOut(k, p.startX, shift.x, p.factor, p.outNX)
                              : sincOut(k, p.startY, shift.y, p.factor, p.outNY);
    const int inN = axis == 0 ? p.inNX : p.inNY;
    const size_t base = ((size_t)(img * 2 + axis) * p.size + k) * p.intplength;
    float sum = 0.0f;
    for (int i = 0; i < p.intplength; i++) {
        int in;
        const float c = sincTap(out, i, inN, filter, p, in);
        tapIndex[base + i] = in;
        tapCoef[base + i] = c;
        sum += c;
    }
    tapSum[(img * 2 + axis) * p.size + k] = sum;
}

// Separable sinc oversampling (cuSincOverSamplerR2R::execute), pass 1:
// every input row interpolated along y; rows[img][row][ky]
kernel void sincRows(device const float *in [[buffer(0)]],
                     device float *rows [[buffer(1)]],
                     device const int *tapIndex [[buffer(2)]],
                     device const float *tapCoef [[buffer(3)]],
                     constant SincParams &p [[buffer(4)]],
                     uint3 gid [[thread_position_in_grid]])
{
    const int ky = gid.x, row = gid.y, img = gid.z;
    if (ky >= p.size || row >= p.inNX) return;
    const size_t base = ((size_t)(img * 2 + 1) * p.size + ky) * p.intplength;
    device const float *line = in + ((size_t)img * p.inNX + row) * p.inNY;
    float v = 0.0f;
    for (int j = 0; j < p.intplength; j++) v += line[tapIndex[base + j]] * tapCoef[base + j];
    rows[((size_t)img * p.inNX + row) * p.size + ky] = v;
}

// pass 2: interpolation along x, normalized by the product of tap sums;
// compact output: only the window around the peak (rest of surface is 0)
kernel void sincCols(device const float *rows [[buffer(0)]],
                     device float *out [[buffer(1)]],
                     device const int *tapIndex [[buffer(2)]],
                     device const float *tapCoef [[buffer(3)]],
                     device const float *tapSum [[buffer(4)]],
                     constant SincParams &p [[buffer(5)]],
                     uint3 gid [[thread_position_in_grid]])
{
    const int ky = gid.x, kx = gid.y, img = gid.z;
    if (ky >= p.size || kx >= p.size) return;
    const size_t base = ((size_t)(img * 2) * p.size + kx) * p.intplength;
    device const float *r = rows + (size_t)img * p.inNX * p.size + ky;
    float v = 0.0f;
    for (int i = 0; i < p.intplength; i++) v += r[tapIndex[base + i] * p.size] * tapCoef[base + i];
    const float norm = tapSum[(img * 2) * p.size + kx] * tapSum[(img * 2 + 1) * p.size + ky];
    out[((size_t)img * p.size + kx) * p.size + ky] = v / norm;
}

// whether surface coordinate o is inside the oversampled window along an axis
inline bool inWindow(int o, int start, int shift, int factor, int outN, int size)
{
    int k = o - (start + shift * factor);
    if (k < 0) k += outN;
    return k >= 0 && k < size;
}

// cuArraysMaxloc2D of the full oversampled surface, which is 0 outside the
// sinc window: max of the compact window (ties -> first row-major surface
// index), compared with the first 0 outside the window (one threadgroup per
// image)
kernel void maxlocSinc(device const float *window [[buffer(0)]],
                       device const int2 *centerShift [[buffer(1)]],
                       device int2 *maxloc [[buffer(2)]],
                       device float *maxval [[buffer(3)]],
                       constant SincParams &p [[buffer(4)]],
                       uint img [[threadgroup_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float vals[REDUCE_THREADS];
    threadgroup int idxs[REDUCE_THREADS];
    const int2 shift = centerShift[img];
    const int n = p.size * p.size;
    device const float *w = window + (size_t)img * n;
    float best = -INFINITY;
    int bestIdx = p.outNX * p.outNY;
    for (int i = tid; i < n; i += REDUCE_THREADS) {
        const int kx = i / p.size, ky = i % p.size;
        const int idx = sincOut(kx, p.startX, shift.x, p.factor, p.outNX) * p.outNY +
                        sincOut(ky, p.startY, shift.y, p.factor, p.outNY);
        const float v = w[i];
        if (v > best || (v == best && idx < bestIdx)) { best = v; bestIdx = idx; }
    }
    vals[tid] = best; idxs[tid] = bestIdx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float v = vals[tid + s];
            const int k = idxs[tid + s];
            if (v > vals[tid] || (v == vals[tid] && k < idxs[tid])) {
                vals[tid] = v; idxs[tid] = k;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        best = vals[0];
        bestIdx = idxs[0];
        if (best <= 0.0f) {
            // first surface index outside the window: row 0 if it is outside,
            // else the first column of row 0 outside the window
            int zero = 0;
            if (inWindow(0, p.startX, shift.x, p.factor, p.outNX, p.size)) {
                while (inWindow(zero, p.startY, shift.y, p.factor, p.outNY, p.size)) zero++;
            }
            if (best < 0.0f || zero < bestIdx) { best = 0.0f; bestIdx = zero; }
        }
        maxval[img] = best;
        maxloc[img] = int2(bestIdx / p.outNY, bestIdx % p.outNY);
    }
}
