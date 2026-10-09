// Sinc resampling of an SLC block to given input indices on the GPU; mirrors
// isce3::cuda::image::v2::_resampleToCoordsGlobal (Resample.cu) and the CPU
// isce3::image::v2::resampleToCoords. Metal has no FP64: the double indices
// are split exactly into integer part and fraction from their bits, and the
// native Doppler LUT (constant or bilinear) is evaluated in FP32.
#include <metal_stdlib>
using namespace metal;

constant int SINC_HALF = 4;
constant int SINC_LEN = 8;
constant int SINC_SUB = 8192;

struct Params {
    uint count;
    int inWidth, inLength;
    float fillRe, fillIm;  // fill value (float2 would be 8-byte aligned)
    // Doppler LUT: 0 constant value, 1 bilinear over lutWidth x lutLength;
    // LUT indices x = cx0 + cx1 * range index, y = cy0 + cy1 * azimuth index
    int lutMode, lutWidth, lutLength;
    float lutValue, cx0, cx1, cy0, cy1;
    float dopplerScale;  // 2 pi / prf
};

// Integer part, fraction and nearest sinc filter phase (floor(fraction *
// SINC_SUB), exact) of a double given as its bits. False for NaN, infinite,
// negative and too large values, which all fall outside the input block.
static bool splitIndex(ulong u, thread int &ip, thread float &frac,
                       thread int &phase)
{
    if (u == 0x8000000000000000ul)  // -0
        u = 0;
    if (u >> 63)
        return false;
    const int ex = int(u >> 52);
    if (ex == 0x7FF)
        return false;
    if (ex == 0) {  // zero or subnormal: 0 for the integer part and phase
        ip = 0; frac = 0.f; phase = 0;
        return true;
    }
    const int s = 1075 - ex;  // value = m * 2^-s
    if (s <= 0)
        return false;          // >= 2^52
    const ulong m = (u & 0xFFFFFFFFFFFFFul) | 0x10000000000000ul;
    const ulong i = s >= 64 ? 0ul : m >> s;
    if (i >= 0x80000000ul)
        return false;
    const ulong f = s >= 64 ? m : m & ((1ul << s) - 1);
    ip = int(i);
    frac = ldexp(float(f), -s);
    phase = int(s > 13 ? (s - 13 >= 64 ? 0ul : f >> (s - 13)) : f << (13 - s));
    return true;
}

// LUT2d bilinear interpolation (BilinearInterpolator) at clamped indices
static float bilinear(device const float *z, int w, int l, float x, float y)
{
    x = clamp(x, 0.f, float(w - 1));
    y = clamp(y, 0.f, float(l - 1));
    const int x1 = int(floor(x)), x2 = int(ceil(x));
    const int y1 = int(floor(y)), y2 = int(ceil(y));
    const float q11 = z[y1 * w + x1], q12 = z[y2 * w + x1];
    const float q21 = z[y1 * w + x2], q22 = z[y2 * w + x2];
    if (y1 == y2 && x1 == x2)
        return q11;
    if (y1 == y2)
        return (x2 - x) / (x2 - x1) * q11 + (x - x1) / (x2 - x1) * q21;
    if (x1 == x2)
        return (y2 - y) / (y2 - y1) * q11 + (y - y1) / (y2 - y1) * q12;
    const float d = (x2 - x1) * (y2 - y1);
    return q11 * ((x2 - x) * (y2 - y)) / d + q21 * ((x - x1) * (y2 - y)) / d +
           q12 * ((x2 - x) * (y - y1)) / d + q22 * ((x - x1) * (y - y1)) / d;
}

static float2 cmul(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

kernel void resampleToCoords(device const float2 *input [[buffer(0)]],
                             device const ulong *rangeIndices [[buffer(1)]],
                             device const ulong *azimuthIndices [[buffer(2)]],
                             device const float *sincFilter [[buffer(3)]],
                             device const float *lut [[buffer(4)]],
                             device float2 *output [[buffer(5)]],
                             constant Params &p [[buffer(6)]],
                             uint gid [[thread_position_in_grid]])
{
    if (gid >= p.count)
        return;
    output[gid] = float2(p.fillRe, p.fillIm);

    // integer input indices, fractions and filter phases
    int irg, iaz, phaseRg, phaseAz;
    float frg, faz;
    if (!splitIndex(rangeIndices[gid], irg, frg, phaseRg) ||
        !splitIndex(azimuthIndices[gid], iaz, faz, phaseAz))
        return;
    // sinc chip within the input block
    if (irg < SINC_HALF || irg >= p.inWidth - SINC_HALF ||
        iaz < SINC_HALF || iaz >= p.inLength - SINC_HALF)
        return;

    // native Doppler (Hz) at the input position, then radians per line
    float doppler = p.lutValue;
    if (p.lutMode == 1) {
        const float x = p.cx0 + p.cx1 * (float(irg) + frg);
        const float y = p.cy0 + p.cy1 * (float(iaz) + faz);
        if (!(x >= 0.f && x <= float(p.lutWidth - 1) &&
              y >= 0.f && y <= float(p.lutLength - 1)))
            return;
        doppler = bilinear(lut, p.lutWidth, p.lutLength, x, y);
    }
    const float dopplerFreq = doppler * p.dopplerScale;

    // Sinc interpolation of the Doppler-stripped samples around the
    // position, rows separably as in the CUDA kernel
    device const float *kx = sincFilter + phaseRg * SINC_LEN;
    device const float *ky = sincFilter + phaseAz * SINC_LEN;
    float2 value = 0.f;
    for (int i = 0; i < SINC_LEN; ++i) {
        // chip row offset from the center
        const int drow = SINC_HALF - i;
        const float phase = dopplerFreq * float(drow);
        const float2 dopplerConj(cos(phase), -sin(phase));
        // first tap of the row (input column irg + SINC_HALF)
        device const float2 *row = input + long(p.inWidth) * (iaz + drow) +
                                   (irg + SINC_HALF);
        float2 rowSum = 0.f;
        for (int j = 0; j < SINC_LEN; ++j)
            rowSum += cmul(row[-j], dopplerConj) * kx[j];
        value += rowSum * ky[i];
    }

    // reintroduce the Doppler phase at the interpolated position
    const float phase = dopplerFreq * faz;
    output[gid] = cmul(value, float2(cos(phase), sin(phase)));
}
