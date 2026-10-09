// FP32 height iteration of the mixed-precision rdr2geo on the GPU; mirrors
// rdr2geoResidualIterate (Rdr2GeoMixed.h) with the DEM height from the
// biquintic (order 6) Spline2dInterpolator of DEMInterpolator::interpolateXY
#include <metal_stdlib>
using namespace metal;

struct Line {
    float that[3], chat[3], nhat[3];
    float a, ndotvOverVdott;
};

struct Pixel {
    int col, row;
    float fcol, frow;
    float jac[2][3];
    float n0[3];
    float hT0;
    float tHat[3];
    float tNorm, dh0, c0, b0, r, alpha0, beta0, curvature;
};

struct Params {
    int count, width, nx, ny, maxiter;
    float tol, refHeight;
};

constant int ORDER = 6;

// Spline2dInterpolator::_initSpline
static void initSpline(thread const float *Y, thread float *R)
{
    float Q[ORDER];
    Q[0] = 0.f;
    R[0] = 0.f;
    for (int i = 1; i < ORDER - 1; ++i) {
        const float p = 1.f / (0.5f * Q[i - 1] + 2.f);
        Q[i] = -0.5f * p;
        R[i] = (3.f * (Y[i + 1] - 2.f * Y[i] + Y[i - 1]) - 0.5f * R[i - 1]) * p;
    }
    R[ORDER - 1] = 0.f;
    for (int i = ORDER - 2; i > 0; --i)
        R[i] = Q[i] * R[i + 1] + R[i];
}

// Spline2dInterpolator::_spline for 1 <= x <= ORDER (here x is in [2, 3))
static float spline(float x, thread const float *Y, thread const float *R)
{
    const int j = int(floor(x));
    const float xx = x - j;
    const float t0 = Y[j] - Y[j - 1] - R[j - 1] / 3.f - R[j] / 6.f;
    const float t1 = xx * (R[j - 1] / 2.f + xx * ((R[j] - R[j - 1]) / 6.f));
    return Y[j - 1] + xx * (t0 + t1);
}

// DEMInterpolator::interpolateXY at DEM index (col + fc, row + fr)
static float demHeight(int col, float fc, int row, float fr,
                       device const float *z, constant Params &p)
{
    const float kc = floor(fc), kr = floor(fr);
    const int ic = col + int(kc), ir = row + int(kr);
    fc -= kc;
    fr -= kr;
    if (ir < 2 || ir >= p.ny - 1 || ic < 2 || ic >= p.nx - 1)
        return p.refHeight;
    float A[ORDER], R[ORDER], HC[ORDER];
    for (int i = 0; i < ORDER; ++i) {
        const int indi = clamp(ir - 2 + i, 0, p.ny - 2);
        for (int j = 0; j < ORDER; ++j)
            A[j] = z[(indi + 1) * p.nx + clamp(ic - 2 + j, 0, p.nx - 2) + 1];
        initSpline(A, R);
        HC[i] = spline(fc + 2.f, A, R);
    }
    initSpline(HC, R);
    return spline(fr + 2.f, HC, R);
}

kernel void rdr2geoResidualIterate(device const Line *lines [[buffer(0)]],
                                   device const Pixel *pixels [[buffer(1)]],
                                   device const float *z [[buffer(2)]],
                                   device float *out [[buffer(3)]],
                                   constant Params &p [[buffer(4)]],
                                   uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= p.count)
        return;
    const Pixel s = pixels[gid];
    const Line l = lines[gid / p.width];
    out[gid] = NAN;
    if (isnan(s.r))
        return;
    float dh = 0.f;
    for (int it = 0; it < p.maxiter; ++it) {
        const float dc = -dh * (2.f * s.b0 + dh) / (2.f * l.a * s.r);
        const float dgamma = s.r * dc;
        const float dalpha = -dgamma * l.ndotvOverVdott;
        const float dbeta2 = -s.r * s.r * dc * (2.f * s.c0 + dc) -
                             dalpha * (2.f * s.alpha0 + dalpha);
        const float root = sqrt(max(s.beta0 * s.beta0 + dbeta2, 0.f));
        const float dbeta = dbeta2 / (s.beta0 + copysign(root, s.beta0));
        float dT[3];
        for (int j = 0; j < 3; ++j)
            dT[j] = dalpha * l.that[j] + dbeta * l.chat[j] + dgamma * l.nhat[j];
        const float dx = s.jac[0][0] * dT[0] + s.jac[0][1] * dT[1] + s.jac[0][2] * dT[2];
        const float dy = s.jac[1][0] * dT[0] + s.jac[1][1] * dT[1] + s.jac[1][2] * dT[2];
        const float hdem = demHeight(s.col, s.fcol + dx, s.row, s.frow + dy, z, p);
        const float dn = s.n0[0] * dT[0] + s.n0[1] * dT[1] + s.n0[2] * dT[2];
        const float dT2 = dT[0] * dT[0] + dT[1] * dT[1] + dT[2] * dT[2];
        const float hT = s.hT0 + dn + (dT2 - dn * dn) * s.curvature;
        float tdotx = 0.f, dX2 = 0.f;
        for (int j = 0; j < 3; ++j) {
            const float dX = dT[j] + (hdem - hT) * s.n0[j];
            tdotx += s.tHat[j] * dX;
            dX2 += dX * dX;
        }
        const float dnorm = (2.f * tdotx * s.tNorm + dX2) /
                (s.tNorm + sqrt(s.tNorm * s.tNorm + 2.f * tdotx * s.tNorm + dX2));
        const float dhNew = s.dh0 + dnorm;
        const bool done = fabs(dhNew - dh) < p.tol;
        dh = dhNew;
        if (done) {
            out[gid] = dh;
            return;
        }
    }
}
