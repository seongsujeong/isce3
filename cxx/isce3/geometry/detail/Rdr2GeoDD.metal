// rdr2geo on the GPU in double-float (dd: pair of floats, ~48-bit
// significand) arithmetic: the iteration of detail::rdr2geo (Rdr2Geo.icc)
// with the same update, convergence test and extra iterations. The target
// and DEM positions, heights and normals are algebraic in ECEF (ellipsoidal
// height and normal by Vermeille's method, as Ellipsoid::xyzToLonLat). The
// map projections (DEM indices, output coordinates) are quadratic models in
// the ECEF offset from an FP64 anchor per tile, computed on the CPU.
#include <metal_stdlib>
using namespace metal;

// ---- double-float arithmetic (Dekker / Knuth error-free transformations) ----
struct dd { float hi, lo; };

static dd quickTwoSum(float a, float b)
{
    const float s = a + b;
    return {s, b - (s - a)};
}
static dd twoSum(float a, float b)
{
    const float s = a + b, v = s - a;
    return {s, (a - (s - v)) + (b - v)};
}
static dd operator+(dd a, dd b)
{
    dd s = twoSum(a.hi, b.hi);
    const dd t = twoSum(a.lo, b.lo);
    s.lo += t.hi;
    s = quickTwoSum(s.hi, s.lo);
    s.lo += t.lo;
    return quickTwoSum(s.hi, s.lo);
}
static dd operator-(dd a) { return {-a.hi, -a.lo}; }
static dd operator-(dd a, dd b) { return a + (-b); }
static dd operator*(dd a, dd b)
{
    const float p = a.hi * b.hi;
    float e = fma(a.hi, b.hi, -p);
    e += a.hi * b.lo + a.lo * b.hi;
    return quickTwoSum(p, e);
}
static dd operator*(dd a, float b)
{
    const float p = a.hi * b;
    const float e = fma(a.hi, b, -p) + a.lo * b;
    return quickTwoSum(p, e);
}
static dd operator+(dd a, float b) { return a + dd{b, 0.f}; }
static dd operator-(dd a, float b) { return a + dd{-b, 0.f}; }
static dd operator/(dd a, dd b)
{
    const float q1 = a.hi / b.hi;
    dd r = a - b * q1;
    const float q2 = r.hi / b.hi;
    r = r - b * q2;
    const float q3 = r.hi / b.hi;
    return quickTwoSum(q1, q2) + q3;
}
static dd ddsqrt(dd a)
{
    if (!(a.hi > 0.f))
        return {a.hi == 0.f ? 0.f : NAN, 0.f};
    const float x = rsqrt(a.hi), ax = a.hi * x;
    return dd{ax, 0.f} + (a - dd{ax, 0.f} * dd{ax, 0.f}).hi * (x * 0.5f);
}
static dd ddcbrt(dd a)
{
    dd c = {powr(a.hi, 1.f / 3.f), 0.f};  // a > 0 here (Vermeille)
    for (int k = 0; k < 2; ++k)  // Newton: c - (c^3 - a) / (3 c^2)
        c = c - (c * c * c - a) / (c * c * 3.f);
    return c;
}
static float todouble_hi(dd a) { return a.hi + a.lo; }

struct dd3 { dd x, y, z; };
static dd3 operator+(dd3 a, dd3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static dd3 operator-(dd3 a, dd3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static dd3 operator*(dd3 a, dd s) { return {a.x * s, a.y * s, a.z * s}; }
static dd3 operator*(dd3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static dd dot(dd3 a, dd3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static dd norm(dd3 a) { return ddsqrt(dot(a, a)); }
static float3 tofloat(dd3 a) { return float3(todouble_hi(a.x), todouble_hi(a.y), todouble_hi(a.z)); }

// ---- inputs (layouts match Rdr2GeoMetal.mm) ----
struct Line {
    dd3 pos, that, chat, nhat;
    dd satDist, radius, height;   // |pos|, nadir radius, (1 - eta) |pos|
    dd ndotv, vdott, dopCoef;     // dopfact = dopCoef * range
    dd side;                      // +1 right, -1 left looking
};

// quadratic model of a map in the ECEF offset d from the anchor:
// f(d) = f0 + J d + 1/2 d^T H d, H symmetric as (xx, yy, zz, xy, xz, yz)
struct Tile {
    dd3 anchor;
    int demCol, demRow;           // DEM index of the anchor: integer part
    float demFrac[2];             // and fraction (col, row)
    float demJ[2][3], demH[2][6];
    dd out0[2];                   // output map coordinates of the anchor
    dd outJ[2][3];
    float outH[2][6];
    int valid;
};

struct Params {
    uint count;
    int width, tileSize, tilesPerRow;
    int nx, ny;                   // DEM window
    float refHeight;
    float threshold;
    int maxiter, extraiter;
    dd a, e2;                     // ellipsoid
};

struct Out {
    dd x, y, z;
    float h;                      // radius + h model height (prior of the next lines)
    int converged;
};

// ---- Vermeille: ellipsoidal height and normal of an ECEF position ----
static dd ellipsoidHeight(dd3 X, constant Params &p, thread float3 &normal)
{
    const dd a2 = p.a * p.a, e2 = p.e2, e4 = p.e2 * p.e2;
    const dd rho2 = X.x * X.x + X.y * X.y;
    const dd pp = rho2 / a2;
    const dd q = (dd{1.f, 0.f} - e2) * (X.z * X.z) / a2;
    const dd r = (pp + q - e4) / dd{6.f, 0.f};  // 1/6 is not exact in float
    const dd s = e4 * pp * q / (r * r * r * 4.f);
    const dd t = ddcbrt(dd{1.f, 0.f} + s + ddsqrt(s * (s + 2.f)));
    const dd u = r * (dd{1.f, 0.f} + t + dd{1.f, 0.f} / t);
    const dd rv = ddsqrt(u * u + e4 * q);
    const dd w = e2 * (u + rv - q) / (rv * 2.f);
    const dd k = ddsqrt(u + rv + w * w) - w;
    const dd rho = ddsqrt(rho2);
    const dd d = k * rho / (k + e2);
    const dd dz = ddsqrt(d * d + X.z * X.z);
    // normal (cos lat cos lon, cos lat sin lon, sin lat)
    const float cl = todouble_hi(d / dz), sl = todouble_hi(X.z / dz);
    const float rf = todouble_hi(rho);
    normal = float3(cl * todouble_hi(X.x) / rf, cl * todouble_hi(X.y) / rf, sl);
    return (k + e2 - 1.f) * dz / k;
}

// ---- DEM: biquintic spline as DEMInterpolator::interpolateXY ----
constant int ORDER = 6;
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
static float spline(float x, thread const float *Y, thread const float *R)
{
    const int j = int(floor(x));
    const float xx = x - j;
    const float t0 = Y[j] - Y[j - 1] - R[j - 1] / 3.f - R[j] / 6.f;
    const float t1 = xx * (R[j - 1] / 2.f + xx * ((R[j] - R[j - 1]) / 6.f));
    return Y[j - 1] + xx * (t0 + t1);
}
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

static float quadratic(float3 d, constant const float *J, constant const float *H)
{
    return J[0] * d.x + J[1] * d.y + J[2] * d.z +
           0.5f * (H[0] * d.x * d.x + H[1] * d.y * d.y + H[2] * d.z * d.z) +
           H[3] * d.x * d.y + H[4] * d.x * d.z + H[5] * d.y * d.z;
}

// DEM height at an ECEF position (model of the tile)
static float demAt(dd3 X, constant Tile &t, device const float *z, constant Params &p)
{
    const float3 d = tofloat(X - t.anchor);
    return demHeight(t.demCol, t.demFrac[0] + quadratic(d, t.demJ[0], t.demH[0]),
                     t.demRow, t.demFrac[1] + quadratic(d, t.demJ[1], t.demH[1]), z, p);
}

// target at height h of the radius + h model (updateLLH of detail::rdr2geo)
static dd3 target(dd h, dd r, constant Line &l, dd dopfact)
{
    const dd a = l.satDist, b = l.radius + h;
    const dd c = (a / r + r / a - (b / a) * (b / r)) * 0.5f;
    const dd s = ddsqrt(dd{1.f, 0.f} - c * c);
    const dd gamma = r * c;
    const dd alpha = (dopfact - gamma * l.ndotv) / l.vdott;
    const dd rs = r * s;
    const dd beta = l.side * ddsqrt(rs * rs - alpha * alpha);
    return l.pos + l.that * alpha + l.chat * beta + l.nhat * gamma;
}

kernel void rdr2geoDD(constant Line *lines [[buffer(0)]],
                      device const dd *ranges [[buffer(1)]],
                      device const float *prior [[buffer(2)]],
                      constant Tile *tiles [[buffer(3)]],
                      device const float *z [[buffer(4)]],
                      device Out *out [[buffer(5)]],
                      constant Params &p [[buffer(6)]],
                      uint gid [[thread_position_in_grid]])
{
    if (gid >= p.count)
        return;
    const int line = int(gid) / p.width, bin = int(gid) % p.width;
    constant Line &l = lines[line];
    constant Tile &t = tiles[(line / p.tileSize) * p.tilesPerRow + bin / p.tileSize];
    out[gid].converged = -1;  // tile without a valid anchor: done on the CPU
    if (!t.valid)
        return;
    const dd r = ranges[bin];
    const dd dopfact = l.dopCoef * r;
    const float h0 = prior[bin];
    dd h = isnan(h0) ? l.height : dd{h0, 0.f};

    bool converged = false;
    dd3 xyzOld = {};
    float3 normal;
    for (int i = 0; i < p.maxiter + p.extraiter; ++i) {
        // near nadir test
        if ((l.height - h - r).hi >= 0.f)
            break;
        // target, then snapped to the DEM height at its latitude/longitude
        const dd3 T = target(h, r, l, dopfact);
        const dd hT = ellipsoidHeight(T, p, normal);
        const float hdem = demAt(T, t, z, p);
        dd3 xyzNew = T + dd3{dd{normal.x, 0.f}, dd{normal.y, 0.f}, dd{normal.z, 0.f}} *
                         (dd{hdem, 0.f} - hT);
        h = norm(xyzNew) - l.radius;
        // convergence: slant range of the snapped target
        const dd dr = r - norm(l.pos - xyzNew);
        if (fabs(todouble_hi(dr)) < p.threshold) {
            converged = true;
            break;
        }
        // in extra iterations, average of new and old target
        if (i > p.maxiter) {
            xyzNew = (xyzOld + xyzNew) * 0.5f;
            h = norm(xyzNew) - l.radius;
        }
        xyzOld = xyzNew;
    }

    // final target exactly at the pixel range
    const dd3 T = target(h, r, l, dopfact);
    const dd hT = ellipsoidHeight(T, p, normal);
    const dd3 dd_ = T - t.anchor;
    const float3 d = tofloat(dd_);
    for (int k = 0; k < 2; ++k) {
        const dd v = t.out0[k] + t.outJ[k][0] * dd_.x + t.outJ[k][1] * dd_.y +
                     t.outJ[k][2] * dd_.z +
                     0.5f * (t.outH[k][0] * d.x * d.x + t.outH[k][1] * d.y * d.y +
                             t.outH[k][2] * d.z * d.z) +
                     t.outH[k][3] * d.x * d.y + t.outH[k][4] * d.x * d.z +
                     t.outH[k][5] * d.y * d.z;
        if (k == 0) out[gid].x = v; else out[gid].y = v;
    }
    out[gid].z = hT;
    out[gid].h = todouble_hi(h);
    out[gid].converged = converged ? 1 : 0;
}
