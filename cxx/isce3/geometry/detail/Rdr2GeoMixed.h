// -*- C++ -*-
// Mixed-precision rdr2geo: the height iteration of detail::rdr2geo with the
// iterations in FP32 on residuals around an FP64 starting point, finished by
// a few FP64 iterations. FP32 is enough for the residuals (target motion of
// meters to kilometers, heights), while the large quantities (satellite and
// target positions ~1e6-1e7 m, DEM map coordinates) stay FP64 constants; the
// FP64 iterations then remove the FP32 rounding and the linearizations.
#pragma once

#include <cmath>
#include <limits>

#include <isce3/core/Basis.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Pixel.h>
#include <isce3/core/Projections.h>
#include <isce3/core/Vector.h>
#include <isce3/error/ErrorCode.h>
#include "Rdr2Geo.h"

namespace isce3 { namespace geometry { namespace detail {

/** \internal FP64 constants of one pixel for the FP32 height iteration,
 * linearized at the target T0 of a starting height h0 (see rdr2geoMixed) */
struct Rdr2GeoResidualSetup {
    double h0;          // starting height of the radius + h model
    double demX, demY;  // DEM map coordinates of T0
    float jac[2][3];    // d(DEM map coordinates)/d(ECEF) at T0
    float n0[3];        // ellipsoid normal at T0
    float hT0;          // ellipsoidal height of T0
    float tHat[3];      // T0 / |T0|
    float tNorm;        // |T0|
    float dh0;          // |T0| - radius - h0: height-model change at T0
    float c0, b0, a, r; // cos(look) at h0, radius + h0, |pos|, range
    float alpha0, beta0, ndotvOverVdott;
    float that[3], chat[3], nhat[3];
    float curvature;    // 1 / (2 * radius of curvature)
};

/** \internal FP64 setup of the FP32 iteration: target at height h0 of the
 * radius + h model (as updateLLH of detail::rdr2geo), its DEM map
 * coordinates, Jacobian, normal and the look-geometry constants.
 * Returns false where the look geometry is invalid. */
inline bool rdr2geoResidualSetup(Rdr2GeoResidualSetup& s,
        const isce3::core::Pixel& pixel, const isce3::core::Basis& tcn,
        const isce3::core::Vec3& pos, const isce3::core::Vec3& vel,
        const isce3::core::Ellipsoid& ellipsoid,
        const isce3::core::ProjectionBase& demProj,
        isce3::core::LookSide side, double h0)
{
    using namespace isce3::core;
    const Vec3 vhat = vel.normalized();
    const Vec3 &that = tcn.x0(), &chat = tcn.x1(), &nhat = tcn.x2();
    const double ndotv = nhat.dot(vhat), vdott = vhat.dot(that);
    const double major = ellipsoid.a();
    const double minor = major * std::sqrt(1. - ellipsoid.e2());
    const double satDist = pos.norm();
    const double eta = 1. / std::sqrt((pos[0] / major) * (pos[0] / major) +
                                      (pos[1] / major) * (pos[1] / major) +
                                      (pos[2] / minor) * (pos[2] / minor));
    const double radius = eta * satDist;
    const double r = pixel.range();
    const double b0 = radius + h0;
    const double c0 = 0.5 * (satDist / r + r / satDist - (b0 / satDist) * (b0 / r));
    const double sin0 = std::sqrt(1. - c0 * c0);
    const double gamma0 = r * c0;
    const double alpha0 = (pixel.dopfact() - gamma0 * ndotv) / vdott;
    const double beta2 = (r * sin0) * (r * sin0) - alpha0 * alpha0;
    if (!(beta2 > 0.))
        return false;
    const double beta0 = (side == LookSide::Right ? 1. : -1.) * std::sqrt(beta2);
    const Vec3 t0 = pos + alpha0 * that + beta0 * chat + gamma0 * nhat;
    const Vec3 llh0 = ellipsoid.xyzToLonLat(t0);

    // DEM map coordinates of T0 and their derivatives w.r.t. ECEF:
    // d(lon, lat)/dX from the local east/north vectors and radii of
    // curvature, d(map)/d(lon, lat) by finite differences of the projection
    Vec3 p0, pLon, pLat;
    const double eps = 1e-7;  // rad, ~0.6 m
    demProj.forward(llh0, p0);
    demProj.forward({llh0[0] + eps, llh0[1], 0.}, pLon);
    demProj.forward({llh0[0], llh0[1] + eps, 0.}, pLat);
    const double sinLon = std::sin(llh0[0]), cosLon = std::cos(llh0[0]);
    const double sinLat = std::sin(llh0[1]), cosLat = std::cos(llh0[1]);
    const Vec3 east {-sinLon, cosLon, 0.};
    const Vec3 north {-sinLat * cosLon, -sinLat * sinLon, cosLat};
    const Vec3 up {cosLat * cosLon, cosLat * sinLon, sinLat};
    const double rEast = ellipsoid.rEast(llh0[1]) + llh0[2];
    const double rNorth = ellipsoid.rNorth(llh0[1]) + llh0[2];
    for (int k = 0; k < 2; ++k) {
        const double dmapDlon = (pLon[k] - p0[k]) / eps;
        const double dmapDlat = (pLat[k] - p0[k]) / eps;
        for (int j = 0; j < 3; ++j)
            s.jac[k][j] = static_cast<float>(
                    dmapDlon * east[j] / (rEast * cosLat) +
                    dmapDlat * north[j] / rNorth);
    }
    s.h0 = h0;
    s.demX = p0[0];
    s.demY = p0[1];
    const double tNorm = t0.norm();
    for (int j = 0; j < 3; ++j) {
        s.n0[j] = static_cast<float>(up[j]);
        s.tHat[j] = static_cast<float>(t0[j] / tNorm);
        s.that[j] = static_cast<float>(that[j]);
        s.chat[j] = static_cast<float>(chat[j]);
        s.nhat[j] = static_cast<float>(nhat[j]);
    }
    s.hT0 = static_cast<float>(llh0[2]);
    s.tNorm = static_cast<float>(tNorm);
    s.dh0 = static_cast<float>((tNorm - radius) - h0);
    s.c0 = static_cast<float>(c0);
    s.b0 = static_cast<float>(b0);
    s.a = static_cast<float>(satDist);
    s.r = static_cast<float>(r);
    s.alpha0 = static_cast<float>(alpha0);
    s.beta0 = static_cast<float>(beta0);
    s.ndotvOverVdott = static_cast<float>(ndotv / vdott);
    s.curvature = static_cast<float>(0.5 / std::sqrt(rEast * rNorth));
    return true;
}

/** \internal FP32 height iteration on residuals: target motion dT(dh) for a
 * height change dh of the radius + h model, DEM height at the linearized map
 * coordinates, ellipsoidal height of the target to second order, and the new
 * height of the model. demHeight(x, y) returns the DEM height at map
 * coordinates (the FP64 setup's demX/demY plus FP32 offsets).
 * Returns the height h0 + dh, or NaN if it did not converge to tol. */
template<class DemHeight>
inline double rdr2geoResidualIterate(const Rdr2GeoResidualSetup& s,
        DemHeight&& demHeight, int maxiter, float tol)
{
    float dh = 0.f;
    for (int it = 0; it < maxiter; ++it) {
        // cos(look) change: b^2 - b0^2 = dh (2 b0 + dh)
        const float dc = -dh * (2.f * s.b0 + dh) / (2.f * s.a * s.r);
        const float dgamma = s.r * dc;
        const float dalpha = -dgamma * s.ndotvOverVdott;
        // beta^2 - beta0^2 = -r^2 dc (2 c0 + dc) - dalpha (2 alpha0 + dalpha)
        const float dbeta2 = -s.r * s.r * dc * (2.f * s.c0 + dc) -
                             dalpha * (2.f * s.alpha0 + dalpha);
        const float root = std::sqrt(std::fmax(s.beta0 * s.beta0 + dbeta2, 0.f));
        const float dbeta = dbeta2 / (s.beta0 + std::copysign(root, s.beta0));
        float dT[3];
        for (int j = 0; j < 3; ++j)
            dT[j] = dalpha * s.that[j] + dbeta * s.chat[j] + dgamma * s.nhat[j];
        // DEM height at the target
        const float dx = s.jac[0][0] * dT[0] + s.jac[0][1] * dT[1] + s.jac[0][2] * dT[2];
        const float dy = s.jac[1][0] * dT[0] + s.jac[1][1] * dT[1] + s.jac[1][2] * dT[2];
        const float hdem = static_cast<float>(demHeight(s.demX + dx, s.demY + dy));
        // ellipsoidal height of the target, then onto the DEM along the normal
        const float dn = s.n0[0] * dT[0] + s.n0[1] * dT[1] + s.n0[2] * dT[2];
        const float dT2 = dT[0] * dT[0] + dT[1] * dT[1] + dT[2] * dT[2];
        const float hT = s.hT0 + dn + (dT2 - dn * dn) * s.curvature;
        float dX[3], tdotx = 0.f, dX2 = 0.f;
        for (int j = 0; j < 3; ++j) {
            dX[j] = dT[j] + (hdem - hT) * s.n0[j];
            tdotx += s.tHat[j] * dX[j];
            dX2 += dX[j] * dX[j];
        }
        // |T0 + dX| - |T0| without forming the ~6.4e6 m norms in FP32
        const float dnorm = (2.f * tdotx * s.tNorm + dX2) /
                            (s.tNorm + std::sqrt(s.tNorm * s.tNorm + 2.f * tdotx * s.tNorm + dX2));
        const float dhNew = s.dh0 + dnorm;
        const bool done = std::fabs(dhNew - dh) < tol;
        dh = dhNew;
        if (done)
            return s.h0 + dh;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

}}} // namespace isce3::geometry::detail
