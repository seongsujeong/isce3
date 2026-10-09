// -*- C++ -*-
// Metal (Apple GPU) FP32 height iteration of the mixed-precision rdr2geo
// (detail/Rdr2GeoMixed.h): the CPU fills the per-line and per-pixel FP32
// constants of the FP64 setup, the GPU iterates and returns the height
// change, the CPU finishes in FP64. Two slots let the CPU fill or finish one
// chunk while the GPU runs the other.
#pragma once

#include <cstddef>
#include <memory>

namespace isce3 { namespace geometry { namespace detail {

/** \internal Constants of one azimuth line; layout matches Rdr2GeoMixed.metal */
struct Rdr2GeoMetalLine {
    float that[3], chat[3], nhat[3];
    float a;              // |pos|
    float ndotvOverVdott;
};

/** \internal Constants of one pixel (Rdr2GeoResidualSetup with the DEM
 * coordinates as DEM indices: integer part + FP32 fraction); r is NaN where
 * the setup failed. Layout matches Rdr2GeoMixed.metal */
struct Rdr2GeoMetalPixel {
    int col, row;
    float fcol, frow;
    float jac[2][3];      // d(DEM index)/d(ECEF)
    float n0[3];
    float hT0;
    float tHat[3];
    float tNorm, dh0, c0, b0, r, alpha0, beta0, curvature;
};

class Rdr2GeoMetal {
public:
    /** GPU iteration on the default Metal device; nullptr if none */
    static std::unique_ptr<Rdr2GeoMetal> create();
    ~Rdr2GeoMetal();

    /** Upload the DEM window (row-major, nx columns) and its reference
     * height, returned outside the window as DEMInterpolator does */
    void dem(const float* z, int nx, int ny, float refHeight);

    /** Host-visible buffers of a slot with room for n lines / pixels; the
     * slot must not be running */
    Rdr2GeoMetalLine* lines(int slot, size_t n);
    Rdr2GeoMetalPixel* pixels(int slot, size_t n);

    /** Start the iteration of nPixels pixels of a slot, line-major with
     * width pixels per line */
    void run(int slot, size_t nPixels, int width, int maxiter, float tol);

    /** Wait for a slot; returns the height change of each pixel, NaN where
     * the iteration did not converge */
    const float* wait(int slot);

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
    Rdr2GeoMetal();
};

}}} // namespace isce3::geometry::detail
