// -*- C++ -*-
// rdr2geo on a Metal GPU in double-float arithmetic (Rdr2GeoDD.metal): the
// CPU fills per-line geometry and per-tile FP64 anchors with quadratic
// models of the map projections, the GPU runs the whole iteration.
#pragma once

#include <cstddef>
#include <memory>

namespace isce3 { namespace geometry { namespace detail {

/** \internal double-float number: value = hi + lo */
struct DD {
    float hi = 0.f, lo = 0.f;
    DD() = default;
    DD(double v) : hi(static_cast<float>(v)),
                   lo(static_cast<float>(v - static_cast<float>(v))) {}
    double value() const { return static_cast<double>(hi) + lo; }
};

/** \internal Geometry of one azimuth line; layout matches Rdr2GeoDD.metal */
struct Rdr2GeoDDLine {
    DD pos[3], that[3], chat[3], nhat[3];
    DD satDist, radius, height;   // |pos|, nadir radius, (1 - eta) |pos|
    DD ndotv, vdott, dopCoef;     // dopfact = dopCoef * range
    DD side;                      // +1 right, -1 left looking
};

/** \internal FP64 anchor of a tile and quadratic models (in the ECEF offset
 * d from it) of the DEM indices and output map coordinates; H is symmetric,
 * stored as (xx, yy, zz, xy, xz, yz). Layout matches Rdr2GeoDD.metal */
struct Rdr2GeoDDTile {
    DD anchor[3];
    int demCol, demRow;
    float demFrac[2];
    float demJ[2][3], demH[2][6];
    DD out0[2];
    DD outJ[2][3];
    float outH[2][6];
    int valid;
};

/** \internal Result of a pixel; converged -1 where the tile has no anchor */
struct Rdr2GeoDDOut {
    DD x, y, z;
    float h;                      // radius + h model height
    int converged;
};

/** \internal Parameters of a run */
struct Rdr2GeoDDParams {
    unsigned count;
    int width, tileSize, tilesPerRow;
    int nx, ny;
    float refHeight;
    float threshold;
    int maxiter, extraiter;
    DD a, e2;
};

class Rdr2GeoDDMetal {
public:
    /** GPU rdr2geo on the default Metal device; nullptr if none */
    static std::unique_ptr<Rdr2GeoDDMetal> create();
    ~Rdr2GeoDDMetal();

    /** Upload the DEM window (row-major, nx columns) */
    void dem(const float* z, int nx, int ny);

    /** Host-visible inputs and outputs with room for the given sizes */
    Rdr2GeoDDLine* lines(size_t n);
    DD* ranges(size_t n);
    float* prior(size_t n);
    Rdr2GeoDDTile* tiles(size_t n);
    Rdr2GeoDDOut* out(size_t n);

    /** Run the pixels (line-major, p.width per line) and wait */
    void run(const Rdr2GeoDDParams& p);

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
    Rdr2GeoDDMetal();
};

}}} // namespace isce3::geometry::detail
