// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan V. Riel, Joshua Cohen
// Copyright 2017-2018

#include "Geo2rdr.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <limits>
#include <valarray>

#include <isce3/core/Constants.h>

#include "geometry.h"
#include "Topo.h"
#include "TopoLayers.h"

// pull in some isce3::core namespaces
using isce3::io::Raster;
using isce3::core::LUT1d;
using isce3::core::Vec3;

// Run geo2rdr with no offsets; internal creation of offset rasters
void isce3::geometry::Geo2rdr::
geo2rdr(isce3::io::Raster & topoRaster,
        const std::string & outdir,
        double azshift, double rgshift)
{
    // Cache the size of the DEM images
    const size_t demWidth = topoRaster.width();
    const size_t demLength = topoRaster.length();

    // Create output rasters
    Raster rgoffRaster = Raster(outdir + "/range.off", demWidth, demLength, 1,
        GDT_Float64, "ISCE");
    Raster azoffRaster = Raster(outdir + "/azimuth.off", demWidth, demLength, 1,
        GDT_Float64, "ISCE");

    // Call main geo2rdr with offsets set to zero
    geo2rdr(topoRaster, rgoffRaster, azoffRaster, azshift, rgshift);
}

// Radar grid extents with the constant shifts
isce3::geometry::Geo2rdr::Extents isce3::geometry::Geo2rdr::
_extents(double azshift, double rgshift) const
{
    Extents e;
    // Sensing start and starting range adjusted for the constant shifts
    e.dtaz = 1.0 / _radarGrid.prf();
    e.t0 = _radarGrid.sensingStart() - azshift / _radarGrid.prf();
    e.tend = e.t0 + ((_radarGrid.length() - 1) * e.dtaz);
    e.dmrg = _radarGrid.rangePixelSpacing();
    e.r0 = _radarGrid.startingRange() - rgshift * e.dmrg;
    e.rngend = e.r0 + ((_radarGrid.width() - 1) * e.dmrg);
    return e;
}

size_t isce3::geometry::Geo2rdr::
_geo2rdrBlock(const Extents & e, const double * x, const double * y,
              const double * hgt, size_t lineStart, size_t blockLength,
              size_t width, double * rgoff, double * azoff) const
{
    size_t converged = 0;
    // Loop over DEM lines in block
    #pragma omp parallel for reduction(+:converged)
    for (size_t blockLine = 0; blockLine < blockLength; ++blockLine) {

        // Global line index
        const size_t line = lineStart + blockLine;

        // Initial azimuth time of each pixel: solution of the previous
        // pixel, a search over the orbit (NaN) for the first one or after
        // a failure
        double aztime = std::numeric_limits<double>::quiet_NaN();

        // Loop over DEM pixels
        for (size_t pixel = 0; pixel < width; ++pixel) {

            // Convert topo XYZ to LLH
            const size_t index = blockLine * width + pixel;
            Vec3 xyz{x[index], y[index], hgt[index]};
            Vec3 llh = _projTopo->inverse(xyz);

            // Perform geo->rdr iterations
            double slantRange;
            int geostat = isce3::geometry::geo2rdr(
                llh, _ellipsoid, _orbit, _doppler,  aztime, slantRange,
                _radarGrid.wavelength(), _radarGrid.lookSide(),
                _threshold, _numiter, 1.0e-8
            );

            // Check if solution is out of bounds
            bool isOutside = false;
            if ((aztime < e.t0) || (aztime > e.tend))
                isOutside = true;
            if ((slantRange < e.r0) || (slantRange > e.rngend))
                isOutside = true;

            // Save result if valid
            if (!isOutside) {
                rgoff[index] = ((slantRange - e.r0) / e.dmrg) - static_cast<double>(pixel);
                azoff[index] = ((aztime - e.t0) / e.dtaz) - static_cast<double>(line);
                converged += geostat;
            } else {
                rgoff[index] = NULL_VALUE;
                azoff[index] = NULL_VALUE;
            }
            if (!geostat)
                aztime = std::numeric_limits<double>::quiet_NaN();
        } // end for loop pixels in line
    } // end OMP for loop lines in block
    return converged;
}

// Run geo2rdr with externally created offset rasters
void isce3::geometry::Geo2rdr::
geo2rdr(isce3::io::Raster & topoRaster,
        isce3::io::Raster & rgoffRaster,
        isce3::io::Raster & azoffRaster,
        double azshift, double rgshift)
{
    // Create reusable pyre::journal channels
    pyre::journal::info_t info("isce.geometry.Geo2rdr");

    // Cache the size of the DEM images
    const size_t demWidth = topoRaster.width();
    const size_t demLength = topoRaster.length();

    // Initialize projection for topo results
    _projTopo = isce3::core::createProj(topoRaster.getEPSG());

    // Print out extents; interpolate orbit to middle of the scene as a test
    const Extents e = _extents(azshift, rgshift);
    _printExtents(info, e.t0, e.tend, e.dtaz, e.r0, e.rngend, e.dmrg,
                  demWidth, demLength);
    _checkOrbitInterpolation(0.5 * (e.t0 + e.tend));

    // Adjust block size if DEM has too few lines
    _linesPerBlock = std::min(demLength, _linesPerBlock);

    // Compute number of DEM blocks needed to process image
    size_t nBlocks = demLength / _linesPerBlock;
    if ((demLength % _linesPerBlock) != 0)
        nBlocks += 1;

    // Block extents: first line and number of lines
    auto extent = [&](size_t block) {
        const size_t lineStart = block * _linesPerBlock;
        return std::make_pair(lineStart,
                              std::min(_linesPerBlock, demLength - lineStart));
    };
    // Block of topo data (x, y, height)
    struct TopoBlock { std::valarray<double> x, y, hgt; };
    auto read = [&](size_t block) {
        const auto [lineStart, blockLength] = extent(block);
        const size_t blockSize = blockLength * demWidth;
        TopoBlock b {std::valarray<double>(blockSize),
                     std::valarray<double>(blockSize),
                     std::valarray<double>(blockSize)};
        topoRaster.getBlock(b.x, 0, lineStart, demWidth, blockLength, 1);
        topoRaster.getBlock(b.y, 0, lineStart, demWidth, blockLength, 2);
        topoRaster.getBlock(b.hgt, 0, lineStart, demWidth, blockLength, 3);
        return b;
    };

    // Loop over blocks. The next block is read and the previous one written
    // while the current one is computed, each raster used by one thread at a
    // time, so that the raster I/O overlaps the computation.
    size_t converged = 0;
    auto nextBlock = std::async(std::launch::async, read, 0);
    std::future<void> written;
    for (size_t block = 0; block < nBlocks; ++block) {

        // Get block extents
        const auto ext = extent(block);
        const size_t lineStart = ext.first, blockLength = ext.second;
        const size_t blockSize = blockLength * demWidth;

        // Diagnostics
        const double tblock = _radarGrid.sensingTime(lineStart);
        info << "Processing block: " << block << " " << pyre::journal::newline
             << "  - line start: " << lineStart << pyre::journal::newline
             << "  - line end  : " << lineStart + blockLength << pyre::journal::newline
             << "  - dopplers near mid far: "
             << _doppler.eval(tblock, e.r0) << " "
             << _doppler.eval(tblock, 0.5*(e.r0 + e.rngend)) << " "
             << _doppler.eval(tblock, e.rngend) << " "
             << pyre::journal::endl;

        // Block of topo data; start reading the next one
        const TopoBlock topo = nextBlock.get();
        if (block + 1 < nBlocks)
            nextBlock = std::async(std::launch::async, read, block + 1);

        // geo2rdr of the block
        std::valarray<double> rgoff(blockSize), azoff(blockSize);
        converged += _geo2rdrBlock(e, &topo.x[0], &topo.y[0], &topo.hgt[0],
                                   lineStart, blockLength, demWidth,
                                   &rgoff[0], &azoff[0]);

        // Write block of data after the previous write finished
        if (written.valid())
            written.get();
        written = std::async(std::launch::async,
            [&, rgoff = std::move(rgoff), azoff = std::move(azoff),
             lineStart, blockLength]() mutable {
                rgoffRaster.setBlock(rgoff, 0, lineStart, demWidth, blockLength);
                azoffRaster.setBlock(azoff, 0, lineStart, demWidth, blockLength);
            });

    } // end for loop blocks in DEM image
    written.get();

    // Print out convergence statistics
    info << "Total convergence: " << converged << " out of "
         << (demWidth * demLength) << pyre::journal::endl;
}

// Run topo and geo2rdr of its targets in one pass
void isce3::geometry::Geo2rdr::
geo2rdr(Topo & topo, isce3::io::Raster & demRaster,
        const std::string & outdir, double azshift, double rgshift)
{
    pyre::journal::info_t info("isce.geometry.Geo2rdr");

    // Outputs on the topo radar grid
    const size_t width = topo.radarGridParameters().width();
    const size_t length = topo.radarGridParameters().length();
    Raster rgoffRaster(outdir + "/range.off", width, length, 1, GDT_Float64, "ISCE");
    Raster azoffRaster(outdir + "/azimuth.off", width, length, 1, GDT_Float64, "ISCE");

    _projTopo = isce3::core::createProj(topo.epsgOut());
    const Extents e = _extents(azshift, rgshift);
    _printExtents(info, e.t0, e.tend, e.dtaz, e.r0, e.rngend, e.dmrg,
                  width, length);
    _checkOrbitInterpolation(0.5 * (e.t0 + e.tend));

    // geo2rdr and writing of each topo block in a background task (one at a
    // time) while topo computes the next block
    size_t converged = 0;
    std::future<size_t> pending;
    TopoLayers layers(topo.linesPerBlock(), width);
    topo.computeMask(false);  // only x, y, height are used
    topo.topo(demRaster, layers, [&](size_t lineStart, TopoLayers & block) {
        if (pending.valid())
            converged += pending.get();
        pending = std::async(std::launch::async,
            [&, lineStart, x = std::move(block.x()), y = std::move(block.y()),
             hgt = std::move(block.z())]() {
                const size_t lines = x.size() / width;
                std::valarray<double> rgoff(x.size()), azoff(x.size());
                const size_t n = _geo2rdrBlock(e, &x[0], &y[0], &hgt[0],
                        lineStart, lines, width, &rgoff[0], &azoff[0]);
                rgoffRaster.setBlock(rgoff, 0, lineStart, width, lines);
                azoffRaster.setBlock(azoff, 0, lineStart, width, lines);
                return n;
            });
    });
    if (pending.valid())
        converged += pending.get();

    info << "Total convergence: " << converged << " out of "
         << (width * length) << pyre::journal::endl;
}

// Print extents and image sizes
void isce3::geometry::Geo2rdr::
_printExtents(pyre::journal::info_t & info, double t0, double tend, double dtaz,
              double r0, double rngend, double dmrg, size_t demWidth, size_t demLength)
{
    info << pyre::journal::newline
         << "Starting acquisition time: " << t0 << pyre::journal::newline
         << "Stop acquisition time: " << tend << pyre::journal::newline
         << "Azimuth line spacing in seconds: " << dtaz << pyre::journal::newline
         << "Slant range spacing in meters: " << dmrg << pyre::journal::newline
         << "Near range (m): " << r0 << pyre::journal::newline
         << "Far range (m): " << rngend << pyre::journal::newline
         << "Radar image length: " << _radarGrid.length() << pyre::journal::newline
         << "Radar image width: " << _radarGrid.width() << pyre::journal::newline
         << "Geocoded lines: " << demLength << pyre::journal::newline
         << "Geocoded samples: " << demWidth << pyre::journal::newline;
}

// Check we can interpolate orbit to middle of DEM
void isce3::geometry::Geo2rdr::
_checkOrbitInterpolation(double aztime)
{
    Vec3 pos, vel;
    _orbit.interpolate(&pos, &vel, aztime);
}

// end of file
