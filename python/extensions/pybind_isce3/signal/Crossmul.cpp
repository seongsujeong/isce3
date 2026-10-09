#include "Crossmul.h"

#include <isce3/core/forward.h>
#include <isce3/io/Raster.h>
#include <isce3/product/forward.h>

namespace py = pybind11;

using isce3::io::Raster;
using isce3::signal::Crossmul;

void addbinding(py::class_<Crossmul> & pyCrossmul)
{
    pyCrossmul
        .def(py::init([](const int rg_looks, const int az_looks)
                    {
                        Crossmul crsml;
                        crsml.rangeLooks(rg_looks);
                        crsml.azimuthLooks(az_looks);
                        return crsml;
                    }),
                py::arg("range_looks")=1,
                py::arg("az_looks")=1,
                R"(
    Returns crossmul object with range and and azimuth multilook
    off by default.
                )")
        .def("crossmul", py::overload_cast<isce3::io::Raster&,
                    isce3::io::Raster&, isce3::io::Raster&, isce3::io::Raster&,
                    isce3::io::Raster*>(&Crossmul::crossmul, py::const_),
                py::arg("ref_slc"),
                py::arg("sec_slc"),
                py::arg("interferogram"),
                py::arg("coherence"),
                py::arg("range_offset") = nullptr, R"(
    Crossmultiply reference and secondary SLCs to generate interferogram and coherence products.

    Parameters
    ----------
    ref_slc: Raster
        Input reference SLC raster
    inputRaster: Raster
        Input secondary SLC raster
    interferogram: Raster
        Output interferogram raster
    coherence: Raster
        Output coherence raster
    interferogram: Raster
        Optional range offset raster usef for flattening
                )")
        .def("crossmul_two_looks", py::overload_cast<isce3::io::Raster&,
                    isce3::io::Raster&, isce3::io::Raster&, isce3::io::Raster&,
                    isce3::io::Raster&, isce3::io::Raster&, int, int,
                    isce3::io::Raster*>(&Crossmul::crossmul, py::const_),
                py::arg("ref_slc"),
                py::arg("sec_slc"),
                py::arg("interferogram"),
                py::arg("coherence"),
                py::arg("interferogram2"),
                py::arg("coherence2"),
                py::arg("range_looks2"),
                py::arg("az_looks2"),
                py::arg("range_offset") = nullptr, R"(
    Crossmultiply reference and secondary SLCs into the interferogram and
    coherence of the set looks and, in the same pass, of a second number of
    looks.

    Parameters
    ----------
    ref_slc: Raster
        Input reference SLC raster
    sec_slc: Raster
        Input secondary SLC raster
    interferogram: Raster
        Output interferogram raster
    coherence: Raster
        Output coherence raster
    interferogram2: Raster
        Output interferogram raster of the second looks
    coherence2: Raster
        Output coherence raster of the second looks
    range_looks2: int
        Number of range looks of the second outputs
    az_looks2: int
        Number of azimuth looks of the second outputs
    range_offset: Raster
        Optional range offset raster used for flattening
                )")
        .def("set_dopplers", &Crossmul::doppler,
                py::arg("ref_doppler"),
                py::arg("sec_doppler"))
        .def_property("ref_doppler",
                py::overload_cast<>(&Crossmul::refDoppler, py::const_),
                py::overload_cast<isce3::core::LUT1d<double>>(&Crossmul::refDoppler))
        .def_property("sec_doppler",
                py::overload_cast<>(&Crossmul::secDoppler, py::const_),
                py::overload_cast<isce3::core::LUT1d<double>>(&Crossmul::secDoppler))
        .def_property("ref_sec_offset_starting_range_shift",
                py::overload_cast<>(&Crossmul::startingRangeShift, py::const_),
                py::overload_cast<double>(&Crossmul::startingRangeShift))
        .def_property("range_pixel_spacing",
                py::overload_cast<>(&Crossmul::rangePixelSpacing, py::const_),
                py::overload_cast<double>(&Crossmul::rangePixelSpacing))
        .def_property("wavelength",
                py::overload_cast<>(&Crossmul::wavelength, py::const_),
                py::overload_cast<double>(&Crossmul::wavelength))
        .def_property("range_looks",
                py::overload_cast<>(&Crossmul::rangeLooks, py::const_),
                py::overload_cast<int>(&Crossmul::rangeLooks))
        .def_property("az_looks",
                py::overload_cast<>(&Crossmul::azimuthLooks, py::const_),
                py::overload_cast<int>(&Crossmul::azimuthLooks))
        .def_property("oversample_factor",
                py::overload_cast<>(&Crossmul::oversampleFactor, py::const_),
                py::overload_cast<size_t>(&Crossmul::oversampleFactor))
        .def_property("lines_per_block",
                py::overload_cast<>(&Crossmul::linesPerBlock, py::const_),
                py::overload_cast<size_t>(&Crossmul::linesPerBlock))
        .def_property_readonly("multilook_enabled", &Crossmul::multiLookEnabled)
        ;
}
