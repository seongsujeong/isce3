#include "matchtemplate.h"

#include "pycuampcor.h"

#include <isce3/matchtemplate/pycuampcor/cuMetal.h>

namespace py = pybind11;

void addsubmodule_matchtemplate(py::module& m)
{
    py::module m_matchtemplate = m.def_submodule("matchtemplate");

    addbinding_pycuampcor_cpu(m_matchtemplate);
    m_matchtemplate.def("metal_available",
            &isce3::matchtemplate::pycuampcor::metalAvailable,
            "Whether CPU ampcor can run supported steps on a Metal GPU "
            "(set PyCPUAmpcor.useMetal)");
}
