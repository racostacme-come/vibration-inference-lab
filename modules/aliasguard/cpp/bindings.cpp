#include "fir.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
PYBIND11_MODULE(_core, m) {
    m.doc() = "Stateful causal FIR decimator; double precision, zero prehistory";
    py::class_<aliasguard::Decimator>(m, "Decimator")
        .def(py::init<std::vector<double>, std::size_t>(), py::arg("coefficients"),
             py::arg("factor"))
        .def("process", &aliasguard::Decimator::process, py::arg("samples"))
        .def("reset", &aliasguard::Decimator::reset);
}
