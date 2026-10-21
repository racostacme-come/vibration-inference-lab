#include "state.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
PYBIND11_MODULE(_core, m) {
  m.doc() = "O(N+tail) zero-state forward and Euclidean adjoint recurrences";
  py::class_<Recurrence>(m, "Recurrence")
      .def(py::init<std::array<double, 4>, std::array<double, 2>, std::size_t, std::size_t>())
      .def("forward", &Recurrence::forward)
      .def("adjoint", &Recurrence::adjoint);
}
