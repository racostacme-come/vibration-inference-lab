#include "energy.hpp"
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
PYBIND11_MODULE(_core, m) {
  m.doc() = "Streaming-memory energy scores in C++17";
  m.def(
      "energy_score",
      [](Array x, Array y, bool unbiased) {
        if (x.ndim() != 2 || y.ndim() != 2 || x.shape(1) != y.shape(1))
          throw std::invalid_argument("Expected (members, dimension) and (cases, dimension)");
        auto scores = jointguard::energy(x.data(), static_cast<std::size_t>(x.shape(0)),
                                         y.data(), static_cast<std::size_t>(y.shape(0)),
                                         static_cast<std::size_t>(x.shape(1)), unbiased);
        py::array_t<double> out(scores.size());
        std::copy(scores.begin(), scores.end(), out.mutable_data());
        return out;
      },
      py::arg("ensemble"), py::arg("observations"), py::arg("unbiased") = true,
      "Energy score per observation; unbiased correction requires iid members.");
}
