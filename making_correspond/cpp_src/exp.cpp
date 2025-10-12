#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

using namespace std;

namespace py = pybind11;

std::pair<py::array_t<int>, py::array_t<float>> making_correspondance(
    py::array_t<float, py::array::c_style | py::array::forcecast> points,
    py::array_t<float, py::array::c_style | py::array::forcecast> polyline, 
    float radius);

PYBIND11_MODULE(making_correspond, m) {
    m.def("making_correspondance", &making_correspondance,
          "Compute segment id and local coord t for each point near polyline");
}
