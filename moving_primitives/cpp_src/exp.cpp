#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <vector>

using namespace std;

namespace py = pybind11;

torch::Tensor compute_primitive_displacement(
    py::array_t<float, py::array::c_style | py::array::forcecast> polyline,
    py::array_t<float, py::array::c_style | py::array::forcecast> rest_polyline,
    py::array_t<int, py::array::c_style | py::array::forcecast> primitive_binding_id,
    py::array_t<float, py::array::c_style | py::array::forcecast> primitive_binding_t_value
);

PYBIND11_MODULE(moving_primitives, m) {
    m.def("compute_primitive_displacement", &compute_primitive_displacement,
          "Compute displacement for each primitive via polyline interpolation");
}
