#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <vector>

using namespace std;

namespace py = pybind11;

torch::Tensor compute_rotation_affine_batch(
    py::array_t<float, py::array::c_style | py::array::forcecast> polyline,
    py::array_t<float, py::array::c_style | py::array::forcecast> rest_polyline,
    torch::Tensor primitive_binding_id, 
    torch::Tensor primitive_binding_time
);

PYBIND11_MODULE(rotating_primitives, m) {
    m.def("compute_rotation_affine_batch", &compute_rotation_affine_batch, "Compute rotation matrix from line segment");
}
