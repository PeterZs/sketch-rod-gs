#include <torch/extension.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cmath>

#include <iostream>

namespace py = pybind11;

torch::Tensor compute_primitive_displacement(
    py::array_t<float, py::array::c_style | py::array::forcecast> polyline,
    py::array_t<float, py::array::c_style | py::array::forcecast> rest_polyline,
    py::array_t<int, py::array::c_style | py::array::forcecast> primitive_binding_id,
    py::array_t<float, py::array::c_style | py::array::forcecast> primitive_binding_t_value
) {
    auto pl = polyline.unchecked<2>();
    auto rest_pl = rest_polyline.unchecked<2>();
    auto ids = primitive_binding_id.unchecked<1>();
    auto ts = primitive_binding_t_value.unchecked<1>();

    int64_t M = ids.shape(0);

    TORCH_CHECK(pl.shape(1) == 3, "polyline must have shape (N, 3)");
    TORCH_CHECK(rest_pl.shape(1) == 3, "rest_polyline must have shape (N, 3)");

    torch::Tensor dx = torch::zeros({M, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto dx_acc = dx.packed_accessor32<float, 2>();

    for (int64_t i = 0; i < M; ++i) {
        int j = ids(i);
        float t = ts(i);
        if (j < 0 || j + 1 >= pl.shape(0)) continue;

        float diff[3];
        for (int k = 0; k < 3; ++k) {
            float d1 = pl(j, k) - rest_pl(j, k);
            float d2 = pl(j+1, k) - rest_pl(j+1, k);
            diff[k] = d1 * (1.0f - t) + d2 * t;
        }

        for (int k = 0; k < 3; ++k) {
            dx_acc[i][k] = diff[k];
        }
    }

    return dx;
}
