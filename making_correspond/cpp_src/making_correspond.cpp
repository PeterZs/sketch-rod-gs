#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <cmath>
#include <limits>

namespace py = pybind11;

std::pair<py::array_t<int>, py::array_t<float>> making_correspondance(
    py::array_t<float, py::array::c_style | py::array::forcecast> points,
    py::array_t<float, py::array::c_style | py::array::forcecast> polyline,
    float radius)
{
    auto pts = points.unchecked<2>();      // (N, 3)
    auto pl = polyline.unchecked<2>();     // (M, 3)

    const ssize_t n_pts = pts.shape(0);
    const ssize_t n_pl = pl.shape(0);

    py::array_t<int> primitive_binding_id(n_pts);
    py::array_t<float> primitive_binding_time(n_pts);

    auto id_out = primitive_binding_id.mutable_unchecked<1>();
    auto t_out = primitive_binding_time.mutable_unchecked<1>();

    // Parallelizing using OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (ssize_t i = 0; i < n_pts; i++) {
        id_out(i) = -1;
        t_out(i) = -1.0f;
        for (ssize_t j = 0; j < n_pl - 1; ++j) {
            float p1[3], p2[3], dir[3], v[3];
            for (int k = 0; k < 3; ++k) {
                p1[k] = pl(j, k);
                p2[k] = pl(j + 1, k);
                dir[k] = p2[k] - p1[k];
                v[k] = pts(i, k);
            }

            float denom = dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2];
            if (denom < 1e-8f) continue;

            float t_num = (v[0]-p1[0])*dir[0] + (v[1]-p1[1])*dir[1] + (v[2]-p1[2])*dir[2];
            float t = t_num / denom;
            if (t < 0.0f || t > 1.0f) continue;

            float proj[3] = {
                p1[0] + t * dir[0],
                p1[1] + t * dir[1],
                p1[2] + t * dir[2]
            };

            float dist = std::sqrt(
                (v[0] - proj[0]) * (v[0] - proj[0]) +
                (v[1] - proj[1]) * (v[1] - proj[1]) +
                (v[2] - proj[2]) * (v[2] - proj[2])
            );

            if (dist <= radius) {
                id_out(i) = static_cast<int>(j);
                t_out(i) = t;
            }
        }
    }

    // Parallelizing using OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (ssize_t i = 0; i < n_pts; i++)
    {
        if (id_out(i) != -1) continue;
        for (ssize_t j = 0; j < n_pl; j++) {
            float p[3], v[3];
            for (int k = 0; k < 3; ++k) {
                p[k] = pl(j, k);
                v[k] = pts(i, k);
            }
            float dist = std::sqrt(
                (p[0] - v[0]) * (p[0] - v[0]) + 
                (p[1] - v[1]) * (p[1] - v[1]) + 
                (p[2] - v[2]) * (p[2] - v[2])
            );
            if (dist < radius) {
                if (j == n_pl - 1)
                    id_out(i) = static_cast<int>(j - 1);
                else
                    id_out(i) = static_cast<int>(j);
                t_out(i) = 0;
            }
        }
    }

    return std::make_pair(primitive_binding_id, primitive_binding_time);
}
