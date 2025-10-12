#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

std::tuple<
    py::array_t<int>, 
    py::array_t<float>, 
    py::array_t<float>, 
    std::vector<float>, 
    float, 
    std::vector<std::pair<float, float>>
>
preprocess(
    int W, 
    int H, 
    const std::vector<std::pair<float, float>>& lines,
    py::array_t<float> depth_of_most_contribute,
    float tube_radius,
    bool debug
);

PYBIND11_MODULE(preprocess_cpp, m)
{
    m.def("preprocess", &preprocess, "preprocess function");
}
