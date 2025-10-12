#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

using Point = pair<int, int>;

namespace py = pybind11;

std::tuple<vector<vector<Point>>, int, float, pair<int, int>, bool> dijkstra(
    const Eigen::Matrix<float, Eigen::Dynamic, 3>& points, // Center point of primitives
    py::array_t<int> most_contribute_id, // Primitive id that has strongest contribute on each pixel
    py::array_t<uint8_t> colors, // Rendered image
    py::array_t<int> strip_ids, // Correspond segment on each pixel
    py::array_t<float> strip_dists,  // Distant to each segment of polyline
    py::array_t<float> strip_times,  // Time(location) on each segment of polyline
    const pair<int, int> start, // Start pixel
    const pair<int, int> goal, // Goal pixel
    float dist_3d_threshhold, 
    int width, int height, bool debug
);

PYBIND11_MODULE(dijkstra_cpp, m)
{
    m.def("dijkstra", &dijkstra, "A dijkstra method function with cpp");
}
