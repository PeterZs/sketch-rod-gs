#include <vector>
#include <queue>
#include <iostream>
#include <limits>
#include <cmath>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

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
) {

    if (debug)
        cout << "dijkstra" << endl;

    int start_x = static_cast<int>(start.first);
    int start_y = static_cast<int>(start.second);
    int end_x   = static_cast<int>(goal.first);
    int end_y   = static_cast<int>(goal.second);

    const auto mcid = most_contribute_id.unchecked<2>();

    const auto strip_ids_ = strip_ids.unchecked<2>();
    const auto strip_dists_ = strip_dists.unchecked<2>();
    const auto strip_times_ = strip_times.unchecked<2>();

    const float INF = numeric_limits<float>::infinity();

    vector<vector<float>> dist(height, vector<float>(width, INF));
    vector<vector<bool>> vis(height, vector<bool>(width, false));
    vector<vector<Point>> prev(height, vector<Point>(width, {-1, -1}));

    int head_segment_id = strip_ids_(start_y, start_x);
    if (debug)
        cout << "dijkstra: head_segment_id: " << head_segment_id << endl;
    float head_segment_time = strip_times_(start_y, start_x);
    pair<int, int> head_segment_pixel = pair<int, int>(start_x, start_y);

    bool reach_goal = false;

    using QElem = pair<float, Point>; // {cost, (x, y)}
    priority_queue<QElem, vector<QElem>, greater<QElem>> queue;

    dist[start_y][start_x] = 0.0f;
    queue.push({0.0f, {start_x, start_y}});

    while (!queue.empty()) {
        auto [d, pos] = queue.top(); queue.pop();
        int x = pos.first;
        int y = pos.second;

        if (vis[y][x]) continue;
        vis[y][x] = true;
        if (dist[y][x] < d) continue;

        if (head_segment_id < strip_ids_(y, x) ||
            (head_segment_id == strip_ids_(y, x) && head_segment_time < strip_times_(y, x)))
        {
            head_segment_id = strip_ids_(y, x);
            head_segment_time = strip_times_(y, x);
            head_segment_pixel = pair<int, int>(x, y);
        }

        if (x == end_x && y == end_y) {
            if (debug)
                cout << "achieved!!" << endl;
            reach_goal = true;
            break;
        }

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (abs(dy) != 1 && abs(dx) != 1) continue;
                int nx = x + dx;
                int ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;

                int id  = mcid(y, x);
                int nid = mcid(ny, nx);
                if (nid == -1) continue;

                float dist3d = (points.row(id) - points.row(nid)).norm();

                if (dist_3d_threshhold < dist3d)
                {
                    continue;
                }

                const int sid = strip_ids_(y, x);
                const int snid = strip_ids_(ny, nx);
                if (snid == -1) continue;
                if (snid < sid) continue;

                if (vis[ny][nx]) continue;

                float strip_cost = pow(strip_dists_(ny, nx), 2);
                float cost = dist[y][x] + dist3d + strip_cost;

                if (cost < dist[ny][nx]) {
                    dist[ny][nx] = cost;
                    prev[ny][nx] = {x, y};
                    queue.push({cost, {nx, ny}});
                }
            }
        }
    }

    return {prev, head_segment_id, head_segment_time, head_segment_pixel, reach_goal};
}
