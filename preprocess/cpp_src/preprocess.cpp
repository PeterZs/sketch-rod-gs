#include <vector>
#include <tuple>
#include <cmath>
#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <omp.h>

#include <iostream>

using namespace Eigen;

namespace py = pybind11;

const float INF = std::numeric_limits<float>::infinity();


static Vector2f pix2ndc(float pix_x, float pix_y, int W, int H)
{
    float x = (pix_x / W) * 2.f - 1.f;
    float y = 1.f - (pix_y / H) * 2.f;
    Vector2f point(x, y);
    return point;
}

static float ccw(const Vector2f& a, const Vector2f& b, const Vector2f& c) {
    return (b - a).x() * (c - a).y() - (b - a).y() * (c - a).x();
}

static bool is_intersect(const Vector2f& v1, const Vector2f& v2, const Vector2f& v3, const Vector2f& v4) {
    float d1 = ccw(v1, v2, v3);
    float d2 = ccw(v1, v2, v4);
    float d3 = ccw(v3, v4, v1);
    float d4 = ccw(v3, v4, v2);
    return (d1 * d2 < 0) && (d3 * d4 < 0);
}

std::vector<std::vector<std::pair<float, float>>>
static split_lines(const std::vector<std::pair<float, float>>& lines)
{
    std::vector<std::vector<std::pair<float, float>>> splitted_lines;
    std::vector<std::pair<float, float>> tmp_lines;

    for (size_t i = 0; i + 1 < lines.size(); ++i) {
        Vector2f v1(lines[i].first, lines[i].second);
        Vector2f v2(lines[i + 1].first, lines[i + 1].second);
        bool intersect = false;

        for (size_t j = 0; j + 1 < lines.size(); ++j) {
            if (i == j) continue;
            Vector2f v3(lines[j].first, lines[j].second);
            Vector2f v4(lines[j + 1].first, lines[j + 1].second);
            if (is_intersect(v1, v2, v3, v4)) {
                intersect = true;
                break;
            }
        }

        tmp_lines.push_back(lines[i]);
        if (intersect) {
            splitted_lines.push_back(tmp_lines);
            tmp_lines.clear();
        }
    }

    tmp_lines.push_back(lines.back());
    splitted_lines.push_back(tmp_lines);

    return splitted_lines;
}

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
) {
    const auto depths = depth_of_most_contribute.unchecked<2>();

    // Skip unsequenced line sample according to depth
    std::vector<std::pair<float, float>> modified_lines;
    float prev_depth;
    const float depth_diff_threshoold = tube_radius * 3;
    for (int i = 0; i < lines.size(); i++)
    {
        const int x = static_cast<int>(lines.at(i).first);
        const int y = static_cast<int>(lines.at(i).second);
        const float point_depth = depths(y, x);
        if (i != 0 && depth_diff_threshoold < std::abs(prev_depth - point_depth)) continue;
        modified_lines.push_back(lines.at(i));
        prev_depth = point_depth;
    }
    if (debug)
        std::cout << "preprocess cpp module: lines.size(): " << lines.size() << ", modified_lines.size(): " << modified_lines.size() << std::endl;

    // line NDC calculation
    std::vector<std::pair<float, float>> ndc_lines;
    for (int i = 0; i < modified_lines.size(); i++)
    {
        Vector2f point = pix2ndc(
            modified_lines.at(i).first, modified_lines.at(i).second, W, H
        );
        ndc_lines.push_back({point.x(), point.y()});
    }

    // line length calculation
    float line_len = 0.f;
    for (int i = 0; i < ndc_lines.size() - 1; i++)
    {
        std::pair<float, float> v1 = ndc_lines.at(i);
        std::pair<float, float> v2 = ndc_lines.at(i + 1);
        line_len += sqrt(
            (v1.first - v2.first) * (v1.first - v2.first) + 
            (v1.second - v2.second) * (v1.second - v2.second)
        );
    }
    
    // line time calculation
    std::vector<float> line_times = {0.f};
    float accum_dist = 0.f;
    for (int i = 0; i < ndc_lines.size() - 1; i++)
    {
        std::pair<float, float> v1 = ndc_lines.at(i);
        std::pair<float, float> v2 = ndc_lines.at(i + 1);
        accum_dist += sqrt(
            (v1.first - v2.first) * (v1.first - v2.first) + 
            (v1.second - v2.second) * (v1.second - v2.second)
        );
        line_times.push_back(accum_dist / line_len);
    }

    // Convert to numpy arrays
    const float init_value = -1.0f;
    const int init_value_int = -1;

    auto strip_ids_array = py::array_t<int>({H, W});
    auto strip_dists_array = py::array_t<float>({H, W});
    auto strip_times_array = py::array_t<float>({H, W});

    std::fill_n(static_cast<int*>(strip_ids_array.mutable_unchecked<2>().mutable_data(0, 0)), H * W, init_value_int);
    std::fill_n(static_cast<float*>(strip_dists_array.mutable_unchecked<2>().mutable_data(0, 0)), H * W, init_value);
    std::fill_n(static_cast<float*>(strip_times_array.mutable_unchecked<2>().mutable_data(0, 0)), H * W, init_value);

    auto strip_ids_ptr = static_cast<int*>(strip_ids_array.mutable_unchecked<2>().mutable_data(0, 0));
    auto strip_dists_ptr = static_cast<float*>(strip_dists_array.mutable_unchecked<2>().mutable_data(0, 0));
    auto strip_times_ptr = static_cast<float*>(strip_times_array.mutable_unchecked<2>().mutable_data(0, 0));

    
    // Parallelizing using OpenMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            Vector2f point = pix2ndc(x, y, W, H);
            float pixel_depth = depths(y, x);
            float min_dist = INF;
            int min_id = -1;
            float min_time = -1.f;

            for (int i = 0; i < ndc_lines.size() - 1; i++)
            {
                // skip line that has unsequenced depth
                const int sx = static_cast<int>(modified_lines.at(i).first);
                const int sy = static_cast<int>(modified_lines.at(i).second);
                const int ex = static_cast<int>(modified_lines.at(i + 1).first);
                const int ey = static_cast<int>(modified_lines.at(i + 1).second);
                float s_depth = depths(sy, sx);
                float e_depth = depths(ey, ex);
                float segment_depth = std::min(s_depth, e_depth);
                if (depth_diff_threshoold < std::abs(pixel_depth - segment_depth)) continue;

                // Caluculating pixel projection to line segment
                Vector2f s(ndc_lines[i].first, ndc_lines[i].second);
                Vector2f e(ndc_lines[i + 1].first, ndc_lines[i + 1].second);

                Vector2f dir = e - s;
                const float denom = dir.dot(dir);
                if (denom < 1e-8) continue;
                float t = (point - s).dot(dir) / denom;

                Vector2f proj;
                if (0.f <= t && t <= 1.f)
                {
                    proj = s + t * dir;
                }
                else
                {
                    t = (t < 0.5) ? 0.f : 1.f;
                    proj = s + t * dir;
                }

                float dist = (point - proj).norm();
                if (0.01 < dist) continue;

                if (dist < min_dist)
                {
                    min_dist = dist;
                    min_id = i;
                    min_time = t;
                }
            }

            strip_ids_ptr[y * W + x] = min_id;
            strip_dists_ptr[y * W + x] = min_dist;
            strip_times_ptr[y * W + x] = min_time;
        }
    }

    for (int i = 0; i < modified_lines.size(); i++)
    {
        const int x = static_cast<int>(modified_lines.at(i).first);
        const int y = static_cast<int>(modified_lines.at(i).second);
        strip_ids_ptr[y * W + x] = i;
        strip_dists_ptr[y * W + x] = 0.0;
        strip_times_ptr[y * W + x] = 0.0;
    }

    return {strip_ids_array, strip_dists_array, strip_times_array, line_times, line_len, modified_lines};
}
