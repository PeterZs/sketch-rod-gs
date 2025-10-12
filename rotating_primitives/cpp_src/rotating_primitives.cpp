#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <cmath>
#include <iostream>
#include <array>

namespace py = pybind11;

// Forward declaration for CUDA function (moved to top)
struct Quaternion;
torch::Tensor compute_rotation_affine_batch_cuda(
    torch::Tensor polyline,
    torch::Tensor rest_polyline,
    torch::Tensor primitive_binding_id,
    torch::Tensor primitive_binding_time,
    const std::vector<Quaternion>& vertex_quaternions
);

// Quaternion structure and operations
struct Quaternion {
    float w, x, y, z;
    
    Quaternion(float w = 1.0f, float x = 0.0f, float y = 0.0f, float z = 0.0f) 
        : w(w), x(x), y(y), z(z) {}
    
    // Normalize quaternion
    void normalize() {
        float norm = std::sqrt(w*w + x*x + y*y + z*z);
        if (norm > 1e-8f) {
            w /= norm; x /= norm; y /= norm; z /= norm;
        }
    }
    
    // SLERP interpolation between two quaternions
    static Quaternion slerp(const Quaternion& q1, const Quaternion& q2, float t) {
        float dot = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z;
        
        // If dot product is negative, use -q2 to ensure shorter path
        Quaternion q2_adj = q2;
        if (dot < 0.0f) {
            dot = -dot;
            q2_adj.w = -q2_adj.w;
            q2_adj.x = -q2_adj.x;
            q2_adj.y = -q2_adj.y;
            q2_adj.z = -q2_adj.z;
        }
        
        if (dot > 0.9995f) {
            // Linear interpolation for very close quaternions
            Quaternion result(
                q1.w + t * (q2_adj.w - q1.w),
                q1.x + t * (q2_adj.x - q1.x),
                q1.y + t * (q2_adj.y - q1.y),
                q1.z + t * (q2_adj.z - q1.z)
            );
            result.normalize();
            return result;
        } else {
            // Spherical linear interpolation
            float theta_0 = std::acos(dot);
            float theta = theta_0 * t;
            float sin_theta_0 = std::sin(theta_0);
            float sin_theta = std::sin(theta);
            
            float s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
            float s1 = sin_theta / sin_theta_0;
            
            return Quaternion(
                s0 * q1.w + s1 * q2_adj.w,
                s0 * q1.x + s1 * q2_adj.x,
                s0 * q1.y + s1 * q2_adj.y,
                s0 * q1.z + s1 * q2_adj.z
            );
        }
    }
};

// Convert rotation matrix to quaternion
Quaternion matrix_to_quaternion(const torch::Tensor& R) {
    float r00 = R[0][0].item<float>();
    float r11 = R[1][1].item<float>();
    float r22 = R[2][2].item<float>();
    float trace = r00 + r11 + r22;
    
    if (trace > 0) {
        float s = std::sqrt(trace + 1.0f) * 2; // s = 4 * qw
        return Quaternion(
            0.25f * s,
            (R[2][1].item<float>() - R[1][2].item<float>()) / s,
            (R[0][2].item<float>() - R[2][0].item<float>()) / s,
            (R[1][0].item<float>() - R[0][1].item<float>()) / s
        );
    } else if (r00 > r11 && r00 > r22) {
        float s = std::sqrt(1.0f + r00 - r11 - r22) * 2;
        return Quaternion(
            (R[2][1].item<float>() - R[1][2].item<float>()) / s,
            0.25f * s,
            (R[0][1].item<float>() + R[1][0].item<float>()) / s,
            (R[0][2].item<float>() + R[2][0].item<float>()) / s
        );
    } else if (r11 > r22) {
        float s = std::sqrt(1.0f + r11 - r00 - r22) * 2;
        return Quaternion(
            (R[0][2].item<float>() - R[2][0].item<float>()) / s,
            (R[0][1].item<float>() + R[1][0].item<float>()) / s,
            0.25f * s,
            (R[1][2].item<float>() + R[2][1].item<float>()) / s
        );
    } else {
        float s = std::sqrt(1.0f + r22 - r00 - r11) * 2;
        return Quaternion(
            (R[1][0].item<float>() - R[0][1].item<float>()) / s,
            (R[0][2].item<float>() + R[2][0].item<float>()) / s,
            (R[1][2].item<float>() + R[2][1].item<float>()) / s,
            0.25f * s
        );
    }
}

// Convert quaternion to rotation matrix
torch::Tensor quaternion_to_matrix(const Quaternion& q) {
    torch::Tensor R = torch::zeros({3, 3}, torch::kFloat32);
    
    float w2 = q.w * q.w;
    float x2 = q.x * q.x;
    float y2 = q.y * q.y;
    float z2 = q.z * q.z;
    
    R[0][0] = w2 + x2 - y2 - z2;
    R[0][1] = 2 * (q.x * q.y - q.w * q.z);
    R[0][2] = 2 * (q.x * q.z + q.w * q.y);
    
    R[1][0] = 2 * (q.x * q.y + q.w * q.z);
    R[1][1] = w2 - x2 + y2 - z2;
    R[1][2] = 2 * (q.y * q.z - q.w * q.x);
    
    R[2][0] = 2 * (q.x * q.z - q.w * q.y);
    R[2][1] = 2 * (q.y * q.z + q.w * q.x);
    R[2][2] = w2 - x2 - y2 + z2;
    
    return R;
}

// Weighted average of multiple quaternions using SLERP
Quaternion weighted_quaternion_average(const std::vector<Quaternion>& quaternions, const std::vector<float>& weights) {
    if (quaternions.empty()) {
        return Quaternion(); // Identity quaternion
    }
    if (quaternions.size() == 1) {
        return quaternions[0];
    }
    
    // Start with the first quaternion
    Quaternion result = quaternions[0];
    float total_weight = weights[0];
    
    // Progressively blend with other quaternions
    for (size_t i = 1; i < quaternions.size(); ++i) {
        float new_weight = weights[i];
        float blend_factor = new_weight / (total_weight + new_weight);
        result = Quaternion::slerp(result, quaternions[i], blend_factor);
        total_weight += new_weight;
    }
    
    return result;
}

torch::Tensor rotation_matrix_from_axis_angle(const std::array<float, 3>& axis, float angle) {
    float x = axis[0], y = axis[1], z = axis[2];
    float c = std::cos(angle);
    float s = std::sin(angle);
    float C = 1.0f - c;

    torch::Tensor R = torch::empty({3, 3}, torch::kFloat32);
    R[0][0] = c + x*x*C;
    R[0][1] = x*y*C - z*s;
    R[0][2] = x*z*C + y*s;

    R[1][0] = y*x*C + z*s;
    R[1][1] = c + y*y*C;
    R[1][2] = y*z*C - x*s;

    R[2][0] = z*x*C - y*s;
    R[2][1] = z*y*C + x*s;
    R[2][2] = c + z*z*C;

    return R;
}

// Compute rotation matrix for a single segment
torch::Tensor compute_segment_rotation(
    const std::array<float, 3>& p0, const std::array<float, 3>& p1,
    const std::array<float, 3>& q0, const std::array<float, 3>& q1
) {
    std::array<float, 3> v0, v1;
    float norm_v0 = 0, norm_v1 = 0;
    for (int j = 0; j < 3; ++j) {
        v0[j] = p1[j] - p0[j];
        v1[j] = q1[j] - q0[j];
        norm_v0 += v0[j] * v0[j];
        norm_v1 += v1[j] * v1[j];
    }
    norm_v0 = std::sqrt(norm_v0);
    norm_v1 = std::sqrt(norm_v1);
    
    if (norm_v0 < 1e-8f || norm_v1 < 1e-8f) {
        return torch::eye(3, torch::kFloat32);
    }
    
    for (int j = 0; j < 3; ++j) {
        v0[j] /= norm_v0;
        v1[j] /= norm_v1;
    }

    std::array<float, 3> axis = {
        v0[1]*v1[2] - v0[2]*v1[1],
        v0[2]*v1[0] - v0[0]*v1[2],
        v0[0]*v1[1] - v0[1]*v1[0]
    };
    float sin_theta = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    float cos_theta = v0[0]*v1[0] + v0[1]*v1[1] + v0[2]*v1[2];

    torch::Tensor ret;
    if (sin_theta < 1e-8f) {
        if (cos_theta > 0.9999f) {
            return torch::eye(3, torch::kFloat32);
        } else {
            std::array<float, 3> ortho = {1, 0, 0};
            if (std::abs(v0[0]) > std::abs(v0[1])) {
                if (std::abs(v0[1]) > std::abs(v0[2])) ortho = {0, 0, 1};
                else ortho = {0, 1, 0};
            }

            axis = {
                v0[1]*ortho[2] - v0[2]*ortho[1],
                v0[2]*ortho[0] - v0[0]*ortho[2],
                v0[0]*ortho[1] - v0[1]*ortho[0]
            };
            float norm_axis = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
            for (int j = 0; j < 3; ++j) axis[j] /= norm_axis;

            ret = rotation_matrix_from_axis_angle(axis, M_PI);
        }
    } else {
        for (int j = 0; j < 3; ++j) axis[j] /= sin_theta;
        float angle = std::atan2(sin_theta, cos_theta);
        ret = rotation_matrix_from_axis_angle(axis, angle);
    }
    return ret.transpose(0, 1);
}

torch::Tensor compute_rotation_affine_batch(
    py::array_t<float, py::array::c_style | py::array::forcecast> polyline,
    py::array_t<float, py::array::c_style | py::array::forcecast> rest_polyline,
    torch::Tensor primitive_binding_id, 
    torch::Tensor primitive_binding_time
) {
    auto pl = polyline.unchecked<2>();
    auto rest = rest_polyline.unchecked<2>();

    int64_t L = pl.shape(0);
    int64_t N = primitive_binding_id.size(0);
    int64_t num_segments = L - 1;
    
    // Input validation
    if (L < 2) {
        std::cerr << "Error: polyline must have at least 2 points" << std::endl;
        return torch::zeros({N, 4}, torch::kFloat32);
    }
    
    if (N == 0) {
        std::cerr << "Warning: no primitives to process" << std::endl;
        return torch::zeros({N, 4}, torch::kFloat32);
    }
    
    // Step 1: Compute rotation matrix for each segment (CPU)
    // auto start_step1 = std::chrono::high_resolution_clock::now();
    std::vector<torch::Tensor> segment_rotations(num_segments);
    for (int64_t seg = 0; seg < num_segments; ++seg) {
        std::array<float, 3> p0 = {pl(seg, 0), pl(seg, 1), pl(seg, 2)};
        std::array<float, 3> p1 = {pl(seg + 1, 0), pl(seg + 1, 1), pl(seg + 1, 2)};
        std::array<float, 3> q0 = {rest(seg, 0), rest(seg, 1), rest(seg, 2)};
        std::array<float, 3> q1 = {rest(seg + 1, 0), rest(seg + 1, 1), rest(seg + 1, 2)};
        
        segment_rotations[seg] = compute_segment_rotation(p0, p1, q0, q1);
    }
    // Step 2: Compute rotation matrix for each vertex (CPU)
    std::vector<Quaternion> vertex_quaternions(L);
    
    vertex_quaternions[0] = matrix_to_quaternion(segment_rotations[0]);
    
    for (int64_t v = 1; v < L - 1; ++v) {
        Quaternion q1 = matrix_to_quaternion(segment_rotations[v - 1]);
        Quaternion q2 = matrix_to_quaternion(segment_rotations[v]);
        vertex_quaternions[v] = Quaternion::slerp(q1, q2, 0.5f);
    }
    
    vertex_quaternions[L - 1] = matrix_to_quaternion(segment_rotations[num_segments - 1]);
    
    try {
        // Convert numpy arrays to torch tensors for CUDA processing
        torch::Tensor polyline_tensor = torch::from_blob(
            const_cast<float*>(pl.data(0, 0)), {L, 3}, torch::kFloat32
        ).clone();
        
        torch::Tensor rest_polyline_tensor = torch::from_blob(
            const_cast<float*>(rest.data(0, 0)), {L, 3}, torch::kFloat32
        ).clone();

        // Step 3: Use CUDA for the parallel interpolation
        torch::Tensor result = compute_rotation_affine_batch_cuda(
            polyline_tensor, rest_polyline_tensor, primitive_binding_id, primitive_binding_time, vertex_quaternions
        );

        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in CUDA computation: " << e.what() << std::endl;
        // Return identity quaternions as fallback
        return torch::tensor({{0.0f, 0.0f, 0.0f, 1.0f}}).repeat({N, 1});
    }
}
