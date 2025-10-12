#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <iostream>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Forward declaration of the Quaternion struct from the C++ file
struct Quaternion {
    float w, x, y, z;
    Quaternion(float w = 1.0f, float x = 0.0f, float y = 0.0f, float z = 0.0f) 
        : w(w), x(x), y(y), z(z) {}
};

struct CudaQuaternion {
    float w, x, y, z;
    
    __device__ CudaQuaternion(float w = 1.0f, float x = 0.0f, float y = 0.0f, float z = 0.0f) 
        : w(w), x(x), y(y), z(z) {}
    
    __device__ void normalize() {
        float norm = sqrtf(w*w + x*x + y*y + z*z);
        if (norm > 1e-8f) {
            w /= norm; x /= norm; y /= norm; z /= norm;
        }
    }
    
    __device__ static CudaQuaternion slerp(const CudaQuaternion& q1, const CudaQuaternion& q2, float t) {
        float dot = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z;
        
        CudaQuaternion q2_adj = q2;
        if (dot < 0.0f) {
            dot = -dot;
            q2_adj.w = -q2_adj.w;
            q2_adj.x = -q2_adj.x;
            q2_adj.y = -q2_adj.y;
            q2_adj.z = -q2_adj.z;
        }
        
        if (dot > 0.9995f) {
            CudaQuaternion result(
                q1.w + t * (q2_adj.w - q1.w),
                q1.x + t * (q2_adj.x - q1.x),
                q1.y + t * (q2_adj.y - q1.y),
                q1.z + t * (q2_adj.z - q1.z)
            );
            result.normalize();
            return result;
        } else {
            float theta_0 = acosf(fabs(dot)); // Use fabs to ensure non-negative input
            float theta = theta_0 * t;
            float sin_theta_0 = sinf(theta_0);
            float sin_theta = sinf(theta);
            
            if (sin_theta_0 < 1e-8f) {
                return q1; // Fallback to avoid division by zero
            }
            
            float s0 = cosf(theta) - dot * sin_theta / sin_theta_0;
            float s1 = sin_theta / sin_theta_0;
            
            return CudaQuaternion(
                s0 * q1.w + s1 * q2_adj.w,
                s0 * q1.x + s1 * q2_adj.x,
                s0 * q1.y + s1 * q2_adj.y,
                s0 * q1.z + s1 * q2_adj.z
            );
        }
    }
};

__device__ CudaQuaternion weighted_quaternion_average_cuda(
    const CudaQuaternion* quaternions, const float* weights, int size) {
    if (size <= 0) {
        return CudaQuaternion();
    }
    if (size == 1) {
        return quaternions[0];
    }
    
    // Normalize weights first
    float total_weight = 0.0f;
    for (int i = 0; i < size; ++i) {
        total_weight += weights[i];
    }
    
    if (total_weight < 1e-8f) {
        return CudaQuaternion(); // Return identity if no weight
    }
    
    CudaQuaternion result = quaternions[0];
    float accumulated_weight = weights[0];
    
    for (int i = 1; i < size; ++i) {
        if (weights[i] > 1e-8f) {
            float blend_factor = weights[i] / (accumulated_weight + weights[i]);
            result = CudaQuaternion::slerp(result, quaternions[i], blend_factor);
            accumulated_weight += weights[i];
        }
    }
    
    return result;
}

__global__ void compute_rotation_affine_cuda_kernel(
    const float* __restrict__ polyline,
    const float* __restrict__ rest_polyline,
    const int* __restrict__ primitive_binding_id,
    const float* __restrict__ primitive_binding_time,
    const float* __restrict__ vertex_quaternions_data,
    float* __restrict__ output,
    int64_t N, int64_t L
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= N) return;
    
    int64_t id = primitive_binding_id[i];
    float t = primitive_binding_time[i];
    
    // Bounds checking
    if (id < 0 || id >= L - 1) {
        output[i * 4 + 0] = 0.0f;
        output[i * 4 + 1] = 0.0f;
        output[i * 4 + 2] = 0.0f;
        output[i * 4 + 3] = 1.0f;
        return;
    }
    
    t = fmaxf(0.0f, fminf(1.0f, t));
    
    CudaQuaternion neighbor_quats[4];
    float weights[4];
    
    for (int offset = -1; offset <= 2; ++offset) {
        int64_t vertex_idx = id + offset;
        
        // Bounds checking
        vertex_idx = max((int64_t)0, min(vertex_idx, L - 1));
        
        // Load quaternion data safely
        neighbor_quats[offset + 1] = CudaQuaternion(
            vertex_quaternions_data[vertex_idx * 4 + 3], // w
            vertex_quaternions_data[vertex_idx * 4 + 0], // x
            vertex_quaternions_data[vertex_idx * 4 + 1], // y
            vertex_quaternions_data[vertex_idx * 4 + 2]  // z
        );
        
        float distance = fabsf(offset - t);
        
        float weight;
        if (distance <= 1.0f) {
            weight = 1.0f - 1.5f * distance * distance + 0.75f * distance * distance * distance;
        } else if (distance <= 2.0f) {
            float d = 2.0f - distance;
            weight = 0.25f * d * d * d;
        } else {
            weight = 0.0f;
        }
        
        weight = fmaxf(0.0f, weight);
        weights[offset + 1] = weight;
    }
    
    CudaQuaternion q_interpolated = weighted_quaternion_average_cuda(neighbor_quats, weights, 4);
    
    // Store results
    output[i * 4 + 0] = q_interpolated.x;
    output[i * 4 + 1] = q_interpolated.y;
    output[i * 4 + 2] = q_interpolated.z;
    output[i * 4 + 3] = q_interpolated.w;
}

torch::Tensor compute_rotation_affine_batch_cuda(
    torch::Tensor polyline,
    torch::Tensor rest_polyline,
    torch::Tensor primitive_binding_id,
    torch::Tensor primitive_binding_time,
    const std::vector<Quaternion>& vertex_quaternions
) {
    // Input validation
    if (vertex_quaternions.empty()) {
        throw std::runtime_error("vertex_quaternions is empty");
    }
    
    int64_t L = polyline.size(0);
    int64_t N = primitive_binding_id.size(0);
    
    if (static_cast<int64_t>(vertex_quaternions.size()) != L) {
        throw std::runtime_error("vertex_quaternions size mismatch");
    }
    
    // Ensure tensors are on CUDA and contiguous
    polyline = polyline.to(torch::kCUDA).contiguous();
    rest_polyline = rest_polyline.to(torch::kCUDA).contiguous();
    primitive_binding_id = primitive_binding_id.contiguous();
    primitive_binding_time = primitive_binding_time.contiguous();
    
    torch::Tensor output = torch::zeros({N, 4}, torch::dtype(torch::kFloat32).device(polyline.device()));
    
    // Copy vertex quaternions to GPU with proper layout
    std::vector<float> quat_data(L * 4);
    for (int64_t i = 0; i < L; ++i) {
        quat_data[i * 4 + 0] = vertex_quaternions[i].x;
        quat_data[i * 4 + 1] = vertex_quaternions[i].y;
        quat_data[i * 4 + 2] = vertex_quaternions[i].z;
        quat_data[i * 4 + 3] = vertex_quaternions[i].w;
    }
    
    torch::Tensor vertex_quats_tensor = torch::from_blob(
        quat_data.data(), {L, 4}, torch::kFloat32
    ).to(polyline.device()).clone();
    
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    
    compute_rotation_affine_cuda_kernel<<<grid_size, block_size>>>(
        polyline.data_ptr<float>(),
        rest_polyline.data_ptr<float>(),
        primitive_binding_id.data_ptr<int>(),
        primitive_binding_time.data_ptr<float>(),
        vertex_quats_tensor.data_ptr<float>(),
        output.data_ptr<float>(),
        N, L
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return output;
}
