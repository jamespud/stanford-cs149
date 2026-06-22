// Template for OneDOccupancyDecoder CUDA Kernel Submission
//
// This file shows the expected signature for your CUDA implementation.
// You must implement the kernel_body and custom_kernel functions below.

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <stdexcept>

constexpr int kNumHeads = 12;
constexpr int kHeadDim = 64;
constexpr int kWidth = kNumHeads * kHeadDim;
constexpr int kNumLatents = 1024;
constexpr int kWarpSize = 32;
constexpr float kLayerNormEps = 1e-6f;

template <typename input_t, typename weight_t, typename output_t>
__device__ void linear_layer_forward_strided(const input_t* vec_in,
                                             const weight_t* weight,
                                             const weight_t* bias, output_t* vec_out,
                                             int in_dim, int out_dim, bool apply_silu,
                                             int thread_id, int num_threads) {
    for (int out_idx = thread_id; out_idx < out_dim; out_idx += num_threads) {
        float sum = static_cast<float>(bias[out_idx]);
        for (int in_idx = 0; in_idx < in_dim; ++in_idx) {
            sum += static_cast<float>(vec_in[in_idx]) *
                   static_cast<float>(weight[out_idx * in_dim + in_idx]);
        }
        if (apply_silu) {
            sum = sum / (1.0f + expf(-sum));
        }
        vec_out[out_idx] = static_cast<output_t>(sum);
    }
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

template <typename scalar_t>
__global__ void precompute_kv_kernel(
    const scalar_t* __restrict__ latents, const scalar_t* __restrict__ attn_c_k_weight,
    const scalar_t* __restrict__ attn_c_k_bias, const scalar_t* __restrict__ attn_c_v_weight,
    const scalar_t* __restrict__ attn_c_v_bias, scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache, int64_t batch_size, int64_t num_latents, int64_t width) {
    int batch_idx = blockIdx.y;
    int latent_idx = blockIdx.x;

    if (batch_idx >= batch_size || latent_idx >= num_latents) {
        return;
    }

    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;

    const int64_t latent_offset = (batch_idx * num_latents + latent_idx) * width;
    const scalar_t* latent_row = latents + latent_offset;
    scalar_t* k_row = k_cache + latent_offset;
    scalar_t* v_row = v_cache + latent_offset;

    linear_layer_forward_strided(latent_row, attn_c_k_weight, attn_c_k_bias, k_row,
                                 static_cast<int>(width), static_cast<int>(width), false,
                                 thread_id, num_threads);
    linear_layer_forward_strided(latent_row, attn_c_v_weight, attn_c_v_bias, v_row,
                                 static_cast<int>(width), static_cast<int>(width), false,
                                 thread_id, num_threads);
}

template <typename scalar_t>
__global__ void kernel_body(
    const scalar_t* __restrict__ queries, const scalar_t* __restrict__ k_cache,
    const scalar_t* __restrict__ v_cache,
    const scalar_t* __restrict__ query_in_in_layer_weight,
    const scalar_t* __restrict__ query_in_in_layer_bias,
    const scalar_t* __restrict__ query_in_out_layer_weight,
    const scalar_t* __restrict__ query_in_out_layer_bias,
    const scalar_t* __restrict__ attn_c_q_weight, const scalar_t* __restrict__ attn_c_q_bias,
    const scalar_t* __restrict__ attn_c_proj_weight,
    const scalar_t* __restrict__ attn_c_proj_bias, const scalar_t* __restrict__ out_proj_weight,
    const scalar_t* __restrict__ out_proj_bias, scalar_t* __restrict__ output,
    scalar_t* __restrict__ intermediate, int64_t batch_size, int64_t num_queries,
    int64_t num_latents, int64_t q_in_dim, int64_t width, int64_t num_heads) {
    int batch_idx = blockIdx.y;
    int query_idx = blockIdx.x;
    int head_idx = threadIdx.y;
    int lane = threadIdx.x;

    if (batch_idx >= batch_size || query_idx >= num_queries || head_idx >= num_heads) {
        return;
    }

    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    int head_dim = static_cast<int>(width / num_heads);
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    __shared__ float q_buf0[kWidth];
    __shared__ float q_buf1[kWidth];
    __shared__ float attn_buf[kWidth];
    __shared__ float proj_buf[kWidth];

    const int64_t query_offset = (batch_idx * num_queries + query_idx) * q_in_dim;
    const scalar_t* query_row = queries + query_offset;

    linear_layer_forward_strided(query_row, query_in_in_layer_weight, query_in_in_layer_bias,
                                 q_buf0, static_cast<int>(q_in_dim), static_cast<int>(width),
                                 true, thread_id, num_threads);
    __syncthreads();

    linear_layer_forward_strided(q_buf0, query_in_out_layer_weight, query_in_out_layer_bias,
                                 q_buf1, static_cast<int>(width), static_cast<int>(width),
                                 false, thread_id, num_threads);
    __syncthreads();

    linear_layer_forward_strided(q_buf1, attn_c_q_weight, attn_c_q_bias, q_buf0,
                                 static_cast<int>(width), static_cast<int>(width), false,
                                 thread_id, num_threads);
    __syncthreads();

    const int head_start = head_idx * head_dim;
    float max_score = -INFINITY;

    // First pass: compute max score for stable softmax
    for (int latent_idx = 0; latent_idx < num_latents; ++latent_idx) {
        const int64_t latent_offset = (batch_idx * num_latents + latent_idx) * width + head_start;
        float partial = 0.0f;
        partial += q_buf0[head_start + lane] * static_cast<float>(k_cache[latent_offset + lane]);
        partial += q_buf0[head_start + lane + kWarpSize] *
                   static_cast<float>(k_cache[latent_offset + lane + kWarpSize]);
        float score = warp_reduce_sum(partial);
        if (lane == 0) {
            max_score = fmaxf(max_score, score * scale);
        }
    }
    max_score = __shfl_sync(0xffffffffu, max_score, 0);

    // Second pass: compute sum of exponents
    float sum_exp = 0.0f;
    for (int latent_idx = 0; latent_idx < num_latents; ++latent_idx) {
        const int64_t latent_offset = (batch_idx * num_latents + latent_idx) * width + head_start;
        float partial = 0.0f;
        partial += q_buf0[head_start + lane] * static_cast<float>(k_cache[latent_offset + lane]);
        partial += q_buf0[head_start + lane + kWarpSize] *
                   static_cast<float>(k_cache[latent_offset + lane + kWarpSize]);
        float score = warp_reduce_sum(partial);
        if (lane == 0) {
            sum_exp += expf(score * scale - max_score);
        }
    }
    sum_exp = __shfl_sync(0xffffffffu, sum_exp, 0);
    sum_exp = fmaxf(sum_exp, 1e-20f);

    // Third pass: compute weighted sum of V values
    float out0 = 0.0f;
    float out1 = 0.0f;
    for (int latent_idx = 0; latent_idx < num_latents; ++latent_idx) {
        const int64_t latent_offset = (batch_idx * num_latents + latent_idx) * width + head_start;
        float partial = 0.0f;
        partial += q_buf0[head_start + lane] * static_cast<float>(k_cache[latent_offset + lane]);
        partial += q_buf0[head_start + lane + kWarpSize] *
                   static_cast<float>(k_cache[latent_offset + lane + kWarpSize]);
        float score = warp_reduce_sum(partial);
        float weight = 0.0f;
        if (lane == 0) {
            weight = expf(score * scale - max_score) / sum_exp;
        }
        weight = __shfl_sync(0xffffffffu, weight, 0);
        out0 += weight * static_cast<float>(v_cache[latent_offset + lane]);
        out1 += weight * static_cast<float>(v_cache[latent_offset + lane + kWarpSize]);
    }

    attn_buf[head_start + lane] = out0;
    attn_buf[head_start + lane + kWarpSize] = out1;
    __syncthreads();

    linear_layer_forward_strided(attn_buf, attn_c_proj_weight, attn_c_proj_bias, proj_buf,
                                 static_cast<int>(width), static_cast<int>(width), false,
                                 thread_id, num_threads);
    __syncthreads();

    if (thread_id == 0) {
        float mean = 0.0f;
        for (int i = 0; i < width; ++i) {
            mean += proj_buf[i];
        }
        mean /= static_cast<float>(width);

        float var = 0.0f;
        for (int i = 0; i < width; ++i) {
            float diff = proj_buf[i] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(width);

        float inv_std = rsqrtf(var + kLayerNormEps);
        const int64_t output_offset = (batch_idx * num_queries + query_idx) * width;
        for (int i = 0; i < width; ++i) {
            float norm = (proj_buf[i] - mean) * inv_std;
            proj_buf[i] = norm;
            intermediate[output_offset + i] = static_cast<scalar_t>(norm);
        }

        float final_sum = static_cast<float>(out_proj_bias[0]);
        for (int i = 0; i < width; ++i) {
            final_sum += proj_buf[i] * static_cast<float>(out_proj_weight[i]);
        }

        output[batch_idx * num_queries + query_idx] = static_cast<scalar_t>(final_sum);
    }
}

// Required: Main function that will be called from Python
// Signature must match the updated input_t format with all weights
torch::Tensor custom_kernel(
    torch::Tensor queries, torch::Tensor latents, torch::Tensor query_in_in_layer_weight,
    torch::Tensor query_in_in_layer_bias, torch::Tensor query_in_out_layer_weight,
    torch::Tensor query_in_out_layer_bias, torch::Tensor attn_c_q_weight,
    torch::Tensor attn_c_q_bias, torch::Tensor attn_c_k_weight, torch::Tensor attn_c_k_bias,
    torch::Tensor attn_c_v_weight, torch::Tensor attn_c_v_bias, torch::Tensor attn_c_proj_weight,
    torch::Tensor attn_c_proj_bias, torch::Tensor out_proj_weight, torch::Tensor out_proj_bias) {
    auto batch_size = queries.size(0);
    auto num_queries = queries.size(1);
    auto q_in_dim = queries.size(2);
    auto num_latents = latents.size(1);
    auto width = latents.size(2);

    if (width != kWidth || num_latents != kNumLatents) {
        throw std::runtime_error("This kernel expects width=768 and num_latents=1024.");
    }

    auto output = torch::empty({batch_size, num_queries, 1}, queries.options());
    auto intermediate = torch::empty({batch_size, num_queries, width}, queries.options());
    auto k_cache = torch::empty_like(latents);
    auto v_cache = torch::empty_like(latents);

    dim3 blockDim(kWarpSize, kNumHeads);
    dim3 kvGrid(num_latents, batch_size);
    dim3 queryGrid(num_queries, batch_size);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(queries.scalar_type(), "precompute_kv_kernel", ([&] {
        precompute_kv_kernel<scalar_t><<<kvGrid, blockDim>>>(
            latents.data_ptr<scalar_t>(), attn_c_k_weight.data_ptr<scalar_t>(),
            attn_c_k_bias.data_ptr<scalar_t>(), attn_c_v_weight.data_ptr<scalar_t>(),
            attn_c_v_bias.data_ptr<scalar_t>(), k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(), batch_size, num_latents, width);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(queries.scalar_type(), "kernel_body", ([&] {
        kernel_body<scalar_t><<<queryGrid, blockDim>>>(
            queries.data_ptr<scalar_t>(), k_cache.data_ptr<scalar_t>(), v_cache.data_ptr<scalar_t>(),
            query_in_in_layer_weight.data_ptr<scalar_t>(),
            query_in_in_layer_bias.data_ptr<scalar_t>(),
            query_in_out_layer_weight.data_ptr<scalar_t>(),
            query_in_out_layer_bias.data_ptr<scalar_t>(), attn_c_q_weight.data_ptr<scalar_t>(),
            attn_c_q_bias.data_ptr<scalar_t>(), attn_c_proj_weight.data_ptr<scalar_t>(),
            attn_c_proj_bias.data_ptr<scalar_t>(), out_proj_weight.data_ptr<scalar_t>(),
            out_proj_bias.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            intermediate.data_ptr<scalar_t>(), batch_size, num_queries, num_latents, q_in_dim,
            width, kNumHeads);
    }));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    return output;
}
