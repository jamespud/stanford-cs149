// OneDOccupancyDecoder - optimized CUDA implementation.
//
// Structure
//  1. Query MLP, K/V projections and the final output projection run as fp16
//     GEMMs (cuBLAS through torch ops) - the same building blocks the PyTorch
//     baseline uses.
//  2. The cross-attention is a custom flash-attention style kernel using
//     tensor cores (WMMA): each block owns kBM queries of one head and streams
//     that head's K/V through shared memory in chunks, maintaining a running
//     (fp32) max and sum so softmax is done in a single pass.
//  3. LayerNorm + output projection are fused into one lightweight kernel.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

#include <cmath>
#include <stdexcept>

using namespace nvcuda;

namespace {

constexpr int kNumHeads = 12;
constexpr int kHeadDim = 64;
constexpr int kWidth = 768;
constexpr int kBM = 128;   // queries per block
constexpr int kBK = 128;   // latents processed per iteration
constexpr int kWarps = kBM / 16;  // 8 warps per block
constexpr int kLanes = 32;
// Head-dim rows padded so every WMMA tile pointer stays 16-byte aligned and
// shared-memory bank conflicts are reduced.
constexpr int kVStride = 72;
constexpr float kLayerNormEps = 1e-6f;
constexpr float kAttnScale = 0.125f;  // 1 / sqrt(64)

// Cooperative load of K/V chunk `chunk` (kBK latents) into shared memory.
// Both Ks and Vs are stored row-major as [latent][head_dim + pad].
__device__ __forceinline__ void load_kv_chunk(
    const half* __restrict__ k, const half* __restrict__ v, half* __restrict__ Ks,
    half* __restrict__ Vs, int batch, int head, int chunk, int num_latents,
    int tid, int nthreads) {
  const int s0 = chunk * kBK;
  for (int i = tid; i < kBK * kHeadDim; i += nthreads) {
    const int s = i / kHeadDim;
    const int d = i - s * kHeadDim;
    const long long src = ((long long)batch * num_latents + s0 + s) * kWidth +
                          head * kHeadDim + d;
    Ks[s * kVStride + d] = k[src];
    Vs[s * kVStride + d] = v[src];
  }
}

// Flash-attention style kernel.
//
// Grid:  (ceil(num_queries / kBM), num_heads, batch)
// Block: (32, kWarps); each warp owns 16 query rows and computes all 64 head
// dimensions of those rows for one head.
__global__ void __launch_bounds__(kLanes * kWarps) attention_kernel(
    const half* __restrict__ q, const half* __restrict__ k,
    const half* __restrict__ v, half* __restrict__ out, int num_queries,
    int num_latents) {
  __shared__ __align__(16) half Ks[kBK * kVStride];
  __shared__ __align__(16) half Vs[kBK * kVStride];

  const int batch = blockIdx.z;
  const int head = blockIdx.y;
  const int q0 = blockIdx.x * kBM;
  const int warp = threadIdx.y;
  const int lane = threadIdx.x;

  // Clamp the warp's first row into the valid range so out-of-bounds tiles are
  // never read; rows >= num_queries are skipped at store time.
  const int last_row = num_queries > 16 ? num_queries - 16 : 0;
  const int qrow_base = (q0 + warp * 16 < last_row) ? (q0 + warp * 16) : last_row;

  // This warp's Q rows (16 x 64), split into 4 k-chunks of 16.
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const half* qptr = q + ((long long)batch * num_queries + qrow_base) * kWidth +
                       head * kHeadDim + c * 16;
    wmma::load_matrix_sync(qf[c], qptr, kWidth);
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    wmma::fill_fragment(o_frag[t], 0.0f);
  }

  // Running softmax state; one entry per fragment row group (rows g and g+8).
  float m[2] = {-1e30f, -1e30f};
  float l[2] = {0.0f, 0.0f};

  const int tid = warp * kLanes + lane;
  const int nthreads = blockDim.y * kLanes;
  const int num_chunks = num_latents / kBK;

  load_kv_chunk(k, v, Ks, Vs, batch, head, 0, num_latents, tid, nthreads);
  __syncthreads();

  for (int c = 0; c < num_chunks; ++c) {
    // ---- QK^T: S[16, 128] in 8 accumulator tiles ----
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      wmma::fill_fragment(acc[t], 0.0f);
    }
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int st = 0; st < 8; ++st) {
        // B[k=dim][n=latent] = Ks[latent][dim] stored row-major -> col-major B.
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kf;
        const half* kptr = Ks + st * 16 * kVStride + kk * 16;
        wmma::load_matrix_sync(kf, kptr, kVStride);
        wmma::mma_sync(acc[st], qf[kk], kf, acc[st]);
      }
    }

    // ---- Online softmax in fp32 ----
    // acc element e lives in row g for e in {0,1,4,5} and row g+8 for e in
    // {2,3,6,7} (verified empirically), where g = lane / 4.
    float rmax[2] = {-1e30f, -1e30f};
#pragma unroll
    for (int t = 0; t < 8; ++t) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float s = acc[t].x[e] * kAttnScale;
        acc[t].x[e] = s;
        rmax[(e >> 1) & 1] = fmaxf(rmax[(e >> 1) & 1], s);
      }
    }
    // The 4 lanes with the same lane % 4 share a row: reduce with xor shuffles.
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      rmax[0] = fmaxf(rmax[0], __shfl_xor_sync(0xffffffffu, rmax[0], o));
      rmax[1] = fmaxf(rmax[1], __shfl_xor_sync(0xffffffffu, rmax[1], o));
    }
    const float m_new0 = fmaxf(rmax[0], m[0]);
    const float m_new1 = fmaxf(rmax[1], m[1]);
    const float a0 = __expf(m[0] - m_new0);
    const float a1 = __expf(m[1] - m_new1);

    // Rescale O and l when the running max increases.
#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        o_frag[t].x[e] *= ((e >> 1) & 1) ? a1 : a0;
      }
    }

    // P = exp(S - m_new); the matrix_a fragment shares acc's element layout,
    // so a straight conversion produces the P tile for the PV matmul.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf[8];
    float lsum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int t = 0; t < 8; ++t) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float p =
            __expf(acc[t].x[e] - (((e >> 1) & 1) ? m_new1 : m_new0));
        pf[t].x[e] = __float2half(p);
        lsum[(e >> 1) & 1] += p;
      }
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      lsum[0] += __shfl_xor_sync(0xffffffffu, lsum[0], o);
      lsum[1] += __shfl_xor_sync(0xffffffffu, lsum[1], o);
    }
    l[0] = l[0] * a0 + lsum[0];
    l[1] = l[1] * a1 + lsum[1];
    m[0] = m_new0;
    m[1] = m_new1;

    // ---- PV: O += P @ V ----
#pragma unroll
    for (int vt = 0; vt < 4; ++vt) {
#pragma unroll
      for (int st = 0; st < 8; ++st) {
        // B[k=latent][n=dim] = Vs[latent][dim] stored row-major -> row-major B.
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
        const half* vptr = Vs + st * 16 * kVStride + vt * 16;
        wmma::load_matrix_sync(vf, vptr, kVStride);
        wmma::mma_sync(o_frag[vt], pf[st], vf, o_frag[vt]);
      }
    }

    // All warps are done reading this chunk: stage the next one.
    __syncthreads();
    if (c + 1 < num_chunks) {
      load_kv_chunk(k, v, Ks, Vs, batch, head, c + 1, num_latents, tid, nthreads);
    }
    __syncthreads();
  }

  // ---- Epilogue: divide by the running row sums, store fp16 ----
  const float inv_l0 = 1.0f / fmaxf(l[0], 1e-20f);
  const float inv_l1 = 1.0f / fmaxf(l[1], 1e-20f);
#pragma unroll
  for (int t = 0; t < 4; ++t) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      o_frag[t].x[e] *= ((e >> 1) & 1) ? inv_l1 : inv_l0;
    }
  }

  const int g = lane >> 2;
  const int t4 = lane & 3;
#pragma unroll
  for (int vt = 0; vt < 4; ++vt) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int row = qrow_base + g + 8 * ((e >> 1) & 1);
      if (row < num_queries) {
        const int col = head * kHeadDim + vt * 16 + 2 * t4 + (e & 1) +
                        ((e >> 2) ? 8 : 0);
        half* optr = out + ((long long)batch * num_queries + row) * kWidth + col;
        *optr = __float2half(o_frag[vt].x[e]);
      }
    }
  }
}

// LayerNorm (elementwise_affine=False, eps=1e-6) fused with the final
// projection: out = out_proj_weight @ LN(x) + out_proj_bias.
// One warp per query row; 8 rows per block.
__global__ void __launch_bounds__(kLanes * 8) ln_outproj_kernel(
    const half* __restrict__ x, const half* __restrict__ weight,
    const half* __restrict__ bias, half* __restrict__ out, int num_queries) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  const int lane = threadIdx.x;
  if (row >= num_queries) {
    return;
  }

  const half* xr = x + ((long long)blockIdx.z * num_queries + row) * kWidth;
  const half2* xr2 = reinterpret_cast<const half2*>(xr);
  const half2* wr2 = reinterpret_cast<const half2*>(weight);

  float sum = 0.0f;
  float sumsq = 0.0f;
#pragma unroll
  for (int i = 0; i < kWidth / 2 / kLanes; ++i) {
    const float2 f = __half22float2(xr2[i * kLanes + lane]);
    sum += f.x + f.y;
    sumsq += f.x * f.x + f.y * f.y;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    sum += __shfl_xor_sync(0xffffffffu, sum, o);
    sumsq += __shfl_xor_sync(0xffffffffu, sumsq, o);
  }

  const float mean = sum / kWidth;
  float var = sumsq / kWidth - mean * mean;
  var = fmaxf(var, 0.0f);
  const float inv_std = rsqrtf(var + kLayerNormEps);

  float dot = 0.0f;
#pragma unroll
  for (int i = 0; i < kWidth / 2 / kLanes; ++i) {
    const float2 f = __half22float2(xr2[i * kLanes + lane]);
    const float2 wv = __half22float2(wr2[i * kLanes + lane]);
    dot += (f.x - mean) * inv_std * wv.x + (f.y - mean) * inv_std * wv.y;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    dot += __shfl_down_sync(0xffffffffu, dot, o);
  }
  if (lane == 0) {
    out[(long long)blockIdx.z * num_queries + row] =
        __float2half(__half2float(bias[0]) + dot);
  }
}

}  // namespace

torch::Tensor custom_kernel(
    torch::Tensor queries, torch::Tensor latents,
    torch::Tensor query_in_in_layer_weight, torch::Tensor query_in_in_layer_bias,
    torch::Tensor query_in_out_layer_weight, torch::Tensor query_in_out_layer_bias,
    torch::Tensor attn_c_q_weight, torch::Tensor attn_c_q_bias,
    torch::Tensor attn_c_k_weight, torch::Tensor attn_c_k_bias,
    torch::Tensor attn_c_v_weight, torch::Tensor attn_c_v_bias,
    torch::Tensor attn_c_proj_weight, torch::Tensor attn_c_proj_bias,
    torch::Tensor out_proj_weight, torch::Tensor out_proj_bias) {
  TORCH_CHECK(queries.scalar_type() == torch::kHalf, "expected float16 inputs");
  queries = queries.contiguous();
  latents = latents.contiguous();

  const int64_t batch = queries.size(0);
  const int64_t num_queries = queries.size(1);
  const int64_t num_latents = latents.size(1);
  TORCH_CHECK(latents.size(2) == kWidth, "expected width 768");
  TORCH_CHECK(num_latents % kBK == 0, "num_latents must be a multiple of kBK");

  auto opts = queries.options();
  auto queries2 = queries.reshape({batch * num_queries, queries.size(2)});
  auto latents2 = latents.reshape({batch * num_latents, kWidth});

  // Fold the two consecutive linear layers of the query MLP:
  //   h1 = h0 @ W_out^T + b_out
  //   q  = h1 @ W_q^T   + b_q
  // => q  = h0 @ (W_q @ W_out)^T + (b_q + W_q @ b_out)
  auto w_eff = torch::matmul(attn_c_q_weight, query_in_out_layer_weight);
  auto b_eff =
      torch::matmul(attn_c_q_weight, query_in_out_layer_bias) + attn_c_q_bias;

  auto h0 = torch::silu(
      torch::addmm(query_in_in_layer_bias, queries2, query_in_in_layer_weight.t()));
  auto q = torch::addmm(b_eff, h0, w_eff.t())
               .reshape({batch, num_queries, kWidth});

  auto k = torch::addmm(attn_c_k_bias, latents2, attn_c_k_weight.t())
               .reshape({batch, num_latents, kWidth});
  auto v = torch::addmm(attn_c_v_bias, latents2, attn_c_v_weight.t())
               .reshape({batch, num_latents, kWidth});

  auto attn = torch::empty({batch, num_queries, kWidth}, opts);
  dim3 block(kLanes, kWarps);
  dim3 grid((num_queries + kBM - 1) / kBM, kNumHeads, batch);
  attention_kernel<<<grid, block>>>(
      reinterpret_cast<const half*>(q.data_ptr()),
      reinterpret_cast<const half*>(k.data_ptr()),
      reinterpret_cast<const half*>(v.data_ptr()),
      reinterpret_cast<half*>(attn.data_ptr()), (int)num_queries,
      (int)num_latents);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  auto cproj =
      torch::addmm(attn_c_proj_bias,
                   attn.reshape({batch * num_queries, kWidth}),
                   attn_c_proj_weight.t())
          .reshape({batch, num_queries, kWidth});

  auto out = torch::empty({batch, num_queries, 1}, opts);
  dim3 ln_block(kLanes, 8);
  dim3 ln_grid((num_queries + 7) / 8, 1, batch);
  ln_outproj_kernel<<<ln_grid, ln_block>>>(
      reinterpret_cast<const half*>(cproj.data_ptr()),
      reinterpret_cast<const half*>(out_proj_weight.data_ptr()),
      reinterpret_cast<const half*>(out_proj_bias.data_ptr()),
      reinterpret_cast<half*>(out.data_ptr()), (int)num_queries);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  return out;
}
