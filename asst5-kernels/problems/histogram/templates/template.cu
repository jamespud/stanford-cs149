#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                               \
  do {                                                                 \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;           \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

#define BLOCK_SIZE 16


//
// craete your function: __global__ void kernel(...) here
// Note: input data is of type uint8_t
//
__global__ void kernel(uint8_t* data, int* out, int num_bins, int num_channels, int length) {
    extern __shared__ int local_shared[];
    __shared__ uint8_t local_data[BLOCK_SIZE][BLOCK_SIZE];

    int base_idx = blockIdx.y * blockDim.y;
    int base_jdy = blockIdx.x * blockDim.x;

    int local_idx = threadIdx.y;
    int local_jdx = threadIdx.x;

    int global_idx = base_idx + local_idx;
    int global_jdx = base_jdy + local_jdx;

    if (global_idx < length && global_jdx < num_channels) {
        local_data[local_idx][local_jdx] = data[global_idx * num_channels + global_jdx];
    } else {
        local_data[local_idx][local_jdx] = 0;
    }
    __syncthreads();

    if (local_jdx == 0) {
        for (int i = 0; i < num_bins; i++) {
            local_shared[local_idx * num_bins + i] = 0;
        }
    }

    __syncthreads();

    if (global_idx < length && global_jdx < num_channels) {
        int bin = local_data[local_idx][local_jdx];
        atomicAdd(&local_shared[local_jdx * num_bins + bin], 1);
    }
    __syncthreads();

    if (local_idx == 0) {
        for (int j = 0; j < num_bins; j++) {
            int count = local_shared[local_jdx * num_bins + j];
            if (count > 0) {
                atomicAdd(&out[global_jdx * num_bins + j], count);
            }
        }
    }
}

// Host function to launch kernel
torch::Tensor histogram_kernel(
    torch::Tensor data,  // [length, num_channels], dtype=uint8
    int num_bins
) {
    TORCH_CHECK(data.device().is_cuda(), "Tensor data must be a CUDA tensor");

    const int length = data.size(0);
    const int num_channels = data.size(1);
    
    // Allocate output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(data.device());
    torch::Tensor histogram = torch::zeros({num_channels, num_bins}, options);
    

    ////
    // Launch your kernel here
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((num_channels + BLOCK_SIZE - 1) / BLOCK_SIZE,
     (length + BLOCK_SIZE - 1) / BLOCK_SIZE);
     size_t shared_memory_bytes = BLOCK_SIZE *  num_bins * sizeof(int);

     CHECK_CUDA(kernel<<<gridSize, blockSize, shared_memory_bytes>>>(
         data.data_ptr<uint8_t>(), histogram.data_ptr<int>(), num_bins, num_channels, length));

     ////

     // Check for errors
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         throw std::runtime_error(cudaGetErrorString(err));
     }
    
    return histogram;
}