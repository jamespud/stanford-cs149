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

    int global_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int global_jdx = blockIdx.x * blockDim.x + threadIdx.x;

    int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;
    int shared_size = blockDim.x * num_bins;

    for (int i = linear_tid; i < shared_size; i += total_threads) {
        local_shared[i] = 0;
    }
    __syncthreads();

    if (global_idx < length && global_jdx < num_channels) {
        uint8_t val = data[global_idx * num_channels + global_jdx];
        atomicAdd(&local_shared[threadIdx.x * num_bins + val], 1);
    }
    __syncthreads();

    int base_global_channel = blockIdx.x * blockDim.x;

    for (int i = linear_tid; i < shared_size; i += total_threads) {
        int count = local_shared[i];
        if (count > 0) {
            int local_ch = i / num_bins;
            int bin = i % num_bins;
            int global_ch = base_global_channel + local_ch;

            if (global_ch < num_channels) {
                atomicAdd(&out[global_ch * num_bins + bin], count);
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