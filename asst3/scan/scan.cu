#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void upsweep_kernel(int* output, int two_d, int num_tasks) {
    int task_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_idx >= num_tasks) {
        return;
    }

    int two_dplus1 = 2 * two_d;
    int i = task_idx * two_dplus1;
    output[i + two_dplus1 - 1] += output[i + two_d - 1];
}

__global__ void downsweep_kernel(int* output, int two_d, int num_tasks) {
    int task_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_idx >= num_tasks) {
        return;
    }

    int two_dplus1 = 2 * two_d;
    int i = task_idx * two_dplus1;
    int t = output[i + two_d - 1];
    output[i + two_d - 1] = output[i + two_dplus1 - 1];
    output[i + two_dplus1 - 1] += t;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{
    int paddedN = nextPow2(N);
    cudaMemcpy(result, input, N * sizeof(int), cudaMemcpyDeviceToDevice);
    if (paddedN > N) {
        cudaMemset(result + N, 0, (paddedN - N) * sizeof(int));
    }

    for (int two_d = 1; two_d < paddedN; two_d *= 2) {
        int num_tasks = paddedN / (2 * two_d);
        int numBlocks = (num_tasks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        upsweep_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(result, two_d, num_tasks);
    }

    cudaDeviceSynchronize();
    cudaMemset(result + paddedN - 1, 0, sizeof(int));

    for (int two_d = paddedN / 2; two_d >= 1; two_d /= 2) {
        int num_tasks = paddedN / (2 * two_d);
        int numBlocks = (num_tasks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        downsweep_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(result, two_d, num_tasks);
    }
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}

__global__ void mark_kernel(int* input, int* output, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= length - 1) {
        return;
    }

    output[idx] = (input[idx] == input[idx + 1]) ? 1 : 0;
}

__global__ void scatter_kernel(int* mask, int* scan, int* output, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length - 1) {
        return;
    }

    if (mask[idx]) {
        output[scan[idx]] = idx;
    }

}

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {
    int* mask = new int[1];
    int* scan = new int[1];
    int* deviceMasks;
    int* scanResults;

    int rounded_length = nextPow2(length);
    int numMatches = 0;
    int numBlocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaMalloc((void**)&deviceMasks, sizeof(int) * rounded_length);
    cudaMalloc((void**)&scanResults, sizeof(int) * rounded_length);

    cudaMemset(deviceMasks, 0, sizeof(int) * length);

    mark_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(device_input, deviceMasks, length);

    exclusive_scan(deviceMasks, length, scanResults);

    scatter_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(deviceMasks, scanResults, device_output,
                                                     length);

    cudaMemcpy(mask, deviceMasks + length - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(scan, scanResults + length - 1, sizeof(int), cudaMemcpyDeviceToHost);

    numMatches = mask[0] + scan[0];

    return numMatches;
}

//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
