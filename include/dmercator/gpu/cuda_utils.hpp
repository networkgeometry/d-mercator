#ifndef DMERCATOR_GPU_CUDA_UTILS_HPP
#define DMERCATOR_GPU_CUDA_UTILS_HPP

#include <cuda_runtime.h>

#include <string>

namespace dmercator::gpu {

void set_last_error(std::string message);
const std::string &last_error();
bool check_cuda_call(cudaError_t status, const char *expr, const char *file, int line);
bool check_cuda_kernel(const char *file, int line);

} // namespace dmercator::gpu

#define DMERCATOR_CUDA_CHECK(expr) dmercator::gpu::check_cuda_call((expr), #expr, __FILE__, __LINE__)
#define DMERCATOR_CUDA_CHECK_KERNEL() dmercator::gpu::check_cuda_kernel(__FILE__, __LINE__)

#endif // DMERCATOR_GPU_CUDA_UTILS_HPP
