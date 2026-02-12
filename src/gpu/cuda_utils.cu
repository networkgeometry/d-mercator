#include "../../include/dmercator/gpu/cuda_utils.hpp"

#include <sstream>

namespace dmercator::gpu {

namespace {
std::string &last_error_storage()
{
  static std::string storage;
  return storage;
}
} // namespace

void set_last_error(std::string message)
{
  last_error_storage() = std::move(message);
}

const std::string &last_error()
{
  return last_error_storage();
}

bool check_cuda_call(cudaError_t status, const char *expr, const char *file, int line)
{
  if(status == cudaSuccess)
  {
    return true;
  }
  std::ostringstream oss;
  oss << file << ":" << line << " CUDA call failed for `" << expr << "`: "
      << cudaGetErrorString(status);
  set_last_error(oss.str());
  return false;
}

bool check_cuda_kernel(const char *file, int line)
{
  const cudaError_t status = cudaGetLastError();
  if(status == cudaSuccess)
  {
    return true;
  }
  std::ostringstream oss;
  oss << file << ":" << line << " CUDA kernel launch failed: "
      << cudaGetErrorString(status);
  set_last_error(oss.str());
  return false;
}

} // namespace dmercator::gpu
