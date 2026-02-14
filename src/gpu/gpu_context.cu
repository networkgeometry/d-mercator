#include "../../include/dmercator/gpu/gpu_context.hpp"

#include "../../include/dmercator/gpu/cuda_utils.hpp"
#include "../../include/dmercator/gpu/kernels.cuh"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <sstream>
#include <utility>
#include <vector>

namespace dmercator::gpu {

namespace {

constexpr double PI = 3.141592653589793238462643383279502884197;

template <typename T>
struct DeviceBuffer
{
  T *ptr = nullptr;
  std::size_t capacity = 0;

  bool ensure(std::size_t required)
  {
    if(required <= capacity)
    {
      return true;
    }
    if(ptr != nullptr)
    {
      if(!DMERCATOR_CUDA_CHECK(cudaFree(ptr)))
      {
        return false;
      }
      ptr = nullptr;
      capacity = 0;
    }
    if(required == 0)
    {
      return true;
    }
    if(!DMERCATOR_CUDA_CHECK(cudaMalloc(&ptr, required * sizeof(T))))
    {
      return false;
    }
    capacity = required;
    return true;
  }

  void release()
  {
    if(ptr != nullptr)
    {
      cudaFree(ptr);
      ptr = nullptr;
    }
    capacity = 0;
  }
};

template <typename T>
struct HostPinnedBuffer
{
  T *ptr = nullptr;
  std::size_t capacity = 0;

  bool ensure(std::size_t required)
  {
    if(required <= capacity)
    {
      return true;
    }
    if(ptr != nullptr)
    {
      if(!DMERCATOR_CUDA_CHECK(cudaFreeHost(ptr)))
      {
        return false;
      }
      ptr = nullptr;
      capacity = 0;
    }
    if(required == 0)
    {
      return true;
    }
    if(!DMERCATOR_CUDA_CHECK(cudaMallocHost(&ptr, required * sizeof(T))))
    {
      return false;
    }
    capacity = required;
    return true;
  }

  void release()
  {
    if(ptr != nullptr)
    {
      cudaFreeHost(ptr);
      ptr = nullptr;
    }
    capacity = 0;
  }
};

struct Context
{
  bool initialized = false;
  bool deterministic_mode = true;
  int device_index = 0;
  cudaStream_t stream = nullptr;

  int refine_nb_vertices = 0;
  int refine_position_stride = 0;
  bool refine_s1_ready = false;
  bool refine_sd_ready = false;
  int expected_nb_vertices = 0;
  int expected_position_stride = 0;
  bool expected_s1_ready = false;
  bool expected_sd_ready = false;

  std::vector<double> host_flat_positions;

  DeviceBuffer<double> d_theta;
  DeviceBuffer<double> d_positions;
  DeviceBuffer<double> d_pair_prefactor;
  DeviceBuffer<int> d_neighbors;
  DeviceBuffer<double> d_candidates;
  DeviceBuffer<double> d_scores;
  DeviceBuffer<double> d_kappa;
  DeviceBuffer<double> d_expected;

  HostPinnedBuffer<double> h_scores;
  HostPinnedBuffer<double> h_expected;

  void release()
  {
    d_theta.release();
    d_positions.release();
    d_pair_prefactor.release();
    d_neighbors.release();
    d_candidates.release();
    d_scores.release();
    d_kappa.release();
    d_expected.release();
    h_scores.release();
    h_expected.release();
    if(stream != nullptr)
    {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
    initialized = false;
    refine_nb_vertices = 0;
    refine_position_stride = 0;
    refine_s1_ready = false;
    refine_sd_ready = false;
    expected_nb_vertices = 0;
    expected_position_stride = 0;
    expected_s1_ready = false;
    expected_sd_ready = false;
  }
};

Context &context()
{
  static Context ctx;
  return ctx;
}

bool flatten_positions(int dim,
                       const std::vector<std::vector<double>> &positions,
                       std::vector<double> &flat)
{
  const std::size_t stride = static_cast<std::size_t>(dim + 1);
  flat.resize(positions.size() * stride);
  for(std::size_t v = 0; v < positions.size(); ++v)
  {
    if(positions[v].size() != stride)
    {
      std::ostringstream oss;
      oss << "Invalid position size at vertex " << v << ": expected "
          << stride << ", got " << positions[v].size();
      set_last_error(oss.str());
      return false;
    }
    std::copy(positions[v].begin(), positions[v].end(), flat.begin() + static_cast<std::ptrdiff_t>(v * stride));
  }
  return true;
}

bool ensure_initialized(bool deterministic_mode)
{
  Context &ctx = context();
  if(ctx.initialized)
  {
    return true;
  }

  int device_count = 0;
  if(!DMERCATOR_CUDA_CHECK(cudaGetDeviceCount(&device_count)))
  {
    return false;
  }
  if(device_count <= 0)
  {
    set_last_error("No CUDA device was detected.");
    return false;
  }
  ctx.device_index = 0;
  ctx.deterministic_mode = deterministic_mode;
  if(!DMERCATOR_CUDA_CHECK(cudaSetDevice(ctx.device_index)))
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking)))
  {
    return false;
  }
  ctx.initialized = true;
  return true;
}

bool upload_refine_kappa(Context &ctx, const std::vector<double> &kappa, int expected_vertices)
{
  if(kappa.empty())
  {
    set_last_error("Refinement requested with empty kappa vector.");
    return false;
  }
  if(expected_vertices != static_cast<int>(kappa.size()))
  {
    set_last_error("Refinement theta/position and kappa sizes are inconsistent.");
    return false;
  }
  if(!ctx.d_kappa.ensure(kappa.size()))
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_kappa.ptr,
                                           kappa.data(),
                                           kappa.size() * sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           ctx.stream)))
  {
    return false;
  }
  return true;
}

bool copy_scores_to_host(Context &ctx, std::size_t nb_candidates, std::vector<double> &out_scores)
{
  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.h_scores.ptr,
                                           ctx.d_scores.ptr,
                                           nb_candidates * sizeof(double),
                                           cudaMemcpyDeviceToHost,
                                           ctx.stream)))
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaStreamSynchronize(ctx.stream)))
  {
    return false;
  }
  out_scores.assign(ctx.h_scores.ptr, ctx.h_scores.ptr + static_cast<std::ptrdiff_t>(nb_candidates));
  return true;
}

bool copy_expected_to_host(Context &ctx, std::size_t nb_vertices, std::vector<double> &out_expected_degrees)
{
  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.h_expected.ptr,
                                           ctx.d_expected.ptr,
                                           nb_vertices * sizeof(double),
                                           cudaMemcpyDeviceToHost,
                                           ctx.stream)))
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaStreamSynchronize(ctx.stream)))
  {
    return false;
  }
  out_expected_degrees.assign(ctx.h_expected.ptr, ctx.h_expected.ptr + static_cast<std::ptrdiff_t>(nb_vertices));
  return true;
}

} // namespace

DeviceInitStatus initialize(bool deterministic_mode)
{
  DeviceInitStatus status;
  if(ensure_initialized(deterministic_mode))
  {
    status.available = true;
    status.message = "CUDA runtime initialized.";
  }
  else
  {
    status.available = false;
    status.message = last_error();
  }
  return status;
}

bool begin_refine_s1(const std::vector<double> &theta)
{
  if(theta.empty())
  {
    set_last_error("Cannot start S1 refinement on empty theta vector.");
    return false;
  }
  Context &ctx = context();
  if(!ensure_initialized(ctx.deterministic_mode))
  {
    return false;
  }
  const std::size_t nb_vertices = theta.size();
  if(!ctx.d_theta.ensure(nb_vertices))
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_theta.ptr,
                                           theta.data(),
                                           nb_vertices * sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           ctx.stream)))
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaStreamSynchronize(ctx.stream)))
  {
    return false;
  }
  ctx.refine_nb_vertices = static_cast<int>(nb_vertices);
  ctx.refine_position_stride = 0;
  ctx.refine_s1_ready = true;
  ctx.refine_sd_ready = false;
  return true;
}

bool begin_refine_s1(const std::vector<double> &theta, const std::vector<double> &kappa)
{
  if(!begin_refine_s1(theta))
  {
    return false;
  }
  Context &ctx = context();
  if(!upload_refine_kappa(ctx, kappa, ctx.refine_nb_vertices))
  {
    return false;
  }
  return DMERCATOR_CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
}

bool begin_refine_sd(int dim, const std::vector<std::vector<double>> &positions)
{
  if(dim < 1)
  {
    set_last_error("Cannot start S^D refinement with dim < 1.");
    return false;
  }
  if(positions.empty())
  {
    set_last_error("Cannot start S^D refinement on empty position set.");
    return false;
  }
  Context &ctx = context();
  if(!ensure_initialized(ctx.deterministic_mode))
  {
    return false;
  }

  if(!flatten_positions(dim, positions, ctx.host_flat_positions))
  {
    return false;
  }

  const std::size_t total_values = ctx.host_flat_positions.size();
  if(!ctx.d_positions.ensure(total_values))
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_positions.ptr,
                                           ctx.host_flat_positions.data(),
                                           total_values * sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           ctx.stream)))
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaStreamSynchronize(ctx.stream)))
  {
    return false;
  }
  ctx.refine_nb_vertices = static_cast<int>(positions.size());
  ctx.refine_position_stride = dim + 1;
  ctx.refine_s1_ready = false;
  ctx.refine_sd_ready = true;
  return true;
}

bool begin_refine_sd(int dim, const std::vector<std::vector<double>> &positions, const std::vector<double> &kappa)
{
  if(!begin_refine_sd(dim, positions))
  {
    return false;
  }
  Context &ctx = context();
  if(!upload_refine_kappa(ctx, kappa, ctx.refine_nb_vertices))
  {
    return false;
  }
  return DMERCATOR_CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
}

bool update_refine_theta(int vertex_index, double theta_value)
{
  Context &ctx = context();
  if(!ctx.refine_s1_ready)
  {
    set_last_error("S1 refinement context is not initialized.");
    return false;
  }
  if(vertex_index < 0 || vertex_index >= ctx.refine_nb_vertices)
  {
    set_last_error("Theta update index is out of bounds.");
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_theta.ptr + vertex_index,
                                           &theta_value,
                                           sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           ctx.stream)))
  {
    return false;
  }
  return DMERCATOR_CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
}

bool update_refine_position(int dim, int vertex_index, const std::vector<double> &position)
{
  Context &ctx = context();
  if(!ctx.refine_sd_ready)
  {
    set_last_error("S^D refinement context is not initialized.");
    return false;
  }
  if(dim + 1 != ctx.refine_position_stride)
  {
    set_last_error("S^D refinement update dimension mismatch.");
    return false;
  }
  if(vertex_index < 0 || vertex_index >= ctx.refine_nb_vertices)
  {
    set_last_error("Position update index is out of bounds.");
    return false;
  }
  if(static_cast<int>(position.size()) != ctx.refine_position_stride)
  {
    set_last_error("Position update size mismatch.");
    return false;
  }
  const std::size_t offset = static_cast<std::size_t>(vertex_index) * static_cast<std::size_t>(ctx.refine_position_stride);
  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_positions.ptr + offset,
                                           position.data(),
                                           static_cast<std::size_t>(ctx.refine_position_stride) * sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           ctx.stream)))
  {
    return false;
  }
  return DMERCATOR_CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
}

bool evaluate_refine_s1_candidates(int v1,
                                   double beta,
                                   const std::vector<double> &pair_prefactor,
                                   const std::vector<int> &neighbors,
                                   const std::vector<double> &candidate_angles,
                                   std::vector<double> &out_scores)
{
  Context &ctx = context();
  if(!ctx.refine_s1_ready)
  {
    set_last_error("S1 refinement context is not initialized.");
    return false;
  }
  if(pair_prefactor.size() != static_cast<std::size_t>(ctx.refine_nb_vertices))
  {
    set_last_error("Pair-prefactor size does not match S1 refinement context.");
    return false;
  }
  if(candidate_angles.empty())
  {
    set_last_error("No S1 candidates were provided.");
    return false;
  }

  const std::size_t nb_vertices = pair_prefactor.size();
  const std::size_t neighbor_count = neighbors.size();
  const std::size_t nb_candidates = candidate_angles.size();

  if(!ctx.d_pair_prefactor.ensure(nb_vertices) ||
     !ctx.d_neighbors.ensure(neighbor_count) ||
     !ctx.d_candidates.ensure(nb_candidates) ||
     !ctx.d_scores.ensure(nb_candidates) ||
     !ctx.h_scores.ensure(nb_candidates))
  {
    return false;
  }

  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_pair_prefactor.ptr,
                                           pair_prefactor.data(),
                                           nb_vertices * sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           ctx.stream)))
  {
    return false;
  }
  if(neighbor_count > 0)
  {
    if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_neighbors.ptr,
                                             neighbors.data(),
                                             neighbor_count * sizeof(int),
                                             cudaMemcpyHostToDevice,
                                             ctx.stream)))
    {
      return false;
    }
  }
  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_candidates.ptr,
                                           candidate_angles.data(),
                                           nb_candidates * sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           ctx.stream)))
  {
    return false;
  }

  constexpr int threads_per_block = 128;
  const int blocks = static_cast<int>((nb_candidates + threads_per_block - 1) / threads_per_block);
  kernels::evaluate_refine_s1_candidates_kernel<<<blocks, threads_per_block, 0, ctx.stream>>>(
    ctx.d_theta.ptr,
    ctx.d_pair_prefactor.ptr,
    ctx.d_neighbors.ptr,
    static_cast<int>(neighbor_count),
    ctx.d_candidates.ptr,
    static_cast<int>(nb_candidates),
    v1,
    static_cast<int>(nb_vertices),
    beta,
    ctx.d_scores.ptr);

  if(!DMERCATOR_CUDA_CHECK_KERNEL())
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.h_scores.ptr,
                                           ctx.d_scores.ptr,
                                           nb_candidates * sizeof(double),
                                           cudaMemcpyDeviceToHost,
                                           ctx.stream)))
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaStreamSynchronize(ctx.stream)))
  {
    return false;
  }

  out_scores.assign(ctx.h_scores.ptr, ctx.h_scores.ptr + static_cast<std::ptrdiff_t>(nb_candidates));
  return true;
}

bool evaluate_refine_sd_candidates(int dim,
                                   int v1,
                                   double beta,
                                   double numerical_zero,
                                   const std::vector<double> &pair_prefactor,
                                   const std::vector<int> &neighbors,
                                   const std::vector<double> &candidate_positions_flat,
                                   std::vector<double> &out_scores)
{
  Context &ctx = context();
  if(!ctx.refine_sd_ready)
  {
    set_last_error("S^D refinement context is not initialized.");
    return false;
  }
  const int position_stride = dim + 1;
  if(position_stride != ctx.refine_position_stride)
  {
    set_last_error("S^D refinement candidate dimension mismatch.");
    return false;
  }
  if(pair_prefactor.size() != static_cast<std::size_t>(ctx.refine_nb_vertices))
  {
    set_last_error("Pair-prefactor size does not match S^D refinement context.");
    return false;
  }
  if(candidate_positions_flat.empty())
  {
    set_last_error("No S^D candidates were provided.");
    return false;
  }
  if(candidate_positions_flat.size() % static_cast<std::size_t>(position_stride) != 0)
  {
    set_last_error("Flat S^D candidate buffer has invalid shape.");
    return false;
  }

  const std::size_t nb_vertices = pair_prefactor.size();
  const std::size_t neighbor_count = neighbors.size();
  const std::size_t candidate_value_count = candidate_positions_flat.size();
  const std::size_t nb_candidates = candidate_value_count / static_cast<std::size_t>(position_stride);

  if(!ctx.d_pair_prefactor.ensure(nb_vertices) ||
     !ctx.d_neighbors.ensure(neighbor_count) ||
     !ctx.d_candidates.ensure(candidate_value_count) ||
     !ctx.d_scores.ensure(nb_candidates) ||
     !ctx.h_scores.ensure(nb_candidates))
  {
    return false;
  }

  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_pair_prefactor.ptr,
                                           pair_prefactor.data(),
                                           nb_vertices * sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           ctx.stream)))
  {
    return false;
  }
  if(neighbor_count > 0)
  {
    if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_neighbors.ptr,
                                             neighbors.data(),
                                             neighbor_count * sizeof(int),
                                             cudaMemcpyHostToDevice,
                                             ctx.stream)))
    {
      return false;
    }
  }
  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_candidates.ptr,
                                           candidate_positions_flat.data(),
                                           candidate_value_count * sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           ctx.stream)))
  {
    return false;
  }

  constexpr int threads_per_block = 128;
  const int blocks = static_cast<int>((nb_candidates + threads_per_block - 1) / threads_per_block);
  kernels::evaluate_refine_sd_candidates_kernel<<<blocks, threads_per_block, 0, ctx.stream>>>(
    ctx.d_positions.ptr,
    position_stride,
    ctx.d_pair_prefactor.ptr,
    ctx.d_neighbors.ptr,
    static_cast<int>(neighbor_count),
    ctx.d_candidates.ptr,
    static_cast<int>(nb_candidates),
    v1,
    static_cast<int>(nb_vertices),
    beta,
    numerical_zero,
    ctx.d_scores.ptr);

  if(!DMERCATOR_CUDA_CHECK_KERNEL())
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.h_scores.ptr,
                                           ctx.d_scores.ptr,
                                           nb_candidates * sizeof(double),
                                           cudaMemcpyDeviceToHost,
                                           ctx.stream)))
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaStreamSynchronize(ctx.stream)))
  {
    return false;
  }

  out_scores.assign(ctx.h_scores.ptr, ctx.h_scores.ptr + static_cast<std::ptrdiff_t>(nb_candidates));
  return true;
}

bool evaluate_refine_s1_candidates_from_kappa(int v1,
                                              double beta,
                                              double prefactor_over_kappa_v1,
                                              const std::vector<int> &neighbors,
                                              const std::vector<double> &candidate_angles,
                                              std::vector<double> &out_scores)
{
  Context &ctx = context();
  if(!ctx.refine_s1_ready)
  {
    set_last_error("S1 refinement context is not initialized.");
    return false;
  }
  if(v1 < 0 || v1 >= ctx.refine_nb_vertices)
  {
    set_last_error("S1 refinement vertex index is out of bounds.");
    return false;
  }
  if(candidate_angles.empty())
  {
    set_last_error("No S1 candidates were provided.");
    return false;
  }
  const std::size_t nb_vertices = static_cast<std::size_t>(ctx.refine_nb_vertices);
  if(ctx.d_kappa.ptr == nullptr || ctx.d_kappa.capacity < nb_vertices)
  {
    set_last_error("S1 refinement kappa buffer is not prepared on GPU.");
    return false;
  }

  const std::size_t neighbor_count = neighbors.size();
  const std::size_t nb_candidates = candidate_angles.size();
  if(!ctx.d_pair_prefactor.ensure(nb_vertices) ||
     !ctx.d_neighbors.ensure(neighbor_count) ||
     !ctx.d_candidates.ensure(nb_candidates) ||
     !ctx.d_scores.ensure(nb_candidates) ||
     !ctx.h_scores.ensure(nb_candidates))
  {
    return false;
  }

  if(neighbor_count > 0)
  {
    if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_neighbors.ptr,
                                             neighbors.data(),
                                             neighbor_count * sizeof(int),
                                             cudaMemcpyHostToDevice,
                                             ctx.stream)))
    {
      return false;
    }
  }
  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_candidates.ptr,
                                           candidate_angles.data(),
                                           nb_candidates * sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           ctx.stream)))
  {
    return false;
  }

  constexpr int prep_threads = 256;
  const int prep_blocks = static_cast<int>((nb_vertices + prep_threads - 1) / prep_threads);
  kernels::prepare_pair_prefactor_s1_kernel<<<prep_blocks, prep_threads, 0, ctx.stream>>>(
    ctx.d_kappa.ptr,
    static_cast<int>(nb_vertices),
    v1,
    prefactor_over_kappa_v1,
    ctx.d_pair_prefactor.ptr);
  if(!DMERCATOR_CUDA_CHECK_KERNEL())
  {
    return false;
  }

  constexpr int threads_per_block = 128;
  const int blocks = static_cast<int>((nb_candidates + threads_per_block - 1) / threads_per_block);
  kernels::evaluate_refine_s1_candidates_kernel<<<blocks, threads_per_block, 0, ctx.stream>>>(
    ctx.d_theta.ptr,
    ctx.d_pair_prefactor.ptr,
    ctx.d_neighbors.ptr,
    static_cast<int>(neighbor_count),
    ctx.d_candidates.ptr,
    static_cast<int>(nb_candidates),
    v1,
    static_cast<int>(nb_vertices),
    beta,
    ctx.d_scores.ptr);
  if(!DMERCATOR_CUDA_CHECK_KERNEL())
  {
    return false;
  }

  return copy_scores_to_host(ctx, nb_candidates, out_scores);
}

bool evaluate_refine_sd_candidates_from_kappa(int dim,
                                              int v1,
                                              double beta,
                                              double numerical_zero,
                                              double mu,
                                              double radius,
                                              const std::vector<int> &neighbors,
                                              const std::vector<double> &candidate_positions_flat,
                                              std::vector<double> &out_scores)
{
  Context &ctx = context();
  if(!ctx.refine_sd_ready)
  {
    set_last_error("S^D refinement context is not initialized.");
    return false;
  }
  const int position_stride = dim + 1;
  if(position_stride != ctx.refine_position_stride)
  {
    set_last_error("S^D refinement candidate dimension mismatch.");
    return false;
  }
  if(v1 < 0 || v1 >= ctx.refine_nb_vertices)
  {
    set_last_error("S^D refinement vertex index is out of bounds.");
    return false;
  }
  if(candidate_positions_flat.empty())
  {
    set_last_error("No S^D candidates were provided.");
    return false;
  }
  if(candidate_positions_flat.size() % static_cast<std::size_t>(position_stride) != 0)
  {
    set_last_error("Flat S^D candidate buffer has invalid shape.");
    return false;
  }
  const std::size_t nb_vertices = static_cast<std::size_t>(ctx.refine_nb_vertices);
  if(ctx.d_kappa.ptr == nullptr || ctx.d_kappa.capacity < nb_vertices)
  {
    set_last_error("S^D refinement kappa buffer is not prepared on GPU.");
    return false;
  }

  const std::size_t neighbor_count = neighbors.size();
  const std::size_t candidate_value_count = candidate_positions_flat.size();
  const std::size_t nb_candidates = candidate_value_count / static_cast<std::size_t>(position_stride);

  if(!ctx.d_pair_prefactor.ensure(nb_vertices) ||
     !ctx.d_neighbors.ensure(neighbor_count) ||
     !ctx.d_candidates.ensure(candidate_value_count) ||
     !ctx.d_scores.ensure(nb_candidates) ||
     !ctx.h_scores.ensure(nb_candidates))
  {
    return false;
  }

  if(neighbor_count > 0)
  {
    if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_neighbors.ptr,
                                             neighbors.data(),
                                             neighbor_count * sizeof(int),
                                             cudaMemcpyHostToDevice,
                                             ctx.stream)))
    {
      return false;
    }
  }
  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_candidates.ptr,
                                           candidate_positions_flat.data(),
                                           candidate_value_count * sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           ctx.stream)))
  {
    return false;
  }

  constexpr int prep_threads = 256;
  const int prep_blocks = static_cast<int>((nb_vertices + prep_threads - 1) / prep_threads);
  kernels::prepare_pair_prefactor_sd_kernel<<<prep_blocks, prep_threads, 0, ctx.stream>>>(
    ctx.d_kappa.ptr,
    static_cast<int>(nb_vertices),
    v1,
    mu,
    radius,
    1.0 / static_cast<double>(dim),
    ctx.d_pair_prefactor.ptr);
  if(!DMERCATOR_CUDA_CHECK_KERNEL())
  {
    return false;
  }

  constexpr int threads_per_block = 128;
  const int blocks = static_cast<int>((nb_candidates + threads_per_block - 1) / threads_per_block);
  kernels::evaluate_refine_sd_candidates_kernel<<<blocks, threads_per_block, 0, ctx.stream>>>(
    ctx.d_positions.ptr,
    position_stride,
    ctx.d_pair_prefactor.ptr,
    ctx.d_neighbors.ptr,
    static_cast<int>(neighbor_count),
    ctx.d_candidates.ptr,
    static_cast<int>(nb_candidates),
    v1,
    static_cast<int>(nb_vertices),
    beta,
    numerical_zero,
    ctx.d_scores.ptr);
  if(!DMERCATOR_CUDA_CHECK_KERNEL())
  {
    return false;
  }

  return copy_scores_to_host(ctx, nb_candidates, out_scores);
}

void clear_inferred_expected_degrees_state()
{
  Context &ctx = context();
  ctx.expected_nb_vertices = 0;
  ctx.expected_position_stride = 0;
  ctx.expected_s1_ready = false;
  ctx.expected_sd_ready = false;
}

bool prepare_inferred_expected_degrees_s1(const std::vector<double> &theta)
{
  if(theta.empty())
  {
    set_last_error("S1 expected-degree preparation requires non-empty theta.");
    return false;
  }
  Context &ctx = context();
  if(!ensure_initialized(ctx.deterministic_mode))
  {
    return false;
  }

  const std::size_t nb_vertices = theta.size();
  if(!ctx.d_theta.ensure(nb_vertices))
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_theta.ptr,
                                           theta.data(),
                                           nb_vertices * sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           ctx.stream)))
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaStreamSynchronize(ctx.stream)))
  {
    return false;
  }

  ctx.expected_nb_vertices = static_cast<int>(nb_vertices);
  ctx.expected_position_stride = 0;
  ctx.expected_s1_ready = true;
  ctx.expected_sd_ready = false;
  return true;
}

bool prepare_inferred_expected_degrees_sd(int dim, const std::vector<std::vector<double>> &positions)
{
  if(dim < 1)
  {
    set_last_error("S^D expected-degree preparation requested with dim < 1.");
    return false;
  }
  if(positions.empty())
  {
    set_last_error("S^D expected-degree preparation requires non-empty positions.");
    return false;
  }
  Context &ctx = context();
  if(!ensure_initialized(ctx.deterministic_mode))
  {
    return false;
  }
  if(!flatten_positions(dim, positions, ctx.host_flat_positions))
  {
    return false;
  }

  const std::size_t total_values = ctx.host_flat_positions.size();
  if(!ctx.d_positions.ensure(total_values))
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_positions.ptr,
                                           ctx.host_flat_positions.data(),
                                           total_values * sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           ctx.stream)))
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaStreamSynchronize(ctx.stream)))
  {
    return false;
  }

  ctx.expected_nb_vertices = static_cast<int>(positions.size());
  ctx.expected_position_stride = dim + 1;
  ctx.expected_s1_ready = false;
  ctx.expected_sd_ready = true;
  return true;
}

bool compute_inferred_expected_degrees_s1_prepared(double beta,
                                                   double mu,
                                                   const std::vector<double> &kappa,
                                                   std::vector<double> &out_expected_degrees)
{
  Context &ctx = context();
  if(!ctx.expected_s1_ready)
  {
    set_last_error("S1 expected-degree GPU state is not prepared.");
    return false;
  }
  if(static_cast<int>(kappa.size()) != ctx.expected_nb_vertices)
  {
    set_last_error("S1 expected-degree kappa size mismatch.");
    return false;
  }

  const std::size_t nb_vertices = static_cast<std::size_t>(ctx.expected_nb_vertices);
  if(!ctx.d_kappa.ensure(nb_vertices) ||
     !ctx.d_expected.ensure(nb_vertices) ||
     !ctx.h_expected.ensure(nb_vertices))
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_kappa.ptr,
                                           kappa.data(),
                                           nb_vertices * sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           ctx.stream)))
  {
    return false;
  }

  const double prefactor = static_cast<double>(nb_vertices) / (2.0 * PI * mu);
  constexpr int threads_per_block = 256;
  const int blocks = static_cast<int>((nb_vertices + threads_per_block - 1) / threads_per_block);
  kernels::inferred_expected_degrees_s1_kernel<<<blocks, threads_per_block, 0, ctx.stream>>>(
    ctx.d_theta.ptr,
    ctx.d_kappa.ptr,
    static_cast<int>(nb_vertices),
    beta,
    prefactor,
    ctx.d_expected.ptr);
  if(!DMERCATOR_CUDA_CHECK_KERNEL())
  {
    return false;
  }

  return copy_expected_to_host(ctx, nb_vertices, out_expected_degrees);
}

bool compute_inferred_expected_degrees_sd_prepared(int dim,
                                                   double beta,
                                                   double mu,
                                                   double radius,
                                                   double numerical_zero,
                                                   const std::vector<double> &kappa,
                                                   std::vector<double> &out_expected_degrees)
{
  Context &ctx = context();
  if(!ctx.expected_sd_ready)
  {
    set_last_error("S^D expected-degree GPU state is not prepared.");
    return false;
  }
  if((dim + 1) != ctx.expected_position_stride)
  {
    set_last_error("S^D expected-degree dimension mismatch.");
    return false;
  }
  if(static_cast<int>(kappa.size()) != ctx.expected_nb_vertices)
  {
    set_last_error("S^D expected-degree kappa size mismatch.");
    return false;
  }

  const std::size_t nb_vertices = static_cast<std::size_t>(ctx.expected_nb_vertices);
  if(!ctx.d_kappa.ensure(nb_vertices) ||
     !ctx.d_expected.ensure(nb_vertices) ||
     !ctx.h_expected.ensure(nb_vertices))
  {
    return false;
  }
  if(!DMERCATOR_CUDA_CHECK(cudaMemcpyAsync(ctx.d_kappa.ptr,
                                           kappa.data(),
                                           nb_vertices * sizeof(double),
                                           cudaMemcpyHostToDevice,
                                           ctx.stream)))
  {
    return false;
  }

  constexpr int threads_per_block = 256;
  const int blocks = static_cast<int>((nb_vertices + threads_per_block - 1) / threads_per_block);
  kernels::inferred_expected_degrees_sd_kernel<<<blocks, threads_per_block, 0, ctx.stream>>>(
    ctx.d_positions.ptr,
    dim + 1,
    ctx.d_kappa.ptr,
    static_cast<int>(nb_vertices),
    dim,
    beta,
    mu,
    radius,
    numerical_zero,
    ctx.d_expected.ptr);
  if(!DMERCATOR_CUDA_CHECK_KERNEL())
  {
    return false;
  }

  return copy_expected_to_host(ctx, nb_vertices, out_expected_degrees);
}

bool compute_inferred_expected_degrees_s1(double beta,
                                          double mu,
                                          const std::vector<double> &theta,
                                          const std::vector<double> &kappa,
                                          std::vector<double> &out_expected_degrees)
{
  if(theta.empty() || theta.size() != kappa.size())
  {
    set_last_error("S1 expected-degree input sizes are inconsistent.");
    return false;
  }
  if(!prepare_inferred_expected_degrees_s1(theta))
  {
    return false;
  }
  const bool ok = compute_inferred_expected_degrees_s1_prepared(beta, mu, kappa, out_expected_degrees);
  clear_inferred_expected_degrees_state();
  return ok;
}

bool compute_inferred_expected_degrees_sd(int dim,
                                          double beta,
                                          double mu,
                                          double radius,
                                          double numerical_zero,
                                          const std::vector<std::vector<double>> &positions,
                                          const std::vector<double> &kappa,
                                          std::vector<double> &out_expected_degrees)
{
  if(dim < 1)
  {
    set_last_error("S^D expected-degree requested with dim < 1.");
    return false;
  }
  if(positions.empty() || positions.size() != kappa.size())
  {
    set_last_error("S^D expected-degree input sizes are inconsistent.");
    return false;
  }
  if(!prepare_inferred_expected_degrees_sd(dim, positions))
  {
    return false;
  }
  const bool ok = compute_inferred_expected_degrees_sd_prepared(dim,
                                                                 beta,
                                                                 mu,
                                                                 radius,
                                                                 numerical_zero,
                                                                 kappa,
                                                                 out_expected_degrees);
  clear_inferred_expected_degrees_state();
  return ok;
}

} // namespace dmercator::gpu
