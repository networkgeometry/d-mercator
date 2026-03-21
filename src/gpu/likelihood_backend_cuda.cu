#include "../../include/dmercator/gpu/likelihood_backend.hpp"

#include "../../include/dmercator/gpu/likelihood_kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
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

  bool ensure(std::size_t required, std::string &out_error, const char *name)
  {
    if(required <= capacity)
    {
      return true;
    }
    if(ptr != nullptr)
    {
      if(cudaFree(ptr) != cudaSuccess)
      {
        out_error = std::string("cudaFree failed for ") + name + ".";
        return false;
      }
      ptr = nullptr;
      capacity = 0;
    }
    if(required == 0)
    {
      return true;
    }
    if(cudaMalloc(&ptr, required * sizeof(T)) != cudaSuccess)
    {
      out_error = std::string("cudaMalloc failed for ") + name + ".";
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

  bool ensure(std::size_t required, std::string &out_error, const char *name)
  {
    if(required <= capacity)
    {
      return true;
    }
    if(ptr != nullptr)
    {
      if(cudaFreeHost(ptr) != cudaSuccess)
      {
        out_error = std::string("cudaFreeHost failed for ") + name + ".";
        return false;
      }
      ptr = nullptr;
      capacity = 0;
    }
    if(required == 0)
    {
      return true;
    }
    if(cudaMallocHost(&ptr, required * sizeof(T)) != cudaSuccess)
    {
      out_error = std::string("cudaMallocHost failed for ") + name + ".";
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

class CudaLikelihoodBackend final : public LikelihoodBackend
{
  public:
    CudaLikelihoodBackend() = default;
    ~CudaLikelihoodBackend() override
    {
      release();
    }

    DeviceInitStatus initialize(const CsrGraph &graph, bool deterministic_mode) override
    {
      release();

      deterministic_mode_ = deterministic_mode;
      if(graph.nb_vertices <= 0)
      {
        set_error("Invalid graph size for CUDA backend initialization.");
        return {false, last_error_};
      }
      if(static_cast<int>(graph.row_offsets.size()) != graph.nb_vertices + 1)
      {
        set_error("CSR row_offsets has invalid size.");
        return {false, last_error_};
      }

      int device_count = 0;
      if(!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount"))
      {
        return {false, last_error_};
      }
      if(device_count <= 0)
      {
        set_error("No CUDA device available.");
        return {false, last_error_};
      }
      if(!check_cuda(cudaSetDevice(0), "cudaSetDevice"))
      {
        return {false, last_error_};
      }
      if(!check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags"))
      {
        return {false, last_error_};
      }

      nb_vertices_ = graph.nb_vertices;
      if(!d_row_offsets_.ensure(static_cast<std::size_t>(nb_vertices_ + 1), last_error_, "d_row_offsets"))
      {
        return {false, last_error_};
      }
      if(!d_col_indices_.ensure(graph.col_indices.size(), last_error_, "d_col_indices"))
      {
        return {false, last_error_};
      }
      if(!d_kappa_.ensure(static_cast<std::size_t>(nb_vertices_), last_error_, "d_kappa") ||
         !d_degree_.ensure(static_cast<std::size_t>(nb_vertices_), last_error_, "d_degree") ||
         !d_theta_.ensure(static_cast<std::size_t>(nb_vertices_), last_error_, "d_theta") ||
         !d_expected_.ensure(static_cast<std::size_t>(nb_vertices_), last_error_, "d_expected"))
      {
        return {false, last_error_};
      }

      if(!check_cuda(cudaMemcpyAsync(d_row_offsets_.ptr,
                                     graph.row_offsets.data(),
                                     static_cast<std::size_t>(nb_vertices_ + 1) * sizeof(int),
                                     cudaMemcpyHostToDevice,
                                     stream_),
                     "cudaMemcpyAsync(row_offsets)"))
      {
        return {false, last_error_};
      }
      if(!graph.col_indices.empty())
      {
        if(!check_cuda(cudaMemcpyAsync(d_col_indices_.ptr,
                                       graph.col_indices.data(),
                                       graph.col_indices.size() * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream_),
                       "cudaMemcpyAsync(col_indices)"))
        {
          return {false, last_error_};
        }
      }
      if(!sync_stream("initialize sync"))
      {
        return {false, last_error_};
      }

      initialized_ = true;
      return {true, "CUDA backend initialized."};
    }

    bool is_initialized() const override
    {
      return initialized_;
    }

    const std::string &last_error() const override
    {
      return last_error_;
    }

    bool set_kappa(const std::vector<double> &kappa) override
    {
      if(!check_initialized("set_kappa"))
      {
        return false;
      }
      if(static_cast<int>(kappa.size()) != nb_vertices_)
      {
        set_error("set_kappa size mismatch.");
        return false;
      }
      if(!check_cuda(cudaMemcpyAsync(d_kappa_.ptr,
                                     kappa.data(),
                                     static_cast<std::size_t>(nb_vertices_) * sizeof(double),
                                     cudaMemcpyHostToDevice,
                                     stream_),
                     "cudaMemcpyAsync(kappa)"))
      {
        return false;
      }
      if(!sync_stream("set_kappa sync"))
      {
        return false;
      }
      has_kappa_ = true;
      return true;
    }

    bool set_degree(const std::vector<int> &degree) override
    {
      if(!check_initialized("set_degree"))
      {
        return false;
      }
      if(static_cast<int>(degree.size()) != nb_vertices_)
      {
        set_error("set_degree size mismatch.");
        return false;
      }
      if(!check_cuda(cudaMemcpyAsync(d_degree_.ptr,
                                     degree.data(),
                                     static_cast<std::size_t>(nb_vertices_) * sizeof(int),
                                     cudaMemcpyHostToDevice,
                                     stream_),
                     "cudaMemcpyAsync(degree)"))
      {
        return false;
      }
      if(!sync_stream("set_degree sync"))
      {
        return false;
      }
      has_degree_ = true;
      return true;
    }

    bool set_theta(const std::vector<double> &theta) override
    {
      if(!check_initialized("set_theta"))
      {
        return false;
      }
      if(static_cast<int>(theta.size()) != nb_vertices_)
      {
        set_error("set_theta size mismatch.");
        return false;
      }
      if(!check_cuda(cudaMemcpyAsync(d_theta_.ptr,
                                     theta.data(),
                                     static_cast<std::size_t>(nb_vertices_) * sizeof(double),
                                     cudaMemcpyHostToDevice,
                                     stream_),
                     "cudaMemcpyAsync(theta)"))
      {
        return false;
      }
      if(!sync_stream("set_theta sync"))
      {
        return false;
      }
      has_theta_ = true;
      return true;
    }

    bool set_theta_single(int vertex_index, double theta_value) override
    {
      if(!check_initialized("set_theta_single"))
      {
        return false;
      }
      if(vertex_index < 0 || vertex_index >= nb_vertices_)
      {
        set_error("set_theta_single vertex index out of bounds.");
        return false;
      }
      if(!has_theta_)
      {
        set_error("set_theta_single called before set_theta.");
        return false;
      }
      if(!h_theta_single_.ensure(1, last_error_, "h_theta_single"))
      {
        return false;
      }
      h_theta_single_.ptr[0] = theta_value;
      if(!check_cuda(cudaMemcpyAsync(d_theta_.ptr + vertex_index,
                                     h_theta_single_.ptr,
                                     sizeof(double),
                                     cudaMemcpyHostToDevice,
                                     stream_),
                     "cudaMemcpyAsync(theta single)"))
      {
        return false;
      }
      return true;
    }

    bool set_positions_soa(int dim_plus_one, const std::vector<double> &positions_soa) override
    {
      if(!check_initialized("set_positions_soa"))
      {
        return false;
      }
      if(dim_plus_one <= 0)
      {
        set_error("set_positions_soa received invalid dimension.");
        return false;
      }
      const std::size_t expected_size = static_cast<std::size_t>(dim_plus_one) * static_cast<std::size_t>(nb_vertices_);
      if(positions_soa.size() != expected_size)
      {
        set_error("set_positions_soa size mismatch.");
        return false;
      }
      if(!d_positions_soa_.ensure(expected_size, last_error_, "d_positions_soa"))
      {
        return false;
      }
      if(!check_cuda(cudaMemcpyAsync(d_positions_soa_.ptr,
                                     positions_soa.data(),
                                     expected_size * sizeof(double),
                                     cudaMemcpyHostToDevice,
                                     stream_),
                     "cudaMemcpyAsync(positions_soa)"))
      {
        return false;
      }
      if(!sync_stream("set_positions_soa sync"))
      {
        return false;
      }
      dim_plus_one_ = dim_plus_one;
      has_positions_ = true;
      return true;
    }

    bool set_position_single(int vertex_index, const std::vector<double> &position) override
    {
      if(!check_initialized("set_position_single"))
      {
        return false;
      }
      if(vertex_index < 0 || vertex_index >= nb_vertices_)
      {
        set_error("set_position_single vertex index out of bounds.");
        return false;
      }
      if(!has_positions_ || dim_plus_one_ <= 0)
      {
        set_error("set_position_single called before set_positions_soa.");
        return false;
      }
      if(static_cast<int>(position.size()) != dim_plus_one_)
      {
        set_error("set_position_single dimension mismatch.");
        return false;
      }
      if(!h_position_single_.ensure(static_cast<std::size_t>(dim_plus_one_), last_error_, "h_position_single"))
      {
        return false;
      }
      std::copy(position.begin(), position.end(), h_position_single_.ptr);
      for(int i = 0; i < dim_plus_one_; ++i)
      {
        if(!check_cuda(cudaMemcpyAsync(d_positions_soa_.ptr + static_cast<std::size_t>(i) * static_cast<std::size_t>(nb_vertices_) + static_cast<std::size_t>(vertex_index),
                                       h_position_single_.ptr + i,
                                       sizeof(double),
                                       cudaMemcpyHostToDevice,
                                       stream_),
                       "cudaMemcpyAsync(position single)"))
        {
          return false;
        }
      }
      return true;
    }

    bool score_candidates_s1(int v1,
                             double prefactor,
                             double beta,
                             const std::vector<double> &candidate_theta,
                             const std::vector<int> &negative_vertices,
                             double negative_weight,
                             bool exact_negative_sweep,
                             std::vector<double> &out_scores) override
    {
      if(!check_initialized("score_candidates_s1"))
      {
        return false;
      }
      if(!has_kappa_ || !has_theta_)
      {
        set_error("score_candidates_s1 requires resident kappa and theta.");
        return false;
      }
      if(v1 < 0 || v1 >= nb_vertices_)
      {
        set_error("score_candidates_s1 vertex index out of bounds.");
        return false;
      }
      const int nb_candidates = static_cast<int>(candidate_theta.size());
      if(nb_candidates <= 0)
      {
        out_scores.clear();
        return true;
      }
      if(!exact_negative_sweep && negative_vertices.empty())
      {
        set_error("score_candidates_s1 sampled mode requires at least one negative vertex.");
        return false;
      }

      if(!d_candidate_theta_.ensure(static_cast<std::size_t>(nb_candidates), last_error_, "d_candidate_theta") ||
         !d_negative_vertices_.ensure(negative_vertices.size(), last_error_, "d_negative_vertices") ||
         !d_scores_.ensure(static_cast<std::size_t>(nb_candidates), last_error_, "d_scores") ||
         !h_scores_.ensure(static_cast<std::size_t>(nb_candidates), last_error_, "h_scores"))
      {
        return false;
      }

      if(!check_cuda(cudaMemcpyAsync(d_candidate_theta_.ptr,
                                     candidate_theta.data(),
                                     static_cast<std::size_t>(nb_candidates) * sizeof(double),
                                     cudaMemcpyHostToDevice,
                                     stream_),
                     "cudaMemcpyAsync(candidate_theta)"))
      {
        return false;
      }
      if(!negative_vertices.empty())
      {
        if(!check_cuda(cudaMemcpyAsync(d_negative_vertices_.ptr,
                                       negative_vertices.data(),
                                       negative_vertices.size() * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream_),
                       "cudaMemcpyAsync(negative_vertices_s1)"))
        {
          return false;
        }
      }

      kernels::launch_score_candidates_s1(d_theta_.ptr,
                                          d_kappa_.ptr,
                                          nb_vertices_,
                                          d_row_offsets_.ptr,
                                          d_col_indices_.ptr,
                                          v1,
                                          prefactor,
                                          beta,
                                          d_candidate_theta_.ptr,
                                          nb_candidates,
                                          d_negative_vertices_.ptr,
                                          static_cast<int>(negative_vertices.size()),
                                          negative_weight,
                                          exact_negative_sweep,
                                          d_scores_.ptr,
                                          stream_);
      if(!check_kernel("launch_score_candidates_s1"))
      {
        return false;
      }

      if(!check_cuda(cudaMemcpyAsync(h_scores_.ptr,
                                     d_scores_.ptr,
                                     static_cast<std::size_t>(nb_candidates) * sizeof(double),
                                     cudaMemcpyDeviceToHost,
                                     stream_),
                     "cudaMemcpyAsync(scores_s1 D2H)"))
      {
        return false;
      }
      if(!sync_stream("score_candidates_s1 sync"))
      {
        return false;
      }

      out_scores.assign(h_scores_.ptr, h_scores_.ptr + static_cast<std::ptrdiff_t>(nb_candidates));
      return true;
    }

    bool score_candidates_sd(int dim,
                             int v1,
                             double radius,
                             double mu,
                             double beta,
                             double numerical_zero,
                             const std::vector<double> &candidate_positions_soa,
                             const std::vector<int> &negative_vertices,
                             double negative_weight,
                             bool exact_negative_sweep,
                             std::vector<double> &out_scores) override
    {
      if(!check_initialized("score_candidates_sd"))
      {
        return false;
      }
      if(!has_kappa_ || !has_positions_)
      {
        set_error("score_candidates_sd requires resident kappa and positions.");
        return false;
      }
      if(dim < 1)
      {
        set_error("score_candidates_sd received invalid dim.");
        return false;
      }
      if(v1 < 0 || v1 >= nb_vertices_)
      {
        set_error("score_candidates_sd vertex index out of bounds.");
        return false;
      }
      if(dim_plus_one_ != (dim + 1))
      {
        set_error("score_candidates_sd dimension mismatch with resident positions.");
        return false;
      }
      if(candidate_positions_soa.empty())
      {
        out_scores.clear();
        return true;
      }
      if(candidate_positions_soa.size() % static_cast<std::size_t>(dim_plus_one_) != 0)
      {
        set_error("score_candidates_sd candidate buffer shape is invalid.");
        return false;
      }
      if(!exact_negative_sweep && negative_vertices.empty())
      {
        set_error("score_candidates_sd sampled mode requires at least one negative vertex.");
        return false;
      }

      const int nb_candidates = static_cast<int>(candidate_positions_soa.size() / static_cast<std::size_t>(dim_plus_one_));
      if(!d_candidate_positions_soa_.ensure(candidate_positions_soa.size(), last_error_, "d_candidate_positions_soa") ||
         !d_negative_vertices_.ensure(negative_vertices.size(), last_error_, "d_negative_vertices") ||
         !d_scores_.ensure(static_cast<std::size_t>(nb_candidates), last_error_, "d_scores") ||
         !h_scores_.ensure(static_cast<std::size_t>(nb_candidates), last_error_, "h_scores"))
      {
        return false;
      }

      if(!check_cuda(cudaMemcpyAsync(d_candidate_positions_soa_.ptr,
                                     candidate_positions_soa.data(),
                                     candidate_positions_soa.size() * sizeof(double),
                                     cudaMemcpyHostToDevice,
                                     stream_),
                     "cudaMemcpyAsync(candidate_positions_soa)"))
      {
        return false;
      }
      if(!negative_vertices.empty())
      {
        if(!check_cuda(cudaMemcpyAsync(d_negative_vertices_.ptr,
                                       negative_vertices.data(),
                                       negative_vertices.size() * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream_),
                       "cudaMemcpyAsync(negative_vertices_sd)"))
        {
          return false;
        }
      }

      kernels::launch_score_candidates_sd(d_positions_soa_.ptr,
                                          dim_plus_one_,
                                          d_kappa_.ptr,
                                          nb_vertices_,
                                          d_row_offsets_.ptr,
                                          d_col_indices_.ptr,
                                          v1,
                                          radius,
                                          mu,
                                          beta,
                                          numerical_zero,
                                          d_candidate_positions_soa_.ptr,
                                          nb_candidates,
                                          d_negative_vertices_.ptr,
                                          static_cast<int>(negative_vertices.size()),
                                          negative_weight,
                                          exact_negative_sweep,
                                          d_scores_.ptr,
                                          stream_);
      if(!check_kernel("launch_score_candidates_sd"))
      {
        return false;
      }

      if(!check_cuda(cudaMemcpyAsync(h_scores_.ptr,
                                     d_scores_.ptr,
                                     static_cast<std::size_t>(nb_candidates) * sizeof(double),
                                     cudaMemcpyDeviceToHost,
                                     stream_),
                     "cudaMemcpyAsync(scores_sd D2H)"))
      {
        return false;
      }
      if(!sync_stream("score_candidates_sd sync"))
      {
        return false;
      }

      out_scores.assign(h_scores_.ptr, h_scores_.ptr + static_cast<std::ptrdiff_t>(nb_candidates));
      return true;
    }

    bool compute_expected_degrees_s1(double beta,
                                     double mu,
                                     std::vector<double> &out_expected_degrees) override
    {
      if(!check_initialized("compute_expected_degrees_s1"))
      {
        return false;
      }
      if(!has_theta_ || !has_kappa_)
      {
        set_error("compute_expected_degrees_s1 requires resident theta and kappa.");
        return false;
      }
      if(!h_expected_.ensure(static_cast<std::size_t>(nb_vertices_), last_error_, "h_expected"))
      {
        return false;
      }
      const double prefactor = static_cast<double>(nb_vertices_) / (2.0 * PI * mu);
      kernels::launch_expected_degrees_s1(d_theta_.ptr,
                                          d_kappa_.ptr,
                                          nb_vertices_,
                                          beta,
                                          prefactor,
                                          d_expected_.ptr,
                                          stream_);
      if(!check_kernel("launch_expected_degrees_s1"))
      {
        return false;
      }
      if(!check_cuda(cudaMemcpyAsync(h_expected_.ptr,
                                     d_expected_.ptr,
                                     static_cast<std::size_t>(nb_vertices_) * sizeof(double),
                                     cudaMemcpyDeviceToHost,
                                     stream_),
                     "cudaMemcpyAsync(expected_s1 D2H)"))
      {
        return false;
      }
      if(!sync_stream("compute_expected_degrees_s1 sync"))
      {
        return false;
      }
      out_expected_degrees.assign(h_expected_.ptr, h_expected_.ptr + static_cast<std::ptrdiff_t>(nb_vertices_));
      return true;
    }

    bool compute_expected_degrees_sd(int dim,
                                     double beta,
                                     double mu,
                                     double radius,
                                     double numerical_zero,
                                     std::vector<double> &out_expected_degrees) override
    {
      if(!check_initialized("compute_expected_degrees_sd"))
      {
        return false;
      }
      if(!has_positions_ || !has_kappa_)
      {
        set_error("compute_expected_degrees_sd requires resident positions and kappa.");
        return false;
      }
      if(dim < 1)
      {
        set_error("compute_expected_degrees_sd received invalid dim.");
        return false;
      }
      if(dim_plus_one_ != (dim + 1))
      {
        set_error("compute_expected_degrees_sd dimension mismatch.");
        return false;
      }
      if(!h_expected_.ensure(static_cast<std::size_t>(nb_vertices_), last_error_, "h_expected"))
      {
        return false;
      }

      kernels::launch_expected_degrees_sd(d_positions_soa_.ptr,
                                          dim_plus_one_,
                                          d_kappa_.ptr,
                                          nb_vertices_,
                                          dim,
                                          beta,
                                          mu,
                                          radius,
                                          numerical_zero,
                                          d_expected_.ptr,
                                          stream_);
      if(!check_kernel("launch_expected_degrees_sd"))
      {
        return false;
      }
      if(!check_cuda(cudaMemcpyAsync(h_expected_.ptr,
                                     d_expected_.ptr,
                                     static_cast<std::size_t>(nb_vertices_) * sizeof(double),
                                     cudaMemcpyDeviceToHost,
                                     stream_),
                     "cudaMemcpyAsync(expected_sd D2H)"))
      {
        return false;
      }
      if(!sync_stream("compute_expected_degrees_sd sync"))
      {
        return false;
      }
      out_expected_degrees.assign(h_expected_.ptr, h_expected_.ptr + static_cast<std::ptrdiff_t>(nb_vertices_));
      return true;
    }

  private:
    bool check_initialized(const char *method)
    {
      if(initialized_)
      {
        return true;
      }
      set_error(std::string(method) + " called before initialize.");
      return false;
    }

    void set_error(std::string message)
    {
      last_error_ = std::move(message);
    }

    bool check_cuda(cudaError_t status, const char *what)
    {
      if(status == cudaSuccess)
      {
        return true;
      }
      std::ostringstream oss;
      oss << what << " failed: " << cudaGetErrorString(status);
      set_error(oss.str());
      return false;
    }

    bool check_kernel(const char *what)
    {
      return check_cuda(cudaGetLastError(), what);
    }

    bool sync_stream(const char *what)
    {
      return check_cuda(cudaStreamSynchronize(stream_), what);
    }

    void release()
    {
      d_row_offsets_.release();
      d_col_indices_.release();
      d_kappa_.release();
      d_degree_.release();
      d_theta_.release();
      d_positions_soa_.release();
      d_expected_.release();
      d_candidate_theta_.release();
      d_candidate_positions_soa_.release();
      d_negative_vertices_.release();
      d_scores_.release();
      h_scores_.release();
      h_expected_.release();
      h_theta_single_.release();
      h_position_single_.release();
      if(stream_ != nullptr)
      {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
      }

      nb_vertices_ = 0;
      dim_plus_one_ = 0;
      initialized_ = false;
      has_kappa_ = false;
      has_degree_ = false;
      has_theta_ = false;
      has_positions_ = false;
      deterministic_mode_ = true;
    }

    std::string last_error_;
    bool initialized_ = false;
    bool has_kappa_ = false;
    bool has_degree_ = false;
    bool has_theta_ = false;
    bool has_positions_ = false;
    bool deterministic_mode_ = true;
    int nb_vertices_ = 0;
    int dim_plus_one_ = 0;

    cudaStream_t stream_ = nullptr;

    DeviceBuffer<int> d_row_offsets_;
    DeviceBuffer<int> d_col_indices_;
    DeviceBuffer<double> d_kappa_;
    DeviceBuffer<int> d_degree_;
    DeviceBuffer<double> d_theta_;
    DeviceBuffer<double> d_positions_soa_;
    DeviceBuffer<double> d_expected_;

    DeviceBuffer<double> d_candidate_theta_;
    DeviceBuffer<double> d_candidate_positions_soa_;
    DeviceBuffer<int> d_negative_vertices_;
    DeviceBuffer<double> d_scores_;

    HostPinnedBuffer<double> h_scores_;
    HostPinnedBuffer<double> h_expected_;
    HostPinnedBuffer<double> h_theta_single_;
    HostPinnedBuffer<double> h_position_single_;
};

} // namespace

std::unique_ptr<LikelihoodBackend> create_cuda_likelihood_backend()
{
  return std::make_unique<CudaLikelihoodBackend>();
}

} // namespace dmercator::gpu
