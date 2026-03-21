#include "../../include/dmercator/gpu/likelihood_kernels.cuh"

#include <cmath>

namespace dmercator::gpu::kernels {

namespace {

constexpr double PI = 3.141592653589793238462643383279502884197;

template <int BLOCK_SIZE>
__device__ double block_reduce_sum(double local)
{
  __shared__ double shared[BLOCK_SIZE];
  shared[threadIdx.x] = local;
  __syncthreads();
  for(int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1)
  {
    if(threadIdx.x < offset)
    {
      shared[threadIdx.x] += shared[threadIdx.x + offset];
    }
    __syncthreads();
  }
  return shared[0];
}

__device__ inline double angular_distance_s1(double t1, double t2)
{
  return PI - fabs(PI - fabs(t1 - t2));
}

__device__ inline double angular_distance_sd_vertices(const double *positions_soa,
                                                      int dim_plus_one,
                                                      int nb_vertices,
                                                      int v1,
                                                      int v2,
                                                      double numerical_zero)
{
  double dot = 0.0;
  double norm1 = 0.0;
  double norm2 = 0.0;
  for(int i = 0; i < dim_plus_one; ++i)
  {
    const double p1 = positions_soa[static_cast<size_t>(i) * static_cast<size_t>(nb_vertices) + static_cast<size_t>(v1)];
    const double p2 = positions_soa[static_cast<size_t>(i) * static_cast<size_t>(nb_vertices) + static_cast<size_t>(v2)];
    dot += p1 * p2;
    norm1 += p1 * p1;
    norm2 += p2 * p2;
  }
  norm1 /= sqrt(norm1);
  norm2 /= sqrt(norm2);
  double result = dot / (norm1 * norm2);
  if(fabs(result - 1.0) < numerical_zero)
  {
    return 0.0;
  }
  if(result > 1.0)
  {
    result = 1.0;
  }
  else if(result < -1.0)
  {
    result = -1.0;
  }
  return acos(result);
}

__device__ inline double angular_distance_sd_candidate(const double *candidate_positions_soa,
                                                       int nb_candidates,
                                                       int candidate_index,
                                                       const double *positions_soa,
                                                       int dim_plus_one,
                                                       int nb_vertices,
                                                       int v2,
                                                       double numerical_zero)
{
  double dot = 0.0;
  double norm1 = 0.0;
  double norm2 = 0.0;
  for(int i = 0; i < dim_plus_one; ++i)
  {
    const double p1 = candidate_positions_soa[static_cast<size_t>(i) * static_cast<size_t>(nb_candidates) + static_cast<size_t>(candidate_index)];
    const double p2 = positions_soa[static_cast<size_t>(i) * static_cast<size_t>(nb_vertices) + static_cast<size_t>(v2)];
    dot += p1 * p2;
    norm1 += p1 * p1;
    norm2 += p2 * p2;
  }
  norm1 /= sqrt(norm1);
  norm2 /= sqrt(norm2);
  double result = dot / (norm1 * norm2);
  if(fabs(result - 1.0) < numerical_zero)
  {
    return 0.0;
  }
  if(result > 1.0)
  {
    result = 1.0;
  }
  else if(result < -1.0)
  {
    result = -1.0;
  }
  return acos(result);
}

template <int BLOCK_SIZE>
__global__ void score_s1_nonedge_kernel(const double *theta,
                                        const double *kappa,
                                        int nb_vertices,
                                        const double *candidate_theta,
                                        int nb_candidates,
                                        int v1,
                                        double prefactor,
                                        double beta,
                                        double *out_scores)
{
  const int candidate_index = blockIdx.x;
  if(candidate_index >= nb_candidates)
  {
    return;
  }

  const double angle = candidate_theta[candidate_index];
  const double kappa_v1 = kappa[v1];
  double local = 0.0;
  for(int v2 = threadIdx.x; v2 < nb_vertices; v2 += BLOCK_SIZE)
  {
    if(v2 == v1)
    {
      continue;
    }
    const double dtheta = angular_distance_s1(angle, theta[v2]);
    const double fraction = (prefactor * dtheta) / (kappa_v1 * kappa[v2]);
    local += -log(1.0 + pow(fraction, -beta));
  }

  const double reduced = block_reduce_sum<BLOCK_SIZE>(local);
  if(threadIdx.x == 0)
  {
    out_scores[candidate_index] = reduced;
  }
}

template <int BLOCK_SIZE>
__global__ void score_s1_edge_kernel(const double *theta,
                                     const double *kappa,
                                     const int *row_offsets,
                                     const int *col_indices,
                                     int v1,
                                     double prefactor,
                                     double beta,
                                     const double *candidate_theta,
                                     int nb_candidates,
                                     bool include_nonedge_term,
                                     double *out_scores)
{
  const int candidate_index = blockIdx.x;
  if(candidate_index >= nb_candidates)
  {
    return;
  }

  const double angle = candidate_theta[candidate_index];
  const double kappa_v1 = kappa[v1];
  const int begin = row_offsets[v1];
  const int end = row_offsets[v1 + 1];

  double local = 0.0;
  for(int edge = begin + threadIdx.x; edge < end; edge += BLOCK_SIZE)
  {
    const int v2 = col_indices[edge];
    const double dtheta = angular_distance_s1(angle, theta[v2]);
    const double fraction = (prefactor * dtheta) / (kappa_v1 * kappa[v2]);
    if(include_nonedge_term)
    {
      local += -log(1.0 + pow(fraction, -beta));
    }
    local += -beta * log(fraction);
  }

  const double reduced = block_reduce_sum<BLOCK_SIZE>(local);
  if(threadIdx.x == 0)
  {
    out_scores[candidate_index] += reduced;
  }
}

template <int BLOCK_SIZE>
__global__ void score_s1_sampled_negative_kernel(const double *theta,
                                                 const double *kappa,
                                                 int v1,
                                                 double prefactor,
                                                 double beta,
                                                 const double *candidate_theta,
                                                 int nb_candidates,
                                                 const int *negative_vertices,
                                                 int nb_negative_vertices,
                                                 double negative_weight,
                                                 double *out_scores)
{
  const int candidate_index = blockIdx.x;
  if(candidate_index >= nb_candidates)
  {
    return;
  }

  const double angle = candidate_theta[candidate_index];
  const double kappa_v1 = kappa[v1];
  double local = 0.0;
  for(int idx = threadIdx.x; idx < nb_negative_vertices; idx += BLOCK_SIZE)
  {
    const int v2 = negative_vertices[idx];
    const double dtheta = angular_distance_s1(angle, theta[v2]);
    const double fraction = (prefactor * dtheta) / (kappa_v1 * kappa[v2]);
    local += -log(1.0 + pow(fraction, -beta));
  }

  const double reduced = block_reduce_sum<BLOCK_SIZE>(local);
  if(threadIdx.x == 0)
  {
    out_scores[candidate_index] += negative_weight * reduced;
  }
}

template <int BLOCK_SIZE>
__global__ void score_sd_nonedge_kernel(const double *positions_soa,
                                        int dim_plus_one,
                                        const double *kappa,
                                        int nb_vertices,
                                        int v1,
                                        double radius,
                                        double mu,
                                        double beta,
                                        double inv_dim,
                                        double numerical_zero,
                                        const double *candidate_positions_soa,
                                        int nb_candidates,
                                        double *out_scores)
{
  const int candidate_index = blockIdx.x;
  if(candidate_index >= nb_candidates)
  {
    return;
  }

  const double kappa_v1 = kappa[v1];
  double local = 0.0;
  for(int v2 = threadIdx.x; v2 < nb_vertices; v2 += BLOCK_SIZE)
  {
    if(v2 == v1)
    {
      continue;
    }
    const double dtheta = angular_distance_sd_candidate(candidate_positions_soa,
                                                        nb_candidates,
                                                        candidate_index,
                                                        positions_soa,
                                                        dim_plus_one,
                                                        nb_vertices,
                                                        v2,
                                                        numerical_zero);
    const double chi = radius * dtheta / pow(mu * kappa_v1 * kappa[v2], inv_dim);
    const double prob = 1.0 / (1.0 + pow(chi, beta));
    local += log(1.0 - prob);
  }

  const double reduced = block_reduce_sum<BLOCK_SIZE>(local);
  if(threadIdx.x == 0)
  {
    out_scores[candidate_index] = reduced;
  }
}

template <int BLOCK_SIZE>
__global__ void score_sd_edge_kernel(const double *positions_soa,
                                     int dim_plus_one,
                                     const double *kappa,
                                     const int *row_offsets,
                                     const int *col_indices,
                                     int nb_vertices,
                                     int v1,
                                     double radius,
                                     double mu,
                                     double beta,
                                     double inv_dim,
                                     double numerical_zero,
                                     const double *candidate_positions_soa,
                                     int nb_candidates,
                                     bool include_nonedge_term,
                                     double *out_scores)
{
  const int candidate_index = blockIdx.x;
  if(candidate_index >= nb_candidates)
  {
    return;
  }

  const double kappa_v1 = kappa[v1];
  const int begin = row_offsets[v1];
  const int end = row_offsets[v1 + 1];

  double local = 0.0;
  for(int edge = begin + threadIdx.x; edge < end; edge += BLOCK_SIZE)
  {
    const int v2 = col_indices[edge];
    const double dtheta = angular_distance_sd_candidate(candidate_positions_soa,
                                                        nb_candidates,
                                                        candidate_index,
                                                        positions_soa,
                                                        dim_plus_one,
                                                        nb_vertices,
                                                        v2,
                                                        numerical_zero);
    const double chi = radius * dtheta / pow(mu * kappa_v1 * kappa[v2], inv_dim);
    const double prob = 1.0 / (1.0 + pow(chi, beta));
    if(include_nonedge_term)
    {
      local += log(1.0 - prob);
    }
    local += log(prob);
  }

  const double reduced = block_reduce_sum<BLOCK_SIZE>(local);
  if(threadIdx.x == 0)
  {
    out_scores[candidate_index] += reduced;
  }
}

template <int BLOCK_SIZE>
__global__ void score_sd_sampled_negative_kernel(const double *positions_soa,
                                                 int dim_plus_one,
                                                 const double *kappa,
                                                 int nb_vertices,
                                                 int v1,
                                                 double radius,
                                                 double mu,
                                                 double beta,
                                                 double inv_dim,
                                                 double numerical_zero,
                                                 const double *candidate_positions_soa,
                                                 int nb_candidates,
                                                 const int *negative_vertices,
                                                 int nb_negative_vertices,
                                                 double negative_weight,
                                                 double *out_scores)
{
  const int candidate_index = blockIdx.x;
  if(candidate_index >= nb_candidates)
  {
    return;
  }

  const double kappa_v1 = kappa[v1];
  double local = 0.0;
  for(int idx = threadIdx.x; idx < nb_negative_vertices; idx += BLOCK_SIZE)
  {
    const int v2 = negative_vertices[idx];
    const double dtheta = angular_distance_sd_candidate(candidate_positions_soa,
                                                        nb_candidates,
                                                        candidate_index,
                                                        positions_soa,
                                                        dim_plus_one,
                                                        nb_vertices,
                                                        v2,
                                                        numerical_zero);
    const double chi = radius * dtheta / pow(mu * kappa_v1 * kappa[v2], inv_dim);
    const double prob = 1.0 / (1.0 + pow(chi, beta));
    local += log(1.0 - prob);
  }

  const double reduced = block_reduce_sum<BLOCK_SIZE>(local);
  if(threadIdx.x == 0)
  {
    out_scores[candidate_index] += negative_weight * reduced;
  }
}

template <int BLOCK_SIZE>
__global__ void expected_degrees_s1_kernel(const double *theta,
                                           const double *kappa,
                                           int nb_vertices,
                                           double beta,
                                           double prefactor,
                                           double *out_expected_degrees)
{
  const int v1 = blockIdx.x * blockDim.x + threadIdx.x;
  if(v1 >= nb_vertices)
  {
    return;
  }

  const double kappa_v1 = kappa[v1];
  const double theta_v1 = theta[v1];
  double expected = 0.0;
  for(int v2 = 0; v2 < nb_vertices; ++v2)
  {
    if(v2 == v1)
    {
      continue;
    }
    const double dtheta = angular_distance_s1(theta_v1, theta[v2]);
    const double chi = (prefactor * dtheta) / (kappa_v1 * kappa[v2]);
    const double prob = 1.0 / (1.0 + pow(chi, beta));
    expected += prob;
  }
  out_expected_degrees[v1] = expected;
}

template <int BLOCK_SIZE>
__global__ void expected_degrees_sd_kernel(const double *positions_soa,
                                           int dim_plus_one,
                                           const double *kappa,
                                           int nb_vertices,
                                           int dim,
                                           double beta,
                                           double mu,
                                           double radius,
                                           double numerical_zero,
                                           double *out_expected_degrees)
{
  const int v1 = blockIdx.x * blockDim.x + threadIdx.x;
  if(v1 >= nb_vertices)
  {
    return;
  }

  const double kappa_v1 = kappa[v1];
  const double inv_dim = 1.0 / static_cast<double>(dim);
  double expected = 0.0;
  for(int v2 = 0; v2 < nb_vertices; ++v2)
  {
    if(v2 == v1)
    {
      continue;
    }
    const double dtheta = angular_distance_sd_vertices(positions_soa,
                                                       dim_plus_one,
                                                       nb_vertices,
                                                       v1,
                                                       v2,
                                                       numerical_zero);
    const double chi = radius * dtheta / pow(mu * kappa_v1 * kappa[v2], inv_dim);
    const double prob = 1.0 / (1.0 + pow(chi, beta));
    expected += prob;
  }
  out_expected_degrees[v1] = expected;
}

} // namespace

void launch_score_candidates_s1(const double *theta,
                                const double *kappa,
                                int nb_vertices,
                                const int *row_offsets,
                                const int *col_indices,
                                int v1,
                                double prefactor,
                                double beta,
                                const double *candidate_theta,
                                int nb_candidates,
                                const int *negative_vertices,
                                int nb_negative_vertices,
                                double negative_weight,
                                bool exact_negative_sweep,
                                double *out_scores,
                                cudaStream_t stream)
{
  if(nb_candidates <= 0)
  {
    return;
  }
  constexpr int block_size = 256;
  if(exact_negative_sweep)
  {
    score_s1_nonedge_kernel<block_size><<<nb_candidates, block_size, 0, stream>>>(theta,
                                                                                    kappa,
                                                                                    nb_vertices,
                                                                                    candidate_theta,
                                                                                    nb_candidates,
                                                                                    v1,
                                                                                    prefactor,
                                                                                    beta,
                                                                                    out_scores);
    score_s1_edge_kernel<block_size><<<nb_candidates, block_size, 0, stream>>>(theta,
                                                                                 kappa,
                                                                                 row_offsets,
                                                                                 col_indices,
                                                                                 v1,
                                                                                 prefactor,
                                                                                 beta,
                                                                                 candidate_theta,
                                                                                 nb_candidates,
                                                                                 false,
                                                                                 out_scores);
    return;
  }

  cudaMemsetAsync(out_scores, 0, static_cast<std::size_t>(nb_candidates) * sizeof(double), stream);
  score_s1_edge_kernel<block_size><<<nb_candidates, block_size, 0, stream>>>(theta,
                                                                               kappa,
                                                                               row_offsets,
                                                                               col_indices,
                                                                               v1,
                                                                               prefactor,
                                                                               beta,
                                                                               candidate_theta,
                                                                               nb_candidates,
                                                                               true,
                                                                               out_scores);
  if(nb_negative_vertices > 0)
  {
    score_s1_sampled_negative_kernel<block_size><<<nb_candidates, block_size, 0, stream>>>(theta,
                                                                                              kappa,
                                                                                              v1,
                                                                                              prefactor,
                                                                                              beta,
                                                                                              candidate_theta,
                                                                                              nb_candidates,
                                                                                              negative_vertices,
                                                                                              nb_negative_vertices,
                                                                                              negative_weight,
                                                                                              out_scores);
  }
}

void launch_score_candidates_sd(const double *positions_soa,
                                int dim_plus_one,
                                const double *kappa,
                                int nb_vertices,
                                const int *row_offsets,
                                const int *col_indices,
                                int v1,
                                double radius,
                                double mu,
                                double beta,
                                double numerical_zero,
                                const double *candidate_positions_soa,
                                int nb_candidates,
                                const int *negative_vertices,
                                int nb_negative_vertices,
                                double negative_weight,
                                bool exact_negative_sweep,
                                double *out_scores,
                                cudaStream_t stream)
{
  if(nb_candidates <= 0)
  {
    return;
  }
  constexpr int block_size = 256;
  const double inv_dim = 1.0 / static_cast<double>(dim_plus_one - 1);
  if(exact_negative_sweep)
  {
    score_sd_nonedge_kernel<block_size><<<nb_candidates, block_size, 0, stream>>>(positions_soa,
                                                                                    dim_plus_one,
                                                                                    kappa,
                                                                                    nb_vertices,
                                                                                    v1,
                                                                                    radius,
                                                                                    mu,
                                                                                    beta,
                                                                                    inv_dim,
                                                                                    numerical_zero,
                                                                                    candidate_positions_soa,
                                                                                    nb_candidates,
                                                                                    out_scores);
    score_sd_edge_kernel<block_size><<<nb_candidates, block_size, 0, stream>>>(positions_soa,
                                                                                 dim_plus_one,
                                                                                 kappa,
                                                                                 row_offsets,
                                                                                 col_indices,
                                                                                 nb_vertices,
                                                                                 v1,
                                                                                 radius,
                                                                                 mu,
                                                                                 beta,
                                                                                 inv_dim,
                                                                                 numerical_zero,
                                                                                 candidate_positions_soa,
                                                                                 nb_candidates,
                                                                                 false,
                                                                                 out_scores);
    return;
  }

  cudaMemsetAsync(out_scores, 0, static_cast<std::size_t>(nb_candidates) * sizeof(double), stream);
  score_sd_edge_kernel<block_size><<<nb_candidates, block_size, 0, stream>>>(positions_soa,
                                                                               dim_plus_one,
                                                                               kappa,
                                                                               row_offsets,
                                                                               col_indices,
                                                                               nb_vertices,
                                                                               v1,
                                                                               radius,
                                                                               mu,
                                                                               beta,
                                                                               inv_dim,
                                                                               numerical_zero,
                                                                               candidate_positions_soa,
                                                                               nb_candidates,
                                                                               true,
                                                                               out_scores);
  if(nb_negative_vertices > 0)
  {
    score_sd_sampled_negative_kernel<block_size><<<nb_candidates, block_size, 0, stream>>>(positions_soa,
                                                                                              dim_plus_one,
                                                                                              kappa,
                                                                                              nb_vertices,
                                                                                              v1,
                                                                                              radius,
                                                                                              mu,
                                                                                              beta,
                                                                                              inv_dim,
                                                                                              numerical_zero,
                                                                                              candidate_positions_soa,
                                                                                              nb_candidates,
                                                                                              negative_vertices,
                                                                                              nb_negative_vertices,
                                                                                              negative_weight,
                                                                                              out_scores);
  }
}

void launch_expected_degrees_s1(const double *theta,
                                const double *kappa,
                                int nb_vertices,
                                double beta,
                                double prefactor,
                                double *out_expected_degrees,
                                cudaStream_t stream)
{
  if(nb_vertices <= 0)
  {
    return;
  }
  constexpr int block_size = 256;
  const int blocks = (nb_vertices + block_size - 1) / block_size;
  expected_degrees_s1_kernel<block_size><<<blocks, block_size, 0, stream>>>(theta,
                                                                              kappa,
                                                                              nb_vertices,
                                                                              beta,
                                                                              prefactor,
                                                                              out_expected_degrees);
}

void launch_expected_degrees_sd(const double *positions_soa,
                                int dim_plus_one,
                                const double *kappa,
                                int nb_vertices,
                                int dim,
                                double beta,
                                double mu,
                                double radius,
                                double numerical_zero,
                                double *out_expected_degrees,
                                cudaStream_t stream)
{
  if(nb_vertices <= 0)
  {
    return;
  }
  constexpr int block_size = 256;
  const int blocks = (nb_vertices + block_size - 1) / block_size;
  expected_degrees_sd_kernel<block_size><<<blocks, block_size, 0, stream>>>(positions_soa,
                                                                              dim_plus_one,
                                                                              kappa,
                                                                              nb_vertices,
                                                                              dim,
                                                                              beta,
                                                                              mu,
                                                                              radius,
                                                                              numerical_zero,
                                                                              out_expected_degrees);
}

} // namespace dmercator::gpu::kernels
