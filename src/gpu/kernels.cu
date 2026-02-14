#include "../../include/dmercator/gpu/kernels.cuh"

#include <cstddef>
#include <cmath>

namespace dmercator::gpu::kernels {

namespace {
constexpr double PI = 3.141592653589793238462643383279502884197;
}

__device__ inline double angular_distance_s1(double t1, double t2)
{
  return PI - fabs(PI - fabs(t1 - t2));
}

__device__ inline double angular_distance_sd(const double *pos1,
                                             const double *pos2,
                                             int position_stride,
                                             double numerical_zero)
{
  double angle = 0.0;
  double norm1 = 0.0;
  double norm2 = 0.0;
  for(int i = 0; i < position_stride; ++i)
  {
    angle += pos1[i] * pos2[i];
    norm1 += pos1[i] * pos1[i];
    norm2 += pos2[i] * pos2[i];
  }
  norm1 /= sqrt(norm1);
  norm2 /= sqrt(norm2);
  const double result = angle / (norm1 * norm2);
  if(fabs(result - 1.0) < numerical_zero)
  {
    return 0.0;
  }
  return acos(result);
}

__global__ void prepare_pair_prefactor_s1_kernel(const double *kappa,
                                                 int nb_vertices,
                                                 int v1,
                                                 double prefactor_over_kappa_v1,
                                                 double *out_pair_prefactor)
{
  const int v2 = blockIdx.x * blockDim.x + threadIdx.x;
  if(v2 >= nb_vertices)
  {
    return;
  }
  if(v2 == v1)
  {
    out_pair_prefactor[v2] = 0.0;
    return;
  }
  out_pair_prefactor[v2] = prefactor_over_kappa_v1 / kappa[v2];
}

__global__ void prepare_pair_prefactor_sd_kernel(const double *kappa,
                                                 int nb_vertices,
                                                 int v1,
                                                 double mu,
                                                 double radius,
                                                 double inv_dim,
                                                 double *out_pair_prefactor)
{
  const int v2 = blockIdx.x * blockDim.x + threadIdx.x;
  if(v2 >= nb_vertices)
  {
    return;
  }
  if(v2 == v1)
  {
    out_pair_prefactor[v2] = 0.0;
    return;
  }
  out_pair_prefactor[v2] = radius / pow(mu * kappa[v1] * kappa[v2], inv_dim);
}

__global__ void evaluate_refine_s1_candidates_kernel(const double *theta,
                                                     const double *pair_prefactor,
                                                     const int *neighbors,
                                                     int neighbor_count,
                                                     const double *candidate_angles,
                                                     int nb_candidates,
                                                     int v1,
                                                     int nb_vertices,
                                                     double beta,
                                                     double *out_scores)
{
  const int candidate_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(candidate_idx >= nb_candidates)
  {
    return;
  }

  const double angle = candidate_angles[candidate_idx];
  double local_loglikelihood = 0.0;

  for(int v2 = 0; v2 < nb_vertices; ++v2)
  {
    if(v2 == v1)
    {
      continue;
    }
    const double da = angular_distance_s1(angle, theta[v2]);
    const double fraction = pair_prefactor[v2] * da;
    local_loglikelihood += -log(1.0 + pow(fraction, -beta));
  }

  for(int idx = 0; idx < neighbor_count; ++idx)
  {
    const int v2 = neighbors[idx];
    const double da = angular_distance_s1(angle, theta[v2]);
    const double fraction = pair_prefactor[v2] * da;
    local_loglikelihood += -beta * log(fraction);
  }

  out_scores[candidate_idx] = local_loglikelihood;
}

__global__ void evaluate_refine_sd_candidates_kernel(const double *positions,
                                                     int position_stride,
                                                     const double *pair_prefactor,
                                                     const int *neighbors,
                                                     int neighbor_count,
                                                     const double *candidate_positions,
                                                     int nb_candidates,
                                                     int v1,
                                                     int nb_vertices,
                                                     double beta,
                                                     double numerical_zero,
                                                     double *out_scores)
{
  const int candidate_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(candidate_idx >= nb_candidates)
  {
    return;
  }

  const double *candidate = candidate_positions + static_cast<size_t>(candidate_idx) * position_stride;
  double local_loglikelihood = 0.0;

  for(int v2 = 0; v2 < nb_vertices; ++v2)
  {
    if(v2 == v1)
    {
      continue;
    }
    const double *pos2 = positions + static_cast<size_t>(v2) * position_stride;
    const double dtheta = angular_distance_sd(candidate, pos2, position_stride, numerical_zero);
    const double chi = pair_prefactor[v2] * dtheta;
    const double prob = 1.0 / (1.0 + pow(chi, beta));
    local_loglikelihood += log(1.0 - prob);
  }

  for(int idx = 0; idx < neighbor_count; ++idx)
  {
    const int v2 = neighbors[idx];
    const double *pos2 = positions + static_cast<size_t>(v2) * position_stride;
    const double dtheta = angular_distance_sd(candidate, pos2, position_stride, numerical_zero);
    const double chi = pair_prefactor[v2] * dtheta;
    const double prob = 1.0 / (1.0 + pow(chi, beta));
    local_loglikelihood += log(prob);
  }

  out_scores[candidate_idx] = local_loglikelihood;
}

__global__ void inferred_expected_degrees_s1_kernel(const double *theta,
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

  const double kappa1 = kappa[v1];
  const double theta1 = theta[v1];
  double expected_degree = 0.0;
  for(int v2 = 0; v2 < nb_vertices; ++v2)
  {
    if(v2 == v1)
    {
      continue;
    }
    const double dtheta = angular_distance_s1(theta1, theta[v2]);
    const double chi = (prefactor * dtheta) / (kappa1 * kappa[v2]);
    const double prob = 1.0 / (1.0 + pow(chi, beta));
    expected_degree += prob;
  }
  out_expected_degrees[v1] = expected_degree;
}

__global__ void inferred_expected_degrees_sd_kernel(const double *positions,
                                                    int position_stride,
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

  const double *pos1 = positions + static_cast<size_t>(v1) * position_stride;
  const double kappa1 = kappa[v1];
  const double inv_dim = 1.0 / static_cast<double>(dim);
  double expected_degree = 0.0;
  for(int v2 = 0; v2 < nb_vertices; ++v2)
  {
    if(v2 == v1)
    {
      continue;
    }
    const double *pos2 = positions + static_cast<size_t>(v2) * position_stride;
    const double dtheta = angular_distance_sd(pos1, pos2, position_stride, numerical_zero);
    const double chi = radius * dtheta / pow(mu * kappa1 * kappa[v2], inv_dim);
    const double prob = 1.0 / (1.0 + pow(chi, beta));
    expected_degree += prob;
  }
  out_expected_degrees[v1] = expected_degree;
}

__global__ void edge_probabilities_s1_kernel(const double *theta,
                                             const double *pair_prefactor,
                                             int nb_vertices,
                                             int v1,
                                             double beta,
                                             double *out_probabilities)
{
  const int v2 = blockIdx.x * blockDim.x + threadIdx.x;
  if(v2 >= nb_vertices)
  {
    return;
  }
  if(v2 == v1)
  {
    out_probabilities[v2] = 0.0;
    return;
  }

  const double dtheta = angular_distance_s1(theta[v1], theta[v2]);
  const double chi = pair_prefactor[v2] * dtheta;
  out_probabilities[v2] = 1.0 / (1.0 + pow(chi, beta));
}

__global__ void edge_probabilities_sd_kernel(const double *positions,
                                             int position_stride,
                                             const double *pair_prefactor,
                                             int nb_vertices,
                                             int v1,
                                             double beta,
                                             double numerical_zero,
                                             double *out_probabilities)
{
  const int v2 = blockIdx.x * blockDim.x + threadIdx.x;
  if(v2 >= nb_vertices)
  {
    return;
  }
  if(v2 == v1)
  {
    out_probabilities[v2] = 0.0;
    return;
  }

  const double *pos1 = positions + static_cast<size_t>(v1) * position_stride;
  const double *pos2 = positions + static_cast<size_t>(v2) * position_stride;
  const double dtheta = angular_distance_sd(pos1, pos2, position_stride, numerical_zero);
  const double chi = pair_prefactor[v2] * dtheta;
  out_probabilities[v2] = 1.0 / (1.0 + pow(chi, beta));
}

} // namespace dmercator::gpu::kernels
