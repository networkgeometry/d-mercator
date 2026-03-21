#include "dmercator/gpu/likelihood_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884197;

dmercator::gpu::CsrGraph make_test_graph()
{
  dmercator::gpu::CsrGraph graph;
  graph.nb_vertices = 5;
  graph.row_offsets = {0, 2, 4, 7, 9, 10};
  graph.col_indices = {1, 2, 0, 2, 0, 1, 3, 2, 4, 3};
  return graph;
}

std::vector<double> positions_to_soa(const std::vector<std::vector<double>> &positions)
{
  const std::size_t n = positions.size();
  const std::size_t stride = positions.front().size();
  std::vector<double> soa(stride * n, 0.0);
  for(std::size_t vertex = 0; vertex < n; ++vertex)
  {
    for(std::size_t axis = 0; axis < stride; ++axis)
    {
      soa[axis * n + vertex] = positions[vertex][axis];
    }
  }
  return soa;
}

double angular_distance_s1(double lhs, double rhs)
{
  return kPi - std::fabs(kPi - std::fabs(lhs - rhs));
}

double score_candidate_s1_host(const std::vector<double> &theta,
                               const std::vector<double> &kappa,
                               const std::vector<int> &neighbors,
                               int v1,
                               double prefactor,
                               double beta,
                               double candidate_theta,
                               const std::vector<int> &negative_vertices,
                               double negative_weight,
                               bool exact_negative_sweep)
{
  double score = 0.0;
  if(exact_negative_sweep)
  {
    for(int v2 = 0; v2 < static_cast<int>(theta.size()); ++v2)
    {
      if(v2 == v1)
      {
        continue;
      }
      const double fraction = (prefactor * angular_distance_s1(candidate_theta, theta[v2])) / (kappa[v1] * kappa[v2]);
      score += -std::log(1.0 + std::pow(fraction, -beta));
    }
    for(const int v2 : neighbors)
    {
      const double fraction = (prefactor * angular_distance_s1(candidate_theta, theta[v2])) / (kappa[v1] * kappa[v2]);
      score += -beta * std::log(fraction);
    }
    return score;
  }

  for(const int v2 : neighbors)
  {
    const double fraction = (prefactor * angular_distance_s1(candidate_theta, theta[v2])) / (kappa[v1] * kappa[v2]);
    score += -std::log(1.0 + std::pow(fraction, -beta));
    score += -beta * std::log(fraction);
  }
  double sampled_negative = 0.0;
  for(const int v2 : negative_vertices)
  {
    const double fraction = (prefactor * angular_distance_s1(candidate_theta, theta[v2])) / (kappa[v1] * kappa[v2]);
    sampled_negative += -std::log(1.0 + std::pow(fraction, -beta));
  }
  score += negative_weight * sampled_negative;
  return score;
}

double compute_radius(int dim, int n)
{
  const double inside = n / (2.0 * std::pow(kPi, (dim + 1) / 2.0)) * std::tgamma((dim + 1) / 2.0);
  return std::pow(inside, 1.0 / dim);
}

double angular_distance_sd(const std::vector<double> &lhs, const std::vector<double> &rhs, double numerical_zero)
{
  double dot = 0.0;
  double norm1 = 0.0;
  double norm2 = 0.0;
  for(std::size_t axis = 0; axis < lhs.size(); ++axis)
  {
    dot += lhs[axis] * rhs[axis];
    norm1 += lhs[axis] * lhs[axis];
    norm2 += rhs[axis] * rhs[axis];
  }
  norm1 /= std::sqrt(norm1);
  norm2 /= std::sqrt(norm2);
  const double result = dot / (norm1 * norm2);
  if(std::fabs(result - 1.0) < numerical_zero)
  {
    return 0.0;
  }
  return std::acos(std::max(-1.0, std::min(1.0, result)));
}

double score_candidate_sd_host(const std::vector<std::vector<double>> &positions,
                               const std::vector<double> &kappa,
                               const std::vector<int> &neighbors,
                               int dim,
                               int v1,
                               double radius,
                               double mu,
                               double beta,
                               double numerical_zero,
                               const std::vector<double> &candidate_position,
                               const std::vector<int> &negative_vertices,
                               double negative_weight,
                               bool exact_negative_sweep)
{
  const double inv_dim = 1.0 / static_cast<double>(dim);
  double score = 0.0;
  auto pair_prob = [&](int v2) -> double
  {
    const double dtheta = angular_distance_sd(candidate_position, positions[v2], numerical_zero);
    const double chi = radius * dtheta / std::pow(mu * kappa[v1] * kappa[v2], inv_dim);
    return 1.0 / (1.0 + std::pow(chi, beta));
  };

  if(exact_negative_sweep)
  {
    for(int v2 = 0; v2 < static_cast<int>(positions.size()); ++v2)
    {
      if(v2 == v1)
      {
        continue;
      }
      score += std::log(1.0 - pair_prob(v2));
    }
    for(const int v2 : neighbors)
    {
      score += std::log(pair_prob(v2));
    }
    return score;
  }

  for(const int v2 : neighbors)
  {
    const double prob = pair_prob(v2);
    score += std::log(1.0 - prob);
    score += std::log(prob);
  }
  double sampled_negative = 0.0;
  for(const int v2 : negative_vertices)
  {
    sampled_negative += std::log(1.0 - pair_prob(v2));
  }
  score += negative_weight * sampled_negative;
  return score;
}

bool approx_equal(const std::vector<double> &lhs, const std::vector<double> &rhs, double tolerance)
{
  if(lhs.size() != rhs.size())
  {
    return false;
  }
  for(std::size_t i = 0; i < lhs.size(); ++i)
  {
    if(std::fabs(lhs[i] - rhs[i]) > tolerance)
    {
      return false;
    }
  }
  return true;
}

int verify_s1_backend(dmercator::gpu::LikelihoodBackend &backend)
{
  const std::vector<double> kappa = {1.2, 1.5, 1.8, 1.4, 1.1};
  const std::vector<double> theta = {0.1, 0.9, 2.1, 3.0, 4.2};
  const std::vector<int> neighbors = {1, 2};
  const std::vector<double> candidates = {0.25, 1.75};
  const std::vector<int> negatives = {3};
  const double mu = 0.75;
  const double beta = 2.0;
  const double prefactor = static_cast<double>(theta.size()) / (2.0 * kPi * mu);
  const double negative_weight = 2.0;
  const int v1 = 0;

  if(!backend.set_kappa(kappa) || !backend.set_theta(theta))
  {
    std::cerr << "S1 backend setup failed: " << backend.last_error() << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<double> gpu_exact;
  if(!backend.score_candidates_s1(v1,
                                  prefactor,
                                  beta,
                                  candidates,
                                  {},
                                  1.0,
                                  true,
                                  gpu_exact))
  {
    std::cerr << "S1 exact backend scoring failed: " << backend.last_error() << std::endl;
    return EXIT_FAILURE;
  }
  std::vector<double> host_exact;
  for(const double candidate : candidates)
  {
    host_exact.push_back(score_candidate_s1_host(theta,
                                                 kappa,
                                                 neighbors,
                                                 v1,
                                                 prefactor,
                                                 beta,
                                                 candidate,
                                                 {},
                                                 1.0,
                                                 true));
  }

  std::vector<double> gpu_sampled;
  if(!backend.score_candidates_s1(v1,
                                  prefactor,
                                  beta,
                                  candidates,
                                  negatives,
                                  negative_weight,
                                  false,
                                  gpu_sampled))
  {
    std::cerr << "S1 sampled backend scoring failed: " << backend.last_error() << std::endl;
    return EXIT_FAILURE;
  }
  std::vector<double> host_sampled;
  for(const double candidate : candidates)
  {
    host_sampled.push_back(score_candidate_s1_host(theta,
                                                   kappa,
                                                   neighbors,
                                                   v1,
                                                   prefactor,
                                                   beta,
                                                   candidate,
                                                   negatives,
                                                   negative_weight,
                                                   false));
  }

  constexpr double kTolerance = 1e-8;
  if(!approx_equal(host_exact, gpu_exact, kTolerance) || !approx_equal(host_sampled, gpu_sampled, kTolerance))
  {
    std::cerr << "S1 backend mismatch" << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int verify_sd_backend(dmercator::gpu::LikelihoodBackend &backend)
{
  const int dim = 2;
  const double numerical_zero = 1e-10;
  const double radius = compute_radius(dim, 5);
  const std::vector<double> kappa = {1.2, 1.5, 1.8, 1.4, 1.1};
  const std::vector<std::vector<double>> positions = {
    {radius, 0.0, 0.0},
    {0.0, radius, 0.0},
    {0.0, 0.0, radius},
    {-radius, 0.0, 0.0},
    {0.0, -radius, 0.0},
  };
  const std::vector<int> neighbors = {1, 2};
  const std::vector<std::vector<double>> candidates = {
    {radius / std::sqrt(2.0), radius / std::sqrt(2.0), 0.0},
    {radius / std::sqrt(2.0), 0.0, radius / std::sqrt(2.0)},
  };
  const std::vector<int> negatives = {3};
  const double mu = 0.55;
  const double beta = 4.0;
  const double negative_weight = 2.0;
  const int v1 = 0;

  if(!backend.set_kappa(kappa) || !backend.set_positions_soa(dim + 1, positions_to_soa(positions)))
  {
    std::cerr << "S^D backend setup failed: " << backend.last_error() << std::endl;
    return EXIT_FAILURE;
  }

  const auto candidate_soa = positions_to_soa(candidates);
  std::vector<double> gpu_exact;
  if(!backend.score_candidates_sd(dim,
                                  v1,
                                  radius,
                                  mu,
                                  beta,
                                  numerical_zero,
                                  candidate_soa,
                                  {},
                                  1.0,
                                  true,
                                  gpu_exact))
  {
    std::cerr << "S^D exact backend scoring failed: " << backend.last_error() << std::endl;
    return EXIT_FAILURE;
  }
  std::vector<double> host_exact;
  for(const auto &candidate : candidates)
  {
    host_exact.push_back(score_candidate_sd_host(positions,
                                                 kappa,
                                                 neighbors,
                                                 dim,
                                                 v1,
                                                 radius,
                                                 mu,
                                                 beta,
                                                 numerical_zero,
                                                 candidate,
                                                 {},
                                                 1.0,
                                                 true));
  }

  std::vector<double> gpu_sampled;
  if(!backend.score_candidates_sd(dim,
                                  v1,
                                  radius,
                                  mu,
                                  beta,
                                  numerical_zero,
                                  candidate_soa,
                                  negatives,
                                  negative_weight,
                                  false,
                                  gpu_sampled))
  {
    std::cerr << "S^D sampled backend scoring failed: " << backend.last_error() << std::endl;
    return EXIT_FAILURE;
  }
  std::vector<double> host_sampled;
  for(const auto &candidate : candidates)
  {
    host_sampled.push_back(score_candidate_sd_host(positions,
                                                   kappa,
                                                   neighbors,
                                                   dim,
                                                   v1,
                                                   radius,
                                                   mu,
                                                   beta,
                                                   numerical_zero,
                                                   candidate,
                                                   negatives,
                                                   negative_weight,
                                                   false));
  }

  constexpr double kTolerance = 1e-8;
  if(!approx_equal(host_exact, gpu_exact, kTolerance) || !approx_equal(host_sampled, gpu_sampled, kTolerance))
  {
    std::cerr << "S^D backend mismatch" << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

} // namespace

int main()
{
  auto backend = dmercator::gpu::create_likelihood_backend();
  if(!backend)
  {
    std::cerr << "backend creation failed" << std::endl;
    return EXIT_FAILURE;
  }

  const auto status = backend->initialize(make_test_graph(), true);
  if(!status.available)
  {
    std::cerr << "CUDA backend unavailable: " << status.message << std::endl;
    return EXIT_FAILURE;
  }

  if(verify_s1_backend(*backend) != EXIT_SUCCESS)
  {
    return EXIT_FAILURE;
  }
  if(verify_sd_backend(*backend) != EXIT_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  std::cout << "negative sampling backend smoke passed" << std::endl;
  return EXIT_SUCCESS;
}
