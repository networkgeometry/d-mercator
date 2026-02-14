#ifndef DMERCATOR_GPU_CONTEXT_HPP
#define DMERCATOR_GPU_CONTEXT_HPP

#include <string>
#include <vector>

namespace dmercator::gpu {

struct DeviceInitStatus
{
  bool available = false;
  std::string message;
};

DeviceInitStatus initialize(bool deterministic_mode);
const std::string &last_error();

bool begin_refine_s1(const std::vector<double> &theta, const std::vector<double> &kappa);
bool begin_refine_sd(int dim, const std::vector<std::vector<double>> &positions, const std::vector<double> &kappa);
bool begin_refine_s1(const std::vector<double> &theta);
bool begin_refine_sd(int dim, const std::vector<std::vector<double>> &positions);

bool update_refine_theta(int vertex_index, double theta_value);
bool update_refine_position(int dim, int vertex_index, const std::vector<double> &position);

bool evaluate_refine_s1_candidates(int v1,
                                   double beta,
                                   const std::vector<double> &pair_prefactor,
                                   const std::vector<int> &neighbors,
                                   const std::vector<double> &candidate_angles,
                                   std::vector<double> &out_scores);

bool evaluate_refine_sd_candidates(int dim,
                                   int v1,
                                   double beta,
                                   double numerical_zero,
                                   const std::vector<double> &pair_prefactor,
                                   const std::vector<int> &neighbors,
                                   const std::vector<double> &candidate_positions_flat,
                                   std::vector<double> &out_scores);

bool evaluate_refine_s1_candidates_from_kappa(int v1,
                                              double beta,
                                              double prefactor_over_kappa_v1,
                                              const std::vector<int> &neighbors,
                                              const std::vector<double> &candidate_angles,
                                              std::vector<double> &out_scores);

bool evaluate_refine_sd_candidates_from_kappa(int dim,
                                              int v1,
                                              double beta,
                                              double numerical_zero,
                                              double mu,
                                              double radius,
                                              const std::vector<int> &neighbors,
                                              const std::vector<double> &candidate_positions_flat,
                                              std::vector<double> &out_scores);

bool evaluate_generation_s1_probabilities_from_kappa(int v1,
                                                     double beta,
                                                     double mu,
                                                     std::vector<double> &out_probabilities);

bool evaluate_generation_sd_probabilities_from_kappa(int dim,
                                                     int v1,
                                                     double beta,
                                                     double numerical_zero,
                                                     double mu,
                                                     double radius,
                                                     std::vector<double> &out_probabilities);

bool prepare_inferred_expected_degrees_s1(const std::vector<double> &theta);
bool prepare_inferred_expected_degrees_sd(int dim, const std::vector<std::vector<double>> &positions);
void clear_inferred_expected_degrees_state();

bool compute_inferred_expected_degrees_s1_prepared(double beta,
                                                   double mu,
                                                   const std::vector<double> &kappa,
                                                   std::vector<double> &out_expected_degrees);

bool compute_inferred_expected_degrees_sd_prepared(int dim,
                                                   double beta,
                                                   double mu,
                                                   double radius,
                                                   double numerical_zero,
                                                   const std::vector<double> &kappa,
                                                   std::vector<double> &out_expected_degrees);

bool compute_inferred_expected_degrees_s1(double beta,
                                          double mu,
                                          const std::vector<double> &theta,
                                          const std::vector<double> &kappa,
                                          std::vector<double> &out_expected_degrees);

bool compute_inferred_expected_degrees_sd(int dim,
                                          double beta,
                                          double mu,
                                          double radius,
                                          double numerical_zero,
                                          const std::vector<std::vector<double>> &positions,
                                          const std::vector<double> &kappa,
                                          std::vector<double> &out_expected_degrees);

} // namespace dmercator::gpu

#endif // DMERCATOR_GPU_CONTEXT_HPP
