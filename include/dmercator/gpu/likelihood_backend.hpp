#ifndef DMERCATOR_GPU_LIKELIHOOD_BACKEND_HPP
#define DMERCATOR_GPU_LIKELIHOOD_BACKEND_HPP

#include <memory>
#include <string>
#include <vector>

namespace dmercator::gpu {

struct DeviceInitStatus
{
  bool available = false;
  std::string message;
};

struct CsrGraph
{
  int nb_vertices = 0;
  std::vector<int> row_offsets;
  std::vector<int> col_indices;
};

class LikelihoodBackend
{
  public:
    virtual ~LikelihoodBackend() = default;

    virtual DeviceInitStatus initialize(const CsrGraph &graph, bool deterministic_mode) = 0;
    virtual bool is_initialized() const = 0;
    virtual const std::string &last_error() const = 0;

    virtual bool set_kappa(const std::vector<double> &kappa) = 0;
    virtual bool set_degree(const std::vector<int> &degree) = 0;
    virtual bool set_theta(const std::vector<double> &theta) = 0;
    virtual bool set_theta_single(int vertex_index, double theta_value) = 0;

    // Positions are stored in SoA layout:
    // [coord0(all vertices), coord1(all vertices), ...].
    virtual bool set_positions_soa(int dim_plus_one, const std::vector<double> &positions_soa) = 0;
    virtual bool set_position_single(int vertex_index, const std::vector<double> &position) = 0;

    virtual bool score_candidates_s1(int v1,
                                     double prefactor,
                                     double beta,
                                     const std::vector<double> &candidate_theta,
                                     const std::vector<int> &negative_vertices,
                                     double negative_weight,
                                     bool exact_negative_sweep,
                                     std::vector<double> &out_scores) = 0;

    // Candidate positions must be SoA:
    // [coord0(all candidates), coord1(all candidates), ...].
    virtual bool score_candidates_sd(int dim,
                                     int v1,
                                     double radius,
                                     double mu,
                                     double beta,
                                     double numerical_zero,
                                     const std::vector<double> &candidate_positions_soa,
                                     const std::vector<int> &negative_vertices,
                                     double negative_weight,
                                     bool exact_negative_sweep,
                                     std::vector<double> &out_scores) = 0;

    virtual bool compute_expected_degrees_s1(double beta,
                                             double mu,
                                             std::vector<double> &out_expected_degrees) = 0;

    virtual bool compute_expected_degrees_sd(int dim,
                                             double beta,
                                             double mu,
                                             double radius,
                                             double numerical_zero,
                                             std::vector<double> &out_expected_degrees) = 0;
};

std::unique_ptr<LikelihoodBackend> create_likelihood_backend();

} // namespace dmercator::gpu

#endif // DMERCATOR_GPU_LIKELIHOOD_BACKEND_HPP
