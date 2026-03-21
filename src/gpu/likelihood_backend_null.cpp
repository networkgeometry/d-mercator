#include "../../include/dmercator/gpu/likelihood_backend.hpp"

#include <memory>
#include <string>
#include <utility>

namespace dmercator::gpu {

namespace {

class NullLikelihoodBackend final : public LikelihoodBackend
{
  public:
    DeviceInitStatus initialize(const CsrGraph &, bool) override
    {
      last_error_ = "CUDA backend is unavailable in this build.";
      return {false, last_error_};
    }

    bool is_initialized() const override
    {
      return false;
    }

    const std::string &last_error() const override
    {
      return last_error_;
    }

    bool set_kappa(const std::vector<double> &) override
    {
      return fail("set_kappa");
    }

    bool set_degree(const std::vector<int> &) override
    {
      return fail("set_degree");
    }

    bool set_theta(const std::vector<double> &) override
    {
      return fail("set_theta");
    }

    bool set_theta_single(int, double) override
    {
      return fail("set_theta_single");
    }

    bool set_positions_soa(int, const std::vector<double> &) override
    {
      return fail("set_positions_soa");
    }

    bool set_position_single(int, const std::vector<double> &) override
    {
      return fail("set_position_single");
    }

    bool score_candidates_s1(int,
                             double,
                             double,
                             const std::vector<double> &,
                             const std::vector<int> &,
                             double,
                             bool,
                             std::vector<double> &) override
    {
      return fail("score_candidates_s1");
    }

    bool score_candidates_sd(int,
                             int,
                             double,
                             double,
                             double,
                             double,
                             const std::vector<double> &,
                             const std::vector<int> &,
                             double,
                             bool,
                             std::vector<double> &) override
    {
      return fail("score_candidates_sd");
    }

    bool compute_expected_degrees_s1(double, double, std::vector<double> &) override
    {
      return fail("compute_expected_degrees_s1");
    }

    bool compute_expected_degrees_sd(int,
                                     double,
                                     double,
                                     double,
                                     double,
                                     std::vector<double> &) override
    {
      return fail("compute_expected_degrees_sd");
    }

  private:
    bool fail(const char *method)
    {
      last_error_ = std::string(method) + " is unavailable because CUDA support is disabled.";
      return false;
    }

    std::string last_error_;
};

} // namespace

#if defined(DMERCATOR_USE_CUDA)
std::unique_ptr<LikelihoodBackend> create_cuda_likelihood_backend();
#endif

std::unique_ptr<LikelihoodBackend> create_likelihood_backend()
{
#if defined(DMERCATOR_USE_CUDA)
  return create_cuda_likelihood_backend();
#else
  return std::make_unique<NullLikelihoodBackend>();
#endif
}

} // namespace dmercator::gpu
