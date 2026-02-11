#ifndef DMERCATOR_CORE_TIMING_HPP
#define DMERCATOR_CORE_TIMING_HPP

#include <chrono>

namespace dmercator::core::timing {

struct StageTimingSummary
{
  double total_time_ms = 0.0;
  double initialization_ms = 0.0;
  double parameter_inference_ms = 0.0;
  double initial_positions_ms = 0.0;
  double refining_positions_ms = 0.0;
  double adjusting_kappas_ms = 0.0;
  double io_ms = 0.0;
};

class ScopedTimer
{
 public:
  explicit ScopedTimer(double &target_ms)
      : target_ms_(target_ms), start_(std::chrono::steady_clock::now())
  {
  }

  ScopedTimer(const ScopedTimer &) = delete;
  ScopedTimer &operator=(const ScopedTimer &) = delete;

  ~ScopedTimer()
  {
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start_);
    target_ms_ += elapsed.count();
  }

 private:
  double &target_ms_;
  std::chrono::steady_clock::time_point start_;
};

} // namespace dmercator::core::timing

#endif
