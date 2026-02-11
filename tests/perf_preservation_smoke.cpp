#include "embeddingSD.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::filesystem::path write_tiny_connected_graph(const std::filesystem::path &root)
{
  std::filesystem::create_directories(root);
  const auto edge_path = root / "tiny.edge";

  constexpr int kNodes = 28;
  std::ofstream out(edge_path);
  if(!out.is_open()) {
    throw std::runtime_error("failed to open tiny.edge for writing");
  }

  for(int i = 0; i < kNodes; ++i) {
    const int j = (i + 1) % kNodes;
    out << "v" << i << " " << "v" << j << "\n";
  }
  for(int i = 0; i < kNodes; ++i) {
    const int j = (i + 5) % kNodes;
    out << "v" << i << " " << "v" << j << "\n";
  }
  for(int i = 0; i < kNodes; ++i) {
    const int j = (i + 9) % kNodes;
    out << "v" << i << " " << "v" << j << "\n";
  }

  return edge_path;
}

double circular_abs_diff(double a, double b)
{
  constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
  const double raw = std::fmod(std::fabs(a - b), kTwoPi);
  return std::min(raw, kTwoPi - raw);
}

template <typename DiffFn>
double max_abs_diff(const std::vector<double> &lhs, const std::vector<double> &rhs, DiffFn diff_fn)
{
  if(lhs.size() != rhs.size()) {
    throw std::runtime_error("vector size mismatch");
  }
  double worst = 0.0;
  for(std::size_t i = 0; i < lhs.size(); ++i) {
    worst = std::max(worst, diff_fn(lhs[i], rhs[i]));
  }
  return worst;
}

embeddingSD_t::EmbeddingStateSnapshot run_embedding(const std::filesystem::path &edge_path,
                                                    const std::filesystem::path &output_root,
                                                    bool optimized)
{
  embeddingSD_t graph;
  graph.EDGELIST_FILENAME = edge_path.string();
  graph.CUSTOM_OUTPUT_ROOTNAME_MODE = true;
  graph.ROOTNAME_OUTPUT = output_root.string();
  graph.CUSTOM_SEED = true;
  graph.SEED = 424242;
  graph.CUSTOM_BETA = true;
  graph.beta = 4.2;
  graph.DIMENSION = 1;
  graph.QUIET_MODE = true;
  graph.VALIDATION_MODE = false;
  graph.CHARACTERIZATION_MODE = false;
  graph.OPTIMIZED_PERF_MODE = optimized;

  graph.embed();
  return graph.snapshot_state();
}

} // namespace

int main()
{
#if defined(_WIN32)
  _putenv_s("OMP_NUM_THREADS", "1");
#else
  setenv("OMP_NUM_THREADS", "1", 1);
#endif

  const auto root = std::filesystem::temp_directory_path() / "dmercator_perf_smoke";
  const auto edge_path = write_tiny_connected_graph(root);

  const auto baseline = run_embedding(edge_path, root / "baseline", false);
  const auto optimized = run_embedding(edge_path, root / "optimized", true);

  const double kappa_max = max_abs_diff(baseline.kappa, optimized.kappa, [](double a, double b) {
    return std::fabs(a - b);
  });
  const double theta_max = max_abs_diff(baseline.theta, optimized.theta, [](double a, double b) {
    return circular_abs_diff(a, b);
  });

  constexpr double kTolerance = 1e-8;
  if(kappa_max > kTolerance || theta_max > kTolerance) {
    std::cerr << "perf preservation smoke failed:"
              << " kappa_max=" << kappa_max
              << " theta_max=" << theta_max
              << " tolerance=" << kTolerance
              << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "perf preservation smoke passed"
            << " kappa_max=" << kappa_max
            << " theta_max=" << theta_max
            << std::endl;
  return EXIT_SUCCESS;
}
