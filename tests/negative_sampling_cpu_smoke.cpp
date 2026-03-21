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
  const auto edge_path = root / "tiny_negative_sampling.edge";

  constexpr int kNodes = 24;
  std::ofstream out(edge_path);
  if(!out.is_open())
  {
    throw std::runtime_error("failed to open tiny_negative_sampling.edge for writing");
  }

  for(int i = 0; i < kNodes; ++i)
  {
    out << "v" << i << " " << "v" << ((i + 1) % kNodes) << "\n";
    out << "v" << i << " " << "v" << ((i + 5) % kNodes) << "\n";
    out << "v" << i << " " << "v" << ((i + 9) % kNodes) << "\n";
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
  if(lhs.size() != rhs.size())
  {
    throw std::runtime_error("vector size mismatch");
  }
  double worst = 0.0;
  for(std::size_t i = 0; i < lhs.size(); ++i)
  {
    worst = std::max(worst, diff_fn(lhs[i], rhs[i]));
  }
  return worst;
}

std::vector<double> flatten_positions(const std::vector<std::vector<double>> &positions)
{
  std::vector<double> flat;
  for(const auto &position : positions)
  {
    flat.insert(flat.end(), position.begin(), position.end());
  }
  return flat;
}

embeddingSD_t::EmbeddingStateSnapshot run_embedding(const std::filesystem::path &edge_path,
                                                    const std::filesystem::path &output_root,
                                                    int dimension,
                                                    int negative_samples)
{
  embeddingSD_t graph;
  graph.EDGELIST_FILENAME = edge_path.string();
  graph.CUSTOM_OUTPUT_ROOTNAME_MODE = true;
  graph.ROOTNAME_OUTPUT = output_root.string();
  graph.CUSTOM_SEED = true;
  graph.SEED = 424242;
  graph.CUSTOM_BETA = true;
  graph.beta = 2.0 * static_cast<double>(dimension);
  graph.DIMENSION = dimension;
  graph.QUIET_MODE = true;
  graph.VALIDATION_MODE = false;
  graph.CHARACTERIZATION_MODE = false;
  graph.REFINE_NEGATIVE_SAMPLES = negative_samples;
  graph.embed();
  return graph.snapshot_state();
}

int verify_dimension(const std::filesystem::path &edge_path, const std::filesystem::path &root, int dimension)
{
  const auto exact = run_embedding(edge_path, root / ("d" + std::to_string(dimension) + "_exact"), dimension, 0);
  const auto oversampled = run_embedding(edge_path, root / ("d" + std::to_string(dimension) + "_oversampled"), dimension, 1000);
  const auto sampled_a = run_embedding(edge_path, root / ("d" + std::to_string(dimension) + "_sampled_a"), dimension, 4);
  const auto sampled_b = run_embedding(edge_path, root / ("d" + std::to_string(dimension) + "_sampled_b"), dimension, 4);

  constexpr double kExactTolerance = 1e-8;
  constexpr double kDeterministicTolerance = 1e-8;

  const double exact_kappa_max = max_abs_diff(exact.kappa, oversampled.kappa, [](double a, double b) {
    return std::fabs(a - b);
  });
  const double sampled_kappa_max = max_abs_diff(sampled_a.kappa, sampled_b.kappa, [](double a, double b) {
    return std::fabs(a - b);
  });

  double exact_position_max = 0.0;
  double sampled_position_max = 0.0;
  if(dimension == 1)
  {
    exact_position_max = max_abs_diff(exact.theta, oversampled.theta, [](double a, double b) {
      return circular_abs_diff(a, b);
    });
    sampled_position_max = max_abs_diff(sampled_a.theta, sampled_b.theta, [](double a, double b) {
      return circular_abs_diff(a, b);
    });
  }
  else
  {
    const auto exact_flat = flatten_positions(exact.d_positions);
    const auto oversampled_flat = flatten_positions(oversampled.d_positions);
    const auto sampled_a_flat = flatten_positions(sampled_a.d_positions);
    const auto sampled_b_flat = flatten_positions(sampled_b.d_positions);
    exact_position_max = max_abs_diff(exact_flat, oversampled_flat, [](double a, double b) {
      return std::fabs(a - b);
    });
    sampled_position_max = max_abs_diff(sampled_a_flat, sampled_b_flat, [](double a, double b) {
      return std::fabs(a - b);
    });
  }

  if(exact_kappa_max > kExactTolerance || exact_position_max > kExactTolerance)
  {
    std::cerr << "negative sampling exact-collapse failed for dimension " << dimension
              << " kappa_max=" << exact_kappa_max
              << " position_max=" << exact_position_max
              << " tolerance=" << kExactTolerance
              << std::endl;
    return EXIT_FAILURE;
  }
  if(sampled_kappa_max > kDeterministicTolerance || sampled_position_max > kDeterministicTolerance)
  {
    std::cerr << "negative sampling determinism failed for dimension " << dimension
              << " kappa_max=" << sampled_kappa_max
              << " position_max=" << sampled_position_max
              << " tolerance=" << kDeterministicTolerance
              << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

} // namespace

int main()
{
#if defined(_WIN32)
  _putenv_s("OMP_NUM_THREADS", "1");
#else
  setenv("OMP_NUM_THREADS", "1", 1);
#endif

  const auto root = std::filesystem::temp_directory_path() / "dmercator_negative_sampling_cpu_smoke";
  const auto edge_path = write_tiny_connected_graph(root);

  if(verify_dimension(edge_path, root, 1) != EXIT_SUCCESS)
  {
    return EXIT_FAILURE;
  }
  if(verify_dimension(edge_path, root, 2) != EXIT_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  std::cout << "negative sampling cpu smoke passed" << std::endl;
  return EXIT_SUCCESS;
}
