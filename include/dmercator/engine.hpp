
#ifndef DMERCATOR_ENGINE_HPP
#define DMERCATOR_ENGINE_HPP

#include <algorithm>
#include <cmath>
#include <ctime>
#include <complex>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <filesystem>
#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "Spectra/GenEigsSolver.h"
#include "Spectra/MatOp/SparseGenMatProd.h"
#include "hyp2f1.hpp"
#include "integrate_expected_degree.hpp"
#include "readjust_positions.hpp"
#include "dmercator/core/timing.hpp"

class embeddingSD_t
{
  private:

    const double PI = 3.141592653589793238462643383279502884197;

  public:

    bool CHARACTERIZATION_MODE = false;

    bool CLEAN_RAW_OUTPUT_MODE = false;

    bool CUSTOM_BETA = false;

    bool CUSTOM_CHARACTERIZATION_NB_GRAPHS = false;

    bool CUSTOM_INFERRED_COORDINATES = false;

    bool CUSTOM_OUTPUT_ROOTNAME_MODE = false;

    bool CUSTOM_SEED = false;

    bool KAPPA_POST_INFERENCE_MODE = true;

    bool MAXIMIZATION_MODE = true;

    bool QUIET_MODE = false;

    bool REFINE_MODE = false;

    bool VALIDATION_MODE = false;

    bool VERBOSE_MODE = false;

    // Runtime switch used by preservation tests:
    // - true: optimized implementation path
    // - false: baseline/original implementation path
    bool OPTIMIZED_PERF_MODE = true;

    // Print one machine-readable timing JSON line per embedding run.
    bool TIMING_JSON_MODE = false;

    int DIMENSION = 2;

    bool ONLY_KAPPAS = false;

  public:

    std::string ALREADY_INFERRED_PARAMETERS_FILENAME;

    const double BETA_ABS_MAX = 25;
    const double BETA_ABS_MIN = 1.01;

    int CHARACTERIZATION_NB_GRAPHS = 100;

    const int MIN_NB_ANGLES_TO_TRY = 100;

    std::string EDGELIST_FILENAME;

    const int EXP_CLUST_NB_INTEGRATION_MC_STEPS = 600;

    const int EXP_DIST_NB_INTEGRATION_STEPS = 1000;

    const int KAPPA_MAX_NB_ITER_CONV =  500;
    const int KAPPA_MAX_NB_ITER_CONV_2 = 500;

    const double MIN_TWO_SIGMAS_NORMAL_DIST = PI;

    const double NUMERICAL_CONVERGENCE_THRESHOLD_1 = 1e-2;
    const double NUMERICAL_CONVERGENCE_THRESHOLD_2 = 5e-5;
    const double NUMERICAL_CONVERGENCE_THRESHOLD_3 = 0.5;

    const double NUMERICAL_ZERO = 1e-10;

    std::string ROOTNAME_OUTPUT;

    int SEED;
  private:

    std::string_view VERSION = "0.9";

    std::string_view TAB = "    ";

  private:

    std::mt19937 engine;
    std::uniform_real_distribution<double> uniform_01;
    std::normal_distribution<double> normal_01;

    std::map< std::string, int > Name2Num;
    std::vector<std::string> Num2Name;

    std::set<int> degree_class;

    std::map<int, std::map<double, int, std::less<>>> cumul_prob_kgkp;

    std::vector<int> ordered_list_of_vertices;

    double time0, time1, time2, time3, time4, time5, time6, time7;
    time_t time_started, time_ended;

    std::ofstream logfile;
    std::streambuf *old_rdbuf;

    int width_names;
    int width_values;

  private:

    int nb_vertices;
    int nb_vertices_degree_gt_one;

    int nb_edges;

    double average_degree;

    double average_clustering;

    std::vector<double> sum_degree_of_neighbors;

    std::vector<double> nbtriangles;

    std::vector< std::set<int> > adjacency_list;

    // Contiguous adjacency cache (sorted, copied from adjacency_list) for
    // inner-loop iteration and membership scans.
    std::vector<std::vector<int>> adjacency_flat_list;

    std::vector<int> degree;

    std::map<int, std::vector<int> > degree2vertices;

  public:

    double beta;
  private:

    double random_ensemble_average_degree;

    double random_ensemble_average_clustering;

    std::map<int, double> random_ensemble_expected_degree_per_degree_class;

    std::vector<double> inferred_ensemble_expected_degree;

    std::map<int, double> random_ensemble_kappa_per_degree_class;

    double mu;

    std::vector<double> kappa;

    std::vector<double> theta;

    std::vector<std::vector<double>> d_positions;

  private:

    std::vector< std::set<int> > simulated_adjacency_list;

    std::vector<double> simulated_degree;

    std::vector<double> simulated_sum_degree_of_neighbors;

    std::vector<double> simulated_nb_triangles;

    std::map<int, double> simulated_stat_degree;

    std::map<int, double> simulated_stat_sum_degree_neighbors;

    std::map<int, double> simulated_stat_avg_degree_neighbors;

    std::map<int, double> simulated_stat_nb_triangles;

    std::map<int, double> simulated_stat_clustering;

    std::vector< std::vector< std::vector<double> > > characterizing_inferred_ensemble_vprops;

    std::map< int, std::vector<double> > characterizing_inferred_ensemble_vstat;

    dmercator::core::timing::StageTimingSummary stage_timing_summary;

  private:

    void analyze_degrees();

    void compute_clustering();

    void load_edgelist();

    void load_already_inferred_parameters();
    void load_already_inferred_parameters(int dim);

    void build_cumul_dist_for_mc_integration();
    void build_cumul_dist_for_mc_integration(int dim);

    void compute_random_ensemble_average_degree();
    void compute_random_ensemble_clustering(int dim);
    void compute_random_ensemble_clustering();
    double compute_random_ensemble_clustering_for_degree_class(int d1, int dim);
    double compute_random_ensemble_clustering_for_degree_class(int d1);

    void infer_kappas_given_beta_for_all_vertices();
    void infer_kappas_given_beta_for_all_vertices(int dim);

    void infer_kappas_given_beta_for_degree_class();
    void infer_kappas_given_beta_for_degree_class(int dim);

    double compute_pairwise_loglikelihood(int v1, double t1, int v2, double t2, bool neighbors);
    double compute_pairwise_loglikelihood(int dim, int v1, const std::vector<double> &pos1, int v2, const std::vector<double> &pos2, bool neighbors, double radius);

    void find_initial_ordering(std::vector<int> &ordering, std::vector<double> &raw_theta);
    void find_initial_ordering(std::vector<std::vector<double>> &positions, int dim);
    void infer_initial_positions();
    void infer_initial_positions(int dim);

    void refine_positions();
    int refine_angle(int v1);
    void refine_positions(int dim);
    int refine_angle(int dim, int v1, double radius);

    void finalize();

    void initialize();

    void infer_parameters();

    void infer_parameters(int dim);

    void order_vertices();

    void compute_inferred_ensemble_expected_degrees();
    void compute_inferred_ensemble_expected_degrees(int dim, double radius);

    void analyze_simulated_adjacency_list();
    void generate_simulated_adjacency_list();
    void generate_simulated_adjacency_list(int dim, bool random_positions);

    void extract_onion_decomposition(std::vector<int> &coreness, std::vector<int> &od_layer);

    std::string format_time(time_t _time);

    double time_since_epoch_in_seconds();

    void save_inferred_connection_probability();
    void save_inferred_connection_probability(int dim);

    void save_inferred_ensemble_characterization();
    void save_inferred_ensemble_characterization(int dim, bool random_positions);

    void save_inferred_theta_density();

    void save_inferred_coordinates();
    void save_inferred_coordinates(int dim);

    int get_root(int i, std::vector<int> &clust_id);
    void merge_clusters(std::vector<int> &size, std::vector<int> &clust_id);
    void check_connected_components();

    inline double calculateMu() const;

    std::pair<int, double> degree_of_random_vertex_and_prob_conn(int d1, double R);
    std::pair<int, double> degree_of_random_vertex_and_prob_conn(int d1, double R, int dim);

    double draw_random_angular_distance(int d1, int d2, double R, double p12);
    double draw_random_angular_distance(int d1, int d2, double R, double p12, int dim);

    inline double compute_radius(int dim, int N) const;

    inline double calculate_mu(int dim) const;

    std::vector<double> generate_random_d_vector(int dim, double radius);

    std::vector<double> generate_random_d_vector_with_first_coordinate(int dim, double angle, double radius);

    double compute_angle_d_vectors(const std::vector<double> &v1, const std::vector<double> &v2);

    double ks_stats(const std::map<int, double> &cdf1, const std::map<int, double> &cdf2);

    bool ks_test(const std::map<int, double> &cdf1, const std::map<int, double> &cdf2, double alpha=0.05);

    void normalize_and_rescale_vector(std::vector<double> &v, double radius);

    double euclidean_distance(const std::vector<double> &v1, const std::vector<double> &v2);

    void save_kappas_and_exit();

    void rebuild_adjacency_flat_list();

  private:

    // Reused scratch buffers to avoid repeated allocations in hot loops.
    std::vector<int> scratch_neighbors_int;
    std::vector<double> scratch_pair_prefactor;
    std::vector<double> scratch_mean_vector;
    std::vector<double> scratch_proposed_vector;

  public:

    struct EmbeddingStateSnapshot
    {
      std::vector<double> kappa;
      std::vector<double> theta;
      std::vector<std::vector<double>> d_positions;
    };

    EmbeddingStateSnapshot snapshot_state() const;

    embeddingSD_t() {};

    ~embeddingSD_t() {};

    void embed();
    void embed(int dim);
    void embed(std::string edgelist_filename) { EDGELIST_FILENAME = edgelist_filename; embed(); };
};

#include "dmercator/utils.hpp"
#include "dmercator/initialize.hpp"
#include "dmercator/infer_parameters.hpp"
#include "dmercator/infer_initial_positions.hpp"
#include "dmercator/refine_positions.hpp"
#include "dmercator/validation.hpp"

void embeddingSD_t::embed()
{
  embed(DIMENSION);
}

void embeddingSD_t::embed(int dim)
{
  if(dim < 1)
  {
    std::cerr << "ERROR: embedding dimension must be >= 1." << std::endl;
    std::terminate();
  }

  const bool is_s1 = (dim == 1);

  stage_timing_summary = {};
  {
    dmercator::core::timing::ScopedTimer total_timer(stage_timing_summary.total_time_ms);

    // Keep execution order aligned with legacy to preserve RNG and numeric behavior.
    time0 = time_since_epoch_in_seconds();
    time_started = std::time(nullptr);

    {
      dmercator::core::timing::ScopedTimer timer(stage_timing_summary.initialization_ms);
      initialize();
    }

    time1 = time_since_epoch_in_seconds();
    if(ONLY_KAPPAS)
    {
      if(REFINE_MODE)
      {
        const auto custom_beta = beta;
        load_already_inferred_parameters(dim);
        beta = custom_beta;
        mu = is_s1 ? calculateMu() : calculate_mu(dim);
        {
          dmercator::core::timing::ScopedTimer timer(stage_timing_summary.adjusting_kappas_ms);
          infer_kappas_given_beta_for_all_vertices(dim);
        }
      } else {
        {
          dmercator::core::timing::ScopedTimer timer(stage_timing_summary.parameter_inference_ms);
          infer_parameters(dim);
        }
        if(!is_s1)
        {
          dmercator::core::timing::ScopedTimer timer(stage_timing_summary.adjusting_kappas_ms);
          infer_kappas_given_beta_for_all_vertices(dim);
        }
      }
      save_kappas_and_exit();
    }

    if(REFINE_MODE)
    {
      load_already_inferred_parameters(dim);
    }
    if(!REFINE_MODE)
    {
      dmercator::core::timing::ScopedTimer timer(stage_timing_summary.parameter_inference_ms);
      infer_parameters(dim);
    }

    time2 = time_since_epoch_in_seconds();
    if(!REFINE_MODE)
    {
      dmercator::core::timing::ScopedTimer timer(stage_timing_summary.initial_positions_ms);
      infer_initial_positions(dim);
    }
    time3 = time_since_epoch_in_seconds();

    if(MAXIMIZATION_MODE)
    {
      dmercator::core::timing::ScopedTimer timer(stage_timing_summary.refining_positions_ms);
      refine_positions(dim);
    }
    time4 = time_since_epoch_in_seconds();
    if(KAPPA_POST_INFERENCE_MODE)
    {
      dmercator::core::timing::ScopedTimer timer(stage_timing_summary.adjusting_kappas_ms);
      infer_kappas_given_beta_for_all_vertices(dim);
    }
    time5 = time_since_epoch_in_seconds();

    if(is_s1)
    {
      time_ended = std::time(nullptr);
    }

    {
      dmercator::core::timing::ScopedTimer timer(stage_timing_summary.io_ms);
      save_inferred_coordinates(dim);

      if(VALIDATION_MODE)
      {
        save_inferred_connection_probability(dim);
        if(is_s1)
        {
          save_inferred_theta_density();
        }
      }
      time6 = time_since_epoch_in_seconds();
      if(CHARACTERIZATION_MODE)
      {
        save_inferred_ensemble_characterization(dim, false);
      }
      time7 = time_since_epoch_in_seconds();
    }
    time_ended = std::time(nullptr);
  }
  finalize();
}
#endif
