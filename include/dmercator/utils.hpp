#ifndef DMERCATOR_UTILS_HPP
#define DMERCATOR_UTILS_HPP

std::string embeddingSD_t::format_time(time_t _time)
{

  struct tm *aTime = gmtime(& _time);
  int year    = aTime->tm_year + 1900;
  int month   = aTime->tm_mon + 1;
  int day     = aTime->tm_mday;
  int hours   = aTime->tm_hour;
  int minutes = aTime->tm_min;

  std::string the_time = std::to_string(year) + "/";
  if(month < 10)
    the_time += "0";
  the_time += std::to_string(month) + "/";
  if(day < 10)
    the_time += "0";
  the_time += std::to_string(day) + " " + std::to_string(hours) + ":";
  if(minutes < 10)
    the_time += "0";
  the_time += std::to_string(minutes) + " UTC";

  return the_time;
}
inline double embeddingSD_t::calculateMu() const
{
  return beta * std::sin(PI / beta) / (2.0 * PI * average_degree);
}

inline double embeddingSD_t::compute_radius(int dim, int N) const
{
  const auto inside = N / (2 * std::pow(PI, (dim + 1) / 2.0)) * std::tgamma((dim + 1) / 2.0);
  return std::pow(inside, 1.0 / dim);
}

inline double embeddingSD_t::calculate_mu(int dim) const
{
  const auto top = beta * std::tgamma(dim / 2.0) * std::sin(dim * PI / beta);
  const auto bottom = average_degree * 2 * std::pow(PI, 1 + dim / 2.0);
  return top / bottom;
}

std::vector<double> embeddingSD_t::generate_random_d_vector(int dim, double radius) {
  std::vector<double> positions;
  positions.resize(dim + 1);
  double norm{0};
  for (auto &pos : positions) {
    pos = normal_01(engine);
    norm += pos * pos;
  }
  norm /= std::sqrt(norm);

  for (auto &pos: positions)
    pos = pos / norm * radius;

  return positions;
}

std::vector<double> embeddingSD_t::generate_random_d_vector_with_first_coordinate(int dim, double angle, double radius) {

  std::vector<double> positions;
  positions.resize(dim + 1);
  double norm{0};
  for (int i = 1; i < dim + 1; i++) {
    positions[i] = normal_01(engine);
    norm += positions[i] * positions[i];
  }
  const auto firstNorm = norm / std::sqrt(norm);
  positions[0] = 1.0 / std::tan(angle) * firstNorm;
  normalize_and_rescale_vector(positions, radius);
  return positions;
}

double embeddingSD_t::compute_angle_d_vectors(const std::vector<double> &v1, const std::vector<double> &v2) {
  double angle{0}, norm1{0}, norm2{0};
  for (int i = 0; i < v1.size(); ++i) {
    angle += v1[i] * v2[i];
    norm1 += v1[i] * v1[i];
    norm2 += v2[i] * v2[i];
  }
  norm1 /= sqrt(norm1);
  norm2 /= sqrt(norm2);

  const auto result = angle / (norm1 * norm2);
  if (std::fabs(result - 1) < NUMERICAL_ZERO)
    return 0;
  else
    return std::acos(result);
}

double embeddingSD_t::ks_stats(const std::map<int, double> &inf_ccdf,
                               const std::map<int, double> &obs_ccdf) {

  double statistic = 0;
  for (auto inf_vals = inf_ccdf.begin(); inf_vals != inf_ccdf.end(); ++inf_vals) {
    for (auto obs_vals = obs_ccdf.begin(); obs_vals != obs_ccdf.end(); ++obs_vals) {
      const int d_inf = inf_vals->first;
      const int d_obs = obs_vals->first;
      if (d_inf == d_obs) {
        const double diff = std::fabs(inf_vals->second - obs_vals->second);
        if (diff > statistic)
          statistic = diff;
      }
    }
  }
  return statistic;
}

bool embeddingSD_t::ks_test(const std::map<int, double> &inf_ccdf,
                            const std::map<int, double> &obs_ccdf,
                            double alpha) {
  const double n = inf_ccdf.size();
  const double m = obs_ccdf.size();
  const double statistic = ks_stats(inf_ccdf, obs_ccdf);
  const double c_alpha = std::sqrt(-std::log(alpha / 2) * 0.5);
  const double critical_value = c_alpha * std::sqrt((n + m) / (n * m));
  if (!QUIET_MODE) {
    std::clog << "Test statistic: " << statistic << ", critical_value: " << critical_value << std::endl;
  }
  return statistic > critical_value;
}

void embeddingSD_t::normalize_and_rescale_vector(std::vector<double> &v, double radius) {
  int dim = v.size() - 1;
  double norm=0;
  for (int i=0; i<dim + 1; ++i)
    norm += v[i] * v[i];

  norm = std::sqrt(norm);
  for (int i=0; i<dim + 1; ++i)
    v[i] /= norm;

  for (int i=0; i<dim + 1; ++i)
    v[i] *= radius;
}

double embeddingSD_t::euclidean_distance(const std::vector<double> &v1, const std::vector<double> &v2) {
  double distance=0;
  for (int i=0; i<v1.size(); ++i)
    distance += (v1[i] - v2[i]) * (v1[i] - v2[i]);
  return std::sqrt(distance);
}

embeddingSD_t::EmbeddingStateSnapshot embeddingSD_t::snapshot_state() const
{
  return EmbeddingStateSnapshot{kappa, theta, d_positions};
}

void embeddingSD_t::save_kappas_and_exit() {

  auto path = std::filesystem::path(EDGELIST_FILENAME);
  auto dirname = path.parent_path().string();
  auto filename = path.filename().string();
  std::string kappas_file = filename.substr(0, filename.find("."));
  kappas_file.append(".kappas");
  dirname.append("/");
  dirname.append(kappas_file.c_str());
  std::ofstream outfile(dirname);
  std::for_each(kappa.begin(), kappa.end(), [&outfile](auto x){
    outfile << x << '\n';
  });
  outfile.close();
  std::exit(13);
}

void embeddingSD_t::rebuild_adjacency_flat_list()
{
  adjacency_flat_list.clear();
  adjacency_flat_list.resize(nb_vertices);
  for(int v = 0; v < nb_vertices; ++v)
  {
    auto &flat = adjacency_flat_list[v];
    const auto &neighbors = adjacency_list[v];
    flat.clear();
    flat.reserve(neighbors.size());
    flat.insert(flat.end(), neighbors.begin(), neighbors.end());
  }
}

void embeddingSD_t::rebuild_adjacency_csr()
{
  adjacency_row_offsets.clear();
  adjacency_col_indices.clear();

  adjacency_row_offsets.resize(static_cast<std::size_t>(nb_vertices) + 1, 0);
  for(int v = 0; v < nb_vertices; ++v)
  {
    adjacency_row_offsets[static_cast<std::size_t>(v + 1)] =
      adjacency_row_offsets[static_cast<std::size_t>(v)] + static_cast<int>(adjacency_flat_list[static_cast<std::size_t>(v)].size());
  }

  adjacency_col_indices.resize(static_cast<std::size_t>(adjacency_row_offsets.back()), 0);
  for(int v = 0; v < nb_vertices; ++v)
  {
    const auto &neighbors = adjacency_flat_list[static_cast<std::size_t>(v)];
    const int offset = adjacency_row_offsets[static_cast<std::size_t>(v)];
    std::copy(neighbors.begin(),
              neighbors.end(),
              adjacency_col_indices.begin() + static_cast<std::ptrdiff_t>(offset));
  }
}

#endif
