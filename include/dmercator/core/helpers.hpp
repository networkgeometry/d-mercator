#ifndef DMERCATOR_CORE_HELPERS_HPP
#define DMERCATOR_CORE_HELPERS_HPP

int embeddingSD_t::get_root(int i, std::vector<int> &clust_id)
{
  while(i != clust_id[i])
  {
    clust_id[i] = clust_id[clust_id[i]];
    i = clust_id[i];
  }
  return i;
}

void embeddingSD_t::merge_clusters(std::vector<int> &size, std::vector<int> &clust_id)
{

  int v1, v2, v3, v4;

  std::set<int>::iterator it, end;

  for(int i(0); i<nb_vertices; ++i)
  {

    it  = adjacency_list[i].begin();
    end = adjacency_list[i].end();
    for(; it!=end; ++it)
    {
      if(get_root(i, clust_id) != get_root(*it, clust_id))
      {

        v1 = i;
        v2 = *it;
        if(size[v2] > size[v1])
          std::swap(v1, v2);
        v3 = get_root(v1, clust_id);
        v4 = get_root(v2, clust_id);
        clust_id[v4] = v3;
        size[v3] += size[v4];
      }
    }
  }
}

void embeddingSD_t::check_connected_components()
{

  std::vector<double> Vertex2Prop(nb_vertices, -1);

  std::vector<int> connected_components_size;

  std::set< std::pair<int, int> > ordered_connected_components;

  std::vector<int> clust_id(nb_vertices);
  std::vector<int> clust_size(nb_vertices, 1);
  for(int v(0); v<nb_vertices; ++v)
  {
    clust_id[v] = v;
  }

  merge_clusters(clust_size, clust_id);
  clust_size.clear();

  int nb_conn_comp = 0;
  int comp_id;
  std::map<int, int> CompID;
  for(int v(0); v<nb_vertices; ++v)
  {
    comp_id = get_root(v, clust_id);
    if(CompID.find(comp_id) == CompID.end())
    {
      CompID[comp_id] = nb_conn_comp;
      connected_components_size.push_back(0);
      ++nb_conn_comp;
    }
    Vertex2Prop[v] = CompID[comp_id];
    connected_components_size[CompID[comp_id]] += 1;
  }

  for(int c(0); c<nb_conn_comp; ++c)
  {
    ordered_connected_components.insert( std::make_pair(connected_components_size[c], c) );
  }

  int lcc_id = (--ordered_connected_components.end())->second;
  int lcc_size = (--ordered_connected_components.end())->first;

  if(lcc_size != nb_vertices)
  {
    if(!QUIET_MODE) { std::clog << std::endl; }
    if(!QUIET_MODE) { std::clog << TAB << "- More than one component found!!" << std::endl; }
    if(!QUIET_MODE) { std::clog << TAB << "- " << lcc_size << "/" << nb_vertices << " vertices in the largest component." << std::endl; }
    std::cerr << std::endl;
    std::cerr << "More than one component found (" << lcc_size << "/" << nb_vertices << ") vertices in the largest component." << std::endl;

    std::string edgelist_rootname;
    size_t lastdot = EDGELIST_FILENAME.find_last_of(".");
    if(lastdot == std::string::npos)
    {
      edgelist_rootname = EDGELIST_FILENAME;
    }
    edgelist_rootname = EDGELIST_FILENAME.substr(0, lastdot);

    std::string edgelist_filename = edgelist_rootname + "_GC.edge";

    std::fstream edgelist_file(edgelist_filename.c_str(), std::fstream::out);
    if( !edgelist_file.is_open() )
    {
      std::cerr << "Could not open file: " << edgelist_filename << "." << std::endl;
      std::terminate();
    }

    std::set<int>::iterator it, end;
    width_names = 14;
    for(int v1(0), v2, c1, c2; v1<nb_vertices; ++v1)
    {
      c1 = Vertex2Prop[v1];
      if(c1 == lcc_id)
      {
        it  = adjacency_list[v1].begin();
        end = adjacency_list[v1].end();
        for(; it!=end; ++it)
        {
          v2 = *it;
          c2 = Vertex2Prop[v2];
          if(c2 == lcc_id)
          {
            if(v1 < v2)
            {
              edgelist_file << std::setw(width_names) << Num2Name[v1] << " ";
              edgelist_file << std::setw(width_names) << Num2Name[v2] << " ";
              edgelist_file << std::endl;
            }
          }
        }
      }
    }

    edgelist_file.close();

    if(!QUIET_MODE) { std::clog << TAB << "- Edges belonging to the largest component saved to " << edgelist_rootname + "_GC.edge." << std::endl; }
    if(!QUIET_MODE) { std::clog << TAB << "- Please rerun the program using this new edgelist." << std::endl; }
    if(!QUIET_MODE) { std::clog << std::endl; }

    if(QUIET_MODE)  { std::clog << std::endl; }
    std::cerr << "Edges belonging to the largest component saved to " << edgelist_rootname + "_GC.edge. Please rerun the program using this new edgelist." << std::endl;
    std::cerr << std::endl;

    std::exit(12);
  }
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

#endif
