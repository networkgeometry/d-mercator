#ifndef DMERCATOR_VALIDATION_HPP
#define DMERCATOR_VALIDATION_HPP

void embeddingSD_t::analyze_simulated_adjacency_list()
{

  simulated_degree.clear();
  simulated_degree.resize(nb_vertices);
  simulated_sum_degree_of_neighbors.clear();
  simulated_sum_degree_of_neighbors.resize(nb_vertices, 0);
  simulated_nb_triangles.clear();
  simulated_nb_triangles.resize(nb_vertices, 0);
  simulated_stat_degree.clear();
  simulated_stat_sum_degree_neighbors.clear();
  simulated_stat_avg_degree_neighbors.clear();
  simulated_stat_nb_triangles.clear();
  simulated_stat_clustering.clear();

  std::set<int>::iterator it, end;
  for(int v1(0), d1; v1<nb_vertices; ++v1)
  {
    d1 = simulated_adjacency_list[v1].size();
    simulated_degree[v1] = d1;
    it = simulated_adjacency_list[v1].begin();
    end = simulated_adjacency_list[v1].end();
    for(; it!=end; ++it)
    {
      simulated_sum_degree_of_neighbors[*it] += d1;
    }
  }

  double nb_triangles, tmp_val;

  std::vector<int> intersection;

  std::set<int> neighbors_v2;

  std::set<int>::iterator it1, end1, it2, end2;
  std::map<int, std::vector<int> >::iterator it3, end3;
  std::vector<int>::iterator it4;

  for(int v1(0), d1; v1<nb_vertices; ++v1)
  {

    nb_triangles = 0;

    d1 = simulated_degree[v1];
    if( d1 > 1 )
    {

      it1 = simulated_adjacency_list[v1].begin();
      end1 = simulated_adjacency_list[v1].end();
      for(; it1!=end1; ++it1)
      {

        if( simulated_degree[*it1] > 1 )
        {

          it2 = simulated_adjacency_list[*it1].begin();
          end2 = simulated_adjacency_list[*it1].end();
          neighbors_v2.clear();
          for(; it2!=end2; ++it2)
          {
            if(*it1 < *it2)
            {
              neighbors_v2.insert(*it2);
            }
          }

          if(neighbors_v2.size() > 0)
          {
            intersection.clear();
            intersection.resize(std::min(simulated_adjacency_list[v1].size(), neighbors_v2.size()));
            it4 = std::set_intersection(simulated_adjacency_list[v1].begin(), simulated_adjacency_list[v1].end(),
                                        neighbors_v2.begin(), neighbors_v2.end(), intersection.begin());
            intersection.resize(it4-intersection.begin());
            nb_triangles += intersection.size();
          }
        }
      }

      simulated_nb_triangles[v1] = nb_triangles;

    }
  }

  for(int v1(0), d1; v1<nb_vertices; ++v1)
  {

    d1 = simulated_degree[v1];

    if( simulated_stat_degree.find(d1) == simulated_stat_degree.end() )
    {
      simulated_stat_degree[d1] = 0;
      simulated_stat_sum_degree_neighbors[d1] = 0;
      simulated_stat_avg_degree_neighbors[d1] = 0;
      simulated_stat_nb_triangles[d1] = 0;
      simulated_stat_clustering[d1] = 0;
    }

    simulated_stat_degree[d1] += 1;
    if(d1 > 0)
    {
      simulated_stat_sum_degree_neighbors[d1] += simulated_sum_degree_of_neighbors[v1];
      simulated_stat_avg_degree_neighbors[d1] += simulated_sum_degree_of_neighbors[v1] / d1;
    }
    if(d1 > 1)
    {
      simulated_stat_nb_triangles[d1] += simulated_nb_triangles[v1];
      simulated_stat_clustering[d1] += 2 * simulated_nb_triangles[v1] / d1 / (d1 - 1);
    }
  }
}
void embeddingSD_t::compute_inferred_ensemble_expected_degrees(int dim, double radius)
{
#if defined(DMERCATOR_USE_CUDA)
  if(CUDA_MODE)
  {
    bool computed_on_gpu = false;
    if(dim == 1)
    {
      computed_on_gpu = dmercator::gpu::compute_inferred_expected_degrees_s1(beta, mu, theta, kappa, inferred_ensemble_expected_degree);
    }
    else
    {
      computed_on_gpu = dmercator::gpu::compute_inferred_expected_degrees_sd(dim,
                                                                              beta,
                                                                              mu,
                                                                              radius,
                                                                              NUMERICAL_ZERO,
                                                                              d_positions,
                                                                              kappa,
                                                                              inferred_ensemble_expected_degree);
    }
    if(computed_on_gpu)
    {
      return;
    }
    CUDA_MODE = false;
    CUDA_REFINEMENT_ACTIVE = false;
    if(!QUIET_MODE)
    {
      std::clog << TAB << "WARNING: CUDA expected-degree computation failed; switching to CPU. "
                << dmercator::gpu::last_error() << std::endl;
    }
  }
#endif

  if(dim == 1)
  {

    double kappa1, theta1, dtheta, prob;
    const double prefactor = nb_vertices / (2 * PI * mu);
    inferred_ensemble_expected_degree.clear();
    inferred_ensemble_expected_degree.resize(nb_vertices, 0);
    for(int v1(0); v1<nb_vertices; ++v1)
    {
      kappa1 = kappa[v1];
      theta1 = theta[v1];
      for(int v2(v1 + 1); v2<nb_vertices; ++v2)
      {
        dtheta = PI - std::fabs(PI - std::fabs(theta1 - theta[v2]));
        prob = 1 / (1 + std::pow((prefactor * dtheta) / (kappa1 * kappa[v2]), beta));
        inferred_ensemble_expected_degree[v1] += prob;
        inferred_ensemble_expected_degree[v2] += prob;
      }
    }
    return;
  }

  inferred_ensemble_expected_degree.clear();
  inferred_ensemble_expected_degree.resize(nb_vertices, 0);
  const double inv_dim = 1.0 / static_cast<double>(dim);
  for(int v1=0; v1<nb_vertices; ++v1) {
    const auto &pos1 = d_positions[v1];
    const double kappa1 = kappa[v1];
    for(int v2(v1 + 1); v2<nb_vertices; ++v2) {
      const auto dtheta = compute_angle_d_vectors(pos1, d_positions[v2]);
      const auto chi = radius * dtheta / std::pow(mu * kappa1 * kappa[v2], inv_dim);
      const auto prob = 1 / (1 + std::pow(chi, beta));
      inferred_ensemble_expected_degree[v1] += prob;
      inferred_ensemble_expected_degree[v2] += prob;
    }
  }
}

void embeddingSD_t::compute_inferred_ensemble_expected_degrees()
{
  compute_inferred_ensemble_expected_degrees(1, nb_vertices / (2 * PI));
}
void embeddingSD_t::generate_simulated_adjacency_list(int dim, bool random_positions=true)
{
  if(dim == 1)
  {

    simulated_adjacency_list.clear();
    simulated_adjacency_list.resize(nb_vertices);

    double kappa1, theta1, dtheta, prob;
    double prefactor = nb_vertices / (2 * PI * mu);
    for(int v1(0); v1<nb_vertices; ++v1)
    {
      kappa1 = kappa[v1];
      theta1 = theta[v1];
      for(int v2(v1 + 1); v2<nb_vertices; ++v2)
      {
        dtheta = PI - std::fabs(PI - std::fabs(theta1 - theta[v2]));
        prob = 1 / (1 + std::pow((prefactor * dtheta) / (kappa1 * kappa[v2]), beta));
        if(uniform_01(engine) < prob)
        {
          simulated_adjacency_list[v1].insert(v2);
          simulated_adjacency_list[v2].insert(v1);
        }
      }
    }
    return;
  }

  const auto radius = compute_radius(dim, nb_vertices);
  if (random_positions) {
    d_positions.clear();
    d_positions.resize(nb_vertices);
    for (int i = 0; i < nb_vertices; ++i)
      d_positions[i] = generate_random_d_vector(dim, radius);
  }
  mu = calculate_mu(dim);

  simulated_adjacency_list.clear();
  simulated_adjacency_list.resize(nb_vertices);
  const double inv_dim = 1.0 / static_cast<double>(dim);

  for (int v1=0; v1 < nb_vertices; ++v1) {
    const auto &positions1 = d_positions[v1];
    const double kappa1 = kappa[v1];
    for (int v2 = v1 + 1; v2 < nb_vertices; ++v2) {
      const auto dtheta = compute_angle_d_vectors(positions1, d_positions[v2]);
      const auto chi = radius * dtheta / std::pow(mu * kappa1 * kappa[v2], inv_dim);
      const auto prob = 1 / (1 + std::pow(chi, beta));
      if (uniform_01(engine) < prob) {
        simulated_adjacency_list[v1].insert(v2);
        simulated_adjacency_list[v2].insert(v1);
      }
    }
  }
}

void embeddingSD_t::generate_simulated_adjacency_list()
{
  generate_simulated_adjacency_list(1, true);
}
void embeddingSD_t::save_inferred_connection_probability(int dim)
{
  if(dim == 1)
  {

    std::map<double, int> bins;
    std::map<double, int>::iterator it;
    int bound = 20;
    int cnt = 0;
    double dt = 0.05;
    for(double t(-bound), tt(bound + 0.000001); t<tt; t+=dt, ++cnt)
    {
      bins[std::pow(10, t)] = cnt;
    }

    std::vector<double> n(bins.size(), 0);
    std::vector<double> p(bins.size(), 0);
    std::vector<double> x(bins.size(), 0);

    double k1;
    double t1;
    double da;
    double dist;
    for(int v1(0), i; v1<nb_vertices; ++v1)
    {
      k1 = kappa[v1];
      t1 = theta[v1];
      const auto &neighbors_v1 = adjacency_flat_list[v1];
      auto neigh_it = neighbors_v1.begin();
      const auto neigh_end = neighbors_v1.end();
      while(neigh_it != neigh_end && *neigh_it <= v1)
      {
        ++neigh_it;
      }
      for(int v2(v1 + 1); v2<nb_vertices; ++v2)
      {
        da = PI - std::fabs( PI - std::fabs(t1 - theta[v2]) );
        dist = (nb_vertices * da) / (2 * PI * mu * k1 * kappa[v2]);
        i = bins.lower_bound(dist)->second;
        n[i] += 1;
        x[i] += dist;
        bool connected = false;
        if(!OPTIMIZED_PERF_MODE)
        {
          connected = (adjacency_list[v1].find(v2) != adjacency_list[v1].end());
        }
        else
        {
          while(neigh_it != neigh_end && *neigh_it < v2)
          {
            ++neigh_it;
          }
          connected = (neigh_it != neigh_end && *neigh_it == v2);
        }
        if(connected)
        {
          p[i] += 1;
        }
      }
    }

    std::string pconn_filename = ROOTNAME_OUTPUT + ".inf_pconn";
    std::fstream pconn_file(pconn_filename.c_str(), std::fstream::out);
    if( !pconn_file.is_open() )
    {
      std::cerr << "Could not open file: " << pconn_filename << "." << std::endl;
      std::terminate();
    }
    pconn_file << "#";
    pconn_file << std::setw(width_values - 1) << "RescaledDist" << " ";
    pconn_file << std::setw(width_values)     << "InfConnProb"  << " ";
    pconn_file << std::setw(width_values)     << "ThConnProb"   << " ";
    pconn_file << std::endl;
    for(int i(0), ii(n.size()); i<ii; ++i)
    {
      if(n[i] > 0)
      {
        pconn_file << std::setw(width_values) << x[i] / n[i]           << " ";
        pconn_file << std::setw(width_values) << p[i] / n[i]           << " ";
        pconn_file << std::setw(width_values) << 1 / (1 + std::pow(x[i] / n[i], beta) ) << " ";
        pconn_file << std::endl;
      }
    }

    pconn_file.close();
    if(!QUIET_MODE) { std::clog << std::endl; }
    if(!QUIET_MODE) { std::clog << TAB << "=> Inferred connection probability saved to " << ROOTNAME_OUTPUT + ".inf_pconn" << std::endl; }
    return;
  }

  std::map<double, int> bins;
  std::map<double, int>::iterator it;
  int bound = 20;
  int cnt = 0;
  double dt = 0.05;
  const auto radius = compute_radius(dim, nb_vertices);
  for(double t(-bound), tt(bound + 0.000001); t<tt; t+=dt, ++cnt)
  {
    bins[std::pow(10, t)] = cnt;
  }

  std::vector<double> n(bins.size(), 0);
  std::vector<double> p(bins.size(), 0);
  std::vector<double> x(bins.size(), 0);

  double k1;
  double t1;
  double da;
  double dist;
  for(int v1(0), i; v1<nb_vertices; ++v1)
  {
    k1 = kappa[v1];
    const auto &pos1 = d_positions[v1];
    const auto &neighbors_v1 = adjacency_flat_list[v1];
    auto neigh_it = neighbors_v1.begin();
    const auto neigh_end = neighbors_v1.end();
    while(neigh_it != neigh_end && *neigh_it <= v1)
    {
      ++neigh_it;
    }
    for(int v2(v1 + 1); v2<nb_vertices; ++v2)
    {
      da = compute_angle_d_vectors(pos1, d_positions[v2]);
      dist = (radius * da) / std::pow(mu * k1 * kappa[v2], 1.0 / dim);
      i = bins.lower_bound(dist)->second;
      n[i] += 1;
      x[i] += dist;
      bool connected = false;
      if(!OPTIMIZED_PERF_MODE)
      {
        connected = (adjacency_list[v1].find(v2) != adjacency_list[v1].end());
      }
      else
      {
        while(neigh_it != neigh_end && *neigh_it < v2)
        {
          ++neigh_it;
        }
        connected = (neigh_it != neigh_end && *neigh_it == v2);
      }
      if(connected)
      {
        p[i] += 1;
      }
    }
  }

  std::string pconn_filename = ROOTNAME_OUTPUT + ".inf_pconn";
  std::fstream pconn_file(pconn_filename.c_str(), std::fstream::out);
  if( !pconn_file.is_open() )
  {
    std::cerr << "Could not open file: " << pconn_filename << "." << std::endl;
    std::terminate();
  }
  pconn_file << "#";
  pconn_file << std::setw(width_values - 1) << "RescaledDist" << " ";
  pconn_file << std::setw(width_values)     << "InfConnProb"  << " ";
  pconn_file << std::setw(width_values)     << "ThConnProb"   << " ";
  pconn_file << std::endl;
  for(int i(0), ii(n.size()); i<ii; ++i)
  {
    if(n[i] > 0)
    {
      pconn_file << std::setw(width_values) << x[i] / n[i]           << " ";
      pconn_file << std::setw(width_values) << p[i] / n[i]           << " ";
      pconn_file << std::setw(width_values) << 1 / (1 + std::pow(x[i] / n[i], beta) ) << " ";
      pconn_file << std::endl;
    }
  }

  pconn_file.close();
  if(!QUIET_MODE) { std::clog << std::endl; }
  if(!QUIET_MODE) { std::clog << TAB << "=> Inferred connection probability saved to " << ROOTNAME_OUTPUT + ".inf_pconn" << std::endl; }
}

void embeddingSD_t::save_inferred_connection_probability()
{
  save_inferred_connection_probability(1);
}

void embeddingSD_t::save_inferred_coordinates(int dim)
{
  // Preserve the original S1 output schema for backward compatibility.
  if(dim == 1)
  {

    double kappa_min = *std::min_element(kappa.begin(), kappa.end());
    double kappa_max = *std::max_element(kappa.begin(), kappa.end());

    double hyp_radius = 2 * std::log( nb_vertices / (PI * mu * kappa_min * kappa_min) );
    double min_radial_position = hyp_radius - 2 * std::log( kappa_min / kappa_max );
    bool warning = false;
    if(min_radial_position < 0)
    {
      hyp_radius += std::fabs(min_radial_position);
      warning = true;
    }

    std::string coordinates_filename = ROOTNAME_OUTPUT + ".inf_coord";

    std::fstream coordinates_file(coordinates_filename.c_str(), std::fstream::out);
    if( !coordinates_file.is_open() )
    {
      std::cerr << "Could not open file: " << coordinates_filename << "." << std::endl;
      std::terminate();
    }

    coordinates_file << "# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=" << std::endl;
    coordinates_file << "# Embedding started at:  " << format_time(time_started)    << std::endl;
    coordinates_file << "# Ended at:              " << format_time(time_ended)      << std::endl;
    coordinates_file << "# Elapsed CPU time:      " << time5 - time0 << " seconds"  << std::endl;
    coordinates_file << "# Edgelist file:         " << EDGELIST_FILENAME            << std::endl;
    coordinates_file << "#"                                                         << std::endl;
    coordinates_file << "# Parameters"                                              << std::endl;
    coordinates_file << "#   - nb. vertices:      " << nb_vertices                  << std::endl;
    coordinates_file << "#   - beta:              " << beta                         << std::endl;
    coordinates_file << "#   - mu:                " << mu                           << std::endl;
    coordinates_file << "#   - radius_S1:         " << nb_vertices / (2 * PI)       << std::endl;
    coordinates_file << "#   - radius_H2:         " << hyp_radius                   << std::endl;
    coordinates_file << "#   - kappa_min:         " << kappa_min                    << std::endl;
    coordinates_file << "# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=" << std::endl;
    coordinates_file << "#";
    coordinates_file << std::setw(width_names - 1) << "Vertex"          << " ";
    coordinates_file << std::setw(width_values)     << "Inf.Kappa"       << " ";
    coordinates_file << std::setw(width_values)     << "Inf.Theta"       << " ";
    coordinates_file << std::setw(width_values)     << "Inf.Hyp.Rad."    << " ";
    coordinates_file << std::endl;

    struct compare
    {
      bool operator()(const std::pair<std::string, int>& lhs, const std::pair<std::string, int>& rhs) const
      {
        if(lhs.first.size() == rhs.first.size())
        {
          if(lhs.first == rhs.first)
          {
            return lhs.second < rhs.second;
          }
          else
          {
            return lhs.first < rhs.first;
          }
        }
        else
        {
          return lhs.first.size() < rhs.first.size();
        }
      }
    };

    std::set< std::pair<std::string, int>, compare > ordered_names;
    for(int v(0); v<nb_vertices; ++v)
    {
      ordered_names.insert(std::make_pair(Num2Name[v], v));
    }
    auto it  = ordered_names.begin();
    auto end = ordered_names.end();
    for(int v; it!=end; ++it)
    {
      v = it->second;
      coordinates_file << std::setw(width_names) << it->first                                                      << " ";
      coordinates_file << std::setw(width_values) << kappa[v]                                                       << " ";
      coordinates_file << std::setw(width_values) << theta[v]                                                       << " ";
      coordinates_file << std::setw(width_values) << hyp_radius - 2 * std::log( kappa[v] / kappa_min )              << " ";

      coordinates_file << std::endl;
    }
    coordinates_file << "#"                                                                                                          << std::endl;
    coordinates_file << "# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~="        << std::endl;
    coordinates_file << "# Internal parameters and options"                                                                          << std::endl;
    coordinates_file << "# " << TAB << "ALREADY_INFERRED_PARAMETERS_FILENAME   " << ALREADY_INFERRED_PARAMETERS_FILENAME             << std::endl;
    coordinates_file << "# " << TAB << "BETA_ABS_MAX                           " << BETA_ABS_MAX                                     << std::endl;
    coordinates_file << "# " << TAB << "BETA_ABS_MIN                           " << BETA_ABS_MIN                                     << std::endl;
    coordinates_file << "# " << TAB << "CHARACTERIZATION_MODE                  " << (CHARACTERIZATION_MODE       ? "true" : "false") << std::endl;
    coordinates_file << "# " << TAB << "CHARACTERIZATION_NB_GRAPHS             " << CHARACTERIZATION_NB_GRAPHS                       << std::endl;
    coordinates_file << "# " << TAB << "CLEAN_RAW_OUTPUT_MODE                  " << (CLEAN_RAW_OUTPUT_MODE       ? "true" : "false") << std::endl;

    coordinates_file << "# " << TAB << "CUSTOM_BETA                            " << (CUSTOM_BETA                 ? "true" : "false") << std::endl;
    coordinates_file << "# " << TAB << "CUSTOM_INFERRED_COORDINATES            " << (CUSTOM_INFERRED_COORDINATES ? "true" : "false") << std::endl;
    coordinates_file << "# " << TAB << "CUSTOM_OUTPUT_ROOTNAME_MODE            " << (CUSTOM_OUTPUT_ROOTNAME_MODE ? "true" : "false") << std::endl;
    coordinates_file << "# " << TAB << "CUSTOM_SEED                            " << (CUSTOM_SEED                 ? "true" : "false") << std::endl;
    coordinates_file << "# " << TAB << "EDGELIST_FILENAME:                     " << EDGELIST_FILENAME                                << std::endl;
    coordinates_file << "# " << TAB << "EXP_CLUST_NB_INTEGRATION_MC_STEPS      " << EXP_CLUST_NB_INTEGRATION_MC_STEPS                << std::endl;
    coordinates_file << "# " << TAB << "EXP_DIST_NB_INTEGRATION_STEPS          " << EXP_DIST_NB_INTEGRATION_STEPS                    << std::endl;
    coordinates_file << "# " << TAB << "KAPPA_MAX_NB_ITER_CONV                 " << KAPPA_MAX_NB_ITER_CONV                           << std::endl;
    coordinates_file << "# " << TAB << "KAPPA_POST_INFERENCE_MODE              " << (KAPPA_POST_INFERENCE_MODE   ? "true" : "false") << std::endl;

    coordinates_file << "# " << TAB << "MAXIMIZATION_MODE                      " << (MAXIMIZATION_MODE           ? "true" : "false") << std::endl;

    coordinates_file << "# " << TAB << "MIN_NB_ANGLES_TO_TRY                   " << MIN_NB_ANGLES_TO_TRY                             << std::endl;
    coordinates_file << "# " << TAB << "NUMERICAL_CONVERGENCE_THRESHOLD_1      " << NUMERICAL_CONVERGENCE_THRESHOLD_1                << std::endl;
    coordinates_file << "# " << TAB << "NUMERICAL_CONVERGENCE_THRESHOLD_2      " << NUMERICAL_CONVERGENCE_THRESHOLD_2                << std::endl;
    coordinates_file << "# " << TAB << "NUMERICAL_CONVERGENCE_THRESHOLD_3      " << NUMERICAL_CONVERGENCE_THRESHOLD_3                << std::endl;
    coordinates_file << "# " << TAB << "NUMERICAL_ZERO                         " << NUMERICAL_ZERO                                   << std::endl;
    coordinates_file << "# " << TAB << "QUIET_MODE                             " << (QUIET_MODE                  ? "true" : "false") << std::endl;
    coordinates_file << "# " << TAB << "REFINE_MODE                            " << (REFINE_MODE                 ? "true" : "false") << std::endl;

    coordinates_file << "# " << TAB << "ROOTNAME_OUTPUT:                       " << ROOTNAME_OUTPUT                                  << std::endl;
    coordinates_file << "# " << TAB << "SEED                                   " << SEED                                             << std::endl;
    coordinates_file << "# " << TAB << "VALIDATION_MODE                        " << (VALIDATION_MODE             ? "true" : "false") << std::endl;
    coordinates_file << "# " << TAB << "VERBOSE_MODE                           " << (VERBOSE_MODE                ? "true" : "false") << std::endl;
    coordinates_file << "# " << TAB << "VERSION                                " << VERSION                                          << std::endl;
    coordinates_file << "# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~="        << std::endl;

    coordinates_file.close();

    if(!QUIET_MODE) { std::clog << std::endl; }
    if(!QUIET_MODE) { std::clog << TAB << "=> Inferred coordinates saved to " << ROOTNAME_OUTPUT + ".inf_coord" << std::endl; }

    if(CLEAN_RAW_OUTPUT_MODE)
    {

      coordinates_filename = ROOTNAME_OUTPUT + ".inf_coord_raw";

      coordinates_file.open(coordinates_filename.c_str(), std::fstream::out);
      if( !coordinates_file.is_open() )
      {
        std::cerr << "Could not open file: " << coordinates_filename << "." << std::endl;
        std::terminate();
      }

      it  = ordered_names.begin();
      end = ordered_names.end();
      for(int v; it!=end; ++it)
      {
        v = it->second;

        coordinates_file << kappa[v]                                          << " ";
        coordinates_file << theta[v]                                          << " ";
        coordinates_file << hyp_radius - 2 * std::log( kappa[v] / kappa_min ) << " ";
        coordinates_file << std::endl;
      }

      coordinates_file.close();

      if(!QUIET_MODE) { std::clog << std::endl; }
      if(!QUIET_MODE) { std::clog << TAB << "=> Raw inferred coordinates also saved to " << ROOTNAME_OUTPUT + ".inf_coord_raw" << std::endl; }
    }

    if(warning)
    {
      if(!QUIET_MODE) { std::clog << "WARNING: Hyperbolic radius has been adjusted to account for negative radial positions." << std::endl; }
    }
    return;
  }

  const auto R = compute_radius(dim, nb_vertices);

  double kappa_min = *std::min_element(kappa.begin(), kappa.end());
  double kappa_max = *std::max_element(kappa.begin(), kappa.end());

  double hyp_radius = 2 * std::log(2 * R / std::pow(mu * kappa_min * kappa_min, 1.0 / dim));
  double min_radial_position = hyp_radius - (2.0 / dim) * std::log(kappa_max / kappa_min);
  bool warning = false;
  if(min_radial_position < 0)
  {
    hyp_radius += std::fabs(min_radial_position);
    warning = true;
  }

  std::string coordinates_filename = ROOTNAME_OUTPUT + ".inf_coord";

  std::fstream coordinates_file(coordinates_filename.c_str(), std::fstream::out);
  if( !coordinates_file.is_open() )
  {
    std::cerr << "Could not open file: " << coordinates_filename << "." << std::endl;
    std::terminate();
  }

  coordinates_file << "# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=" << std::endl;
  coordinates_file << "# Embedding started at:  " << format_time(time_started)    << std::endl;
  coordinates_file << "# Ended at:              " << format_time(time_ended)      << std::endl;
  coordinates_file << "# Elapsed CPU time:      " << time5 - time0 << " seconds"  << std::endl;
  coordinates_file << "# Edgelist file:         " << EDGELIST_FILENAME            << std::endl;
  coordinates_file << "#"                                                         << std::endl;
  coordinates_file << "# Parameters"                                              << std::endl;
  coordinates_file << "#   - nb. vertices:      " << nb_vertices                  << std::endl;
  coordinates_file << "#   - beta:              " << beta                         << std::endl;
  coordinates_file << "#   - mu:                " << mu                           << std::endl;
  coordinates_file << "#   - radius_S^D:        " << R                            << std::endl;
  coordinates_file << "#   - radius_H^D+1       " << hyp_radius                   << std::endl;
  coordinates_file << "#   - kappa_min:         " << kappa_min                    << std::endl;
  coordinates_file << "# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=" << std::endl;
  coordinates_file << "#";
  coordinates_file << std::setw(width_names - 1)  << "Vertex"          << " ";
  coordinates_file << std::setw(width_values)     << "Inf.Kappa"       << " ";
  coordinates_file << std::setw(width_values)     << "Inf.Hyp.Rad"       << " ";
  for (int i=0; i<dim+1; ++i) {
    coordinates_file << std::setw(width_values)     << "Inf.Pos." << i+1   << " ";
  }

  coordinates_file << std::endl;

  struct compare
  {
    bool operator()(const std::pair<std::string, int>& lhs, const std::pair<std::string, int>& rhs) const
    {
      if(lhs.first.size() == rhs.first.size())
      {
        if(lhs.first == rhs.first)
        {
          return lhs.second < rhs.second;
        }
        else
        {
          return lhs.first < rhs.first;
        }
      }
      else
      {
        return lhs.first.size() < rhs.first.size();
      }
    }
  };

  std::set< std::pair<std::string, int>, compare > ordered_names;
  for(int v(0); v<nb_vertices; ++v)
  {
    ordered_names.insert(std::make_pair(Num2Name[v], v));
  }
  auto it  = ordered_names.begin();
  auto end = ordered_names.end();
  for(int v; it!=end; ++it)
  {
    v = it->second;
    coordinates_file << std::setw(width_names) << it->first                                                      << " ";
    coordinates_file << std::setw(width_values) << kappa[v]                                                       << " ";
    coordinates_file << std::setw(width_values) << hyp_radius - (2.0 / dim) * std::log(kappa[v] / kappa_min) << " ";
    for (int i=0; i<dim+1; ++i) {
      coordinates_file << std::setw(width_values) << d_positions[v][i]                                            << " ";
    }
    coordinates_file << std::endl;
  }
  coordinates_file << "#"                                                                                                          << std::endl;
  coordinates_file << "# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~="        << std::endl;
  coordinates_file << "# Internal parameters and options"                                                                          << std::endl;
  coordinates_file << "# " << TAB << "ALREADY_INFERRED_PARAMETERS_FILENAME   " << ALREADY_INFERRED_PARAMETERS_FILENAME             << std::endl;
  coordinates_file << "# " << TAB << "BETA_ABS_MAX                           " << BETA_ABS_MAX                                     << std::endl;
  coordinates_file << "# " << TAB << "BETA_ABS_MIN                           " << BETA_ABS_MIN                                     << std::endl;
  coordinates_file << "# " << TAB << "CHARACTERIZATION_MODE                  " << (CHARACTERIZATION_MODE       ? "true" : "false") << std::endl;
  coordinates_file << "# " << TAB << "CHARACTERIZATION_NB_GRAPHS             " << CHARACTERIZATION_NB_GRAPHS                       << std::endl;
  coordinates_file << "# " << TAB << "CLEAN_RAW_OUTPUT_MODE                  " << (CLEAN_RAW_OUTPUT_MODE       ? "true" : "false") << std::endl;

  coordinates_file << "# " << TAB << "CUSTOM_BETA                            " << (CUSTOM_BETA                 ? "true" : "false") << std::endl;
  coordinates_file << "# " << TAB << "CUSTOM_INFERRED_COORDINATES            " << (CUSTOM_INFERRED_COORDINATES ? "true" : "false") << std::endl;
  coordinates_file << "# " << TAB << "CUSTOM_OUTPUT_ROOTNAME_MODE            " << (CUSTOM_OUTPUT_ROOTNAME_MODE ? "true" : "false") << std::endl;
  coordinates_file << "# " << TAB << "CUSTOM_SEED                            " << (CUSTOM_SEED                 ? "true" : "false") << std::endl;
  coordinates_file << "# " << TAB << "EDGELIST_FILENAME:                     " << EDGELIST_FILENAME                                << std::endl;
  coordinates_file << "# " << TAB << "EXP_CLUST_NB_INTEGRATION_MC_STEPS      " << EXP_CLUST_NB_INTEGRATION_MC_STEPS                << std::endl;
  coordinates_file << "# " << TAB << "EXP_DIST_NB_INTEGRATION_STEPS          " << EXP_DIST_NB_INTEGRATION_STEPS                    << std::endl;
  coordinates_file << "# " << TAB << "KAPPA_MAX_NB_ITER_CONV                 " << KAPPA_MAX_NB_ITER_CONV                           << std::endl;
  coordinates_file << "# " << TAB << "KAPPA_POST_INFERENCE_MODE              " << (KAPPA_POST_INFERENCE_MODE   ? "true" : "false") << std::endl;

  coordinates_file << "# " << TAB << "MAXIMIZATION_MODE                      " << (MAXIMIZATION_MODE           ? "true" : "false") << std::endl;

  coordinates_file << "# " << TAB << "MIN_NB_ANGLES_TO_TRY                   " << MIN_NB_ANGLES_TO_TRY                             << std::endl;
  coordinates_file << "# " << TAB << "NUMERICAL_CONVERGENCE_THRESHOLD_1      " << NUMERICAL_CONVERGENCE_THRESHOLD_1                << std::endl;
  coordinates_file << "# " << TAB << "NUMERICAL_CONVERGENCE_THRESHOLD_2      " << NUMERICAL_CONVERGENCE_THRESHOLD_2                << std::endl;
  coordinates_file << "# " << TAB << "NUMERICAL_CONVERGENCE_THRESHOLD_3      " << NUMERICAL_CONVERGENCE_THRESHOLD_3                << std::endl;
  coordinates_file << "# " << TAB << "NUMERICAL_ZERO                         " << NUMERICAL_ZERO                                   << std::endl;
  coordinates_file << "# " << TAB << "QUIET_MODE                             " << (QUIET_MODE                  ? "true" : "false") << std::endl;
  coordinates_file << "# " << TAB << "REFINE_MODE                            " << (REFINE_MODE                 ? "true" : "false") << std::endl;

  coordinates_file << "# " << TAB << "ROOTNAME_OUTPUT:                       " << ROOTNAME_OUTPUT                                  << std::endl;
  coordinates_file << "# " << TAB << "SEED                                   " << SEED                                             << std::endl;
  coordinates_file << "# " << TAB << "VALIDATION_MODE                        " << (VALIDATION_MODE             ? "true" : "false") << std::endl;
  coordinates_file << "# " << TAB << "VERBOSE_MODE                           " << (VERBOSE_MODE                ? "true" : "false") << std::endl;
  coordinates_file << "# " << TAB << "DIMENSION                              " << dim                                          << std::endl;
  coordinates_file << "# " << TAB << "VERSION                                " << VERSION                                          << std::endl;
  coordinates_file << "# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~="        << std::endl;

  coordinates_file.close();

  if(!QUIET_MODE) { std::clog << std::endl; }
  if(!QUIET_MODE) { std::clog << TAB << "=> Inferred coordinates saved to " << ROOTNAME_OUTPUT + ".inf_coord" << std::endl; }

  if(CLEAN_RAW_OUTPUT_MODE)
  {

    coordinates_filename = ROOTNAME_OUTPUT + ".inf_coord_raw";

    coordinates_file.open(coordinates_filename.c_str(), std::fstream::out);
    if( !coordinates_file.is_open() )
    {
      std::cerr << "Could not open file: " << coordinates_filename << "." << std::endl;
      std::terminate();
    }

    it  = ordered_names.begin();
    end = ordered_names.end();
    for(int v; it!=end; ++it)
    {
      v = it->second;

      coordinates_file << kappa[v]                                          << " ";
      for (int i=0; i<dim+1; ++i) {
          coordinates_file << d_positions[v][i]                             << " ";
      }

      coordinates_file << std::endl;
    }

    coordinates_file.close();

    if(!QUIET_MODE) { std::clog << std::endl; }
    if(!QUIET_MODE) { std::clog << TAB << "=> Raw inferred coordinates also saved to " << ROOTNAME_OUTPUT + ".inf_coord_raw" << std::endl; }
  }

  if(warning)
  {
    if(!QUIET_MODE) { std::clog << "WARNING: Hyperbolic radius has been adjusted to account for negative radial positions." << std::endl; }
  }
}

void embeddingSD_t::save_inferred_coordinates()
{
  save_inferred_coordinates(1);
}

void embeddingSD_t::save_inferred_ensemble_characterization()
{
  save_inferred_ensemble_characterization(1, false);
}

void embeddingSD_t::save_inferred_ensemble_characterization(int dim, bool random_positions=true)
{
  if(!QUIET_MODE) { std::clog << std::endl; }
  if(!QUIET_MODE) { std::clog << "Characterizing the inferred ensemble..." << std::endl; }

  std::map<int, double>::iterator it2, end2;
  std::map<int, std::vector<double> >::iterator it3, end3;

  characterizing_inferred_ensemble_vprops.clear();
  characterizing_inferred_ensemble_vprops.resize(5);
  for(auto &elem : characterizing_inferred_ensemble_vprops)
  {
    elem.clear();
    elem.resize(nb_vertices, std::vector<double>(2, 0));
  }
  characterizing_inferred_ensemble_vstat.clear();

  std::vector<double> single_comp_cumul_degree_dist;
  std::vector<double> avg_comp_cumul_degree_dist;
  std::vector<double> std_comp_cumul_degree_dist;
  std::vector<int> nb_comp_cumul_degree_dist;

  if(!CUSTOM_CHARACTERIZATION_NB_GRAPHS)
  {
    if     (nb_vertices < 500  ) { CHARACTERIZATION_NB_GRAPHS = 1000; }
    else if(nb_vertices < 1000 ) { CHARACTERIZATION_NB_GRAPHS = 500;  }
    else if(nb_vertices < 10000) { CHARACTERIZATION_NB_GRAPHS = 100;  }
    else                         { CHARACTERIZATION_NB_GRAPHS = 10;   }
  }
  if(!QUIET_MODE) { std::clog << std::endl; }
  if(!QUIET_MODE) { std::clog << TAB << "A total of " << CHARACTERIZATION_NB_GRAPHS << " graphs will be generated (chosen in function of the total number" << std::endl; }
  if(!QUIET_MODE) { std::clog << TAB << "of vertices). To change this value, set the flag 'CUSTOM_CHARACTERIZATION_NB_GRAPHS'" << std::endl; }
  if(!QUIET_MODE) { std::clog << TAB << "to 'true' and set the variable 'CHARACTERIZATION_NB_GRAPHS' to the desired value." << std::endl; }
  if(!QUIET_MODE) { std::clog << std::endl; }

  int delta_nb_graphs = CHARACTERIZATION_NB_GRAPHS / 19.999999;
  if(delta_nb_graphs < 1) { delta_nb_graphs = 1; }
  int width = 2 * (std::log10(CHARACTERIZATION_NB_GRAPHS) + 1) + 6;
  double d1, value;
  std::string graph_range;
  double start_time, stop_time;
  for(int g_i(0), g_f, d_max; g_i<CHARACTERIZATION_NB_GRAPHS;)
  {
    g_f = (g_i + delta_nb_graphs);
    g_f = (g_f > CHARACTERIZATION_NB_GRAPHS) ? CHARACTERIZATION_NB_GRAPHS : g_f;
    start_time = time_since_epoch_in_seconds();
    if(!QUIET_MODE) { graph_range = "[" + std::to_string(g_i+1) + "," + std::to_string(g_f) + "]..."; }
    if(!QUIET_MODE) { std::clog << TAB << "Generating and analyzing graphs " << std::setw(width) << graph_range; }
    for(; g_i<g_f; ++g_i)
    {
      single_comp_cumul_degree_dist.clear();
      if (dim == 1)
        generate_simulated_adjacency_list();
      else
        generate_simulated_adjacency_list(dim, random_positions);
      analyze_simulated_adjacency_list();
      for(int v1(0); v1<nb_vertices; ++v1)
      {

        d1 = simulated_degree[v1];
        characterizing_inferred_ensemble_vprops[0][v1][0] += d1;
        characterizing_inferred_ensemble_vprops[0][v1][1] += d1 * d1;
        if(d1 > 0)
        {

          value = simulated_sum_degree_of_neighbors[v1];
          characterizing_inferred_ensemble_vprops[1][v1][0] += value;
          characterizing_inferred_ensemble_vprops[1][v1][1] += value * value;

          value /= d1;
          characterizing_inferred_ensemble_vprops[2][v1][0] += value;
          characterizing_inferred_ensemble_vprops[2][v1][1] += value * value;
        }
        if(d1 > 1)
        {

          value = simulated_nb_triangles[v1];
          characterizing_inferred_ensemble_vprops[3][v1][0] += value;
          characterizing_inferred_ensemble_vprops[3][v1][1] += value * value;

          value /= d1 * (d1 - 1) / 2;
          characterizing_inferred_ensemble_vprops[4][v1][0] += value;
          characterizing_inferred_ensemble_vprops[4][v1][1] += value * value;
        }
      }

      it2 = simulated_stat_degree.begin();
      end2 = simulated_stat_degree.end();
      d_max = -1;

      for(int d, norm; it2!=end2; ++it2)
      {

        d = it2->first;

        if( characterizing_inferred_ensemble_vstat.find(d) == characterizing_inferred_ensemble_vstat.end() )
        {
          characterizing_inferred_ensemble_vstat[d] = std::vector<double>((2 * 5) + 1, 0);
        }

        if(d > d_max)
        {
          d_max = d;
          single_comp_cumul_degree_dist.resize(d_max + 1, 0);
        }

        norm = it2->second;

        value = simulated_stat_degree[d] / nb_vertices;
        characterizing_inferred_ensemble_vstat[d][0] += value;
        characterizing_inferred_ensemble_vstat[d][1] += value * value;

        for(int q(0); q<=d; ++q)
        {
          single_comp_cumul_degree_dist[q] += value;

        }

        value = simulated_stat_sum_degree_neighbors[d] / norm;
        characterizing_inferred_ensemble_vstat[d][2] += value;
        characterizing_inferred_ensemble_vstat[d][3] += value * value;

        value = simulated_stat_avg_degree_neighbors[d] / norm;
        characterizing_inferred_ensemble_vstat[d][4] += value;
        characterizing_inferred_ensemble_vstat[d][5] += value * value;

        value = simulated_stat_nb_triangles[d] / norm;
        characterizing_inferred_ensemble_vstat[d][6] += value;
        characterizing_inferred_ensemble_vstat[d][7] += value * value;

        value = simulated_stat_clustering[d] / norm;
        characterizing_inferred_ensemble_vstat[d][8] += value;
        characterizing_inferred_ensemble_vstat[d][9] += value * value;

        characterizing_inferred_ensemble_vstat[d][10] += 1;
      }

      if((d_max + 1) > nb_comp_cumul_degree_dist.size())
      {
        avg_comp_cumul_degree_dist.resize(d_max + 1, 0);
        std_comp_cumul_degree_dist.resize(d_max + 1, 0);
        nb_comp_cumul_degree_dist.resize(d_max + 1, 0);
      }
      for(int r(0); r<=d_max; ++r)
      {
        avg_comp_cumul_degree_dist[r] += single_comp_cumul_degree_dist[r];
        std_comp_cumul_degree_dist[r] += single_comp_cumul_degree_dist[r] * single_comp_cumul_degree_dist[r];
        nb_comp_cumul_degree_dist[r] += 1;
      }
    }

    stop_time = time_since_epoch_in_seconds();
    if(!QUIET_MODE) { std::clog << "...done in " << std::setw(6) << std::fixed << stop_time - start_time << " seconds" << std::endl; }
  }

  for(int i(0); i<characterizing_inferred_ensemble_vprops.size(); ++i)
  {
    for(int v1(0); v1<nb_vertices; ++v1)
    {
      characterizing_inferred_ensemble_vprops[i][v1][0] /= CHARACTERIZATION_NB_GRAPHS;
      characterizing_inferred_ensemble_vprops[i][v1][1] /= CHARACTERIZATION_NB_GRAPHS;
      characterizing_inferred_ensemble_vprops[i][v1][1] -= characterizing_inferred_ensemble_vprops[i][v1][0] * characterizing_inferred_ensemble_vprops[i][v1][0];
      characterizing_inferred_ensemble_vprops[i][v1][1] *= CHARACTERIZATION_NB_GRAPHS / (CHARACTERIZATION_NB_GRAPHS - 1);
      characterizing_inferred_ensemble_vprops[i][v1][1] = std::sqrt( characterizing_inferred_ensemble_vprops[i][v1][1] );
    }
  }
  it3 = characterizing_inferred_ensemble_vstat.begin();
  end3 = characterizing_inferred_ensemble_vstat.end();
  for(int norm, d; it3!=end3; ++it3)
  {
    d = it3->first;
    norm = characterizing_inferred_ensemble_vstat[d][10];
    for(int i(0); i<10; ++++i)
    {
      characterizing_inferred_ensemble_vstat[d][i + 0] /= norm;
      if(norm > 1)
      {
        characterizing_inferred_ensemble_vstat[d][i + 1] /= norm;
        characterizing_inferred_ensemble_vstat[d][i + 1] -= characterizing_inferred_ensemble_vstat[d][i + 0] * characterizing_inferred_ensemble_vstat[d][i + 0];
        characterizing_inferred_ensemble_vstat[d][i + 1] *= norm / (norm - 1);
        if( characterizing_inferred_ensemble_vstat[d][i + 1] < 0 )
        {
          characterizing_inferred_ensemble_vstat[d][i + 1] = 0;
        }
        else
        {
          characterizing_inferred_ensemble_vstat[d][i + 1] = std::sqrt( characterizing_inferred_ensemble_vstat[d][i + 1] );
        }
      }
      else
      {
        characterizing_inferred_ensemble_vstat[d][i + 1] = 0;
      }
    }
  }

  for(int i(0), ii(nb_comp_cumul_degree_dist.size()); i<ii; ++i)
  {
    if(nb_comp_cumul_degree_dist[i] > 0)
    {
      avg_comp_cumul_degree_dist[i] /= nb_comp_cumul_degree_dist[i];
      if(nb_comp_cumul_degree_dist[i] > 1)
      {
        std_comp_cumul_degree_dist[i] /= nb_comp_cumul_degree_dist[i];
        std_comp_cumul_degree_dist[i] -= avg_comp_cumul_degree_dist[i] * avg_comp_cumul_degree_dist[i];
        std_comp_cumul_degree_dist[i] *= nb_comp_cumul_degree_dist[i] / (nb_comp_cumul_degree_dist[i] - 1);
        if(std_comp_cumul_degree_dist[i] < 0)
        {
          std_comp_cumul_degree_dist[i] = 0;
        }
        else
        {
          std_comp_cumul_degree_dist[i] = std::sqrt(std_comp_cumul_degree_dist[i]);
        }
      }
      else
      {
        std_comp_cumul_degree_dist[i] = 0;
      }
    }
  }
  if(!QUIET_MODE) { std::clog << "                                       ...............................................done." << std::endl; }

  std::string vertex_properties_filename = ROOTNAME_OUTPUT + ".inf_vprop";

  std::fstream vertex_properties_file(vertex_properties_filename.c_str(), std::fstream::out);
  if( !vertex_properties_file.is_open() )
  {
    std::cerr << "Could not open file: " << vertex_properties_filename << "." << std::endl;
    std::terminate();
  }

  vertex_properties_file << "#";
  vertex_properties_file << std::setw(width_names - 1)  << "Vertex"          << " ";
  vertex_properties_file << std::setw(width_values)     << "Degree"          << " ";
  vertex_properties_file << std::setw(width_values)     << "Avg.Degree"      << " ";
  vertex_properties_file << std::setw(width_values)     << "Std.Degree"      << " ";
  vertex_properties_file << std::setw(width_values)     << "Sum.Deg.N"       << " ";
  vertex_properties_file << std::setw(width_values)     << "Avg.Sum.Deg.N"   << " ";
  vertex_properties_file << std::setw(width_values)     << "Std.Sum.Deg.N"   << " ";
  vertex_properties_file << std::setw(width_values)     << "Avg.Deg.N"       << " ";
  vertex_properties_file << std::setw(width_values)     << "Avg.Avg.Deg.N"   << " ";
  vertex_properties_file << std::setw(width_values)     << "Std.Avg.Deg.N"   << " ";
  vertex_properties_file << std::setw(width_values)     << "NbTriang"        << " ";
  vertex_properties_file << std::setw(width_values)     << "Avg.NbTriang"    << " ";
  vertex_properties_file << std::setw(width_values)     << "Std.NbTriang"    << " ";
  vertex_properties_file << std::setw(width_values)     << "Clustering"      << " ";
  vertex_properties_file << std::setw(width_values)     << "Avg.Clustering"  << " ";
  vertex_properties_file << std::setw(width_values)     << "Std.Clustering"  << " ";
  vertex_properties_file << std::endl;

  struct compare
  {
    bool operator()(const std::pair<std::string, int>& lhs, const std::pair<std::string, int>& rhs) const
    {
      if(lhs.first.size() == rhs.first.size())
      {
        if(lhs.first == rhs.first)
        {
          return lhs.second < rhs.second;
        }
        else
        {
          return lhs.first < rhs.first;
        }
      }
      else
      {
        return lhs.first.size() < rhs.first.size();
      }
    }
  };

  std::set< std::pair<std::string, int>, compare > ordered_names;
  for(int v(0); v<nb_vertices; ++v)
  {
    ordered_names.insert(std::make_pair(Num2Name[v], v));
  }
  auto it  = ordered_names.begin();
  auto end = ordered_names.end();
  for(int v, d; it!=end; ++it)
  {
    v = it->second;
    d = degree[v];
    vertex_properties_file << std::setw(width_names) << it->first                                        << " ";
    vertex_properties_file << std::setw(width_values) << degree[v]                                        << " ";
    vertex_properties_file << std::setw(width_values) << characterizing_inferred_ensemble_vprops[0][v][0] << " ";
    vertex_properties_file << std::setw(width_values) << characterizing_inferred_ensemble_vprops[0][v][1] << " ";
    vertex_properties_file << std::setw(width_values) << sum_degree_of_neighbors[v]                       << " ";
    vertex_properties_file << std::setw(width_values) << characterizing_inferred_ensemble_vprops[1][v][0] << " ";
    vertex_properties_file << std::setw(width_values) << characterizing_inferred_ensemble_vprops[1][v][1] << " ";
    if(d > 0)
    {
      vertex_properties_file << std::setw(width_values) << sum_degree_of_neighbors[v] / d                 << " ";
    }
    else
    {
      vertex_properties_file << std::setw(width_values) << sum_degree_of_neighbors[v]                     << " ";
    }
    vertex_properties_file << std::setw(width_values) << characterizing_inferred_ensemble_vprops[2][v][0] << " ";
    vertex_properties_file << std::setw(width_values) << characterizing_inferred_ensemble_vprops[2][v][1] << " ";
    vertex_properties_file << std::setw(width_values) << nbtriangles[v]                                   << " ";
    vertex_properties_file << std::setw(width_values) << characterizing_inferred_ensemble_vprops[3][v][0] << " ";
    vertex_properties_file << std::setw(width_values) << characterizing_inferred_ensemble_vprops[3][v][1] << " ";
    if(d > 1)
    {
      vertex_properties_file << std::setw(width_values) << nbtriangles[v] / (d * (d-1) / 2)               << " ";
    }
    else
    {
      vertex_properties_file << std::setw(width_values) << nbtriangles[v]                                 << " ";
    }
    vertex_properties_file << std::setw(width_values) << characterizing_inferred_ensemble_vprops[4][v][0] << " ";
    vertex_properties_file << std::setw(width_values) << characterizing_inferred_ensemble_vprops[4][v][1] << " ";
    vertex_properties_file << std::endl;
  }
  vertex_properties_file.close();
  if(!QUIET_MODE) { std::clog << std::endl; }
  if(!QUIET_MODE) { std::clog << TAB << "=> Vertices properties of the inferred ensemble saved to " << ROOTNAME_OUTPUT + ".inf_vprop" << std::endl; }

  std::string vertex_stat_filename = ROOTNAME_OUTPUT + ".inf_vstat";

  std::fstream vertex_stat_file(vertex_stat_filename.c_str(), std::fstream::out);
  if( !vertex_stat_file.is_open() )
  {
    std::cerr << "Could not open file: " << vertex_stat_filename << "." << std::endl;
    std::terminate();
  }

  vertex_stat_file << "#";
  vertex_stat_file << std::setw(width_values - 1) << "Degree"          << " ";

  vertex_stat_file << std::setw(width_values)     << "DegDistEns"      << " ";
  vertex_stat_file << std::setw(width_values)     << "DegDistEnsStd"   << " ";
  vertex_stat_file << std::setw(width_values)     << "CDegDistEns"      << " ";
  vertex_stat_file << std::setw(width_values)     << "CDegDistEnsStd"   << " ";

  vertex_stat_file << std::setw(width_values)     << "SumDegNEns"      << " ";
  vertex_stat_file << std::setw(width_values)     << "SumDegNEnsStd"   << " ";

  vertex_stat_file << std::setw(width_values)     << "AvgDegNEns"      << " ";
  vertex_stat_file << std::setw(width_values)     << "AvgDegNEnsStd"   << " ";

  vertex_stat_file << std::setw(width_values)     << "NbTriangEns"     << " ";
  vertex_stat_file << std::setw(width_values)     << "NbTriangEnsStd"  << " ";

  vertex_stat_file << std::setw(width_values)     << "ClustEns"        << " ";
  vertex_stat_file << std::setw(width_values)     << "ClustEnsStd"     << " ";
  vertex_stat_file << std::endl;

  it3 = characterizing_inferred_ensemble_vstat.begin();
  end3 = characterizing_inferred_ensemble_vstat.end();
  std::map<int, double> inf_degree_ccdf;
  for(int v, d; it3!=end3; ++it3)
  {
    d = it3->first;
    vertex_stat_file << std::setw(width_values) << d                                            << " ";
    vertex_stat_file << std::setw(width_values) << characterizing_inferred_ensemble_vstat[d][0] << " ";
    vertex_stat_file << std::setw(width_values) << characterizing_inferred_ensemble_vstat[d][1] << " ";
    vertex_stat_file << std::setw(width_values) << avg_comp_cumul_degree_dist[d]                << " ";
    vertex_stat_file << std::setw(width_values) << std_comp_cumul_degree_dist[d]                << " ";
    vertex_stat_file << std::setw(width_values) << characterizing_inferred_ensemble_vstat[d][2] << " ";
    vertex_stat_file << std::setw(width_values) << characterizing_inferred_ensemble_vstat[d][3] << " ";
    vertex_stat_file << std::setw(width_values) << characterizing_inferred_ensemble_vstat[d][4] << " ";
    vertex_stat_file << std::setw(width_values) << characterizing_inferred_ensemble_vstat[d][5] << " ";
    vertex_stat_file << std::setw(width_values) << characterizing_inferred_ensemble_vstat[d][6] << " ";
    vertex_stat_file << std::setw(width_values) << characterizing_inferred_ensemble_vstat[d][7] << " ";
    vertex_stat_file << std::setw(width_values) << characterizing_inferred_ensemble_vstat[d][8] << " ";
    vertex_stat_file << std::setw(width_values) << characterizing_inferred_ensemble_vstat[d][9] << " ";
    vertex_stat_file << std::endl;
    inf_degree_ccdf.insert(std::make_pair(d, avg_comp_cumul_degree_dist[d]));
  }
  vertex_stat_file.close();
  if(!QUIET_MODE) { std::clog << std::endl; }
  if(!QUIET_MODE) { std::clog << TAB << "=> Inferred ensemble statistics by degree class saved to " << ROOTNAME_OUTPUT + ".inf_vstat" << std::endl; }

  if(!QUIET_MODE) { std::clog << std::endl; }
  if(!QUIET_MODE) { std::clog << "Extracting the original graph statistics by degree class..."; }

  std::map<int, double> original_stat_degree;
  std::map<int, double> original_stat_sum_degree_neighbors;
  std::map<int, double> original_stat_avg_degree_neighbors;
  std::map<int, double> original_stat_nb_triangles;
  std::map<int, double> original_stat_clustering;
  for(int v1(0), d1; v1<nb_vertices; ++v1)
  {

    d1 = degree[v1];

    if( original_stat_degree.find(d1) == original_stat_degree.end() )
    {
      original_stat_degree[d1] = 0;
      original_stat_sum_degree_neighbors[d1] = 0;
      original_stat_avg_degree_neighbors[d1] = 0;
      original_stat_nb_triangles[d1] = 0;
      original_stat_clustering[d1] = 0;
    }

    original_stat_degree[d1] += 1;
    if(d1 > 0)
    {
      original_stat_sum_degree_neighbors[d1] += sum_degree_of_neighbors[v1];
      original_stat_avg_degree_neighbors[d1] += sum_degree_of_neighbors[v1] / d1;
    }
    if(d1 > 1)
    {
      original_stat_nb_triangles[d1] += nbtriangles[v1];
      original_stat_clustering[d1] += 2 * nbtriangles[v1] / d1 / (d1 - 1);
    }
  }
  if(!QUIET_MODE) { std::clog << "...........................done." << std::endl; }

  std::string graph_stat_filename = ROOTNAME_OUTPUT + ".obs_vstat";

  std::fstream graph_stat_file(graph_stat_filename.c_str(), std::fstream::out);
  if( !graph_stat_file.is_open() )
  {
    std::cerr << "Could not open file: " << graph_stat_filename << "." << std::endl;
    std::terminate();
  }

  graph_stat_file << "#";
  graph_stat_file << std::setw(width_values - 1) << "Degree"          << " ";
  graph_stat_file << std::setw(width_values)     << "DegDist"         << " ";
  graph_stat_file << std::setw(width_values)     << "CDegDist"        << " ";
  graph_stat_file << std::setw(width_values)     << "SumDegN"         << " ";
  graph_stat_file << std::setw(width_values)     << "AvgDegN"         << " ";
  graph_stat_file << std::setw(width_values)     << "NbTriang"        << " ";
  graph_stat_file << std::setw(width_values)     << "Clust"           << " ";
  graph_stat_file << std::endl;

  double ccdegdist = 1;
  it2 = original_stat_degree.begin();
  end2 = original_stat_degree.end();
  std::map<int, double> obs_degree_ccdf;
  for(int v, d, norm; it2!=end2; ++it2)
  {
    d = it2->first;
    norm = it2->second;
    graph_stat_file << std::setw(width_values) << d                                                   << " ";
    graph_stat_file << std::setw(width_values) << original_stat_degree[d] / nb_vertices               << " ";
    graph_stat_file << std::setw(width_values) << ccdegdist                                           << " ";
    obs_degree_ccdf.insert(std::make_pair(d, ccdegdist));
    ccdegdist -= original_stat_degree[d] / nb_vertices;
    graph_stat_file << std::setw(width_values) << original_stat_sum_degree_neighbors[d] / norm        << " ";
    graph_stat_file << std::setw(width_values) << original_stat_avg_degree_neighbors[d] / norm        << " ";
    graph_stat_file << std::setw(width_values) << original_stat_nb_triangles[d] / norm                << " ";
    graph_stat_file << std::setw(width_values) << original_stat_clustering[d] / norm                  << " ";
    graph_stat_file << std::endl;
  }
  graph_stat_file.close();

  if(!QUIET_MODE) { std::clog << std::endl; }
  if(!QUIET_MODE) { std::clog << TAB << "=> Original graph statistics by degree class saved to " << ROOTNAME_OUTPUT + ".obs_vstat" << std::endl; }

  double alpha = 0.05;
  if (!QUIET_MODE) {
    std::clog << std::endl;
    std::clog << "Computing two-sample Kolmogorov-Smirnov test......" << std::endl;
  }
  const auto ks_result = ks_test(inf_degree_ccdf, obs_degree_ccdf, alpha);
  if (!QUIET_MODE) {
    if (ks_result) {
      std::clog << "The null hypothesis is rejected at level " << alpha << std::endl;
      std::clog << "Two data samples does NOT come from the same distribution" << std::endl;
    } else {
      std::clog << "The null hypothesis is not rejected at level " << alpha << std::endl;
      std::clog << "Two data samples does come from the same distribution" << std::endl;
    }
  }
}

void embeddingSD_t::save_inferred_theta_density()
{

  std::map<double, int> bins;
  std::map<double, int>::iterator it;
  int cnt = 0;
  int nb_bins = 25;
  double dt = 2 * PI / nb_bins;
  for(double t(dt), tt(2 * PI + 0.001); t<tt; t+=dt, ++cnt)
  {
    bins[t] = cnt;
  }

  std::vector<double> n(bins.size(), 0);

  for(int v1(0), i; v1<nb_vertices; ++v1)
  {
    i = bins.upper_bound(theta[v1])->second;
    n[i] += 1;
  }

  std::string theta_density_filename = ROOTNAME_OUTPUT + ".inf_theta_density";
  std::fstream theta_density_file(theta_density_filename.c_str(), std::fstream::out);
  if( !theta_density_file.is_open() )
  {
    std::cerr << "Could not open file: " << theta_density_filename << "." << std::endl;
    std::terminate();
  }
  theta_density_file << "#";
  theta_density_file << std::setw(width_values - 1) << "Theta"       << " ";
  theta_density_file << std::setw(width_values)     << "InfDensity"  << " ";
  theta_density_file << std::setw(width_values)     << "ThDensity"   << " ";
  theta_density_file << std::endl;
  for(int i(0), ii(n.size()); i<ii; ++i)
  {
    theta_density_file << std::setw(width_values) << ((i + 0.5) * dt)    << " ";
    theta_density_file << std::setw(width_values) << n[i] / nb_vertices  << " ";
    theta_density_file << std::setw(width_values) << 1.0 / (nb_bins - 1) << " ";
    theta_density_file << std::endl;
  }

  theta_density_file.close();
  if(!QUIET_MODE) { std::clog << std::endl; }
  if(!QUIET_MODE) { std::clog << TAB << "=> Inferred theta density saved to " << ROOTNAME_OUTPUT + ".inf_theta_density" << std::endl; }
}

double embeddingSD_t::time_since_epoch_in_seconds()
{

  clock_t t = clock();
  return ((float)t) / (CLOCKS_PER_SEC);
}

void embeddingSD_t::finalize()
{

  std::clog << std::resetiosflags(std::ios::floatfield | std::ios::fixed | std::ios::showpoint);

  if (!QUIET_MODE) {
      std::clog << std::endl;
      std::clog << "Internal parameters and options" << std::endl;
      std::clog << TAB << "ALREADY_INFERRED_PARAMETERS_FILENAME   " << ALREADY_INFERRED_PARAMETERS_FILENAME
                << std::endl;
      std::clog << TAB << "BETA_ABS_MAX                           " << BETA_ABS_MAX << std::endl;
      std::clog << TAB << "BETA_ABS_MIN                           " << BETA_ABS_MIN << std::endl;
      std::clog << TAB << "CHARACTERIZATION_MODE                  " << (CHARACTERIZATION_MODE ? "true" : "false")
                << std::endl;
      std::clog << TAB << "CHARACTERIZATION_NB_GRAPHS             " << CHARACTERIZATION_NB_GRAPHS << std::endl;
      std::clog << TAB << "CLEAN_RAW_OUTPUT_MODE                  " << (CLEAN_RAW_OUTPUT_MODE ? "true" : "false")
                << std::endl;

      std::clog << TAB << "CUSTOM_BETA                            " << (CUSTOM_BETA ? "true" : "false") << std::endl;
      std::clog << TAB << "CUSTOM_CHARACTERIZATION_NB_GRAPHS      "
                << (CUSTOM_CHARACTERIZATION_NB_GRAPHS ? "true" : "false") << std::endl;
      std::clog << TAB << "CUSTOM_INFERRED_COORDINATES            " << (CUSTOM_INFERRED_COORDINATES ? "true" : "false")
                << std::endl;
      std::clog << TAB << "CUSTOM_OUTPUT_ROOTNAME_MODE            " << (CUSTOM_OUTPUT_ROOTNAME_MODE ? "true" : "false")
                << std::endl;
      std::clog << TAB << "CUSTOM_SEED                            " << (CUSTOM_SEED ? "true" : "false") << std::endl;
      std::clog << TAB << "CUDA_MODE                              " << (CUDA_MODE ? "true" : "false") << std::endl;
      std::clog << TAB << "CUDA_DETERMINISTIC_MODE                " << (CUDA_DETERMINISTIC_MODE ? "true" : "false") << std::endl;
      std::clog << TAB << "CUDA_RUNTIME_AVAILABLE                 " << (CUDA_RUNTIME_AVAILABLE ? "true" : "false") << std::endl;
      std::clog << TAB << "EDGELIST_FILENAME:                     " << EDGELIST_FILENAME << std::endl;
      std::clog << TAB << "EXP_CLUST_NB_INTEGRATION_MC_STEPS      " << EXP_CLUST_NB_INTEGRATION_MC_STEPS << std::endl;
      std::clog << TAB << "EXP_DIST_NB_INTEGRATION_STEPS          " << EXP_DIST_NB_INTEGRATION_STEPS << std::endl;
      std::clog << TAB << "KAPPA_MAX_NB_ITER_CONV                 " << KAPPA_MAX_NB_ITER_CONV << std::endl;
      std::clog << TAB << "KAPPA_POST_INFERENCE_MODE              " << (KAPPA_POST_INFERENCE_MODE ? "true" : "false")
                << std::endl;

      std::clog << TAB << "MAXIMIZATION_MODE                      " << (MAXIMIZATION_MODE ? "true" : "false")
                << std::endl;

      std::clog << TAB << "MIN_NB_ANGLES_TO_TRY                   " << MIN_NB_ANGLES_TO_TRY << std::endl;
      std::clog << TAB << "NUMERICAL_CONVERGENCE_THRESHOLD_1      " << NUMERICAL_CONVERGENCE_THRESHOLD_1 << std::endl;
      std::clog << TAB << "NUMERICAL_CONVERGENCE_THRESHOLD_2      " << NUMERICAL_CONVERGENCE_THRESHOLD_2 << std::endl;
      std::clog << TAB << "NUMERICAL_CONVERGENCE_THRESHOLD_3      " << NUMERICAL_CONVERGENCE_THRESHOLD_3 << std::endl;
      std::clog << TAB << "NUMERICAL_ZERO                         " << NUMERICAL_ZERO << std::endl;
      std::clog << TAB << "QUIET_MODE                             " << (QUIET_MODE ? "true" : "false") << std::endl;
      std::clog << TAB << "REFINE_MODE                            " << (REFINE_MODE ? "true" : "false") << std::endl;

      std::clog << TAB << "ROOTNAME_OUTPUT:                       " << ROOTNAME_OUTPUT << std::endl;
      std::clog << TAB << "SEED                                   " << SEED << std::endl;
      std::clog << TAB << "VALIDATION_MODE                        " << (VALIDATION_MODE ? "true" : "false")
                << std::endl;
      std::clog << TAB << "VERBOSE_MODE                           " << (VERBOSE_MODE ? "true" : "false") << std::endl;
      std::clog << TAB << "VERSION                                " << VERSION << std::endl;
      std::clog << std::endl;
      std::clog << "Ended on: " << format_time(time_ended) << std::endl;
      std::clog << "Elapsed CPU time (embedding):            " << std::setw(10) << std::fixed << time5 - time0
                << " seconds" << std::endl;
      std::clog << TAB << "initialization:                      " << std::setw(10) << std::fixed << time1 - time0
                << " seconds" << std::endl;

      if (!REFINE_MODE) {
          std::clog << TAB << "parameters inference:                " << std::setw(10) << std::fixed << time2 - time1 << " seconds" << std::endl;
          std::clog << TAB << "initial positions:                   " << std::setw(10) << std::fixed << time3 - time2 << " seconds" << std::endl;
      }

      if (REFINE_MODE)
          std::clog << TAB << "loading previous positions:          " << std::setw(10) << std::fixed << time3 - time1 << " seconds" << std::endl;

      if (MAXIMIZATION_MODE)
          std::clog << TAB << "refining positions:                  " << std::setw(10) << std::fixed << time4 - time3 << " seconds" << std::endl;

      std::clog << TAB << "adjusting kappas:                    " << std::setw(10) << std::fixed << time5 - time4 << " seconds" << std::endl;

      if (VALIDATION_MODE || CHARACTERIZATION_MODE)
          std::clog << "Elapsed CPU time (validation):           " << std::setw(10) << std::fixed << time7 - time5 << " seconds" << std::endl;

      if (VALIDATION_MODE)
          std::clog << TAB << "validating embedding:                " << std::setw(10) << std::fixed << time6 - time5 << " seconds" << std::endl;

      if (CHARACTERIZATION_MODE)
          std::clog << TAB << "characterizing ensemble:             " << std::setw(10) << std::fixed << time7 - time6 << " seconds" << std::endl;

      std::clog << "===========================================================================================" << std::endl;

      if (!VERBOSE_MODE) {
          logfile.close();

          std::clog.rdbuf(old_rdbuf);
      }
  }

  if(TIMING_JSON_MODE) {
      std::cout << std::fixed << std::setprecision(6);
      std::cout << "{"
                << "\"mode\":\"" << (OPTIMIZED_PERF_MODE ? "optimized" : "baseline") << "\","
                << "\"backend\":\"" << (CUDA_MODE ? "cuda" : "cpu") << "\","
                << "\"dimension\":" << DIMENSION << ","
                << "\"seed\":" << SEED << ","
                << "\"nb_vertices\":" << nb_vertices << ","
                << "\"nb_edges\":" << nb_edges << ","
                << "\"total_time_ms\":" << stage_timing_summary.total_time_ms << ","
                << "\"initialization_ms\":" << stage_timing_summary.initialization_ms << ","
                << "\"parameter_inference_ms\":" << stage_timing_summary.parameter_inference_ms << ","
                << "\"initial_positions_ms\":" << stage_timing_summary.initial_positions_ms << ","
                << "\"refining_positions_ms\":" << stage_timing_summary.refining_positions_ms << ","
                << "\"adjusting_kappas_ms\":" << stage_timing_summary.adjusting_kappas_ms << ","
                << "\"io_ms\":" << stage_timing_summary.io_ms
                << "}" << std::endl;
  }
}

#endif
