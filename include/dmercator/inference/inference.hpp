#ifndef DMERCATOR_INFERENCE_INFERENCE_HPP
#define DMERCATOR_INFERENCE_INFERENCE_HPP

void embeddingSD_t::analyze_degrees()
{

  average_degree = 0;

  nb_vertices_degree_gt_one = 0;

  degree.clear();
  degree.resize(nb_vertices);
  if(VALIDATION_MODE)
  {
    sum_degree_of_neighbors.clear();
    sum_degree_of_neighbors.resize(nb_vertices, 0);
  }

  std::set<int>::iterator it, end;
  for(int n(0), k; n<nb_vertices; ++n)
  {
    k = adjacency_list[n].size();
    degree[n] = k;
    average_degree += k;
    degree2vertices[k].push_back(n);
    degree_class.insert(k);
    if(k > 1)
    {
      ++nb_vertices_degree_gt_one;
    }
    if(VALIDATION_MODE)
    {
      it = adjacency_list[n].begin();
      end = adjacency_list[n].end();
      for(; it!=end; ++it)
      {
        sum_degree_of_neighbors[*it] +=     k;
      }
    }
  }

  average_degree /= nb_vertices;
}

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

void embeddingSD_t::build_cumul_dist_for_mc_integration(int dim) {

  double tmp_val, tmp_cumul;

  std::map<int, double> nkkp;

  cumul_prob_kgkp.clear();

  std::set<int>::iterator it1, end1, it2, end2;

  if(dim == 1)
  {
    const double R = nb_vertices / (2 * PI);
    mu = calculateMu();

    it1 = degree_class.begin();
    end1 = degree_class.end();
    while(*it1 < 2) { ++it1; }
    for(; it1!=end1; ++it1)
    {

      nkkp.clear();

      it2 = degree_class.begin();
      end2 = degree_class.end();
      for(; it2!=end2; ++it2)
      {

        nkkp[*it2] = 0;
      }

      it2 = degree_class.begin();
      end2 = degree_class.end();
      for(; it2!=end2; ++it2)
      {
        tmp_val = hyp2f1a(beta, -std::pow((PI * R) / (mu * random_ensemble_kappa_per_degree_class[*it1] * random_ensemble_kappa_per_degree_class[*it2]), beta));
        nkkp[*it2] = degree2vertices[*it2].size() * tmp_val / random_ensemble_expected_degree_per_degree_class[*it1];
      }

      tmp_cumul = 0;

      cumul_prob_kgkp[*it1];

      it2 = degree_class.begin();
      end2 = degree_class.end();
      for(; it2!=end2; ++it2)
      {

        tmp_val = nkkp[*it2];
        if(tmp_val > NUMERICAL_ZERO)
        {

          tmp_cumul += tmp_val;

          cumul_prob_kgkp[*it1][tmp_cumul] = *it2;
        }
      }
    }
    return;
  }

  const double R = compute_radius(dim, nb_vertices);
  mu = calculate_mu(dim);

  it1 = degree_class.begin();
  end1 = degree_class.end();
  while(*it1 < 2) { ++it1; }
  for(; it1!=end1; ++it1)
  {

    nkkp.clear();

    it2 = degree_class.begin();
    end2 = degree_class.end();
    for(; it2!=end2; ++it2) {

      nkkp[*it2] = 0;
    }

    it2 = degree_class.begin();
    end2 = degree_class.end();
    for(; it2!=end2; ++it2) {
      const auto kappa1 = random_ensemble_kappa_per_degree_class[*it1];
      const auto kappa2 = random_ensemble_kappa_per_degree_class[*it2];
      tmp_val = compute_integral_expected_degree_dimensions(dim, R, mu, beta, kappa1, kappa2);
      nkkp[*it2] = degree2vertices[*it2].size() * tmp_val / random_ensemble_expected_degree_per_degree_class[*it1];
    }

    tmp_cumul = 0;

    cumul_prob_kgkp[*it1];

    it2 = degree_class.begin();
    end2 = degree_class.end();
    for(; it2!=end2; ++it2) {
      tmp_val = nkkp[*it2];
      if(tmp_val > NUMERICAL_ZERO)
      {

        tmp_cumul += tmp_val;

        cumul_prob_kgkp[*it1][tmp_cumul] = *it2;
      }
    }
  }
}

void embeddingSD_t::build_cumul_dist_for_mc_integration()
{
  build_cumul_dist_for_mc_integration(1);
}

void embeddingSD_t::compute_clustering()
{
  average_clustering = 0;

  double nb_triangles, tmp_val;

  std::vector<int> intersection;

  std::set<int> neighbors_v2;

  std::vector<int>::iterator it;
  std::set<int>::iterator it1, end1, it2, end2;
  std::map<int, std::vector<int> >::iterator it3, end3;
  if(VALIDATION_MODE)
  {

    nbtriangles.clear();
    nbtriangles.resize(nb_vertices, 0);
  }

  for(int v1(0), d1; v1<nb_vertices; ++v1)
  {

    nb_triangles = 0;

    d1 = degree[v1];
    if( d1 > 1 )
    {

      it1 = adjacency_list[v1].begin();
      end1 = adjacency_list[v1].end();
      for(; it1!=end1; ++it1)
      {

        if( degree[*it1] > 1 )
        {

          it2 = adjacency_list[*it1].begin();
          end2 = adjacency_list[*it1].end();
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
            intersection.resize(std::min(adjacency_list[v1].size(), neighbors_v2.size()));
            it = std::set_intersection(adjacency_list[v1].begin(), adjacency_list[v1].end(),
                                  neighbors_v2.begin(), neighbors_v2.end(), intersection.begin());
            intersection.resize(it-intersection.begin());
            nb_triangles += intersection.size();
          }
        }
      }

      tmp_val = 2 * nb_triangles / (d1 * (d1 - 1));
      average_clustering += tmp_val;
      if(VALIDATION_MODE)
      {
        nbtriangles[v1] = nb_triangles;
      }
    }
  }

  average_clustering /= nb_vertices_degree_gt_one;
}

void embeddingSD_t::compute_inferred_ensemble_expected_degrees(int dim, double radius)
{
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

void embeddingSD_t::compute_random_ensemble_average_degree()
{

  random_ensemble_average_degree = 0;
  for (const auto &it2: random_ensemble_expected_degree_per_degree_class)
    random_ensemble_average_degree += it2.second * degree2vertices[it2.first].size();
  random_ensemble_average_degree /= nb_vertices;
}

void embeddingSD_t::compute_random_ensemble_clustering(int dim)
{

  random_ensemble_average_clustering = 0;

  auto it = degree_class.begin();
  auto end = degree_class.end();
  while(*it < 2) { ++it; }
  for(; it!=end; ++it)
  {

    double p23;
    if(dim == 1)
      p23 = compute_random_ensemble_clustering_for_degree_class(*it);
    else
      p23 = compute_random_ensemble_clustering_for_degree_class(*it, dim);
    random_ensemble_average_clustering += p23 * degree2vertices[*it].size();
  }

  random_ensemble_average_clustering /= nb_vertices_degree_gt_one;
}

void embeddingSD_t::compute_random_ensemble_clustering()
{
  compute_random_ensemble_clustering(1);
}

std::pair<int, double> embeddingSD_t::degree_of_random_vertex_and_prob_conn(int d1, double R, int dim)
{
  if(dim == 1)
  {
    auto d = cumul_prob_kgkp[d1].lower_bound(uniform_01(engine))->second;
    auto p = hyp2f1a(beta, -std::pow((PI * R) / (mu * random_ensemble_kappa_per_degree_class[d1] * random_ensemble_kappa_per_degree_class[d]), beta));
    return std::make_pair(d, p);
  }

  auto d = cumul_prob_kgkp[d1].lower_bound(uniform_01(engine))->second;
  const auto kappa1 = random_ensemble_kappa_per_degree_class[d1];
  const auto kappa2 = random_ensemble_kappa_per_degree_class[d];
  auto p = compute_integral_expected_degree_dimensions(dim, R, mu, beta, kappa1, kappa2);
  return std::make_pair(d, p);
}

std::pair<int, double> embeddingSD_t::degree_of_random_vertex_and_prob_conn(int d1, double R)
{
  return degree_of_random_vertex_and_prob_conn(d1, R, 1);
}

double embeddingSD_t::draw_random_angular_distance(int d1, int d2, double R, double p12, int dim)
{
  if(dim == 1)
  {
    double pc = uniform_01(engine);
    double zmin = 0, zmax = PI, z, pz;
    while((zmax - zmin) > NUMERICAL_CONVERGENCE_THRESHOLD_2)
    {
      z = (zmax + zmin) / 2;
      pz = (z / PI) * hyp2f1a(beta, -std::pow((z * R) / (mu * random_ensemble_kappa_per_degree_class[d1] * random_ensemble_kappa_per_degree_class[d2]), beta)) / p12;
      if(pz > pc)
        zmax = z;
      else
        zmin = z;
    }
    return (zmax + zmin) / 2;
  }

  double pc = uniform_01(engine);
  double zmin = 0, zmax = PI, z, pz;
  const auto kappa1 = random_ensemble_kappa_per_degree_class[d1];
  const auto kappa2 = random_ensemble_kappa_per_degree_class[d2];
  while((zmax - zmin) > NUMERICAL_CONVERGENCE_THRESHOLD_2)
  {
    z = (zmax + zmin) / 2;
    pz = compute_integral_expected_degree_dimensions(dim, R, mu, beta, kappa1, kappa2, z) / p12;
    if(pz > pc)
      zmax = z;
    else
      zmin = z;
  }
  return (zmax + zmin) / 2;
}

double embeddingSD_t::draw_random_angular_distance(int d1, int d2, double R, double p12)
{
  return draw_random_angular_distance(d1, d2, R, p12, 1);
}

double embeddingSD_t::compute_random_ensemble_clustering_for_degree_class(int d1, int dim)
{
  if(dim == 1)
  {

    double z12, z13, da;
    double p23 = 0;

    int nb_points = EXP_CLUST_NB_INTEGRATION_MC_STEPS;
    const double R = nb_vertices / (2 * PI);
    mu = calculateMu();

#pragma omp parallel for default(shared) reduction(+:p23)
    for(int i=0; i<nb_points; ++i)
    {

      const auto [d2, p12] = degree_of_random_vertex_and_prob_conn(d1, R);
      const auto [d3, p13] = degree_of_random_vertex_and_prob_conn(d1, R);

      z12 = draw_random_angular_distance(d1, d2, R, p12);
      z13 = draw_random_angular_distance(d1, d3, R, p13);

      if(uniform_01(engine) < 0.5)
        da = std::fabs(z12 + z13);
      else
        da = std::fabs(z12 - z13);

      da = std::min(da, (2.0 * PI) - da);
      if(da < NUMERICAL_ZERO)
        p23 += 1;
      else
        p23 += 1.0 / (1.0 + std::pow((da * R) / (mu * random_ensemble_kappa_per_degree_class[d2] * random_ensemble_kappa_per_degree_class[d3]), beta));
    }

    return p23 / nb_points;
  }

  double z12, z13, da;

  double p23 = 0;
  const int nb_points = EXP_CLUST_NB_INTEGRATION_MC_STEPS;
  const double R = compute_radius(dim, nb_vertices);
  mu = calculate_mu(dim);

#pragma omp parallel for default(shared) reduction(+:p23)
  for(int i=0; i<nb_points; ++i)
  {

    const auto [d2, p12] = degree_of_random_vertex_and_prob_conn(d1, R, dim);
    const auto [d3, p13] = degree_of_random_vertex_and_prob_conn(d1, R, dim);

    z12 = draw_random_angular_distance(d1, d2, R, p12, dim);
    z13 = draw_random_angular_distance(d1, d3, R, p13, dim);

    const auto v1 = generate_random_d_vector_with_first_coordinate(dim, z12, R);
    const auto v2 = generate_random_d_vector_with_first_coordinate(dim, z13, R);
    const auto d_angle = compute_angle_d_vectors(v1, v2);
    if (d_angle < NUMERICAL_ZERO) {
      p23 += 1;
    } else {
      const auto kappa1 = random_ensemble_kappa_per_degree_class[d2];
      const auto kappa2 = random_ensemble_kappa_per_degree_class[d3];
      const auto inside = (R * d_angle / std::pow(mu * kappa1 * kappa2, 1.0 / dim));
      p23 += 1.0 / (1 + std::pow(inside, beta));
    }
  }

  return p23 / nb_points;
}

double embeddingSD_t::compute_random_ensemble_clustering_for_degree_class(int d1)
{
  return compute_random_ensemble_clustering_for_degree_class(d1, 1);
}

void embeddingSD_t::infer_kappas_given_beta_for_all_vertices(int dim)
{
  if(!QUIET_MODE) { std::clog << "Updating values of kappa based on inferred positions..." << std::endl; }
  if(!QUIET_MODE) { std::clog.flush(); }
  if(dim == 1)
  {

    int cnt = 0;
    bool keep_going = true;
    while( keep_going && (cnt < KAPPA_MAX_NB_ITER_CONV) )
    {

      compute_inferred_ensemble_expected_degrees();

      keep_going = false;
      for(int v(0); v<nb_vertices; ++v)
      {
        if(std::fabs(inferred_ensemble_expected_degree[v] - degree[v]) > NUMERICAL_CONVERGENCE_THRESHOLD_3)
        {
          keep_going = true;
          continue;
        }
      }

      if(keep_going)
      {
        for(int v(0); v<nb_vertices; ++v)
        {
          kappa[v] += (degree[v] - inferred_ensemble_expected_degree[v]) * uniform_01(engine);
          kappa[v] = std::fabs(kappa[v]);
        }
      }
      ++cnt;
    }

    if(cnt >= KAPPA_MAX_NB_ITER_CONV)
    {
      if(!QUIET_MODE) { std::clog << TAB << "WARNING: maximum number of iterations reached before convergence. This limit can be"  << std::endl; }
      if(!QUIET_MODE) { std::clog << TAB << "         adjusted by setting the parameters KAPPA_MAX_NB_ITER_CONV to desired value." << std::endl; }
    }
    else
    {
      if(!QUIET_MODE) { std::clog << TAB << "Convergence reached after " << cnt << " iterations." << std::endl; }
    }
    if(!QUIET_MODE) { std::clog << "                                                       ...............................done." << std::endl; }
    return;
  }

  int cnt = 0;
  bool keep_going = true;
  const double radius = compute_radius(dim, nb_vertices);
  while (keep_going && (cnt < KAPPA_MAX_NB_ITER_CONV))
  {

    compute_inferred_ensemble_expected_degrees(dim, radius);

    keep_going = false;
    for(int v(0); v<nb_vertices; ++v)
    {
      if(std::fabs(inferred_ensemble_expected_degree[v] - degree[v]) > NUMERICAL_CONVERGENCE_THRESHOLD_3)
      {
        keep_going = true;
        continue;
      }
    }

    if(keep_going)
    {
      for(int v(0); v<nb_vertices; ++v)
      {
        kappa[v] += (degree[v] - inferred_ensemble_expected_degree[v]) * uniform_01(engine);
        kappa[v] = std::fabs(kappa[v]);
      }
    }
    ++cnt;
  }

  if(cnt >= KAPPA_MAX_NB_ITER_CONV)
  {
    if(!QUIET_MODE) { std::clog << TAB << "WARNING: maximum number of iterations reached before convergence. This limit can be"  << std::endl; }
    if(!QUIET_MODE) { std::clog << TAB << "         adjusted by setting the parameters KAPPA_MAX_NB_ITER_CONV to desired value." << std::endl; }
  }
  else
  {
    if(!QUIET_MODE) { std::clog << TAB << "Convergence reached after " << cnt << " iterations." << std::endl; }
  }
  if(!QUIET_MODE) { std::clog << "                                                       ...............................done." << std::endl; }
}

void embeddingSD_t::infer_kappas_given_beta_for_all_vertices()
{
  infer_kappas_given_beta_for_all_vertices(1);
}

void embeddingSD_t::infer_kappas_given_beta_for_degree_class(int dim)
{
  if(!OPTIMIZED_PERF_MODE)
  {
    if(dim == 1)
    {

      double prob_conn;

      mu = calculateMu();

      std::set<int>::iterator it1, it2, end;

      it1 = degree_class.begin();
      end = degree_class.end();
      for(; it1!=end; ++it1)
      {
        random_ensemble_kappa_per_degree_class[*it1] = *it1;
      }

      int cnt = 0;
      bool keep_going = true;
      while( keep_going && (cnt < KAPPA_MAX_NB_ITER_CONV) )
      {

        it1 = degree_class.begin();
        end = degree_class.end();
        for(; it1!=end; ++it1)
        {
          random_ensemble_expected_degree_per_degree_class[*it1] = 0;
        }

        it1 = degree_class.begin();
        end = degree_class.end();
        for(; it1!=end; ++it1)
        {
          it2 = it1;
          prob_conn = hyp2f1a(beta, -std::pow(nb_vertices / (2.0 * mu * random_ensemble_kappa_per_degree_class[*it1] * random_ensemble_kappa_per_degree_class[*it2]), beta));
          random_ensemble_expected_degree_per_degree_class[*it1] += prob_conn * (degree2vertices[*it2].size() - 1);
          for(++it2; it2!=end; ++it2)
          {
            prob_conn = hyp2f1a(beta, -std::pow(nb_vertices / (2.0 * mu * random_ensemble_kappa_per_degree_class[*it1] * random_ensemble_kappa_per_degree_class[*it2]), beta));
            random_ensemble_expected_degree_per_degree_class[*it1] += prob_conn * degree2vertices[*it2].size();
            random_ensemble_expected_degree_per_degree_class[*it2] += prob_conn * degree2vertices[*it1].size();
          }
        }

        keep_going = false;
        it1 = degree_class.begin();
        end = degree_class.end();
        for(; it1!=end; ++it1)
        {
          if(std::fabs(random_ensemble_expected_degree_per_degree_class[*it1] - *it1) > NUMERICAL_CONVERGENCE_THRESHOLD_1)
          {
            keep_going = true;
            break;
          }
        }

        if(keep_going)
        {
          it1 = degree_class.begin();
          end = degree_class.end();
          for(; it1!=end; ++it1)
          {
            random_ensemble_kappa_per_degree_class[*it1] += (*it1 - random_ensemble_expected_degree_per_degree_class[*it1]) * uniform_01(engine);
            random_ensemble_kappa_per_degree_class[*it1] = std::fabs(random_ensemble_kappa_per_degree_class[*it1]);
          }
        }
        ++cnt;
      }
      if(cnt >= KAPPA_MAX_NB_ITER_CONV)
      {
        if(!QUIET_MODE) {
          std::clog << std::endl;
          std::clog << TAB << "WARNING: maximum number of iterations reached before convergence. This limit can be"  << std::endl;
          std::clog << TAB << "         adjusted by setting the parameters KAPPA_MAX_NB_ITER_CONV to desired value." << std::endl;
          std::clog << TAB << std::fixed << std::setw(11) << " " << " ";
        }
      }
      return;
    }

    double prob_conn;
    const auto radius = compute_radius(dim, nb_vertices);
    mu = calculate_mu(dim);

    std::set<int>::iterator it1, it2, end;

    it1 = degree_class.begin();
    end = degree_class.end();

    for(; it1!=end; ++it1)
    {
      random_ensemble_kappa_per_degree_class[*it1] = *it1;
    }

    int cnt = 0;
    bool keep_going = true;
    while (keep_going && (cnt < KAPPA_MAX_NB_ITER_CONV_2))
    {

      it1 = degree_class.begin();
      end = degree_class.end();
      for(; it1!=end; ++it1)
      {
        random_ensemble_expected_degree_per_degree_class[*it1] = 0;
      }

      end = degree_class.end();
      for(it1=degree_class.begin(); it1!=end; ++it1)

      {
        it2 = it1;
        auto kappa_i = random_ensemble_kappa_per_degree_class[*it1];
        auto kappa_j = random_ensemble_kappa_per_degree_class[*it2];
        auto integral = compute_integral_expected_degree_dimensions(dim, radius, mu, beta, kappa_i, kappa_j);
        prob_conn = integral;

        random_ensemble_expected_degree_per_degree_class[*it1] += prob_conn * (degree2vertices[*it2].size() - 1);
        for(++it2; it2!=end; ++it2)
        {
          kappa_i = random_ensemble_kappa_per_degree_class[*it1];
          kappa_j = random_ensemble_kappa_per_degree_class[*it2];
          integral = compute_integral_expected_degree_dimensions(dim, radius, mu, beta, kappa_i, kappa_j);
          prob_conn = integral;
          random_ensemble_expected_degree_per_degree_class[*it1] += prob_conn * degree2vertices[*it2].size();
          random_ensemble_expected_degree_per_degree_class[*it2] += prob_conn * degree2vertices[*it1].size();
        }
      }

      keep_going = false;
      it1 = degree_class.begin();
      end = degree_class.end();
      for(; it1!=end; ++it1)
      {
        if(std::fabs(random_ensemble_expected_degree_per_degree_class[*it1] - *it1) > NUMERICAL_CONVERGENCE_THRESHOLD_1) {
          keep_going = true;
          break;
        }
      }

      if(keep_going)
      {
        it1 = degree_class.begin();
        end = degree_class.end();
        for(; it1!=end; ++it1) {
          random_ensemble_kappa_per_degree_class[*it1] += (*it1 - random_ensemble_expected_degree_per_degree_class[*it1]) * uniform_01(engine);
          random_ensemble_kappa_per_degree_class[*it1] = std::fabs(random_ensemble_kappa_per_degree_class[*it1]);
        }
      }
      ++cnt;
      }
      if (cnt >= KAPPA_MAX_NB_ITER_CONV_2) {
        if (!QUIET_MODE) {
          std::clog << std::endl;
          std::clog << TAB << "WARNING: maximum number of iterations reached before convergence. This limit can be"  << std::endl;
          std::clog << TAB << "         adjusted by setting the parameters KAPPA_MAX_NB_ITER_CONV_2 to desired value." << std::endl;
          std::clog << TAB << std::fixed << std::setw(11) << " " << " ";
        }
      }
    return;
  }

  std::vector<int> degree_classes;
  degree_classes.reserve(degree_class.size());
  for(const int d : degree_class)
  {
    degree_classes.push_back(d);
  }
  const int nb_degree_classes = static_cast<int>(degree_classes.size());

  std::vector<double> class_sizes(nb_degree_classes, 0.0);
  for(int i = 0; i < nb_degree_classes; ++i)
  {
    class_sizes[i] = static_cast<double>(degree2vertices[degree_classes[i]].size());
  }

  if(dim == 1)
  {
    mu = calculateMu();

    for(const int d : degree_classes)
    {
      random_ensemble_kappa_per_degree_class[d] = d;
    }

    int cnt = 0;
    bool keep_going = true;
    while(keep_going && (cnt < KAPPA_MAX_NB_ITER_CONV))
    {
      for(const int d : degree_classes)
      {
        random_ensemble_expected_degree_per_degree_class[d] = 0.0;
      }

      for(int i = 0; i < nb_degree_classes; ++i)
      {
        const int d_i = degree_classes[i];
        const double kappa_i = random_ensemble_kappa_per_degree_class[d_i];

        const int d_j0 = degree_classes[i];
        const double kappa_j0 = random_ensemble_kappa_per_degree_class[d_j0];
        double prob_conn = hyp2f1a(beta, -std::pow(nb_vertices / (2.0 * mu * kappa_i * kappa_j0), beta));
        random_ensemble_expected_degree_per_degree_class[d_i] += prob_conn * (class_sizes[i] - 1.0);

        for(int j = i + 1; j < nb_degree_classes; ++j)
        {
          const int d_j = degree_classes[j];
          const double kappa_j = random_ensemble_kappa_per_degree_class[d_j];
          prob_conn = hyp2f1a(beta, -std::pow(nb_vertices / (2.0 * mu * kappa_i * kappa_j), beta));
          random_ensemble_expected_degree_per_degree_class[d_i] += prob_conn * class_sizes[j];
          random_ensemble_expected_degree_per_degree_class[d_j] += prob_conn * class_sizes[i];
        }
      }

      keep_going = false;
      for(const int d : degree_classes)
      {
        if(std::fabs(random_ensemble_expected_degree_per_degree_class[d] - d) > NUMERICAL_CONVERGENCE_THRESHOLD_1)
        {
          keep_going = true;
          break;
        }
      }

      if(keep_going)
      {
        for(const int d : degree_classes)
        {
          random_ensemble_kappa_per_degree_class[d] += (d - random_ensemble_expected_degree_per_degree_class[d]) * uniform_01(engine);
          random_ensemble_kappa_per_degree_class[d] = std::fabs(random_ensemble_kappa_per_degree_class[d]);
        }
      }
      ++cnt;
    }

    if(cnt >= KAPPA_MAX_NB_ITER_CONV)
    {
      if(!QUIET_MODE)
      {
        std::clog << std::endl;
        std::clog << TAB << "WARNING: maximum number of iterations reached before convergence. This limit can be"  << std::endl;
        std::clog << TAB << "         adjusted by setting the parameters KAPPA_MAX_NB_ITER_CONV to desired value." << std::endl;
        std::clog << TAB << std::fixed << std::setw(11) << " " << " ";
      }
    }
    return;
  }

  const double radius = compute_radius(dim, nb_vertices);
  mu = calculate_mu(dim);

  for(const int d : degree_classes)
  {
    random_ensemble_kappa_per_degree_class[d] = d;
  }

  int cnt = 0;
  bool keep_going = true;
  while(keep_going && (cnt < KAPPA_MAX_NB_ITER_CONV_2))
  {
    for(const int d : degree_classes)
    {
      random_ensemble_expected_degree_per_degree_class[d] = 0.0;
    }

    for(int i = 0; i < nb_degree_classes; ++i)
    {
      const int d_i = degree_classes[i];
      const double kappa_i = random_ensemble_kappa_per_degree_class[d_i];

      const int d_j0 = degree_classes[i];
      const double kappa_j0 = random_ensemble_kappa_per_degree_class[d_j0];
      double prob_conn = compute_integral_expected_degree_dimensions(dim, radius, mu, beta, kappa_i, kappa_j0);
      random_ensemble_expected_degree_per_degree_class[d_i] += prob_conn * (class_sizes[i] - 1.0);

      for(int j = i + 1; j < nb_degree_classes; ++j)
      {
        const int d_j = degree_classes[j];
        const double kappa_j = random_ensemble_kappa_per_degree_class[d_j];
        prob_conn = compute_integral_expected_degree_dimensions(dim, radius, mu, beta, kappa_i, kappa_j);
        random_ensemble_expected_degree_per_degree_class[d_i] += prob_conn * class_sizes[j];
        random_ensemble_expected_degree_per_degree_class[d_j] += prob_conn * class_sizes[i];
      }
    }

    keep_going = false;
    for(const int d : degree_classes)
    {
      if(std::fabs(random_ensemble_expected_degree_per_degree_class[d] - d) > NUMERICAL_CONVERGENCE_THRESHOLD_1)
      {
        keep_going = true;
        break;
      }
    }

    if(keep_going)
    {
      for(const int d : degree_classes)
      {
        random_ensemble_kappa_per_degree_class[d] += (d - random_ensemble_expected_degree_per_degree_class[d]) * uniform_01(engine);
        random_ensemble_kappa_per_degree_class[d] = std::fabs(random_ensemble_kappa_per_degree_class[d]);
      }
    }
    ++cnt;
  }

  if(cnt >= KAPPA_MAX_NB_ITER_CONV_2)
  {
    if(!QUIET_MODE)
    {
      std::clog << std::endl;
      std::clog << TAB << "WARNING: maximum number of iterations reached before convergence. This limit can be"  << std::endl;
      std::clog << TAB << "         adjusted by setting the parameters KAPPA_MAX_NB_ITER_CONV_2 to desired value." << std::endl;
      std::clog << TAB << std::fixed << std::setw(11) << " " << " ";
    }
  }
}

void embeddingSD_t::infer_kappas_given_beta_for_degree_class()
{
  infer_kappas_given_beta_for_degree_class(1);
}

void embeddingSD_t::infer_parameters(int dim)
{
  if(dim == 1)
  {
    if(!QUIET_MODE) { std::clog << "Inferring parameters..."; }
    if(!CUSTOM_BETA)
    {
      if (!QUIET_MODE) {
        std::clog << std::endl;
        std::clog << TAB;
        std::clog << std::fixed << std::setw(11) << "beta" << " ";
        std::clog << std::fixed << std::setw(20) << "avg. clustering" << " \n";
      }

      beta = 2 + uniform_01(engine);

      double beta_max = -1;
      double beta_min = 1;
      random_ensemble_average_clustering = 10;
      while( true )
      {
        if(!QUIET_MODE) {
          std::clog << TAB;
          std::clog << std::fixed << std::setw(11) << beta << " ";
          std::clog.flush();
        }

        infer_kappas_given_beta_for_degree_class();

        build_cumul_dist_for_mc_integration();

        compute_random_ensemble_clustering();
        if(!QUIET_MODE) { std::clog << std::fixed << std::setw(20) << random_ensemble_average_clustering << " \n"; }

        if( std::fabs(random_ensemble_average_clustering - average_clustering) < NUMERICAL_CONVERGENCE_THRESHOLD_1 )
          break;

        if(random_ensemble_average_clustering > average_clustering)
        {
          beta_max = beta;
          beta = (beta_max + beta_min) / 2;
          if(beta < BETA_ABS_MIN)
          {
            if(!QUIET_MODE)
              std::clog << "WARNING: value too close to 1, using beta = " << std::fixed << std::setw(11) << beta << ".\n";
            break;
          }
        }
        else
        {
          beta_min = beta;
          if(beta_max == -1)
            beta *= 1.5;
          else
            beta = (beta_max + beta_min) / 2;
        }
        if(beta > BETA_ABS_MAX)
        {
          if(!QUIET_MODE)
            std::clog << "WARNING: value too high, using beta = " << std::fixed << std::setw(11) << beta << ".\n";
          break;
        }
      }
    }
    else
    {

      infer_kappas_given_beta_for_degree_class();

      build_cumul_dist_for_mc_integration();

      compute_random_ensemble_clustering();
    }

    compute_random_ensemble_average_degree();

    kappa.clear();
    kappa.resize(nb_vertices);
    for(int v=0; v<nb_vertices; ++v)
      kappa[v] = random_ensemble_kappa_per_degree_class[degree[v]];

    if(!QUIET_MODE) {
      if(!CUSTOM_BETA)
        std::clog << "                       ";
      std::clog << "...............................................................done."                                         << std::endl;
      std::clog                                                                                                                   << std::endl;
      std::clog << "Inferred ensemble (random positions)"                                                                         << std::endl;
      std::clog << TAB << "Average degree:                 " << random_ensemble_average_degree                                    << std::endl;
      std::clog << TAB << "Minimum degree:                 " << random_ensemble_expected_degree_per_degree_class.begin()->first   << std::endl;
      std::clog << TAB << "Maximum degree:                 " << (--random_ensemble_expected_degree_per_degree_class.end())->first << std::endl;
      std::clog << TAB << "Average clustering:             " << random_ensemble_average_clustering                                << std::endl;
      std::clog << TAB << "Parameters"                                                                                            << std::endl;
      if(!CUSTOM_BETA)
        std::clog << TAB << "  - beta:                       " << beta                                                            << std::endl;
      else
        std::clog << TAB << "  - beta:                       " << beta  << " (custom)"                                            << std::endl;
      std::clog << TAB << "  - mu:                         " << mu                                                                << std::endl;
      std::clog << TAB << "  - radius_S1 (R):              " << nb_vertices / (2 * PI)                                            << std::endl;
      std::clog                                                                                                                   << std::endl;
    }

    cumul_prob_kgkp.clear();
    degree2vertices.clear();
    random_ensemble_expected_degree_per_degree_class.clear();
    return;
  }

  if(!QUIET_MODE) { std::clog << "Inferring parameters..."; }

  if (!QUIET_MODE) {
    std::clog << std::endl;
    std::clog << TAB;
    std::clog << std::fixed << std::setw(11) << "beta" << " ";
    std::clog << std::fixed << std::setw(20) << "avg. clustering" << " \n";
  }

  const double BETA_ABS_MIN_DIM = dim + 0.01;
  const double BETA_ABS_MAX_DIM = dim + 100;

  if (!CUSTOM_BETA) {

    beta = dim + uniform_01(engine);

    double beta_max = -1;
    double beta_min = dim;
    random_ensemble_average_clustering = 10;

    while(true) {
      if(!QUIET_MODE) {
        std::clog << TAB;
        std::clog << std::fixed << std::setw(11) << beta << " ";
        std::clog.flush();
      }

      infer_kappas_given_beta_for_degree_class(dim);

      build_cumul_dist_for_mc_integration(dim);

      compute_random_ensemble_clustering(dim);
      if(!QUIET_MODE) { std::clog << std::fixed << std::setw(20) << random_ensemble_average_clustering << " \n"; }

      if (std::fabs(random_ensemble_average_clustering - average_clustering) < NUMERICAL_CONVERGENCE_THRESHOLD_1)
        break;

      if(random_ensemble_average_clustering > average_clustering)
      {
        beta_max = beta;
        beta = (beta_max + beta_min) / 2;
        if(beta < BETA_ABS_MIN_DIM)
        {
          beta = BETA_ABS_MIN_DIM;
          infer_kappas_given_beta_for_degree_class(dim);
          if(!QUIET_MODE)
            std::clog << "WARNING: value too close to D, using beta = " << std::fixed << std::setw(11) << beta << ".\n";
          break;
        }
      }
      else
      {
        beta_min = beta;
        if(beta_max == -1)
          beta *= 1.5;
        else
          beta = (beta_max + beta_min) / 2;
      }
      if(beta > BETA_ABS_MAX_DIM)
      {
        if(!QUIET_MODE)
          std::clog << "WARNING: value too high, using beta = " << std::fixed << std::setw(11) << beta << ".\n";
        break;
      }
    }
  } else {
    infer_kappas_given_beta_for_degree_class(dim);
    build_cumul_dist_for_mc_integration(dim);
    compute_random_ensemble_clustering(dim);
  }

  compute_random_ensemble_average_degree();

  kappa.clear();
  kappa.resize(nb_vertices);
  for(int v=0; v < nb_vertices; ++v)
    kappa[v] = random_ensemble_kappa_per_degree_class[degree[v]];

  const auto radius = compute_radius(dim, nb_vertices);
  if(!QUIET_MODE) {
    if(!CUSTOM_BETA)
      std::clog << "                       ";
    std::clog << "...............................................................done."                                         << std::endl;
    std::clog                                                                                                                   << std::endl;
    std::clog << "Inferred ensemble (random positions)"                                                                         << std::endl;
    std::clog << TAB << "Average degree:                 " << random_ensemble_average_degree                                    << std::endl;
    std::clog << TAB << "Minimum degree:                 " << random_ensemble_expected_degree_per_degree_class.begin()->first   << std::endl;
    std::clog << TAB << "Maximum degree:                 " << (--random_ensemble_expected_degree_per_degree_class.end())->first << std::endl;
    std::clog << TAB << "Average clustering:             " << random_ensemble_average_clustering                                << std::endl;
    std::clog << TAB << "Parameters"                                                                                            << std::endl;
    if(!CUSTOM_BETA)
      std::clog << TAB << "  - beta:                       " << beta                                                            << std::endl;
    else
      std::clog << TAB << "  - beta:                       " << beta  << " (custom)"                                            << std::endl;
    std::clog << TAB << "  - mu:                           " << mu                                                              << std::endl;
    std::clog << TAB << "  - radius_S^D (R):               " << radius                              << std::endl;
    std::clog                                                                                                                   << std::endl;
  }

  d_positions.clear();
  d_positions.resize(nb_vertices);
  for (int i=0; i<nb_vertices; ++i)
    d_positions[i] = generate_random_d_vector(dim, radius);
}

void embeddingSD_t::infer_parameters()
{
  infer_parameters(1);
}


#endif
