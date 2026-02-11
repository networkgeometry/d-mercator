#ifndef DMERCATOR_REFINE_REFINEMENT_HPP
#define DMERCATOR_REFINE_REFINEMENT_HPP

double embeddingSD_t::compute_pairwise_loglikelihood(int dim, int v1, const std::vector<double> &pos1, int v2, const std::vector<double> &pos2, bool neighbors, double radius)
{

  if(v1 == v2)
    return 0;
  const auto dtheta = compute_angle_d_vectors(pos1, pos2);
  const auto chi = radius * dtheta / std::pow(mu * kappa[v1] * kappa[v2], 1.0 / dim);
  const auto prob = 1 / (1 + std::pow(chi, beta));
  if (neighbors) {
    return std::log(prob);
  } else {
    return std::log(1 - prob);
  }
}

double embeddingSD_t::compute_pairwise_loglikelihood(int v1, double t1, int v2, double t2, bool neighbors)
{

  if(v1 == v2)
  {
    return 0;
  }

  double da = PI - std::fabs(PI - std::fabs(t1 - t2));

  double fraction = (nb_vertices * da) / (2 * PI * mu * kappa[v1] * kappa[v2]);

  if(neighbors)
  {
    return -beta * std::log(fraction);

  }
  else
  {
    return -std::log(1 + std::pow(fraction, -beta));
  }
}


void embeddingSD_t::find_initial_ordering(std::vector<std::vector<double>> &positions, int dim)
{
  const auto radius = compute_radius(dim, nb_vertices);
  mu = calculate_mu(dim);
  if(!QUIET_MODE) { std::clog << std::endl << TAB << "Building the weights matrix..."; }

  Eigen::SparseMatrix<double> L(nb_vertices_degree_gt_one, nb_vertices_degree_gt_one);
  L.reserve(Eigen::VectorXi::Constant(nb_vertices_degree_gt_one, 3));

  std::vector<int> newID(nb_vertices, -1);
  int n=0;
  for(int v=0; v<nb_vertices; ++v)
  {
    if(degree[v] > 1)
    {
      newID[v] = n;
      ++n;
    }
  }
  if(n != nb_vertices_degree_gt_one)
    std::cout << "There is something wrong here." << std::endl;

  double k1, k2, expected_distance, val, t(0), norm(0);
  for(int v1(0), v2, d1, d2, n1, n2; v1<nb_vertices; ++v1)
  {
    n1 = newID[v1];
    if(n1 != -1)
    {

      d1 = degree[v1];
      k1 = random_ensemble_kappa_per_degree_class[d1];

      const auto &neighbors = adjacency_flat_list[v1];
      for(const int v2_candidate : neighbors)
      {
        v2 = v2_candidate;
        n2 = newID[v2];
        if(n2 != -1)
        {

          if(v1 < v2)
          {

            d2 = degree[v2];
            k2 = random_ensemble_kappa_per_degree_class[d2];

            const auto top_expected_distance = compute_integral_expected_theta(dim, radius, mu, beta, k1, k2);
            const auto bottom_expected_distance = compute_integral_expected_degree_dimensions(dim, radius, mu, beta, k1, k2);
            expected_distance = top_expected_distance / bottom_expected_distance;
            if(expected_distance < 0 || expected_distance > PI)
            {

              std::cerr << "Warning. Expected angular distance out of range." << std::endl;
              std::terminate();
            }

            expected_distance = 2 * std::sin(expected_distance / 2);

            L.insert(n1, n2) = expected_distance;
            L.insert(n2, n1) = expected_distance;

            t += 2 * expected_distance * expected_distance;
            norm += 2;
          }
        }
      }
    }
  }
  t /= norm;

  double value;
  std::vector<double> strength(nb_vertices_degree_gt_one, 0);
  for(int k(0), kk(L.outerSize()), v1, v2; k<kk; ++k)
  {
    for(Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
    {
      v1 = it.row();
      v2 = it.col();
      value = it.value();
      value = std::exp(-1 * value * value / t);
      L.coeffRef(v1, v2) = value;
      strength[v1] += value;
    }
  }

  for(int k(0), kk(L.outerSize()), v1, v2; k<kk; ++k)
  {
    for(Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
    {
      v1 = it.row();
      v2 = it.col();
      value = it.value();
      L.coeffRef(v1, v2) = -1 * value / strength[v1];
    }
  }

  for(int v1(0); v1<nb_vertices_degree_gt_one; ++v1)
    L.insert(v1, v1) = 1;

  if(!QUIET_MODE) { std::clog << " Matrix built." << std::endl; }

  Spectra::SparseGenMatProd<double> op(L);

  Eigen::MatrixXcd evectors;

  int ncv = 10 + dim;

  if(!QUIET_MODE) { std::clog << std::endl; }
  bool keep_going = true;
  while(keep_going)
  {

    Spectra::GenEigsSolver< double, Spectra::SMALLEST_MAGN, Spectra::SparseGenMatProd<double> > eigs(&op, dim + 2, ncv);

    if(!QUIET_MODE) { std::clog << TAB << "Computing eigenvectors using Spectra parameter ncv = " << ncv << "..."; }
    eigs.init();
    int nconv = eigs.compute();

    evectors = eigs.eigenvectors();

    if(eigs.info() != Spectra::SUCCESSFUL)
    {

      if(ncv == nb_vertices)
      {
        std::cerr << std::endl << "The algorithm computing the eigenvectors (Spectra library) cannot converge at all... Exiting." << std::endl << std::endl;
        std::terminate();
      }
      if(!QUIET_MODE) { std::clog << " Convergence not reached." << std::endl; }

      ncv = std::pow(ncv, 1.5);
      if(ncv > nb_vertices)
      {
        ncv = nb_vertices;
      }
    }
    else
      keep_going = false;
  }
  if(!QUIET_MODE) { std::clog << " Convergence reached." << std::endl; }

  std::vector<std::pair<std::vector<double>, int>> ordering_set;
  positions.clear();
  positions.resize(nb_vertices);
  ordering_set.clear();
  for(int v1(0), n1; v1<nb_vertices; ++v1) {
    n1 = newID[v1];
    if(n1 != -1)
    {
      std::vector<double> pos;
      double norm = 0;
      for (int i=0; i<dim+1; ++i) {
        const auto v = evectors(n1, i).real();
        pos.push_back(v);
      }

      normalize_and_rescale_vector(pos, radius);
      positions[v1] = pos;
      ordering_set.push_back(std::make_pair(pos, v1));
    }
  }

  std::vector<int> list_neigh_degree_one;
  std::vector<double> axis(dim + 1, 0);
  axis[0] = 1;
  auto it2 = ordering_set.begin();
  auto end2 = ordering_set.end();
  for (int v1; it2!=end2; ++it2) {

    v1 = it2->second;

    list_neigh_degree_one.clear();
    list_neigh_degree_one.reserve(degree[v1]);
    const auto &neighbors = adjacency_flat_list[v1];
    for(int v2 : neighbors) {
      if(degree[v2] == 1)
        list_neigh_degree_one.push_back(v2);
    }
    if(list_neigh_degree_one.empty())
    {
      continue;
    }

    const auto kappa2 = kappa[v1];
    auto vec1 = positions[v1];
    normalize_and_rescale_vector(vec1, 1);
    auto rotation_matrix = compute_rotation_matrix(axis, vec1);
    for (size_t v = 0; v < list_neigh_degree_one.size(); ++v) {

      const auto kappa1 = kappa[list_neigh_degree_one[v]];
      const auto p12 = compute_integral_expected_degree_dimensions(dim, radius, mu, beta, kappa1, kappa2);
      const auto random_angle = draw_random_angular_distance(degree[v1], 1, radius, p12, dim);
      auto new_position = generate_random_d_vector_with_first_coordinate(dim, random_angle, radius);

      auto new_rotated_positions = rotate_vector(rotation_matrix, new_position);

      positions[list_neigh_degree_one[v]] = new_rotated_positions;
    }
  }
}

void embeddingSD_t::find_initial_ordering(std::vector<int> &ordering, std::vector<double> &raw_theta)
{
  if(!QUIET_MODE) { std::clog << std::endl << TAB << "Building the weights matrix..."; }

  Eigen::SparseMatrix<double> L(nb_vertices_degree_gt_one, nb_vertices_degree_gt_one);
  L.reserve(Eigen::VectorXi::Constant(nb_vertices_degree_gt_one, 3));

  std::vector<int> newID(nb_vertices, -1);
  int n=0;
  for(int v=0; v<nb_vertices; ++v)
  {
    if(degree[v] > 1)
    {
      newID[v] = n;
      ++n;
    }
  }
  if(n != nb_vertices_degree_gt_one)
    std::cout << "There is something wrong here." << std::endl;

  double k1, k2, expected_distance, val, t(0), norm(0);
  for(int v1(0), v2, d1, d2, n1, n2; v1<nb_vertices; ++v1)
  {
    n1 = newID[v1];
    if(n1 != -1)
    {

      d1 = degree[v1];
      k1 = random_ensemble_kappa_per_degree_class[d1];

      const auto &neighbors = adjacency_flat_list[v1];
      for(const int v2_candidate : neighbors)
      {
        v2 = v2_candidate;
        n2 = newID[v2];
        if(n2 != -1)
        {

          if(v1 < v2)
          {

            d2 = degree[v2];
            k2 = random_ensemble_kappa_per_degree_class[d2];

            expected_distance = PI * hyp2f1b(beta, -std::pow(nb_vertices / (2.0 * mu * k1 * k2), beta));
            expected_distance /= 2 * hyp2f1a(beta, -std::pow(nb_vertices / (2.0 * mu * k1 * k2), beta));
            if(expected_distance < 0 || expected_distance > PI)
            {

              std::cerr << "Warning. Expected angular distance out of range." << std::endl;
              std::terminate();
            }

            expected_distance = 2 * std::sin(expected_distance / 2);

            L.insert(n1, n2) = expected_distance;
            L.insert(n2, n1) = expected_distance;

            t += 2 * expected_distance * expected_distance;
            norm += 2;
          }
        }
      }
    }
  }
  t /= norm;

  double value;
  std::vector<double> strength(nb_vertices_degree_gt_one, 0);
  for(int k(0), kk(L.outerSize()), v1, v2; k<kk; ++k)
  {
    for(Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
    {
      v1 = it.row();
      v2 = it.col();
      value = it.value();
      value = std::exp(-1 * value * value / t);
      L.coeffRef(v1, v2) = value;
      strength[v1] += value;
    }
  }

  for(int k(0), kk(L.outerSize()), v1, v2; k<kk; ++k)
  {
    for(Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
    {
      v1 = it.row();
      v2 = it.col();
      value = it.value();
      L.coeffRef(v1, v2) = -1 * value / strength[v1];
    }
  }

  for(int v1(0); v1<nb_vertices_degree_gt_one; ++v1)
    L.insert(v1, v1) = 1;

  if(!QUIET_MODE) { std::clog << " Matrix built." << std::endl; }

  Spectra::SparseGenMatProd<double> op(L);

  Eigen::MatrixXcd evectors;

  int ncv = 7;

  if(!QUIET_MODE) { std::clog << std::endl; }
  bool keep_going = true;
  while(keep_going)
  {

    Spectra::GenEigsSolver< double, Spectra::SMALLEST_MAGN, Spectra::SparseGenMatProd<double> > eigs(&op, 3, ncv);

    if(!QUIET_MODE) { std::clog << TAB << "Computing eigenvectors using Spectra parameter ncv = " << ncv << "..."; }
    eigs.init();
    int nconv = eigs.compute();

    evectors = eigs.eigenvectors();

    if(eigs.info() != Spectra::SUCCESSFUL)
    {

      if(ncv == nb_vertices)
      {
        std::cerr << std::endl << "The algorithm computing the eigenvectors (Spectra library) cannot converge at all... Exiting." << std::endl << std::endl;
        std::terminate();
      }
      if(!QUIET_MODE) { std::clog << " Convergence not reached." << std::endl; }

      ncv = std::pow(ncv, 1.5);
      if(ncv > nb_vertices)
      {
        ncv = nb_vertices;
      }
    }
    else
      keep_going = false;
  }
  if(!QUIET_MODE) { std::clog << " Convergence reached." << std::endl; }

  raw_theta.clear();
  raw_theta.resize(nb_vertices);

  double angle;
  std::set< std::pair<double, int> > ordering_set;
  ordering_set.clear();
  for(int v1(0), n1; v1<nb_vertices; ++v1)
  {
    n1 = newID[v1];
    if(n1 != -1)
    {

      angle = std::atan2(evectors(n1, 0).real(), evectors(n1, 1).real()) + PI;
      ordering_set.insert(std::make_pair(angle, v1));
      raw_theta[v1] = angle;
    }
  }

  ordering.clear();
  ordering.reserve(nb_vertices);
  std::vector<int> list_neigh_degree_one;
  auto it2 = ordering_set.begin();
  auto end2 = ordering_set.end();
  for(int v1; it2!=end2; ++it2)
  {

    v1 = it2->second;

    list_neigh_degree_one.clear();
    list_neigh_degree_one.reserve(degree[v1]);
    const auto &neighbors = adjacency_flat_list[v1];
    for(int v2 : neighbors)
    {
      if(degree[v2] == 1)
        list_neigh_degree_one.push_back(v2);
    }

    for(int v(0), vv(list_neigh_degree_one.size() / 2); v<vv; ++v)
    {
      ordering.push_back(list_neigh_degree_one[v]);
      raw_theta[list_neigh_degree_one[v]] = raw_theta[v1];
    }
    ordering.push_back(v1);
    for(int v(list_neigh_degree_one.size() / 2), vv(list_neigh_degree_one.size()); v<vv; ++v)
    {
      ordering.push_back(list_neigh_degree_one[v]);
      raw_theta[list_neigh_degree_one[v]] = raw_theta[v1];
    }
  }
}


void embeddingSD_t::infer_initial_positions(int dim)
{
  if(!QUIET_MODE) {
    std::clog << "Finding initial positions/ordering...";
    std::clog.flush();
  }
  if(dim == 1)
  {

    std::vector<int> ordering;
    std::vector<double> raw_theta;
    find_initial_ordering(ordering, raw_theta);

    if(ordering.size() != nb_vertices)
      std::cout << TAB << "WARNING: All degree-one vertices have not all been reinserted. Does the original edgelist have more than one connected component." << std::endl;

    theta.clear();
    theta.resize(nb_vertices);

    int v0, v1, n(0);
    double factor, int1, int2, b, tmp;
    double norm = 0;
    double possible_dtheta1, possible_dtheta2;
    double dx = PI / EXP_DIST_NB_INTEGRATION_STEPS;
    double prefactor = nb_vertices / (2 * PI * mu);
    double avg_gap = 2 * PI / nb_vertices;
    auto it = ordering.begin();
    auto end = ordering.end();

    v0 = *ordering.rbegin();
    for(; it!=end; ++it, ++n)
    {

      v1 = *it;

      int1 = 0;
      int2 = 0;
      tmp = 0;
      factor = prefactor / ( random_ensemble_kappa_per_degree_class[degree[v0]] * random_ensemble_kappa_per_degree_class[degree[v1]] );

      b = beta;
      if(adjacency_list[v0].find(v1) == adjacency_list[v0].end())
      {
        b = -beta;
      }

      if(b > 0)
      {
        int2 += 0.5;
      }

      tmp = std::exp(-PI / avg_gap) / ( 1 + std::pow(factor * PI, b) );
      int1 += PI * tmp / 2;
      int2 += tmp / 2;

      for(double da = dx; da < PI; da += dx)
      {
        tmp = std::exp(-da / avg_gap) / ( 1 + std::pow(factor * da, b) );
        int1 += da * tmp;
        int2 += tmp;
      }

      possible_dtheta1 = int1 / int2;
      possible_dtheta2 = PI - std::fabs(PI - std::fabs(raw_theta[v1] - raw_theta[v0]));
      if(possible_dtheta1 > possible_dtheta2)
      {
        norm += possible_dtheta1;
      }
      else
      {
        norm += possible_dtheta2;
      }

      theta[v1] = norm;

      v0 = v1;
    }
    if(!QUIET_MODE) { std::clog << std::endl << TAB << "Sum of the angular positions (before adjustment): " << norm << std::endl; }

    norm /= 2 * PI;
    for(int v(0); v<nb_vertices; ++v)
    {
      theta[v] /= norm;
    }

    theta[v1] = 0;
    if(!QUIET_MODE)
      std::clog << "                                     .................................................done.\n\n";
    return;
  }

  find_initial_ordering(d_positions, dim);
}

void embeddingSD_t::infer_initial_positions()
{
  infer_initial_positions(1);
}


int embeddingSD_t::refine_angle(int v1)
{
  if(!OPTIMIZED_PERF_MODE)
  {
    int has_moved = 0;
    double best_angle = theta[v1];
    std::vector<int> neighbors(adjacency_list[v1].begin(), adjacency_list[v1].end());

    const double prefactor = nb_vertices / (2 * PI * mu);
    const double prefactor_over_kappa_v1 = prefactor / kappa[v1];
    std::vector<double> pair_prefactor(nb_vertices);
    for(int v2(0); v2<nb_vertices; ++v2)
    {
      pair_prefactor[v2] = prefactor_over_kappa_v1 / kappa[v2];
    }

    auto compute_local_loglikelihood = [&](double angle) -> double
    {
      double local_loglikelihood = 0;
      for(int v2(0); v2<nb_vertices; ++v2)
      {
        if(v2 == v1)
        {
          continue;
        }
        const double da = PI - std::fabs(PI - std::fabs(angle - theta[v2]));
        const double fraction = pair_prefactor[v2] * da;
        local_loglikelihood += -std::log(1 + std::pow(fraction, -beta));
      }
      for(const int v2 : neighbors)
      {
        const double da = PI - std::fabs(PI - std::fabs(angle - theta[v2]));
        const double fraction = pair_prefactor[v2] * da;
        local_loglikelihood += -beta * std::log(fraction);
      }
      return local_loglikelihood;
    };

    double best_loglikelihood = compute_local_loglikelihood(best_angle);

    double da;
    double sum_sin_theta = 0;
    double sum_cos_theta = 0;
    for(const int v2 : neighbors)
    {
      const double t2 = theta[v2];
      const double inv_k2_squared = 1.0 / (kappa[v2] * kappa[v2]);

      sum_sin_theta += std::sin(t2) * inv_k2_squared;
      sum_cos_theta += std::cos(t2) * inv_k2_squared;
    }
    double average_theta = std::atan2(sum_sin_theta, sum_cos_theta) + PI;
    while(average_theta > (2 * PI))
      average_theta = average_theta - (2 * PI);
    while(average_theta < 0)
      average_theta = average_theta + (2 * PI);

    double max_angle = MIN_TWO_SIGMAS_NORMAL_DIST;
    for(const int v2 : neighbors)
    {
      da = PI - std::fabs(PI - std::fabs(average_theta - theta[v2]));
      if(da > max_angle)
      {
        max_angle = da;
      }
    }
    max_angle /= 2;

    int _nb_new_angles_to_try = MIN_NB_ANGLES_TO_TRY * std::max(1.0, std::log(static_cast<double>(nb_vertices)));
    for(int e(0); e<_nb_new_angles_to_try; ++e)
    {

      double tmp_angle = (normal_01(engine) * max_angle) + average_theta;
      while(tmp_angle > (2 * PI))
        tmp_angle = tmp_angle - (2 * PI);
      while(tmp_angle < 0)
        tmp_angle = tmp_angle + (2 * PI);

      const double tmp_loglikelihood = compute_local_loglikelihood(tmp_angle);

      if(tmp_loglikelihood > best_loglikelihood)
      {
        best_loglikelihood = tmp_loglikelihood;
        best_angle = tmp_angle;
        has_moved = 1;
      }
    }

    theta[v1] = best_angle;
    return has_moved;
  }

  int has_moved = 0;
  double best_angle = theta[v1];
  const auto &neighbors = adjacency_flat_list[v1];

  const double prefactor = nb_vertices / (2 * PI * mu);
  const double prefactor_over_kappa_v1 = prefactor / kappa[v1];
  if(scratch_pair_prefactor.size() != static_cast<size_t>(nb_vertices))
  {
    scratch_pair_prefactor.assign(nb_vertices, 0.0);
  }
  auto &pair_prefactor = scratch_pair_prefactor;
  for(int v2(0); v2<nb_vertices; ++v2)
  {
    pair_prefactor[v2] = prefactor_over_kappa_v1 / kappa[v2];
  }

  auto compute_local_loglikelihood = [&](double angle) -> double
  {
    double local_loglikelihood = 0;
    for(int v2(0); v2<nb_vertices; ++v2)
    {
      if(v2 == v1)
      {
        continue;
      }
      const double da = PI - std::fabs(PI - std::fabs(angle - theta[v2]));
      const double fraction = pair_prefactor[v2] * da;
      local_loglikelihood += -std::log(1 + std::pow(fraction, -beta));
    }
    for(const int v2 : neighbors)
    {
      const double da = PI - std::fabs(PI - std::fabs(angle - theta[v2]));
      const double fraction = pair_prefactor[v2] * da;
      local_loglikelihood += -beta * std::log(fraction);
    }
    return local_loglikelihood;
  };

  double best_loglikelihood = compute_local_loglikelihood(best_angle);

  double da;
  double sum_sin_theta = 0;
  double sum_cos_theta = 0;
  for(const int v2 : neighbors)
  {
    const double t2 = theta[v2];
    const double inv_k2_squared = 1.0 / (kappa[v2] * kappa[v2]);

    sum_sin_theta += std::sin(t2) * inv_k2_squared;
    sum_cos_theta += std::cos(t2) * inv_k2_squared;
  }
  double average_theta = std::atan2(sum_sin_theta, sum_cos_theta) + PI;
  while(average_theta > (2 * PI))
    average_theta = average_theta - (2 * PI);
  while(average_theta < 0)
    average_theta = average_theta + (2 * PI);

  double max_angle = MIN_TWO_SIGMAS_NORMAL_DIST;
  for(const int v2 : neighbors)
  {
    da = PI - std::fabs(PI - std::fabs(average_theta - theta[v2]));
    if(da > max_angle)
    {
      max_angle = da;
    }
  }
  max_angle /= 2;

  int _nb_new_angles_to_try = MIN_NB_ANGLES_TO_TRY * std::max(1.0, std::log(static_cast<double>(nb_vertices)));
  for(int e(0); e<_nb_new_angles_to_try; ++e)
  {

    double tmp_angle = (normal_01(engine) * max_angle) + average_theta;
    while(tmp_angle > (2 * PI))
      tmp_angle = tmp_angle - (2 * PI);
    while(tmp_angle < 0)
      tmp_angle = tmp_angle + (2 * PI);

    const double tmp_loglikelihood = compute_local_loglikelihood(tmp_angle);

    if(tmp_loglikelihood > best_loglikelihood)
    {
      best_loglikelihood = tmp_loglikelihood;
      best_angle = tmp_angle;
      has_moved = 1;
    }
  }

  theta[v1] = best_angle;

  return has_moved;
}

int embeddingSD_t::refine_angle(int dim, int v1, double radius)
{
  if(!OPTIMIZED_PERF_MODE)
  {
    int has_moved = 0;
    auto best_position = d_positions[v1];
    std::vector<int> neighbors(adjacency_list[v1].begin(), adjacency_list[v1].end());

    const double inv_dim = 1.0 / static_cast<double>(dim);
    std::vector<double> pair_prefactor(nb_vertices, 0);
    for(int v2(0); v2<nb_vertices; ++v2)
    {
      if(v2 == v1)
      {
        continue;
      }
      pair_prefactor[v2] = radius / std::pow(mu * kappa[v1] * kappa[v2], inv_dim);
    }

    auto compute_local_loglikelihood = [&](const std::vector<double> &position) -> double
    {
      double local_loglikelihood = 0;
      for(int v2(0); v2<nb_vertices; ++v2)
      {
        if(v2 == v1)
        {
          continue;
        }
        const double dtheta = compute_angle_d_vectors(position, d_positions[v2]);
        const double chi = pair_prefactor[v2] * dtheta;
        const double prob = 1 / (1 + std::pow(chi, beta));
        local_loglikelihood += std::log(1 - prob);
      }
      for(const int v2 : neighbors)
      {
        const double dtheta = compute_angle_d_vectors(position, d_positions[v2]);
        const double chi = pair_prefactor[v2] * dtheta;
        const double prob = 1 / (1 + std::pow(chi, beta));
        local_loglikelihood += std::log(prob);
      }
      return local_loglikelihood;
    };

    double best_loglikelihood = compute_local_loglikelihood(best_position);

    std::vector<double> mean_vector(dim + 1, 0);
    for(const int v2 : neighbors)
    {
      const auto &pos2 = d_positions[v2];
      const double inv_k2_squared = 1.0 / (kappa[v2] * kappa[v2]);
      for (int i=0; i<dim+1; ++i)
        mean_vector[i] += pos2[i] * inv_k2_squared;
    }
    normalize_and_rescale_vector(mean_vector, radius);

    double max_angle = MIN_TWO_SIGMAS_NORMAL_DIST;
    for(const int v2 : neighbors)
    {
      const double dtheta = compute_angle_d_vectors(mean_vector, d_positions[v2]);
      if(dtheta > max_angle)
        max_angle = dtheta;
    }
    max_angle /= 2;

    int _nb_new_angles_to_try = MIN_NB_ANGLES_TO_TRY * std::max(1.0, std::log(static_cast<double>(nb_vertices)));
    std::vector<double> proposed_position(dim + 1, 0);
    const double inv_radius = 1.0 / radius;
    for(int e(0); e<_nb_new_angles_to_try; ++e)
    {

      for (int i=0; i<dim+1; ++i)
        proposed_position[i] = max_angle * normal_01(engine) + mean_vector[i] * inv_radius;
      normalize_and_rescale_vector(proposed_position, radius);

      const double tmp_loglikelihood = compute_local_loglikelihood(proposed_position);

      if(tmp_loglikelihood > best_loglikelihood)
      {
        best_loglikelihood = tmp_loglikelihood;
        best_position = proposed_position;
        has_moved = 1;
      }
    }

    d_positions[v1] = best_position;
    return has_moved;
  }

  int has_moved = 0;
  auto best_position = d_positions[v1];
  const auto &neighbors = adjacency_flat_list[v1];

  const double inv_dim = 1.0 / static_cast<double>(dim);
  if(scratch_pair_prefactor.size() != static_cast<size_t>(nb_vertices))
  {
    scratch_pair_prefactor.assign(nb_vertices, 0.0);
  }
  auto &pair_prefactor = scratch_pair_prefactor;
  std::fill(pair_prefactor.begin(), pair_prefactor.end(), 0.0);
  for(int v2(0); v2<nb_vertices; ++v2)
  {
    if(v2 == v1)
    {
      continue;
    }
    pair_prefactor[v2] = radius / std::pow(mu * kappa[v1] * kappa[v2], inv_dim);
  }

  auto compute_local_loglikelihood = [&](const std::vector<double> &position) -> double
  {
    double local_loglikelihood = 0;
    for(int v2(0); v2<nb_vertices; ++v2)
    {
      if(v2 == v1)
      {
        continue;
      }
      const double dtheta = compute_angle_d_vectors(position, d_positions[v2]);
      const double chi = pair_prefactor[v2] * dtheta;
      const double prob = 1 / (1 + std::pow(chi, beta));
      local_loglikelihood += std::log(1 - prob);
    }
    for(const int v2 : neighbors)
    {
      const double dtheta = compute_angle_d_vectors(position, d_positions[v2]);
      const double chi = pair_prefactor[v2] * dtheta;
      const double prob = 1 / (1 + std::pow(chi, beta));
      local_loglikelihood += std::log(prob);
    }
    return local_loglikelihood;
  };

  double best_loglikelihood = compute_local_loglikelihood(best_position);

  if(scratch_mean_vector.size() != static_cast<size_t>(dim + 1))
  {
    scratch_mean_vector.assign(dim + 1, 0.0);
  }
  else
  {
    std::fill(scratch_mean_vector.begin(), scratch_mean_vector.end(), 0.0);
  }
  auto &mean_vector = scratch_mean_vector;
  for(const int v2 : neighbors)
  {
    const auto &pos2 = d_positions[v2];
    const double inv_k2_squared = 1.0 / (kappa[v2] * kappa[v2]);
    for (int i=0; i<dim+1; ++i)
      mean_vector[i] += pos2[i] * inv_k2_squared;
  }
  normalize_and_rescale_vector(mean_vector, radius);

  double max_angle = MIN_TWO_SIGMAS_NORMAL_DIST;
  for(const int v2 : neighbors)
  {
    const double dtheta = compute_angle_d_vectors(mean_vector, d_positions[v2]);
    if(dtheta > max_angle)
      max_angle = dtheta;
  }
  max_angle /= 2;

  int _nb_new_angles_to_try = MIN_NB_ANGLES_TO_TRY * std::max(1.0, std::log(static_cast<double>(nb_vertices)));
  if(scratch_proposed_vector.size() != static_cast<size_t>(dim + 1))
  {
    scratch_proposed_vector.assign(dim + 1, 0.0);
  }
  auto &proposed_position = scratch_proposed_vector;
  const double inv_radius = 1.0 / radius;
  for(int e(0); e<_nb_new_angles_to_try; ++e)
  {

    for (int i=0; i<dim+1; ++i)
      proposed_position[i] = max_angle * normal_01(engine) + mean_vector[i] * inv_radius;
    normalize_and_rescale_vector(proposed_position, radius);

    const double tmp_loglikelihood = compute_local_loglikelihood(proposed_position);

    if(tmp_loglikelihood > best_loglikelihood)
    {
      best_loglikelihood = tmp_loglikelihood;
      best_position = proposed_position;
      has_moved = 1;
    }
  }

  d_positions[v1] = best_position;

  return has_moved;
}

void embeddingSD_t::refine_positions(int dim)
{
  if(!QUIET_MODE) { std::clog << "Refining the positions..."; }
  if(!QUIET_MODE) { std::clog << std::endl; }

  if(dim == 1)
  {
    double start_time, stop_time;
    std::string vertices_range;
    int delta_nb_vertices = nb_vertices / 19.999999;
    if(delta_nb_vertices < 1) { delta_nb_vertices = 1; }
    int width = 2 * (std::log10(nb_vertices) + 1) + 6;
    for(int v_i(0), v_f(0), v_m, n_v; v_f<nb_vertices;)
    {
      v_f = (v_i + delta_nb_vertices);
      v_f = (v_f > nb_vertices) ? nb_vertices : v_f;
      n_v = v_f - v_i;
      start_time = time_since_epoch_in_seconds();
      if(!QUIET_MODE) { vertices_range = "[" + std::to_string(v_i+1) + "," + std::to_string(v_f) + "]..."; }
      if(!QUIET_MODE) { std::clog << TAB << "...of vertices " << std::setw(width) << vertices_range; }
      for(v_m = 0; v_i<v_f; ++v_i)
      {
        v_m += refine_angle( ordered_list_of_vertices[v_i] );
      }
      stop_time = time_since_epoch_in_seconds();
      if(!QUIET_MODE) { std::clog << "...done in " << std::setw(6) << std::fixed << stop_time - start_time << " seconds (" << std::setw(std::log10(delta_nb_vertices) + 1) << v_m << "/" << std::setw(std::log10(delta_nb_vertices) + 1) << n_v << " changed position)" << std::endl; }
    }
    if(!QUIET_MODE) { std::clog << "                         .............................................................done." << std::endl; }
    if(!QUIET_MODE) { std::clog << std::endl; }
    return;
  }

  double start_time, stop_time;
  std::string vertices_range;
  int delta_nb_vertices = nb_vertices / 19.999999;
  if(delta_nb_vertices < 1) { delta_nb_vertices = 1; }
  int width = 2 * (std::log10(nb_vertices) + 1) + 6;
  const auto radius = compute_radius(dim, nb_vertices);
  for(int v_i(0), v_f(0), v_m, n_v; v_f<nb_vertices;)
  {
    v_f = (v_i + delta_nb_vertices);
    v_f = (v_f > nb_vertices) ? nb_vertices : v_f;
    n_v = v_f - v_i;
    start_time = time_since_epoch_in_seconds();
    if(!QUIET_MODE) { vertices_range = "[" + std::to_string(v_i+1) + "," + std::to_string(v_f) + "]..."; }
    if(!QUIET_MODE) { std::clog << TAB << "...of vertices " << std::setw(width) << vertices_range; }
    for(v_m = 0; v_i<v_f; ++v_i)
    {
      v_m += refine_angle(dim, ordered_list_of_vertices[v_i], radius);
    }
    stop_time = time_since_epoch_in_seconds();
    if(!QUIET_MODE) { std::clog << "...done in " << std::setw(6) << std::fixed << stop_time - start_time << " seconds (" << std::setw(std::log10(delta_nb_vertices) + 1) << v_m << "/" << std::setw(std::log10(delta_nb_vertices) + 1) << n_v << " changed position)" << std::endl; }
  }

  if(!QUIET_MODE) { std::clog << "                         .............................................................done." << std::endl; }
  if(!QUIET_MODE) { std::clog << std::endl; }
}

void embeddingSD_t::refine_positions()
{
  refine_positions(1);
}

#endif
