#ifndef DMERCATOR_INFER_INITIAL_POSITIONS_HPP
#define DMERCATOR_INFER_INITIAL_POSITIONS_HPP

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

#endif
