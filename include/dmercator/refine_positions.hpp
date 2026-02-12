#ifndef DMERCATOR_REFINE_POSITIONS_HPP
#define DMERCATOR_REFINE_POSITIONS_HPP

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
