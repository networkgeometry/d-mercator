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

  bool pair_prefactor_ready = false;
  auto ensure_pair_prefactor = [&]() -> const std::vector<double> &
  {
    if(!pair_prefactor_ready)
    {
      if(scratch_pair_prefactor.size() != static_cast<size_t>(nb_vertices))
      {
        scratch_pair_prefactor.assign(nb_vertices, 0.0);
      }
      auto &pair_prefactor = scratch_pair_prefactor;
      for(int v2(0); v2<nb_vertices; ++v2)
      {
        pair_prefactor[v2] = prefactor_over_kappa_v1 / kappa[v2];
      }
      pair_prefactor_ready = true;
    }
    return scratch_pair_prefactor;
  };

  auto compute_local_loglikelihood = [&](double angle) -> double
  {
    const auto &pair_prefactor = ensure_pair_prefactor();
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
  auto &candidate_angles = scratch_candidate_angles;
  candidate_angles.clear();
  candidate_angles.reserve(static_cast<size_t>(_nb_new_angles_to_try) + 1);
  candidate_angles.push_back(best_angle);
  for(int e(0); e<_nb_new_angles_to_try; ++e)
  {
    double tmp_angle = (normal_01(engine) * max_angle) + average_theta;
    while(tmp_angle > (2 * PI))
      tmp_angle = tmp_angle - (2 * PI);
    while(tmp_angle < 0)
      tmp_angle = tmp_angle + (2 * PI);
    candidate_angles.push_back(tmp_angle);
  }

  auto &candidate_scores = scratch_candidate_scores;
  candidate_scores.assign(candidate_angles.size(), 0.0);
#if defined(DMERCATOR_USE_CUDA)
  bool scored_on_gpu = false;
  if(CUDA_MODE && CUDA_REFINEMENT_ACTIVE && likelihood_backend)
  {
    scored_on_gpu = likelihood_backend->score_candidates_s1(v1,
                                                            prefactor,
                                                            beta,
                                                            candidate_angles,
                                                            candidate_scores);
    if(!scored_on_gpu)
    {
      CUDA_REFINEMENT_ACTIVE = false;
      if(!QUIET_MODE)
      {
        std::clog << TAB << "WARNING: CUDA S1 refinement scoring failed; falling back to CPU for remaining vertices. "
                  << likelihood_backend->last_error() << std::endl;
      }
    }
  }
  if(!scored_on_gpu)
#endif
  {
    for(size_t idx = 0; idx < candidate_angles.size(); ++idx)
    {
      candidate_scores[idx] = compute_local_loglikelihood(candidate_angles[idx]);
    }
  }

  double best_loglikelihood = candidate_scores[0];
  for(size_t idx = 1; idx < candidate_angles.size(); ++idx)
  {
    const double tmp_loglikelihood = candidate_scores[idx];
    if(tmp_loglikelihood > best_loglikelihood)
    {
      best_loglikelihood = tmp_loglikelihood;
      best_angle = candidate_angles[idx];
      has_moved = 1;
    }
  }

  theta[v1] = best_angle;
#if defined(DMERCATOR_USE_CUDA)
  if(CUDA_MODE && CUDA_REFINEMENT_ACTIVE && likelihood_backend)
  {
    if(!likelihood_backend->set_theta_single(v1, best_angle))
    {
      CUDA_REFINEMENT_ACTIVE = false;
      if(!QUIET_MODE)
      {
        std::clog << TAB << "WARNING: CUDA S1 refinement state update failed; falling back to CPU for remaining vertices. "
                  << likelihood_backend->last_error() << std::endl;
      }
    }
  }
#endif
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

  const int position_stride = dim + 1;
  const double inv_dim = 1.0 / static_cast<double>(dim);
  bool pair_prefactor_ready = false;
  auto ensure_pair_prefactor = [&]() -> const std::vector<double> &
  {
    if(!pair_prefactor_ready)
    {
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
      pair_prefactor_ready = true;
    }
    return scratch_pair_prefactor;
  };

  auto compute_angle_with_position = [&](const double *pos1, const std::vector<double> &pos2) -> double
  {
    double angle = 0;
    double norm1 = 0;
    double norm2 = 0;
    for(int i = 0; i < position_stride; ++i)
    {
      angle += pos1[i] * pos2[i];
      norm1 += pos1[i] * pos1[i];
      norm2 += pos2[i] * pos2[i];
    }
    norm1 /= std::sqrt(norm1);
    norm2 /= std::sqrt(norm2);

    const auto result = angle / (norm1 * norm2);
    if(std::fabs(result - 1) < NUMERICAL_ZERO)
      return 0;
    return std::acos(result);
  };

  auto compute_local_loglikelihood = [&](const double *position_ptr) -> double
  {
    const auto &pair_prefactor = ensure_pair_prefactor();
    double local_loglikelihood = 0;
    for(int v2(0); v2<nb_vertices; ++v2)
    {
      if(v2 == v1)
      {
        continue;
      }
      const double dtheta = compute_angle_with_position(position_ptr, d_positions[v2]);
      const double chi = pair_prefactor[v2] * dtheta;
      const double prob = 1 / (1 + std::pow(chi, beta));
      local_loglikelihood += std::log(1 - prob);
    }
    for(const int v2 : neighbors)
    {
      const double dtheta = compute_angle_with_position(position_ptr, d_positions[v2]);
      const double chi = pair_prefactor[v2] * dtheta;
      const double prob = 1 / (1 + std::pow(chi, beta));
      local_loglikelihood += std::log(prob);
    }
    return local_loglikelihood;
  };

  if(scratch_mean_vector.size() != static_cast<size_t>(position_stride))
  {
    scratch_mean_vector.assign(position_stride, 0.0);
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
    for (int i = 0; i < position_stride; ++i)
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
  if(scratch_proposed_vector.size() != static_cast<size_t>(position_stride))
  {
    scratch_proposed_vector.assign(position_stride, 0.0);
  }
  auto &proposed_position = scratch_proposed_vector;
  const double inv_radius = 1.0 / radius;

  auto &candidate_positions_flat = scratch_candidate_positions_flat;
  candidate_positions_flat.clear();
  candidate_positions_flat.reserve(static_cast<size_t>(_nb_new_angles_to_try + 1) * static_cast<size_t>(position_stride));
  candidate_positions_flat.insert(candidate_positions_flat.end(), best_position.begin(), best_position.end());
  for(int e(0); e<_nb_new_angles_to_try; ++e)
  {
    for(int i = 0; i < position_stride; ++i)
      proposed_position[i] = max_angle * normal_01(engine) + mean_vector[i] * inv_radius;
    normalize_and_rescale_vector(proposed_position, radius);
    candidate_positions_flat.insert(candidate_positions_flat.end(), proposed_position.begin(), proposed_position.end());
  }

  const size_t nb_candidates = candidate_positions_flat.size() / static_cast<size_t>(position_stride);
  auto &candidate_scores = scratch_candidate_scores;
  candidate_scores.assign(nb_candidates, 0.0);
#if defined(DMERCATOR_USE_CUDA)
  bool scored_on_gpu = false;
  if(CUDA_MODE && CUDA_REFINEMENT_ACTIVE && likelihood_backend)
  {
    auto &candidate_positions_soa = scratch_candidate_positions_soa;
    candidate_positions_soa.assign(candidate_positions_flat.size(), 0.0);
    for(size_t candidate_idx = 0; candidate_idx < nb_candidates; ++candidate_idx)
    {
      const size_t in_offset = candidate_idx * static_cast<size_t>(position_stride);
      for(int axis = 0; axis < position_stride; ++axis)
      {
        candidate_positions_soa[static_cast<size_t>(axis) * nb_candidates + candidate_idx] =
          candidate_positions_flat[in_offset + static_cast<size_t>(axis)];
      }
    }
    scored_on_gpu = likelihood_backend->score_candidates_sd(dim,
                                                            v1,
                                                            radius,
                                                            mu,
                                                            beta,
                                                            NUMERICAL_ZERO,
                                                            candidate_positions_soa,
                                                            candidate_scores);
    if(!scored_on_gpu)
    {
      CUDA_REFINEMENT_ACTIVE = false;
      if(!QUIET_MODE)
      {
        std::clog << TAB << "WARNING: CUDA S^D refinement scoring failed; falling back to CPU for remaining vertices. "
                  << likelihood_backend->last_error() << std::endl;
      }
    }
  }
  if(!scored_on_gpu)
#endif
  {
    for(size_t idx = 0; idx < nb_candidates; ++idx)
    {
      const double *candidate_ptr = candidate_positions_flat.data() + idx * static_cast<size_t>(position_stride);
      candidate_scores[idx] = compute_local_loglikelihood(candidate_ptr);
    }
  }

  double best_loglikelihood = candidate_scores[0];
  for(size_t idx = 1; idx < nb_candidates; ++idx)
  {
    const double tmp_loglikelihood = candidate_scores[idx];
    if(tmp_loglikelihood > best_loglikelihood)
    {
      best_loglikelihood = tmp_loglikelihood;
      const size_t offset = idx * static_cast<size_t>(position_stride);
      std::copy(candidate_positions_flat.begin() + static_cast<std::ptrdiff_t>(offset),
                candidate_positions_flat.begin() + static_cast<std::ptrdiff_t>(offset + position_stride),
                best_position.begin());
      has_moved = 1;
    }
  }

  d_positions[v1] = best_position;
#if defined(DMERCATOR_USE_CUDA)
  if(CUDA_MODE && CUDA_REFINEMENT_ACTIVE && likelihood_backend)
  {
    if(!likelihood_backend->set_position_single(v1, best_position))
    {
      CUDA_REFINEMENT_ACTIVE = false;
      if(!QUIET_MODE)
      {
        std::clog << TAB << "WARNING: CUDA S^D refinement state update failed; falling back to CPU for remaining vertices. "
                  << likelihood_backend->last_error() << std::endl;
      }
    }
  }
#endif
  return has_moved;
}

void embeddingSD_t::refine_positions(int dim)
{
  if(!QUIET_MODE) { std::clog << "Refining the positions..."; }
  if(!QUIET_MODE) { std::clog << std::endl; }

  CUDA_REFINEMENT_ACTIVE = false;

  if(dim == 1)
  {
#if defined(DMERCATOR_USE_CUDA)
  if(OPTIMIZED_PERF_MODE && CUDA_MODE)
  {
    CUDA_REFINEMENT_ACTIVE = (likelihood_backend != nullptr) &&
                             likelihood_backend->set_kappa(kappa) &&
                             likelihood_backend->set_theta(theta);
    if(!CUDA_REFINEMENT_ACTIVE)
    {
      CUDA_MODE = false;
      if(!QUIET_MODE)
      {
        std::clog << TAB << "WARNING: CUDA S1 refinement initialization failed; continuing on CPU. "
                  << (likelihood_backend ? likelihood_backend->last_error() : std::string("backend not available")) << std::endl;
      }
    }
  }
#endif

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
    CUDA_REFINEMENT_ACTIVE = false;
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
#if defined(DMERCATOR_USE_CUDA)
  if(OPTIMIZED_PERF_MODE && CUDA_MODE)
  {
    auto &positions_soa = scratch_positions_soa;
    const int position_stride = dim + 1;
    positions_soa.assign(static_cast<size_t>(position_stride) * static_cast<size_t>(nb_vertices), 0.0);
    for(int v = 0; v < nb_vertices; ++v)
    {
      for(int axis = 0; axis < position_stride; ++axis)
      {
        positions_soa[static_cast<size_t>(axis) * static_cast<size_t>(nb_vertices) + static_cast<size_t>(v)] =
          d_positions[static_cast<size_t>(v)][static_cast<size_t>(axis)];
      }
    }
    CUDA_REFINEMENT_ACTIVE = (likelihood_backend != nullptr) &&
                             likelihood_backend->set_kappa(kappa) &&
                             likelihood_backend->set_positions_soa(position_stride, positions_soa);
    if(!CUDA_REFINEMENT_ACTIVE)
    {
      CUDA_MODE = false;
      if(!QUIET_MODE)
      {
        std::clog << TAB << "WARNING: CUDA S^D refinement initialization failed; continuing on CPU. "
                  << (likelihood_backend ? likelihood_backend->last_error() : std::string("backend not available")) << std::endl;
      }
    }
  }
#endif

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

  CUDA_REFINEMENT_ACTIVE = false;
  if(!QUIET_MODE) { std::clog << "                         .............................................................done." << std::endl; }
  if(!QUIET_MODE) { std::clog << std::endl; }
}

void embeddingSD_t::refine_positions()
{
  refine_positions(1);
}

#endif
