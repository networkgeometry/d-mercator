#ifndef DMERCATOR_EMBEDDING_ORCHESTRATOR_HPP
#define DMERCATOR_EMBEDDING_ORCHESTRATOR_HPP

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

void embeddingSD_t::extract_onion_decomposition(std::vector<int> &coreness, std::vector<int> &od_layer)
{

  std::vector<int> DegreeVec(nb_vertices);
  std::set<std::pair<int, int> > DegreeSet;
  for(int v(0); v<nb_vertices; ++v)
  {
    DegreeSet.insert(std::make_pair(degree[v], v));
    DegreeVec[v] = degree[v];
  }

  int v1, v2, d1, d2;
  int current_layer = 0;

  std::set<int>::iterator it1, end;
  std::set< std::pair<int, int> > LayerSet;

  std::set< std::pair<int, int> >::iterator m_it;

  while(!DegreeSet.empty())
  {

    m_it = DegreeSet.begin();
    d1 = m_it->first;

    current_layer += 1;

    while(m_it->first == d1 && m_it != DegreeSet.end())
    {

      v1 = m_it->second;
      coreness[v1] = d1;
      od_layer[v1] = current_layer;

      ++m_it;
    }

    LayerSet.insert(DegreeSet.begin(), m_it);

    DegreeSet.erase(DegreeSet.begin(), m_it);

    while(!LayerSet.empty())
    {

      v1 = LayerSet.begin()->second;

      it1 = adjacency_list[v1].begin();
      end = adjacency_list[v1].end();
      for(; it1!=end; ++it1)
      {

        v2 = *it1;
        d2 = DegreeVec[v2];

        m_it = DegreeSet.find(std::make_pair(d2, v2));
        if(m_it != DegreeSet.end())
        {
          if(d2 > d1)
          {
            DegreeVec[v2] = d2 - 1;
            DegreeSet.erase(m_it);
            DegreeSet.insert(std::make_pair(d2 - 1, v2));
          }
        }
      }

      LayerSet.erase(LayerSet.begin());
    }
  }
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


#endif
