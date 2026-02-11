#ifndef DMERCATOR_INIT_INITIALIZATION_HPP
#define DMERCATOR_INIT_INITIALIZATION_HPP

void embeddingSD_t::initialize()
{

  if(!CUSTOM_OUTPUT_ROOTNAME_MODE)
  {
    size_t lastdot = EDGELIST_FILENAME.find_last_of(".");
    if(lastdot == std::string::npos)
    {
      ROOTNAME_OUTPUT = EDGELIST_FILENAME;
    }
    ROOTNAME_OUTPUT = EDGELIST_FILENAME.substr(0, lastdot);
  }

  if(!CUSTOM_SEED)
  {
    SEED = std::time(nullptr);
  }
  engine.seed(SEED);

  if(!QUIET_MODE)
  {
    if(!VERBOSE_MODE)
    {
     logfile.open(ROOTNAME_OUTPUT + ".inf_log");

     old_rdbuf = std::clog.rdbuf();

     std::clog.rdbuf(logfile.rdbuf());
    }
  }

  if(!QUIET_MODE) { std::clog                                                                                                  << std::endl; }
  if(!QUIET_MODE) { std::clog << "===========================================================================================" << std::endl; }
  if(!QUIET_MODE) { std::clog << "D-Mercator: accurate embeddings of graphs in the SD space"                                     << std::endl; }
  if(!QUIET_MODE) { std::clog << "version: "           << VERSION                                                              << std::endl; }
  if(!QUIET_MODE) { std::clog << "started on: "        << format_time(time_started)                                            << std::endl; }
  if(!QUIET_MODE) { std::clog << "edgelist filename: " << EDGELIST_FILENAME                                                    << std::endl; }
  if(REFINE_MODE)
  {
    if(!QUIET_MODE) { std::clog << "inferred positions filename: " << ALREADY_INFERRED_PARAMETERS_FILENAME                     << std::endl; }
  }
  if(!QUIET_MODE) { std::clog << "seed: "              << SEED                                                                 << std::endl; }

  if(!QUIET_MODE) { std::clog                                                                                                  << std::endl; }
  if(!QUIET_MODE) { std::clog << "Loading edgelist..."; }
  load_edgelist();
  if(!QUIET_MODE) { std::clog << "...................................................................done."                    << std::endl; }

  if(!QUIET_MODE) { std::clog                                                                                                  << std::endl; }
  if(!QUIET_MODE) { std::clog << "Checking number of connected components..."; }
  check_connected_components();
  if(!QUIET_MODE) { std::clog << "............................................done."                                           << std::endl; }

  if(!QUIET_MODE) { std::clog                                                                                                  << std::endl; }
  if(!QUIET_MODE) { std::clog << "Analyzing degrees..."; }
  analyze_degrees();
  if(!QUIET_MODE) { std::clog << "..................................................................done."                     << std::endl; }

  if(!QUIET_MODE) { std::clog                                                                                                  << std::endl; }
  if(!QUIET_MODE) { std::clog << "Computing local clustering..."; }
  compute_clustering();
  if(!QUIET_MODE) { std::clog << ".........................................................done."                              << std::endl; }

  if(!QUIET_MODE) { std::clog                                                                                                  << std::endl; }
  if(!QUIET_MODE) { std::clog << "Ordering vertices..."; }
  order_vertices();
  if(!QUIET_MODE) { std::clog << "..................................................................done."                     << std::endl; }
  if(!QUIET_MODE) { std::clog                                                                                                  << std::endl; }

  std::clog.precision(4);

  width_values = 15;
  width_names = 14;
  for(int v(0), l; v<nb_vertices; ++v)
  {
    l = Num2Name[v].length();
    if(l > width_names)
    {
      width_names = l;
    }
  }
  width_names += 1;

  if (!QUIET_MODE) {
      std::clog << "Properties of the graph" << std::endl;
      std::clog << TAB << "Nb vertices:                    " << nb_vertices << std::endl;
      std::clog << TAB << "Nb edges:                       " << nb_edges << std::endl;
      std::clog << TAB << "Average degree:                 " << average_degree << std::endl;
      std::clog << TAB << "Minimum degree:                 " << *(degree_class.begin()) << std::endl;
      std::clog << TAB << "Maximum degree:                 " << *(--degree_class.end()) << std::endl;
      std::clog << TAB << "Nb of degree class:             " << degree_class.size() << std::endl;
      std::clog << TAB << "Average clustering:             " << average_clustering << std::endl;
      std::clog << std::endl;
  }
}

void embeddingSD_t::load_already_inferred_parameters()
{
  load_already_inferred_parameters(1);
}

void embeddingSD_t::load_already_inferred_parameters(int dim)
{

  std::stringstream one_line;

  std::string full_line, name1_str, name2_str, name3_str;

  kappa.clear();
  kappa.resize(nb_vertices);

  // S1 files store theta; higher-dimensional files store radial + position vectors.
  if(dim == 1)
  {
    theta.clear();
    theta.resize(nb_vertices);
  }
  else
  {
    d_positions.clear();
    d_positions.resize(nb_vertices);
  }

  std::fstream hidden_variables_file(ALREADY_INFERRED_PARAMETERS_FILENAME.c_str(), std::fstream::in);
  if( !hidden_variables_file.is_open() )
  {
    std::cerr << "Could not open file: " << ALREADY_INFERRED_PARAMETERS_FILENAME << "." << std::endl;
    std::terminate();
  }

  for(int l(0); l<8; ++l)
  {
    std::getline(hidden_variables_file, full_line);
  }

  std::getline(hidden_variables_file, full_line);
  hidden_variables_file >> std::ws;
  one_line.str(full_line);
  one_line >> std::ws;
  one_line >> name1_str >> std::ws;
  one_line >> name1_str >> std::ws;
  one_line >> name1_str >> std::ws;
  one_line >> name1_str >> std::ws;
  beta = std::stod(name1_str);
  one_line.clear();

  std::getline(hidden_variables_file, full_line);
  hidden_variables_file >> std::ws;
  one_line.str(full_line);
  one_line >> std::ws;
  one_line >> name1_str >> std::ws;
  one_line >> name1_str >> std::ws;
  one_line >> name1_str >> std::ws;
  one_line >> name1_str >> std::ws;
  mu = std::stod(name1_str);
  one_line.clear();

  while( !hidden_variables_file.eof() )
  {

    std::getline(hidden_variables_file, full_line);
    hidden_variables_file >> std::ws;
    one_line.str(full_line);
    one_line >> std::ws;
    one_line >> name1_str >> std::ws;

    if(name1_str == "#")
    {
      one_line.clear();
      continue;
    }
    one_line >> name2_str >> std::ws;
    kappa[ Name2Num[name1_str] ] = std::stod(name2_str);

    if(dim == 1)
    {
      one_line >> name3_str >> std::ws;
      theta[ Name2Num[name1_str] ] = std::stod(name3_str);
    }
    else
    {
      one_line >> name2_str >> std::ws;
      std::vector<double> tmp_positions;
      tmp_positions.reserve(dim + 1);
      for (int i=0; i<dim+1; ++i) {
        one_line >> name3_str >> std::ws;
        tmp_positions.push_back(std::stod(name3_str));
      }
      d_positions[Name2Num[name1_str]] = tmp_positions;
    }

    one_line.clear();
  }

  hidden_variables_file.close();

  Name2Num.clear();
}

void embeddingSD_t::load_edgelist()
{

  std::ifstream edgelist_file;
  std::stringstream one_line;

  int v1, v2;

  std::string full_line, name1_str, name2_str;

  std::map< std::string, int >::iterator name_it;

  nb_vertices = 0;
  nb_edges = 0;

  Name2Num.clear();
  Num2Name.clear();
  adjacency_list.clear();

  edgelist_file.open(EDGELIST_FILENAME.c_str(), std::ios_base::in);
  if( !edgelist_file.is_open() )
  {
    std::cerr << "Could not open file: " << EDGELIST_FILENAME << "." << std::endl;
    std::terminate();
  }
  else
  {

    while( !edgelist_file.eof() )
    {

      std::getline(edgelist_file, full_line); edgelist_file >> std::ws;
      one_line.str(full_line); one_line >> std::ws;
      one_line >> name1_str >> std::ws;

      if(name1_str == "#")
      {
        one_line.clear();
        continue;
      }
      one_line >> name2_str >> std::ws;
      one_line.clear();

      if(name1_str != name2_str)
      {

        name_it = Name2Num.find(name1_str);
        if( name_it == Name2Num.end() )
        {

          v1 = nb_vertices;
          Name2Num[name1_str] = v1;
          Num2Name.push_back(name1_str);
          adjacency_list.emplace_back();
          ++nb_vertices;
        }
        else
        {

          v1 = name_it->second;
        }

        name_it = Name2Num.find(name2_str);
        if( name_it == Name2Num.end() )
        {

          v2 = nb_vertices;
          Name2Num[name2_str] = v2;
          Num2Name.push_back(name2_str);
          adjacency_list.emplace_back();
          ++nb_vertices;
        }
        else
        {

          v2 = name_it->second;
        }

        std::pair< std::set<int>::iterator, bool > add1 = adjacency_list[v1].insert(v2);
        std::pair< std::set<int>::iterator, bool > add2 = adjacency_list[v2].insert(v1);
        if(add1.second && add2.second)
        {
          ++nb_edges;
        }
      }
    }
  }

  edgelist_file.close();
  rebuild_adjacency_flat_list();
  if(!REFINE_MODE)
  {

    Name2Num.clear();
  }
}

void embeddingSD_t::order_vertices()
{

  std::vector<int> coreness(nb_vertices);
  std::vector<int> od_layer(nb_vertices);

  extract_onion_decomposition(coreness, od_layer);

  std::set< std::pair<int, std::pair<double, int> > > layer_set;
  for(int v(0); v<nb_vertices; ++v)
  {
    layer_set.insert(std::make_pair(od_layer[v], std::make_pair(uniform_01(engine), v)));
  }

  ordered_list_of_vertices.resize(nb_vertices);
  auto it = layer_set.rbegin();
  auto end = layer_set.rend();
  for(int v(0); it!=end; ++it, ++v)
  {
    ordered_list_of_vertices[v] = it->second.second;
  }
  layer_set.clear();
}

#endif
