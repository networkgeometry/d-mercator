#ifndef DMERCATOR_INITIALIZE_HPP
#define DMERCATOR_INITIALIZE_HPP

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

#endif
