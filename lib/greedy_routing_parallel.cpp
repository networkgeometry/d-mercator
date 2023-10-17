#include <iostream>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <complex>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <filesystem>
#include <list>


std::vector<std::string> Num2Name;
std::map<std::string, int> Name2Num;
std::map<int, std::vector<int>> adjacency_list;


void print_help() {
    constexpr auto help_message = R"(
        ./greedy_routing_parallel [dim] [path_to_coords] [path_to_edgelist] [is_modified] [n_runs]
        
        dim                          -- dimension of S^D model
        path_to_coords               -- path to the file with the nodes' coordinates (.inf_coord)
        path_to_edgelist             -- path to the edgelist (.edge) 
        is_modified                  -- whether to run modified version of greedy routing (0 or 1)
        n_runs                       -- number of greedy routing rounds (default=10*network size)


        Compile with: `g++ --std=c++17 -O3 -fopenmp greedy_routing_parallel.cpp`
    )";
    std::cout << help_message << std::endl;
}

void load_coords(int dim, 
                 const std::string &coords_path, 
                 std::vector<double> &radii, 
                 std::vector<double> &thetas, 
                 std::vector<std::vector<double>> &positions) {

    std::stringstream one_line;
    std::string full_line, name1_str, name2_str, name3_str;

    std::fstream hidden_variables_file(coords_path.c_str(), std::fstream::in);
    if( !hidden_variables_file.is_open() )
    {
        std::cerr << "Could not open file: " << coords_path << "." << std::endl;
        std::terminate();
    }

    int n_nodes = 0;
    while(!hidden_variables_file.eof()) {
        // Reads a line of the file.
        std::getline(hidden_variables_file, full_line);
        hidden_variables_file >> std::ws;
        one_line.str(full_line);
        one_line >> std::ws;
        one_line >> name1_str >> std::ws;
        // Skips lines of comment.
        if(name1_str == "#")
        {
            one_line.clear();
            continue;
        }

        one_line >> name2_str >> std::ws; // omit kappas
        Num2Name.push_back(name1_str);
        Name2Num[name1_str] = n_nodes;
        n_nodes++;

        if (dim == 1) {
            one_line >> name3_str >> std::ws;
            thetas.push_back(std::stod(name3_str));
            one_line >> name3_str >> std::ws;
            radii.push_back(std::stod(name3_str));
        } else {
            one_line >> name3_str >> std::ws;
            radii.push_back(std::stod(name3_str));

            std::vector<double> tmp_position;
            for (int i=0; i<dim+1; ++i) {
                one_line >> name3_str >> std::ws;
                tmp_position.push_back(std::stod(name3_str));
            }
            positions.push_back(tmp_position);
        }
        one_line.clear();
    }
    hidden_variables_file.close();
}


void load_edgelist(const std::string& edgelist_path) {
    std::stringstream one_line;
    std::string full_line, source_str, target_str;

    std::fstream edgelist_file(edgelist_path.c_str(), std::fstream::in);
    if( !edgelist_file.is_open() )
    {
        std::cerr << "Could not open file: " << edgelist_path << "." << std::endl;
        std::terminate();
    }

    while(!edgelist_file.eof()) {
        std::getline(edgelist_file, full_line);
        edgelist_file >> std::ws;
        std::istringstream ss(full_line);
        ss >> source_str >> target_str;

        if(source_str == "#" || target_str == "#")
            continue; 
    
        // Assumption: graph is undirected
        adjacency_list[Name2Num[source_str]].push_back(Name2Num[target_str]);
        adjacency_list[Name2Num[target_str]].push_back(Name2Num[source_str]);    
    }
}

bool BFS(int source, int target, int *pred, int *dist) {
    std::list<int> queue;
    const int n = adjacency_list.size();
    bool visited[n];

    for (int i=0; i<n; ++i) {
        visited[i] = false;
        dist[i] = 1000000;
        pred[i] = -1;
    }

    visited[source] = true;
    dist[source] = 0;
    queue.push_back(source);

    while(!queue.empty()) {
        int u = queue.front();
        queue.pop_front();
        for (int i=0; i < adjacency_list[u].size(); ++i) {
            int node = adjacency_list[u][i];
            if (!visited[node]) {
                visited[node] = true;
                dist[node] = dist[u] + 1;
                pred[node] = u;
                queue.push_back(node);

                if (node == target)
                    return true;
            }
        }
    }
    return false;
}


int compute_shortest_path_length(int source, int target) {
    const int n = adjacency_list.size();
    int pred[n], dist[n];

    if (!BFS(source, target, pred, dist)) {
        return -1;
    }

    return dist[target];
}

double S1_distance(double r1, double r2, double theta1, double theta2) {
    if ((r1 == r2) && (theta1 == theta2)) {
        return 0;
    }
    double delta_theta = M_PI - std::fabs(M_PI - std::fabs(theta1 - theta2));
    if (delta_theta == 0) {
        return std::fabs(r1 - r2); 
    } else {
        auto dist = 0.5 * ((1 - std::cos(delta_theta)) * std::cosh(r1 + r2) + (1 + std::cos(delta_theta)) * std::cosh(r1 - r2));
        return std::acosh(dist);
    }
}

double compute_angle_d_vectors(const std::vector<double> &v1, const std::vector<double> &v2) {
  double angle{0}, norm1{0}, norm2{0};
  for (int i = 0; i < v1.size(); ++i) {
    angle += v1[i] * v2[i];
    norm1 += v1[i] * v1[i];
    norm2 += v2[i] * v2[i];
  }
  norm1 /= sqrt(norm1);
  norm2 /= sqrt(norm2);
  
  const auto result = angle / (norm1 * norm2);
  if (std::fabs(result - 1) < 1e-15)
    return 0;
  else
    return std::acos(result);
}

double SD_distance(double r1, double r2, const std::vector<double> &pos1, const std::vector<double> &pos2) {
    double delta_theta = compute_angle_d_vectors(pos1, pos2);
    if ((r1 == r2) && delta_theta == 0) {
        return 0; // the same positions
    }
    
    if (delta_theta == 0) {
        return std::fabs(r1 - r2); 
    } else {
        auto dist = 0.5 * ((1 - std::cos(delta_theta)) * std::cosh(r1 + r2) + (1 + std::cos(delta_theta)) * std::cosh(r1 - r2));
        return std::acosh(dist);
    }
}

double compute_distance(int dim, 
                        int v1, 
                        int v2, 
                        const std::vector<double> &radii, 
                        const std::vector<double> &thetas, 
                        const std::vector<std::vector<double>> &positions) {
    if (dim == 1)
        return S1_distance(radii[v1], radii[v2], thetas[v1], thetas[v2]);
    else
        return SD_distance(radii[v1], radii[v2], positions[v1], positions[v2]);
}

void run_original_greedy_routing(int dim, 
                                 const std::vector<double> &radii, 
                                 const std::vector<double> &thetas, 
                                 const std::vector<std::vector<double>> &positions, 
                                 int n_runs) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, Num2Name.size() - 1);

    std::vector<double> all_stretch;
    double p_s = 0;
    double mean_strech = 0;
    double max_strech = 0;
    double mean_hop_length = 0;
    double gr_score = 0;

#pragma omp parallel for reduction(+:p_s,mean_strech,mean_hop_length,gr_score)
    for (int i=0; i<n_runs; ++i) {
        int source=-1, target=-1;
        while (source == target) {
            source = distr(gen);
            target = distr(gen);
        }
        const int org_source = source;

        std::vector<int> hops = {source};
        bool is_package_dropped = false;
        while (source != target) {
            const double distance_s_t = compute_distance(dim, source, target, radii, thetas, positions);
            double smallest_distance = distance_s_t;
            int new_source = source;

            for (const auto n: adjacency_list[source]) {
                if (n == target) {
                    hops.push_back(target);
                    goto found_target1;
                }
                const double distance_n_t = compute_distance(dim, n, target, radii, thetas, positions);
                if (distance_n_t < smallest_distance) {
                    new_source = n;
                    smallest_distance = distance_n_t;
                }
            }            
            if (new_source == source && new_source != target) {
                is_package_dropped = true;
                break;
            } else {
                source = new_source;
            }
            hops.push_back(source);
        }
        found_target1:
        if (!is_package_dropped) {
            ++p_s;
            mean_hop_length += (double)hops.size();
            double strech = (double)hops.size() / compute_shortest_path_length(org_source, target);
            mean_strech += strech;
            gr_score += 1 / strech;
            #pragma omp critical
            {
                if (strech > max_strech)
                    max_strech = strech;
            }
        }
    }
    mean_strech /= p_s;
    mean_hop_length /= p_s;
    p_s /= n_runs;
    gr_score /= n_runs;
    std::cout << "p_s,mean_hop_length,mean_strech,max_strech,gr_score" << std::endl;
    std::cout << p_s << "," << mean_hop_length << "," << mean_strech << "," << max_strech << "," << gr_score << std::endl;
}


void run_modified_greedy_routing(int dim, 
                                 const std::vector<double> &radii, 
                                 const std::vector<double> &thetas, 
                                 const std::vector<std::vector<double>> &positions, 
                                 int n_runs) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, Num2Name.size() - 1);

    std::vector<double> all_stretch;
    double p_s = 0;
    double mean_strech = 0;
    double max_strech = 0;
    double mean_hop_length = 0;
    double gr_score = 0;

#pragma omp parallel for reduction(+:p_s,mean_strech,mean_hop_length,gr_score)
    for (int i=0; i<n_runs; ++i) {
        int source=-1, target=-1;
        while (source == target) {
            source = distr(gen);
            target = distr(gen);
        }
        const int org_source = source;

        std::vector<int> hops = {source};
        bool is_package_dropped = false;
        while (source != target) {
            double smallest_distance = 999999;
            int new_source = source;

            for (const auto n: adjacency_list[source]) {
                if (n == target) {
                    hops.push_back(target);
                    goto found_target2;
                }
                const double distance_n_t = compute_distance(dim, n, target, radii, thetas, positions);                
                if (distance_n_t < smallest_distance) {
                    new_source = n;
                    smallest_distance = distance_n_t;
                }
            }
            source = new_source;

            if (hops.size() > 1) {
                if (hops[hops.size() - 2] == source) {
                    is_package_dropped = true;
                    break;
                }
            }

            hops.push_back(source);
        }
        found_target2: // goto statement

        if (!is_package_dropped) {
            ++p_s;
            mean_hop_length += (double)hops.size();
            double strech = (double)hops.size() / compute_shortest_path_length(org_source, target);
            mean_strech += strech;
            gr_score += 1 / strech;
            #pragma omp critical
            {
                if (strech > max_strech)
                    max_strech = strech;
            }
        }
    }
    mean_strech /= p_s;
    mean_hop_length /= p_s;
    p_s /= n_runs;
    gr_score /= n_runs;
    std::cout << "p_s,mean_hop_length,mean_strech,max_strech,gr_score" << std::endl;
    std::cout << p_s << "," << mean_hop_length << "," << mean_strech << "," << max_strech << "," << gr_score << std::endl;
}





int main(int argc , char *argv[]) {
    if (argc < 4) {
        std::cout << "Error. Wrong number of parameters." << std::endl;
        print_help();
    }

    int dim = std::stoi(argv[1]);
    std::string coords_path = argv[2];
    std::string edgelist_path = argv[3];

    std::vector<double> radii;
    std::vector<double> thetas;
    std::vector<std::vector<double>> positions;

    load_coords(dim, coords_path, radii, thetas, positions);
    load_edgelist(edgelist_path);

    int is_modified = std::stoi(argv[4]);
    
    int n_runs = 10 * adjacency_list.size();
    if (argc == 6)
        n_runs = std::stoi(argv[5]);

    if (is_modified == 1)
        run_modified_greedy_routing(dim, radii, thetas, positions, n_runs);
    else
        run_original_greedy_routing(dim, radii, thetas, positions, n_runs);
}
    