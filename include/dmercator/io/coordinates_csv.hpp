#ifndef DMERCATOR_IO_COORDINATES_CSV_HPP
#define DMERCATOR_IO_COORDINATES_CSV_HPP

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dmercator::io {

struct CoordinateRow {
  std::string node_id;
  double radial = 0.0;
  double kappa = 0.0;
  std::vector<double> coordinates;
};

inline std::string trim_copy(const std::string &value)
{
  std::size_t begin = 0;
  while(begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  if(begin == value.size()) {
    return "";
  }
  std::size_t end = value.size() - 1;
  while(end > begin && std::isspace(static_cast<unsigned char>(value[end]))) {
    --end;
  }
  return value.substr(begin, end - begin + 1);
}

inline bool is_legacy_comment_or_empty(const std::string &line)
{
  const std::string trimmed = trim_copy(line);
  return trimmed.empty() || trimmed[0] == '#';
}

inline std::vector<std::string> split_ws(const std::string &line)
{
  std::vector<std::string> out;
  std::istringstream iss(line);
  std::string token;
  while(iss >> token) {
    out.push_back(token);
  }
  return out;
}

inline std::vector<CoordinateRow> load_legacy_coordinate_table(const std::string &filename,
                                                               int dimension,
                                                               bool generated_file)
{
  if(dimension < 1) {
    throw std::runtime_error("Dimension must be >= 1");
  }

  std::ifstream input(filename);
  if(!input.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  std::vector<CoordinateRow> rows;
  std::string line;
  while(std::getline(input, line)) {
    if(is_legacy_comment_or_empty(line)) {
      continue;
    }

    const std::vector<std::string> tokens = split_ws(line);
    if(tokens.size() < 4) {
      continue;
    }

    CoordinateRow row;
    row.node_id = tokens[0];
    row.kappa = std::stod(tokens[1]);

    if(dimension == 1) {
      // Legacy schema: node kappa theta radial [realdeg expdeg]
      row.coordinates.push_back(std::stod(tokens[2]));
      row.radial = std::stod(tokens[3]);
    } else {
      // Legacy schema: node kappa radial pos0..posD [realdeg expdeg]
      const int expected_coords = dimension + 1;
      if(static_cast<int>(tokens.size()) < 3 + expected_coords) {
        throw std::runtime_error("Malformed coordinate row in: " + filename);
      }
      row.radial = std::stod(tokens[2]);
      row.coordinates.reserve(expected_coords);
      for(int i = 0; i < expected_coords; ++i) {
        row.coordinates.push_back(std::stod(tokens[3 + i]));
      }
    }

    if(generated_file && row.coordinates.empty()) {
      throw std::runtime_error("Generated coordinate row missing coordinates in: " + filename);
    }

    rows.push_back(std::move(row));
  }

  std::sort(rows.begin(), rows.end(), [](const CoordinateRow &lhs, const CoordinateRow &rhs) {
    if(lhs.node_id == rhs.node_id) {
      return lhs.kappa < rhs.kappa;
    }
    return lhs.node_id < rhs.node_id;
  });

  return rows;
}

inline void write_coordinate_csv(const std::string &filename,
                                 int dimension,
                                 const std::vector<CoordinateRow> &rows)
{
  std::ofstream out(filename);
  if(!out.is_open()) {
    throw std::runtime_error("Could not open output file: " + filename);
  }

  out << "node_id,r,kappa";
  if(dimension == 1) {
    out << ",theta_0";
  } else {
    for(int i = 0; i < dimension + 1; ++i) {
      out << ",pos_" << i;
    }
  }
  out << '\n';

  out << std::setprecision(17);
  for(const auto &row : rows) {
    out << row.node_id << ',' << row.radial << ',' << row.kappa;
    for(const double value : row.coordinates) {
      out << ',' << value;
    }
    out << '\n';
  }
}

inline void convert_legacy_coordinates_to_csv(const std::string &legacy_filename,
                                              const std::string &csv_filename,
                                              int dimension,
                                              bool generated_file)
{
  const auto rows = load_legacy_coordinate_table(legacy_filename, dimension, generated_file);
  write_coordinate_csv(csv_filename, dimension, rows);
}

inline std::string strip_extension(const std::string &path)
{
  const std::size_t last_dot = path.find_last_of('.');
  if(last_dot == std::string::npos) {
    return path;
  }
  return path.substr(0, last_dot);
}

} // namespace dmercator::io

#endif // DMERCATOR_IO_COORDINATES_CSV_HPP
