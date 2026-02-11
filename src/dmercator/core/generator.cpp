#include "../../../include/dmercator/core/generator.hpp"

#include "../../../include/dmercator/io/coordinates_csv.hpp"
#include "../../../include/generatingSD_unix.hpp"

#include <exception>
#include <iostream>
#include <string>

namespace dmercator::core {

int Generator::run_from_cli(int argc, char *argv[])
{
  generatingSD_t graph;
  if(!parse_options(argc, argv, graph)) {
    return EXIT_SUCCESS;
  }

  graph.OUTPUT_VERTICES_PROPERTIES = true;
  graph.load_hidden_variables_for_dimension();
  graph.generate_edgelist_for_dimension();

  try {
    const std::string legacy_path = graph.OUTPUT_ROOTNAME + ".gen_coord";
    const std::string csv_path = graph.OUTPUT_ROOTNAME + ".truth.csv";
    dmercator::io::convert_legacy_coordinates_to_csv(legacy_path, csv_path, graph.DIMENSION, true);
    std::clog << "Machine-readable ground truth saved to " << csv_path << std::endl;
  } catch(const std::exception &e) {
    std::cerr << "ERROR: failed to export truth CSV: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

} // namespace dmercator::core
