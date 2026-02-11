#include "../../../include/dmercator/embedding/embedder.hpp"

#include "../../../include/dmercator/io/coordinates_csv.hpp"
#include "../../../include/embeddingSD_unix.hpp"

#include <exception>
#include <iostream>
#include <string>

namespace dmercator::embedding {

int Embedder::run_from_cli(int argc, char *argv[])
{
  embeddingSD_t graph;
  parse_options(argc, argv, graph);
  graph.embed();

  try {
    const std::string legacy_path = graph.ROOTNAME_OUTPUT + ".inf_coord";
    const std::string csv_path = graph.ROOTNAME_OUTPUT + ".coords.csv";
    dmercator::io::convert_legacy_coordinates_to_csv(legacy_path, csv_path, graph.DIMENSION, false);
    if(!graph.QUIET_MODE) {
      std::clog << "    => Machine-readable coordinates saved to " << csv_path << std::endl;
    }
  } catch(const std::exception &e) {
    std::cerr << "ERROR: failed to export CSV coordinates: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

} // namespace dmercator::embedding
