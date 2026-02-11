#include "../include/legacy/embeddingSD_unix_legacy.hpp"

int main(int argc, char *argv[])
{
  embeddingSD_t the_graph;
  parse_options(argc, argv, the_graph);
  the_graph.embed();
  return EXIT_SUCCESS;
}
