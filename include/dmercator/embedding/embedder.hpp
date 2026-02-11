#ifndef DMERCATOR_EMBEDDING_EMBEDDER_HPP
#define DMERCATOR_EMBEDDING_EMBEDDER_HPP

namespace dmercator::embedding {

class Embedder {
 public:
  static int run_from_cli(int argc, char *argv[]);
};

using EmbedderSD = Embedder;

}

#endif
