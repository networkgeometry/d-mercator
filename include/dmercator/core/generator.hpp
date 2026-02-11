#ifndef DMERCATOR_CORE_GENERATOR_HPP
#define DMERCATOR_CORE_GENERATOR_HPP

namespace dmercator::core {

class Generator {
 public:
  static int run_from_cli(int argc, char *argv[]);
};

using GeneratorSD = Generator;

}

#endif
