
#ifndef EMBEDDINGS1_UNIX_HPP_INCLUDED
#define EMBEDDINGS1_UNIX_HPP_INCLUDED

#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include "embeddingSD.hpp"

void print_usage()
{
  constexpr auto usage = R"(
NAME
      D-Mercator: Inference of high-quality embeddings of complex networks into the
                hyperbolic spaces

SYNOPSIS
      mercator [options] <edgelist_filename>

INPUT
      The structure of the graph is provided by a text file containing it edgelist. Each
      line in the file corresponds to an edge in the graph (i.e., [VERTEX1] [VERTEX2]).
        - The name of the vertices need not be integers (they are stored as std::string).
        - Directed graphs will be converted to undirected.
        - Multiple edges, self-loops and weights will be ignored.
        - Lines starting with '# ' are ignored (i.e., comments).
  )";
  std::cout << usage << '\n';
}

void print_help()
{
  constexpr auto help = R"(
The following options are available:

    -a             Screen mode. Program outputs details about its progress on screen
                   (through std::clog) instead of in a log file. Useful to gather all
                   output in a single file if mercator is a subroutine of a script.
    -b [VALUE]     Specify the value for beta to be used for the embedding. By 
                   default the program infers the value of beta based on the average
                   local clustering coefficient of the original edgelist.
    -c             Clean mode. Writes the inferred coordinates in clean file without
                   any lines starting by # to facilitate their use in custom computer
                   programs.
    -f             Fast mode. Does not infer the positions based on likelihood
                   maximization, rather uses only the EigenMap method.
    -k             No post-processing of the values of kappa based on the inferred
                   angular positions (theta) resulting in every vertices with the same
                   degree ending at the same radial position in the hyperbolic disk.
    -o [ROOTNAME]  Specify the rootname used for all output files. Default: uses the
                   rootname of the edgelist file as (i.e., rootname.edge).
    -r [FILENAME]  Refine mode. Reads the inferred positions from a previous run of
                   this program (file *.inf_coord) and refines the inferred positions.
    -q             Quiet mode. Program does not output information about the network
                   and the embedding procedure.
    -s [SEED]      Program uses a custom seed for the random number generator.
                   Default: EPOCH.
    -v             Validation mode. Validates and characterizes the inferred random
                   network ensemble.
    -d [DIMENSION] Dimension of the embeddings.
    -e             Only infer kappas for a given input network. Then exit and save these 
                   hidden degrees to file.
    -M [MODE]      Performance mode: baseline or optimized. Default: optimized.
    -G, --gpu      Enable CUDA acceleration for supported hotspots (if built with USE_CUDA=ON).
    -C, --cpu      Force CPU-only execution (disables CUDA path even if available).
    -D             Enable deterministic CUDA mode (default).
    -N             Disable deterministic CUDA mode.
    --timing_json  Print machine-readable stage timings (JSON) to stdout.
  )";
  std::cout << help << '\n';
}

void parse_options(int argc , char *argv[], embeddingSD_t &the_graph)
{

  if(argc == 1)
  {
    print_usage();
    print_help();
    std::exit(0);
  }

  the_graph.EDGELIST_FILENAME = argv[argc - 1];

  const option long_options[] = {
    {"mode", required_argument, nullptr, 'M'},
    {"gpu", no_argument, nullptr, 'G'},
    {"cpu", no_argument, nullptr, 'C'},
    {"deterministic", no_argument, nullptr, 'D'},
    {"nondeterministic", no_argument, nullptr, 'N'},
    {"timing_json", no_argument, nullptr, 'T'},
    {nullptr, 0, nullptr, 0}
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "ab:cfko:r:qs:vd:eM:TGCDN", long_options, nullptr)) != -1)
  {
    switch(opt)
    {
      case 'a':
        the_graph.VERBOSE_MODE = true;
        break;

      case 'b':
        the_graph.CUSTOM_BETA = true;
        the_graph.beta = std::stod(optarg);
        break;

      case 'c':
        the_graph.CLEAN_RAW_OUTPUT_MODE = true;
        break;

      case 'f':
        the_graph.MAXIMIZATION_MODE = false;
        break;

      case 'k':
        the_graph.KAPPA_POST_INFERENCE_MODE = false;
        break;
      case 'o':
        the_graph.CUSTOM_OUTPUT_ROOTNAME_MODE = true;
        the_graph.ROOTNAME_OUTPUT = optarg;
        break;

      case 'r':
        the_graph.REFINE_MODE = true;
        the_graph.ALREADY_INFERRED_PARAMETERS_FILENAME = optarg;
        break;

      case 'q':
        the_graph.QUIET_MODE = true;
        break;

      case 's':
        the_graph.CUSTOM_SEED = true;
        the_graph.SEED = std::stoi(optarg);
        break;

      case 'v':
        the_graph.VALIDATION_MODE = true;
        the_graph.CHARACTERIZATION_MODE = true;
        break;
      case 'd':
        the_graph.DIMENSION = std::stoi(optarg);
        break;
      case 'e':
        the_graph.ONLY_KAPPAS = true;
        break;
      case 'M':
      {
        const std::string mode = optarg;
        if(mode == "baseline")
        {
          the_graph.OPTIMIZED_PERF_MODE = false;
        }
        else if(mode == "optimized")
        {
          the_graph.OPTIMIZED_PERF_MODE = true;
        }
        else
        {
          std::cerr << "Invalid mode '" << mode << "'. Use baseline or optimized." << std::endl;
          print_usage();
          print_help();
          std::exit(1);
        }
        break;
      }
      case 'T':
        the_graph.TIMING_JSON_MODE = true;
        break;
      case 'G':
        the_graph.CUDA_MODE = true;
        break;
      case 'C':
        the_graph.CUDA_MODE = false;
        break;
      case 'D':
        the_graph.CUDA_DETERMINISTIC_MODE = true;
        break;
      case 'N':
        the_graph.CUDA_DETERMINISTIC_MODE = false;
        break;
      default:
        print_usage();
        print_help();
        std::exit(0);
    }
  }
}

#endif
