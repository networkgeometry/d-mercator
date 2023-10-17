/*
 *
 * Provides the functions related to UNIX operating system.
 *
 * Author:  Antoine Allard
 * WWW:     antoineallard.info
 * Date:    September 2017
 *
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 */

#ifndef EMBEDDINGS1_UNIX_HPP_INCLUDED
#define EMBEDDINGS1_UNIX_HPP_INCLUDED

// Standard Template Library
#include <cstdlib>
#include <iostream>
#include <string>
// Operating System
#include <unistd.h>
// embeddingSD
#include "embeddingSD.hpp"


// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// Prints the information on the way the command line UNIX program should be used.
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
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


// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// Prints the information about the options of the command line UNIX program should be used.
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
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
    -p             Beta+kappas post-processing step.
    -e             Only infer kappas for a given input network. Then exit and save these 
                   hidden degrees to file.
  )";
  std::cout << help << '\n';
}


// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// Parses the options (for UNIX-like command line use) and returns the filename of the edgelist.
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void parse_options(int argc , char *argv[], embeddingSD_t &the_graph)
{
  // Shows the options if no argument is given.
  if(argc == 1)
  {
    print_usage();
    print_help();
    std::exit(0);
  }

  // <edgelist_filename>
  the_graph.EDGELIST_FILENAME = argv[argc - 1];

  // Parsing options.
  int opt;
  while ((opt = getopt(argc,argv,"ab:cfkpo:r:qs:vd:e")) != -1)
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

      // case 'h':
      //   print_usage();
      //   print_help();
      //   std::exit(0);

      case 'k':
        the_graph.KAPPA_POST_INFERENCE_MODE = false;
        break;
      case 'p':
        the_graph.KAPPA_BETA_POST_INFERENCE_MODE = true;
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
      default:
        print_usage();
        print_help();
        std::exit(0);
    }
  }
}

#endif // EMBEDDINGS1_UNIX_HPP_INCLUDED
