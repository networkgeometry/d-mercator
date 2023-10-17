/*
 *
 * Provides the functions related to UNIX operating system.
 *
 * Author:  Antoine Allard
 * WWW:     antoineallard.info
 * Date:    November 2017
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

#ifndef GENERATINGSD_UNIX_HPP_INCLUDED
#define GENERATINGSD_UNIX_HPP_INCLUDED

// Standard Template Library
#include <cstdlib>
#include <string>
// Operating System
#include <unistd.h>
// embeddingS1
#include "generatingSD.hpp"





// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// Prints the information on the way the command line UNIX program should be used.
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void print_usage()
{
  std::string_view message = R"""(
NAME
  generatingSD -- a program to generate complex networks in the S^D metric space
SYNOPSIS
  generatingSD [options] <hidden_variables_filename>
DESCRIPTION
  [description here, format of input filename]
  )""";
  std::cout << message << std::endl;
}





// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// Prints the information about the options of the command line UNIX program should be used.
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void print_help()
{
  std::string_view help = R"""(
The following options are available:
  -a             Indicates that the file containing the hidden variables comes from the networkS1 embedding program (gets BETA and MU from the file).
  -b [VALUE]     Specifies the value for parameter beta.
  -h             Print this message on screen and exit.
  -m [VALUE]     Specifies the value for parameter mu. Default: MU = BETA * std::sin(PI / BETA) / (2.0 * PI * average_kappa).
  -n             Indicates that the first column of the hidden variables file provides the name of the vertices.
  -o [ROOTNAME]  Specifies the rootname used for all output files. Uses the filename of the hidden variables file as rootname if not specified.
  -s [SEED]      Program uses a custom seed for the random number generator. Default: EPOCH.
  -t             Indicates that the last column of the hidden variables file provides the angular position (i.e., theta) of the vertices.
  -v             Outputs the hidden variables (kappa and theta) used to the generate the network into a file (uses the edgelist's rootname).
  -d [DIMENSION] Specify model's dimension (S^D).
  )""";
  std::cout << help << std::endl;
}





// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// Parses the options (for UNIX-like command line use) and returns the filename of the edgelist or
//   a flag indicating that help dialogues have been shown on screen and further computation is not
//   required.
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
bool parse_options(int argc , char *argv[], generatingSD_t &the_graph)
{
  // Shows the options if no argument is given.
  if(argc == 1)
  {
    print_usage();
    print_help();
    return false;
  }

  // <edgelist_filename>
  the_graph.HIDDEN_VARIABLES_FILENAME = argv[argc - 1];

  // Parsing options.
  int opt;
  while ((opt = getopt(argc,argv,"ab:hm:no:s:tvd:")) != -1)
  {
    switch(opt)
    {
      case 'a':
        the_graph.NATIVE_INPUT_FILE = true;
        the_graph.NAME_PROVIDED = true;
        the_graph.THETA_PROVIDED = true;
        break;

      case 'b':
        the_graph.BETA = std::stod(optarg);
        break;

      case 'h':
        print_usage();
        print_help();
        return false;

      case 'm':
        the_graph.MU = std::stod(optarg);
        break;

      case 'n':
        the_graph.NAME_PROVIDED = true;
        break;

      case 'o':
        the_graph.OUTPUT_ROOTNAME = optarg;
        the_graph.CUSTOM_OUTPUT_ROOTNAME_MODE = true;
        break;

      case 's':
        the_graph.SEED = std::stoi(optarg);
        break;

      case 't':
        the_graph.THETA_PROVIDED = true;
        break;

      case 'v':
        the_graph.OUTPUT_VERTICES_PROPERTIES = true;
        break;
      
      case 'd':
        the_graph.DIMENSION = std::stoi(optarg);
        break;
      
      default:
        print_usage();
        print_help();
        return false;
    }
  }

  // Uses the default rootname for output files.
  if(the_graph.CUSTOM_OUTPUT_ROOTNAME_MODE == false)
  {
    size_t lastdot = the_graph.HIDDEN_VARIABLES_FILENAME.find_last_of(".");
    if(lastdot == std::string::npos)
    {
      the_graph.OUTPUT_ROOTNAME = the_graph.HIDDEN_VARIABLES_FILENAME;
    }
    the_graph.OUTPUT_ROOTNAME = the_graph.HIDDEN_VARIABLES_FILENAME.substr(0, lastdot);
  }

  // Indicates that everything is in order.
  return true;
}


#endif // GENERATINGSD_UNIX_HPP_INCLUDED
