#!/bin/bash

help()
{
   echo "Basic script to build mercator"
   echo
   echo "Syntax: ./build.sh [-b|-s]"
   echo "options:"
   echo "h                 Print this help."
   echo "b [BUILD_TYPE]    Build type: {Release, Debug}."
   echo "s [SYSTEM]        OS type: {win, linux, mac}."
   echo
}

while getopts ":hb:s:" option; do
   case $option in
      h)
        help
        exit;;
      b)
        build_type=$OPTARG;;
      s)
        os_type=$OPTARG;;
     \?)
        echo "Error: Invalid option"
        exit;;
   esac
done

if [[ -z $build_type ]]; then
   build_type="Release"
fi

if [[ -z $os_type ]]; then
   os_type="linux"
fi

mkdir -p build && cd build/ # go to build folder
#export OMP_NUM_THREADS=$(nproc)

# Windows 10
if [[ $os_type == "win" ]]; then
    cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=$build_type
    cmake --build . --config $build_type -j 8
fi

# Linux or Mac
if [[ "$os_type" == "linux" || "$os_type" == "mac" ]]; then
    cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=$build_type
    cmake --build . -j 8
fi

mv mercator ../ # copy mercator to project directory