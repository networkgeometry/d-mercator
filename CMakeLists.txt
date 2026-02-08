cmake_minimum_required(VERSION 3.20)
project(mercator Fortran CXX)

option(DMERCATOR_RELEASE_O3 "Use -O3 in Release builds (default keeps -O2)." OFF)
option(DMERCATOR_ENABLE_NATIVE "Enable -march=native for local tuning." OFF)
option(DMERCATOR_ENABLE_LTO "Enable IPO/LTO in Release builds when supported." OFF)

set(DMERCATOR_RELEASE_OPT_FLAG "-O2")
if(DMERCATOR_RELEASE_O3)
  set(DMERCATOR_RELEASE_OPT_FLAG "-O3")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} ${DMERCATOR_RELEASE_OPT_FLAG} -fopenmp -lpthread")
set(CMAKE_Fortran_FLAGS "-c -fpic -Wall -O3")

if(DMERCATOR_ENABLE_NATIVE)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-march=native>")
endif()

if(DMERCATOR_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT DMERCATOR_IPO_SUPPORTED OUTPUT DMERCATOR_IPO_ERROR)
  if(DMERCATOR_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  else()
    message(WARNING "DMERCATOR_ENABLE_LTO requested but unsupported: ${DMERCATOR_IPO_ERROR}")
  endif()
endif()

include_directories(include)
include_directories(include/*)
add_library(trapecis_Sd include/trapecis_Sd.f)
# add_library(trapecis_Sd_expected include/trapecis_Sd_expected.f)

set(source_dir "${PROJECT_SOURCE_DIR}/src/")
set(source_files "${source_dir}/embeddingSD_unix.cpp")

set(include_dir "${PROJECT_SOURCE_DIR}/include/")
file(GLOB include_files "${include_dir}/*.hpp")

add_executable(mercator ${source_files} ${include_files})
target_link_libraries(mercator trapecis_Sd)
