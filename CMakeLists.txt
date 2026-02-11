cmake_minimum_required(VERSION 3.20)
project(mercator Fortran CXX)
include(CTest)

option(DMERCATOR_RELEASE_O3 "Use -O3 in Release builds (default keeps -O2)." OFF)
option(DMERCATOR_ENABLE_NATIVE "Enable -march=native for local tuning." OFF)
option(DMERCATOR_ENABLE_LTO "Enable IPO/LTO in Release builds when supported." OFF)

set(DMERCATOR_RELEASE_OPT_FLAG "-O2")
if(DMERCATOR_RELEASE_O3)
  set(DMERCATOR_RELEASE_OPT_FLAG "-O3")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} ${DMERCATOR_RELEASE_OPT_FLAG}")
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
add_library(trapecis_Sd include/trapecis_Sd.f)
# add_library(trapecis_Sd_expected include/trapecis_Sd_expected.f)
find_package(Threads REQUIRED)
find_package(OpenMP COMPONENTS CXX)

set(source_dir "${PROJECT_SOURCE_DIR}/src/")
set(source_files "${source_dir}/embeddingSD_unix.cpp")
set(source_generate_files "${source_dir}/generatingSD_unix.cpp")
set(embedder_adapter_source "${source_dir}/dmercator/embedding/embedder.cpp")
set(generator_adapter_source "${source_dir}/dmercator/core/generator.cpp")

set(include_dir "${PROJECT_SOURCE_DIR}/include/")
file(GLOB include_files "${include_dir}/*.hpp")

add_executable(mercator ${source_files} ${include_files})
add_executable(generatingSD ${source_generate_files} ${include_files})
add_executable(mercator_legacy apps/mercator_legacy.cpp include/legacy/embeddingSD_legacy.hpp include/legacy/embeddingSD_unix_legacy.hpp)
add_executable(embed_sd
  apps/embed_sd.cpp
  ${embedder_adapter_source}
  ${include_files}
  include/dmercator/io/coordinates_csv.hpp
  include/dmercator/embedding/embedder.hpp
  include/dmercator/embedding/engine.hpp
  include/dmercator/embedding/orchestrator.hpp
  include/dmercator/init/initialization.hpp
  include/dmercator/inference/inference.hpp
  include/dmercator/refine/refinement.hpp
  include/dmercator/io/embedding_io.hpp
  include/dmercator/core/helpers.hpp
)
add_executable(generate_sd
  apps/generate_sd.cpp
  ${generator_adapter_source}
  ${include_files}
  include/dmercator/io/coordinates_csv.hpp
  include/dmercator/core/generator.hpp
)

target_link_libraries(mercator trapecis_Sd Threads::Threads)
target_link_libraries(generatingSD trapecis_Sd Threads::Threads)
target_link_libraries(mercator_legacy trapecis_Sd Threads::Threads)
target_link_libraries(embed_sd trapecis_Sd Threads::Threads)
target_link_libraries(generate_sd trapecis_Sd Threads::Threads)

if(OpenMP_CXX_FOUND)
  target_link_libraries(mercator OpenMP::OpenMP_CXX)
  target_link_libraries(generatingSD OpenMP::OpenMP_CXX)
  target_link_libraries(mercator_legacy OpenMP::OpenMP_CXX)
  target_link_libraries(embed_sd OpenMP::OpenMP_CXX)
  target_link_libraries(generate_sd OpenMP::OpenMP_CXX)
endif()

if(BUILD_TESTING)
  add_executable(perf_preservation_smoke
    tests/perf_preservation_smoke.cpp
    ${include_files}
  )
  target_link_libraries(perf_preservation_smoke trapecis_Sd Threads::Threads)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(perf_preservation_smoke OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME perf_preservation_smoke COMMAND perf_preservation_smoke)
endif()
