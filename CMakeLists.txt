cmake_minimum_required(VERSION 3.20)
project(mercator Fortran CXX)
include(CTest)

option(DMERCATOR_RELEASE_O3 "Use -O3 in Release builds (default keeps -O2)." OFF)
option(DMERCATOR_ENABLE_NATIVE "Enable -march=native for local tuning." OFF)
option(DMERCATOR_ENABLE_LTO "Enable IPO/LTO in Release builds when supported." OFF)
option(D_MERCATOR_CUDA "Build CUDA acceleration for embedding hotspots." OFF)
option(USE_CUDA "Deprecated alias for D_MERCATOR_CUDA." OFF)

if(USE_CUDA)
  set(D_MERCATOR_CUDA ON)
endif()

set(DMERCATOR_RELEASE_OPT_FLAG "-O2")
if(DMERCATOR_RELEASE_O3)
  set(DMERCATOR_RELEASE_OPT_FLAG "-O3")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} ${DMERCATOR_RELEASE_OPT_FLAG}")
set(CMAKE_Fortran_FLAGS "-c -fpic -Wall -O3")

if(D_MERCATOR_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
  endif()
  find_package(CUDAToolkit REQUIRED)
endif()

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
set(gpu_embedding_sources
  src/gpu/likelihood_backend_null.cpp
)
set(gpu_generator_sources)
if(D_MERCATOR_CUDA)
  set(gpu_embedding_sources
    ${gpu_embedding_sources}
    src/gpu/likelihood_kernels.cu
    src/gpu/likelihood_backend_cuda.cu
  )
  set(gpu_generator_sources
    src/gpu/cuda_utils.cu
    src/gpu/kernels.cu
    src/gpu/gpu_context.cu
  )
endif()

set(include_dir "${PROJECT_SOURCE_DIR}/include/")
file(GLOB include_files "${include_dir}/*.hpp")

add_executable(mercator ${source_files} ${include_files} ${gpu_embedding_sources})
add_executable(generatingSD ${source_generate_files} ${include_files} ${gpu_generator_sources})
add_executable(mercator_legacy apps/mercator_legacy.cpp include/legacy/embeddingSD_legacy.hpp include/legacy/embeddingSD_unix_legacy.hpp)
add_executable(embed_sd
  apps/embed_sd.cpp
  ${embedder_adapter_source}
  ${gpu_embedding_sources}
  ${include_files}
  include/dmercator/io/coordinates_csv.hpp
  include/dmercator/embedding/embedder.hpp
  include/dmercator/engine.hpp
  include/dmercator/initialize.hpp
  include/dmercator/infer_parameters.hpp
  include/dmercator/infer_initial_positions.hpp
  include/dmercator/refine_positions.hpp
  include/dmercator/validation.hpp
  include/dmercator/utils.hpp
)
add_executable(generate_sd
  apps/generate_sd.cpp
  ${generator_adapter_source}
  ${gpu_generator_sources}
  ${include_files}
  include/dmercator/io/coordinates_csv.hpp
  include/dmercator/core/generator.hpp
)

target_link_libraries(mercator trapecis_Sd Threads::Threads)
target_link_libraries(generatingSD trapecis_Sd Threads::Threads)
target_link_libraries(mercator_legacy trapecis_Sd Threads::Threads)
target_link_libraries(embed_sd trapecis_Sd Threads::Threads)
target_link_libraries(generate_sd trapecis_Sd Threads::Threads)

if(D_MERCATOR_CUDA)
  target_compile_definitions(mercator PRIVATE DMERCATOR_USE_CUDA=1)
  target_compile_definitions(embed_sd PRIVATE DMERCATOR_USE_CUDA=1)
  target_compile_definitions(generatingSD PRIVATE DMERCATOR_USE_CUDA=1)
  target_compile_definitions(generate_sd PRIVATE DMERCATOR_USE_CUDA=1)
  target_compile_options(mercator PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  target_compile_options(embed_sd PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  target_compile_options(generatingSD PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  target_compile_options(generate_sd PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  target_link_libraries(mercator CUDA::cudart)
  target_link_libraries(embed_sd CUDA::cudart)
  target_link_libraries(generatingSD CUDA::cudart)
  target_link_libraries(generate_sd CUDA::cudart)
  target_link_libraries(mercator CUDA::cusparse CUDA::cublas CUDA::cusolver)
  target_link_libraries(embed_sd CUDA::cusparse CUDA::cublas CUDA::cusolver)
  target_link_libraries(generatingSD CUDA::cusparse CUDA::cublas CUDA::cusolver)
  target_link_libraries(generate_sd CUDA::cusparse CUDA::cublas CUDA::cusolver)
  set_target_properties(mercator PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(embed_sd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(generatingSD PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(generate_sd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

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
    ${gpu_embedding_sources}
    ${include_files}
  )
  target_link_libraries(perf_preservation_smoke trapecis_Sd Threads::Threads)
  if(D_MERCATOR_CUDA)
    target_compile_definitions(perf_preservation_smoke PRIVATE DMERCATOR_USE_CUDA=1)
    target_compile_options(perf_preservation_smoke PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    target_link_libraries(perf_preservation_smoke CUDA::cudart CUDA::cusparse CUDA::cublas CUDA::cusolver)
    set_target_properties(perf_preservation_smoke PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endif()
  if(OpenMP_CXX_FOUND)
    target_link_libraries(perf_preservation_smoke OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME perf_preservation_smoke COMMAND perf_preservation_smoke)

  add_executable(likelihood_backend_smoke
    tests/likelihood_backend_smoke.cpp
    ${gpu_embedding_sources}
    ${include_files}
  )
  target_link_libraries(likelihood_backend_smoke trapecis_Sd Threads::Threads)
  if(D_MERCATOR_CUDA)
    target_compile_definitions(likelihood_backend_smoke PRIVATE DMERCATOR_USE_CUDA=1)
    target_compile_options(likelihood_backend_smoke PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    target_link_libraries(likelihood_backend_smoke CUDA::cudart CUDA::cusparse CUDA::cublas CUDA::cusolver)
    set_target_properties(likelihood_backend_smoke PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endif()
  if(OpenMP_CXX_FOUND)
    target_link_libraries(likelihood_backend_smoke OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME likelihood_backend_smoke COMMAND likelihood_backend_smoke)

  add_executable(negative_sampling_cpu_smoke
    tests/negative_sampling_cpu_smoke.cpp
    ${gpu_embedding_sources}
    ${include_files}
  )
  target_link_libraries(negative_sampling_cpu_smoke trapecis_Sd Threads::Threads)
  if(D_MERCATOR_CUDA)
    target_compile_definitions(negative_sampling_cpu_smoke PRIVATE DMERCATOR_USE_CUDA=1)
    target_compile_options(negative_sampling_cpu_smoke PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    target_link_libraries(negative_sampling_cpu_smoke CUDA::cudart CUDA::cusparse CUDA::cublas CUDA::cusolver)
    set_target_properties(negative_sampling_cpu_smoke PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endif()
  if(OpenMP_CXX_FOUND)
    target_link_libraries(negative_sampling_cpu_smoke OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME negative_sampling_cpu_smoke COMMAND negative_sampling_cpu_smoke)

  if(D_MERCATOR_CUDA)
    add_executable(negative_sampling_backend_smoke
      tests/negative_sampling_backend_smoke.cpp
      ${gpu_embedding_sources}
      ${include_files}
    )
    target_compile_definitions(negative_sampling_backend_smoke PRIVATE DMERCATOR_USE_CUDA=1)
    target_compile_options(negative_sampling_backend_smoke PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    target_link_libraries(negative_sampling_backend_smoke trapecis_Sd Threads::Threads CUDA::cudart CUDA::cusparse CUDA::cublas CUDA::cusolver)
    set_target_properties(negative_sampling_backend_smoke PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    if(OpenMP_CXX_FOUND)
      target_link_libraries(negative_sampling_backend_smoke OpenMP::OpenMP_CXX)
    endif()
    add_test(NAME negative_sampling_backend_smoke COMMAND negative_sampling_backend_smoke)
  endif()
endif()
