cmake_minimum_required(VERSION 3.20)
project(dmercator Fortran CXX)

get_filename_component(PARENT_DIR ../ ABSOLUTE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2 -fopenmp -lpthread") # with -O3 compiler changes the code...
set(CMAKE_Fortran_FLAGS "-c -fpic -Wall -O3")

include_directories("${PARENT_DIR}/include")
include_directories("${PARENT_DIR}/include/*")
add_library(trapecis_Sd "${PARENT_DIR}/include/trapecis_Sd.f")

set(source_dir "${PARENT_DIR}/src/")
set(source_files "${source_dir}/embeddingSD_unix.cpp")

set(include_dir "${PARENT_DIR}/include/")
file(GLOB include_files "${include_dir}/*.hpp")

add_subdirectory(pybind11)
pybind11_add_module(dmercator "${include_files}" python_bind.cpp)

# add_executable(mercator ${source_files} ${include_files})
target_link_libraries(dmercator PRIVATE trapecis_Sd)
