cmake_minimum_required(VERSION 3.15)
project(SpikingGraph CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library
add_library(spikinggraph SHARED
    src/spikinggraph.cpp
    src/api/api.cpp
)

target_include_directories(spikinggraph PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(spikinggraph PRIVATE SPKG_EXPORTS)
install(FILES include/spikinggraph.h DESTINATION include)
install(FILES src/api/api.h DESTINATION include)

# Example executable
add_executable(run_example examples/run_example.cpp)
target_link_libraries(run_example PRIVATE spikinggraph)

# Install targets
install(TARGETS spikinggraph run_example DESTINATION bin)
