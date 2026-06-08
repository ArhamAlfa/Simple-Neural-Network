# Require a minimum version of CMake
cmake_minimum_required(VERSION 3.10)

# Require CMake to use C++ version 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Name your project
project(NeuralNetSandbox)

# Tell CMake to create an executable file named 'visual_test' from 'main.cpp'
add_executable(visual_test 
    src/orchestrator.cpp
    src/network.cpp
    src/layer.cpp
    src/dataset.cpp
    src/utils.cpp
    src/visualizer.cpp
    src/networkRenderer.cpp
    src/lossRenderer.cpp
    src/controlsRenderer.cpp
    )

# Tell CMake where to find the Eigen headers
# CMAKE_SOURCE_DIR is a variable pointing to the folder containing this CMakeLists.txt
target_include_directories(visual_test PRIVATE ${CMAKE_SOURCE_DIR}/libs/Eigen)

# Since raylib is a compiled library, we need to compile it first then link
# the binary file to our project
add_subdirectory(libs/raylib)
target_link_libraries(visual_test PRIVATE raylib)