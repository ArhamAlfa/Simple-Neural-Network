# Require a minimum version of CMake
cmake_minimum_required(VERSION 3.10)

# Require CMake to use C++ version 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Name your project
project(NeuralNetSandbox)

# Tell CMake to create an executable file named 'nn_test' from 'main.cpp'
add_executable(nn_test 
    src/main.cpp
    src/network.cpp
    src/layer.cpp
    src/dataset.cpp
    )

# Tell CMake where to find the Eigen headers
# CMAKE_SOURCE_DIR is a variable pointing to the folder containing this CMakeLists.txt
target_include_directories(nn_test PRIVATE ${CMAKE_SOURCE_DIR}/libs/Eigen)