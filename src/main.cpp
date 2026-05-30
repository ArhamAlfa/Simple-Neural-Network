#include <iostream>
#include <Eigen/Dense>
#include "network.h"

int main()
{
    // Trying to create a network from user defined topology
    std::cout << "Creating a new neural network! ------------\n"
              << std::endl;
    Network new_network = {{2, 3, 5, 3}, 52, 0.01};
}