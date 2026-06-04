#include <iostream>
#include <Eigen/Dense>
#include "network.h"

int main()
{
    // Trying to create a network from user defined topology
    std::cout << "Creating a new neural network! ------------\n"
              << std::endl;

    Network new_network = {
        /*topology= */ {2, 3, 5, 3},
        /*random_state= */ 52,
        /*learning_rate= */ 0.01,
        /*h_act= */ ActivationType::RELU,
        /*o_act= */ ActivationType::SIGMOID};

    // Attempt forward pass

    Eigen::VectorXd input{{0.2367, 0.9234}};

    new_network.forward_pass(input);
}