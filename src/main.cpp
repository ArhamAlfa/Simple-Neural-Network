#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "network.h"

int main()
{
    // redirect std::out to a txt file
    std::ofstream out("console_output.txt");
    std::streambuf *cout_buffer = std::cout.rdbuf(out.rdbuf());

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
    Eigen::VectorXd predicted = new_network.forward_pass(input);

    // Attempt back pass
    Eigen::VectorXd actual{{0.0784, 0.2340, 0.05583}};
    new_network.backpropagate(predicted, actual);

    // Restore console buffer
    std::cout.rdbuf(cout_buffer);
    return 0;
}