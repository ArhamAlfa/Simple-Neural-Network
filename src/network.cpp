#include <Eigen/Dense>
#include <iostream>
#include "layer.h"
#include "network.h"
#include <vector>
#include <initializer_list>

Network::Network(std::initializer_list<int> topology, int random_state, double learning_rate, ActivationType h_act, ActivationType o_act)
{
    this->random_state = random_state;
    this->learning_rate = learning_rate;

    // Convert initializer_list to a vector
    std::vector<int> top_vec(topology);

    // Start at index 1 as the input layer (index 0) doesn't own any weights.
    for (size_t i = 1; i < top_vec.size() - 1; i++)
    {
        // Grab current and previous node counts
        int prev_nodes = top_vec[i - 1];
        int current_nodes = top_vec[i];

        help_create_layer(prev_nodes, current_nodes, random_state, learning_rate, h_act, i);
    }

    // Final case for the output layer
    help_create_layer(top_vec[top_vec.size() - 2], top_vec[top_vec.size() - 1], random_state, learning_rate, o_act, top_vec.size() - 1);
}

// helper function for cleanliness
void Network::help_create_layer(int prev_nodes, int current_nodes, int random_state, double learning_rate, ActivationType activation, int i)
{
    // Add 'i' to the random state so each layer gets a unique seed
    // since if not the generator will generate the same 4-5 values for all the nodes
    Layer new_layer(prev_nodes, current_nodes, random_state + i, activation, learning_rate);

    layers.push_back(new_layer);

    // Print the matrix to verify the dimensions and random values
    std::cout << "--- Layer " << i << " Weights (" << current_nodes << "x" << prev_nodes << ") ---" << std::endl;
    std::cout << new_layer.get_weights() << "\n"
              << std::endl;
}

Eigen::VectorXd Network::forward_pass(Eigen::VectorXd &input)
{
    std::cout << "\n--- Feedforwarding the following vector ---" << std::endl;
    Eigen::VectorXd output = input;
    std::cout << "[ " << output << " ]" << std::endl;
    std::cout << "\n--- Resulting node outputs for each layer ---" << std::endl;

    // Forward passing logic for each layer
    for (int i = 0; i < layers.size(); i++)
    {
        // Printing
        std::cout << ">> Layer " << i + 1 << std::endl;
        std::cout << ">> Layer:" << std::endl;
        std::cout << layers[i].get_weights() << std::endl;

        std::cout << "\n>> Input: " << std::endl;
        std::cout << output << std::endl;

        // Continually overwrite and pass output to the next layer
        output = layers[i].forward(output);

        // Print the vector for testing purposes
        std::cout << "\n\n>> Output \n[ " << output.transpose() << " ]\n\n--\n"
                  << std::endl;
    }

    return output;
}

void Network::backpropagate(Eigen::VectorXd &predicted, Eigen::VectorXd &actual)
{
    // Backpropagate for output layer
    Eigen::VectorXd error;

    error = 2 * (predicted - actual);

    std::cout << "Starting Backpropagation Process on these actual outputs: \n[" << actual << "]" << std::endl;

    std::cout << "\n>> Output layer-----" << std::endl;
    error = layers[layers.size() - 1].back(error);

    for (int i = layers.size() - 2; i >= 0; i--)
    {
        std::cout << "\n\n---\n"
                  << std::endl;
        std::cout << "\n>> Hidden layer " << i + 1 << "-----" << std::endl;
        error = layers[i].back(error, layers[i + 1].get_weights());
    }

    console_print();
}

void Network::console_print()
{
    for (int i = 0; i < layers.size(); i++)
    {
        std::cout << "\n--- Layer " << i << " Weights (" << layers[i].get_weights().rows() << "x" << layers[i].get_weights().cols() << ") ---" << std::endl;
        std::cout << layers[i].get_weights() << "\n"
                  << std::endl;
    }
}