#include <Eigen/Dense>
#include <iostream>
#include "layer.h"
#include "network.h"
#include <vector>
#include <initializer_list>

Network::Network(std::initializer_list<int> topology, int random_state, double learning_rate)
{
    this->random_state = random_state;
    this->learning_rate = learning_rate;

    // Convert initializer_list to a vector
    std::vector<int> top_vec(topology);

    // Start at index 1 as the input layer (index 0) doesn't own any weights.
    for (size_t i = 1; i < top_vec.size(); i++)
    {
        // Grab current and previous node counts
        int prev_nodes = top_vec[i - 1];
        int current_nodes = top_vec[i];

        // Add 'i' to the random state so each layer gets a unique seed
        // since if not the generator will generate the same 4-5 values for all the nodes
        Layer new_layer(prev_nodes, current_nodes, random_state + i);

        this->layers.push_back(new_layer);

        // Print the matrix to verify the dimensions and random values
        std::cout << "--- Layer " << i << " Weights (" << current_nodes << "x" << prev_nodes << ") ---" << std::endl;
        std::cout << new_layer.get_weights() << "\n"
                  << std::endl;
    }
}
