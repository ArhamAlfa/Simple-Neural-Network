#include <Eigen/Dense>
#include <iostream>
#include "layer.h"
#include "network.h"
#include <vector>
#include <initializer_list>
#include <cmath>

Network::Network(std::vector<int> topology, int random_state, double learning_rate, ActivationType h_act, ActivationType o_act, State &state) : state(state)
{
    this->random_state = random_state;
    this->learning_rate = learning_rate;
    state.topology = topology;

    // Start at index 1 as the input layer (index 0) doesn't own any weights.
    for (size_t i = 1; i < topology.size() - 1; i++)
    {
        // Grab current and previous node counts
        int prev_nodes = topology[i - 1];
        int current_nodes = topology[i];

        help_create_layer(prev_nodes, current_nodes, random_state, learning_rate, h_act, i);
    }

    // Final case for the output layer
    help_create_layer(topology[topology.size() - 2], topology[topology.size() - 1], random_state, learning_rate, o_act, topology.size() - 1);

    std::vector<Eigen::MatrixXd> weights;

    // Add all layers to state
    for (int i = 0; i < layers.size(); i++)
    {
        weights.push_back(layers[i].get_weights());
    }

    state.weights = weights;
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

forward_package Network::forward_pass(Eigen::VectorXd &input)
{
    // std::cout << "\n--- Feedforwarding the following vector ---" << std::endl;
    Eigen::VectorXd output = input;
    std::vector<Eigen::MatrixXd> weights;
    // std::cout << "[ " << output << " ]" << std::endl;
    // std::cout << "\n--- Resulting node outputs for each layer ---" << std::endl;

    // Forward passing logic for each layer
    for (int i = 0; i < layers.size(); i++)
    {
        // Printing
        // std::cout << ">> Layer " << i + 1 << std::endl;
        // std::cout << ">> Layer:" << std::endl;
        // std::cout << layers[i].get_weights() << std::endl;

        // std::cout << "\n>> Input: " << std::endl;
        // std::cout << output << std::endl;

        // Continually overwrite and pass output to the next layer
        weights.push_back(layers[i].get_weights());
        output = layers[i].forward(output);

        // Print the vector for testing purposes
        // std::cout << "\n\n>> Output \n[ " << output.transpose() << " ]\n\n--\n"
        //           << std::endl;
    }

    return {weights, output};
}

void Network::backpropagate(Eigen::VectorXd &predicted, Eigen::VectorXd &actual)
{
    // Backpropagate for output layer
    Eigen::VectorXd error;

    // Derivative of squared error
    error = 2 * (predicted - actual);

    // std::cout << "Starting Backpropagation Process on these actual outputs: \n[" << actual << "]" << std::endl;

    // std::cout << "\n>> Output layer-----" << std::endl;
    error = layers[layers.size() - 1].back(error);

    for (int i = layers.size() - 2; i >= 0; i--)
    {
        // std::cout << "\n\n---\n"
        //           << std::endl;
        // std::cout << "\n>> Hidden layer " << i + 1 << "-----" << std::endl;
        error = layers[i].back(error, layers[i + 1].get_weights());
    }

    // console_print();
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

double Network::compute_loss(Eigen::VectorXd &predicted, Eigen::VectorXd &actual)
{
    Eigen::VectorXd loss = (predicted - actual).array().square();

    double sum = loss.array().sum();

    return std::sqrt(sum / loss.size());
}

void Network::train_model(int epochs, Dataset &dataset, double loss_threshold, State &state, std::mutex &queue_mutex)
{
    int count = 0;
    float computed_loss = 10000; // arbitrary large value

    // Create snapshot
    Snapshot snap;

    // Actual training loop
    while (count < epochs && computed_loss > loss_threshold)
    {
        forward_package forward_results;

        // std::cout << "===== Training Epoch: " << count << " =====" << std::endl;

        // For Stochastic Gradient Descent, loop through each sample in the dataset
        for (int i = 0; i < dataset.inputs.size(); i++)
        {
            // Do a single forward pass of an input
            forward_results = forward_pass(dataset.inputs[i]);

            // backpropagate through a single row from a given forward pass result
            backpropagate(forward_results.prediction, dataset.labels[i]);

            // update computed loss
            computed_loss = compute_loss(forward_results.prediction, dataset.labels[i]);
            // std::cout << ">> Current loss: " << computed_loss << std::endl;
        }

        count++;

        snap.epoch = count;
        snap.loss = computed_loss;
        snap.weights = forward_results.weights;

        std::lock_guard<std::mutex> lock(queue_mutex);
        state.epoch_snapshots.push(snap);
    }

    std::cout << "--- Training Finished! ---" << std::endl;
    std::cout << "\n\n\n"
              << std::endl;

    test_model(dataset);
}

void Network::test_model(Dataset &dataset)
{
    std::cout << "===== Testing Model on " << std::get<std::string>(dataset.metadata["dataset_name"]) << " =====" << std::endl;

    int correct_count = 0;
    int total_samples = dataset.labels.size();

    for (int i = 0; i < total_samples; i++)
    {
        int predicted_index;
        int actual_index;

        // MaxCoeff changes the referenced variables to the index of the max
        // coefficient
        forward_pass(dataset.inputs[i]).prediction.maxCoeff(&predicted_index);
        dataset.labels[i].maxCoeff(&actual_index);

        if (predicted_index == actual_index)
        {
            correct_count++;
        }
    }

    // Calculations
    int missclassified = total_samples - correct_count;
    double accuracy = (static_cast<double>(correct_count) / total_samples) * 100;

    // Print
    std::cout << "\n---- Model Evaluation Report -----" << std::endl;
    std::cout << "Total Samples: " << total_samples << std::endl;
    std::cout << "Correct Predictions: " << correct_count << std::endl;
    std::cout << "Incorrect Predictions: " << missclassified << std::endl;
    std::cout << "Overall Accuracy: " << accuracy << "%" << std::endl;
}
