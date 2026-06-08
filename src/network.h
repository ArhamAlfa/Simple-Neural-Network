#pragma once

#include <iostream>
#include "layer.h"
#include <vector>
#include <initializer_list>
#include "utils.h"
#include <mutex>

class Network
{
private:
    State &state;
    std::vector<Layer> layers;
    int random_state;
    double learning_rate;

    // A dataset variable maybe?

public:
    Network(std::initializer_list<int> topology, int random_state, double learning_rate, ActivationType h_act, ActivationType o_act, State &state);
    void help_create_layer(int prev_nodes, int current_nodes, int random_state, double learning_rate, ActivationType activation, int i = 0);
    forward_package forward_pass(Eigen::VectorXd &input);
    void backpropagate(Eigen::VectorXd &predicted, Eigen::VectorXd &actual);
    double compute_loss(Eigen::VectorXd &predicted, Eigen::VectorXd &actual);
    void console_print();
    void train_model(int epochs, Dataset dataset, double loss_threshold, State &state, std::mutex &queue_mutex);
    void test_model(Dataset dataset);
};