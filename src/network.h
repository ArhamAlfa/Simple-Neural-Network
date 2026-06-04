#pragma once

#include <Eigen/Dense>
#include <iostream>
#include "layer.h"
#include <vector>
#include <initializer_list>
#include "utils.h"

class Network
{
private:
    std::vector<Layer> layers;
    int random_state;
    double learning_rate;

    // A dataset variable maybe?

public:
    Network(std::initializer_list<int> topology, int random_state, double learning_rate, ActivationType h_act, ActivationType o_act);
    void help_create_layer(int prev_nodes, int current_nodes, int random_state, double learning_rate, ActivationType activation, int i = 0);
    Eigen::VectorXd forward_pass(Eigen::VectorXd &input);
    void backpropagate(Eigen::VectorXd &predicted, Eigen::VectorXd &actual);
    void console_print();
};