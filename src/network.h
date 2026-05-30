#pragma once

#include <Eigen/Dense>
#include <iostream>
#include "layer.h"
#include <vector>
#include <initializer_list>

class Network
{
private:
    std::vector<Layer> layers;
    int random_state;
    double learning_rate;
    // A dataset variable maybe?

public:
    Network(std::initializer_list<int> topology, int random_state, double learning_rate);
};