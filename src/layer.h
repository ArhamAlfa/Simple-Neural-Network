#pragma once

#include <Eigen/Dense>
#include "utils.h"

class Layer
{
public:
    enum class LayerType;

private:
    Eigen::MatrixXd weights;
    Eigen::MatrixXd bias;
    LayerType type;
    Activation ActivationFunction;

    
    Eigen::VectorXd unactivated_z;
    Eigen::VectorXd previous_input;
    Eigen::MatrixXd error_signal;
    

public:
    Layer(int incident_count, int node_count, int random_state, ActivationType activation);
    Eigen::MatrixXd &get_weights();
    Eigen::VectorXd forward(Eigen::VectorXd &input_vector);
};