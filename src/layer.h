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

    // For Backprop
    Eigen::VectorXd unactivated_z;
    Eigen::VectorXd previous_input;
    Eigen::MatrixXd error_signal;
    double learning_rate;

public:
    Layer(int incident_count, int node_count, int random_state, ActivationType activation, double learning_rate);
    Eigen::MatrixXd &get_weights();
    Eigen::VectorXd forward(Eigen::VectorXd &input_vector);

    // Back function polymorphism for case: hidden layer and output layer
    Eigen::VectorXd back(Eigen::VectorXd error);
    Eigen::VectorXd back(Eigen::VectorXd error, Eigen::MatrixXd next_weights);

    void update_weights(Eigen::MatrixXd rate_of_loss);
    void update_bias(Eigen::VectorXd error);
};