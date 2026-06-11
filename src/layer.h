#pragma once

#include "utils.h"
#include "activation.h"

class Layer
{
public:
    enum class LayerType;

private:
    Eigen::MatrixXd weights;
    Eigen::VectorXd bias;
    LayerType type;
    Activation ActivationFunction;

    // For Backprop
    Eigen::VectorXd unactivated_z;
    Eigen::VectorXd previous_input;
    double learning_rate;

    Eigen::MatrixXd rate_of_loss;
    Eigen::VectorXd error;

public:
    Layer(int incident_count, int node_count, int random_state, ActivationType activation, double learning_rate);
    Eigen::MatrixXd &get_weights();
    Eigen::VectorXd forward(Eigen::VectorXd &input_vector);

    // Back function polymorphism for case: hidden layer and output layer
    Eigen::VectorXd back(Eigen::VectorXd error);
    Eigen::VectorXd back(Eigen::VectorXd error, Eigen::MatrixXd next_weights);

    void apply_gradient_descent();
    void update_weights(Eigen::MatrixXd rate_of_loss);
    void update_bias(Eigen::VectorXd error);
};