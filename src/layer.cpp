#include <Eigen/Dense>
#include <random>
#include <iostream>
#include "layer.h"
#include "activation.h"

Layer::Layer(int incident_count, int node_count, int random_state, ActivationType activation)
{
    // Initialize generator based on random seed
    std::mt19937 gen(random_state);

    // Create a normal distribution from 0 to 1
    std::normal_distribution<double> dist(0.0, 1.0);

    // Create a lambda function for the nullary expression to use
    auto normal_random_generator = [&gen, &dist]()
    { return dist(gen); };

    // Fill in the matrix values with the nullary expression generator
    // Creating a current_node x incident_node sized matrix to represent each layer
    this->weights = Eigen::MatrixXd::NullaryExpr(node_count, incident_count, normal_random_generator);
    this->bias = Eigen::VectorXd::Constant(node_count, 0.01);

    this->ActivationFunction = get_activation(activation);
}

Eigen::MatrixXd &Layer::get_weights()
{
    return this->weights;
}

Eigen::VectorXd Layer::forward(Eigen::VectorXd &input_vector)
{
    Eigen::VectorXd z = (this->weights * input_vector) + this->bias;

    std::cout << "\n>> Pre-activated output: " << std::endl;
    std::cout << z << std::endl;

    return z.unaryExpr(this->ActivationFunction.apply);
}
