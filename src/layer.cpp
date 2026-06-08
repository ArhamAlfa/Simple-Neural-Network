#include <Eigen/Dense>
#include <random>
#include <iostream>
#include "layer.h"
#include "activation.h"

Layer::Layer(int incident_count, int node_count, int random_state, ActivationType activation, double learning_rate)
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
    weights = Eigen::MatrixXd::NullaryExpr(node_count, incident_count, normal_random_generator);
    bias = Eigen::VectorXd::Constant(node_count, 0.01);

    ActivationFunction = get_activation(activation);
    this->learning_rate = learning_rate;
}

Eigen::MatrixXd &Layer::get_weights()
{
    return this->weights;
}

Eigen::VectorXd Layer::forward(Eigen::VectorXd &input_vector)
{
    previous_input = input_vector;
    Eigen::VectorXd z = (weights * input_vector) + bias;
    unactivated_z = z;

    // std::cout << "\n>> Pre-activated output: " << std::endl;
    // std::cout << z << std::endl;

    return z.unaryExpr(ActivationFunction.apply);
}

Eigen::VectorXd Layer::back(Eigen::VectorXd error)
{
    // std::cout << "\n> Received error = \n"
    //           << error << std::endl;

    // std::cout << "\n> Unactivated z = \n"
    //           << unactivated_z << std::endl;

    // Calculate
    error = error.cwiseProduct(unactivated_z.unaryExpr(ActivationFunction.derivative));

    // std::cout << "\n> New error = \n"
    //           << error << std::endl;

    // Calculate Rate of Loss
    Eigen::MatrixXd rate_of_loss = error * previous_input.transpose();
    // std::cout << "\n> delta = \n"
    //           << rate_of_loss << std::endl;

    // Adjust weights by Gradient Descent
    update_weights(rate_of_loss);

    // Adjust bias by error
    update_bias(error);

    return error;
}

Eigen::VectorXd Layer::back(Eigen::VectorXd error, Eigen::MatrixXd next_weights)
{
    // std::cout << "\n> Received error = \n"
    //           << error << std::endl;

    // std::cout << "\n> Unactivated z = \n"
    //           << unactivated_z << std::endl;
    // Calculate
    error = (next_weights.transpose() * error).cwiseProduct(unactivated_z.unaryExpr(ActivationFunction.derivative));

    // std::cout << "\n> New error = \n"
    //           << error << std::endl;

    // Calculate Rate of Loss
    Eigen::MatrixXd rate_of_loss = error * previous_input.transpose();
    // std::cout << "\n> delta = \n"
    //           << rate_of_loss << std::endl;

    // Adjust weights by Gradient Descent
    update_weights(rate_of_loss);

    // Adjust bias by error
    update_bias(error);

    return error;
}

void Layer::update_weights(Eigen::MatrixXd rate_of_loss)
{
    Eigen::MatrixXd new_weights = weights - (learning_rate * rate_of_loss);
    weights = new_weights;
}

void Layer::update_bias(Eigen::VectorXd error)
{
    Eigen::VectorXd new_bias = bias - (learning_rate * error);
    bias = new_bias;
}