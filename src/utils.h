#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <variant>
#include <Eigen/Dense>

enum class ActivationType
{
    TANH,
    RELU,
    SIGMOID
};

// We bundle together the activation function and its derivative for use in both backprop and forward passing
struct Activation
{
    // They are both functions that take in a double argument and output a double; hence <double(double)>
    std::function<double(double)> apply;
    std::function<double(double)> derivative;
};

// Using variant to allow for different types for the metdata map
using DynamicValues = std::variant<int, double, std::string>;

// Simple SGD approach
struct Dataset
{
    std::vector<Eigen::VectorXd> inputs;
    std::vector<Eigen::VectorXd> labels;
    std::unordered_map<std::string, DynamicValues> metadata;
};