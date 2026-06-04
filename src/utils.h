#pragma once

#include <functional>

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