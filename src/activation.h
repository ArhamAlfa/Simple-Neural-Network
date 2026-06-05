#pragma once
#include <cmath>
#include <algorithm>
#include "utils.h"

// returns the right set of activation functions to be used by the layer
inline Activation get_activation(ActivationType type)
{
    switch (type)
    {
    case ActivationType::RELU:
        return {
            [](double x)
            { return std::max(0.0, x); },
            [](double x)
            { return x > 0.0 ? 1.0 : 0.0; }};

    case ActivationType::SIGMOID:
        return {
            [](double x)
            { return 1.0 / (1.0 + std::exp(-x)); },
            [](double x)
            {
                double sig = 1.0 / (1.0 + std::exp(-x));
                return sig * (1.0 - sig);
            }};

    case ActivationType::TANH:
        return {
            [](double x)
            { return std::tanh(x); },
            [](double x)
            { return 1.0 - (std::tanh(x) * std::tanh(x)); }};

    default:
        throw std::invalid_argument("Unknown activation function!");
    }
}