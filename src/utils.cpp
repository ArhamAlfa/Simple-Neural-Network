#include "utils.h"

// Helper function to return the activation type used
ActivationType get_activation_type(std::string string)
{
    if (string == "TANH")
    {
        return ActivationType::TANH;
    }
    else if (string == "RELU")
    {
        return ActivationType::RELU;
    }
    else if (string == "SIGMOID")
    {
        return ActivationType::SIGMOID;
    }
    else
    {
        // Set it as default for now
        return ActivationType::RELU;
    }
}