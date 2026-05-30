#pragma once

#include <Eigen/Dense>

class Layer
{
public:
    enum class LayerType;
    enum class ActivationType;

private:
    Eigen::MatrixXd weights;
    LayerType type;
    ActivationType activation;

public:
    Layer(int incident_count, int node_count, int random_state);
    Eigen::MatrixXd &get_weights();
};