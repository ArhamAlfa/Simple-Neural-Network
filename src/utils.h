#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <variant>
#include <Eigen/Dense>
#include <queue>
#include <mutex>

enum class ActivationType
{
    TANH,
    RELU,
    SIGMOID
};

// Helper function to return the activation type used
ActivationType get_activation_type(std::string string);

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

struct Snapshot
{
    std::vector<Eigen::MatrixXd> weights;
    double loss;
    int epoch;
};

// shared state for network and graphics
struct State
{
    // Neural Network Visualizer
    std::vector<int> topology;
    std::vector<Eigen::MatrixXd> weights;
    int active_layer_index;
    int previous_layer_index;

    std::queue<Snapshot> epoch_snapshots;
    std::mutex queue_mutex;

    // Loss Curve
    std::vector<double> loss_history;
    std::vector<double> loss_rate_history;
    int current_epoch;

    // Visualize or Not?
    bool do_visualize;

    // Visualizer config
    float tick_speed;
    bool is_backpropping;
};

struct forward_package
{
    std::vector<Eigen::MatrixXd> weights;
    Eigen::VectorXd prediction;
};