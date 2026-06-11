#include "networkRenderer.h"
#include "../libs/raygui.h"
#include <cmath>

struct Node
{
    float x;
    float y;
};

struct NetworkRegion
{
    // Origins
    float x;
    float y;

    // Dimensions
    float width;
    float height;

    // Intervals
    float node_interval;
    float layer_interval;
};

static float vw(Rectangle region, float percentage)
{
    return region.width * (percentage / 100.0f);
}

static float vw(NetworkRegion region, float percentage)
{
    return region.width * (percentage / 100.0f);
}

static float vh(Rectangle region, float percentage)
{
    return region.height * (percentage / 100.0f);
}

static float vh(NetworkRegion region, float percentage)
{
    return region.height * (percentage / 100.0f);
}

void highlight_layer(int layer_index, NetworkRegion &network, std::vector<std::vector<Node>> &network_array)
{
    float layer_height = (network_array[layer_index].size() - 1) * network.node_interval;

    Rectangle highlight_square = {network_array[layer_index][0].x - vw(network, 6),
                                  network_array[layer_index][0].y - vh(network, 9),
                                  vw(network, 12),
                                  layer_height + vh(network, 18)};
    float roundness = 0.4;
    int segments = 0;

    DrawRectangleRounded(highlight_square, roundness, segments, {255, 200, 80, 40});
}

void render_network(Rectangle view_region, std::string box_title, State &state)
{
    GuiGroupBox(view_region, box_title.c_str());

    if (state.active_layer_index == state.previous_layer_index)
    {
        state.is_backpropping = true;
    }

    // Recreating Network from topology as layers and nodes
    std::vector<std::vector<Node>> network_array;
    int max_node = 0;

    for (int node_count : state.topology)
    {
        std::vector<Node> layer_array;

        for (int i = 0; i < node_count; i++)
        {
            layer_array.push_back({});
        }

        network_array.push_back(layer_array);

        if (node_count > max_node)
        {
            max_node = node_count;
        }
    }

    // std::cout << "YAAAAAAAAAA" << std::endl;

    NetworkRegion current_network;

    // Determining NetworkRegion bounds
    float margin_x = vw(view_region, 8);
    float margin_y = vh(view_region, 15);

    current_network.x = view_region.x + margin_x;
    current_network.y = view_region.y + margin_y;

    current_network.width = view_region.width - 2 * (margin_x);
    current_network.height = view_region.height - 2 * (margin_y);

    // Determine single node-interval unit all nodes use
    current_network.node_interval = current_network.height / (max_node);

    // Determine interval between layers
    current_network.layer_interval = current_network.width / (network_array.size() - 1);

    float x_offset = current_network.x;

    // std::cout << "YOOOOOO" << std::endl;

    // Create Nodes
    for (int i = 0; i < network_array.size(); i++)
    {
        // How many node intervals a single layer will use
        float layer_height = (network_array[i].size() - 1) * current_network.node_interval;

        float empty_space = current_network.height - layer_height;

        // Node's top-y offset
        float y_offset = empty_space / 2;

        for (int j = 0; j < network_array[i].size(); j++)
        {
            network_array[i][j].x = x_offset;
            network_array[i][j].y = current_network.y + y_offset;

            // DrawCircle(network_array[i][j].x, network_array[i][j].y, vw(current_network, 2.5), ORANGE);

            // Increase y offset by itself
            y_offset += current_network.node_interval;
        }

        // Increase x_offset
        x_offset += current_network.layer_interval;
    }

    // std::cout << "YAAAAAAAAAA" << std::endl;

    // Draw Highlight
    for (int i = 0; i < network_array.size(); i++)
    {
        // std::cout << state.is_backpropping << std::endl;
        // Highlight layer
        if (i == state.active_layer_index && !state.is_backpropping)
        {
            // std::cout << "Check for highlight" << std::endl;
            highlight_layer(i, current_network, network_array);
        }
    }

    // std::cout << "YOOOOOO" << std::endl;

    // Draw Weights
    if (state.weights.size() == network_array.size() - 1)
    {

        for (int i = 0; i < network_array.size() - 1; i++)
        {
            // std::cout << "Check 1, i: " << i << std::endl;
            // If the next layer is active while backpropagating, then highlight the current set of weights
            bool is_active_connection = (i + 1 == state.active_layer_index);

            // Access each node in layer
            for (int j = 0; j < network_array[i].size(); j++)
            {
                // std::cout << "Check 2, j: " << j << std::endl;
                // Access every other node in adjacent layer
                for (int k = 0; k < network_array[i + 1].size(); k++)
                {
                    // std::cout << "Check 3, k: " << k << std::endl;

                    Vector2 start_pos = {network_array[i][j].x, network_array[i][j].y};
                    Vector2 end_pos = {network_array[i + 1][k].x, network_array[i + 1][k].y};

                    // std::cout << "Nodes accessed" << std::endl;

                    double weight = std::tanh(std::abs(state.weights[i](k, j)));

                    // std::cout << "state.weights accessed" << std::endl;

                    Color line_color;
                    Color circle_color;

                    if (state.is_backpropping && is_active_connection)
                    {
                        line_color = {180, 100, 255, 120};
                        circle_color = {200, 120, 255, 255};
                    }
                    else
                    {
                        line_color = {80, 160, 220, 100};
                        circle_color = {100, 210, 255, 255};
                    }

                    DrawLineEx(start_pos, end_pos, 1 + vw(current_network, weight), line_color);
                    DrawCircle(network_array[i][j].x, network_array[i][j].y, vw(current_network, 2.5), circle_color);
                }
            }
        }
    }

    // std::cout << "YAAAAAAAAAA" << std::endl;

    // Print the last layer
    int last_layer_index = network_array.size() - 1;
    for (int j = 0; j < network_array[last_layer_index].size(); j++)
    {
        Color circle_color;
        if (state.is_backpropping && last_layer_index == state.active_layer_index)
        {
            circle_color = {200, 120, 255, 255};
        }
        else
        {
            circle_color = {100, 210, 255, 255};
        }
        DrawCircle(network_array[last_layer_index][j].x, network_array[last_layer_index][j].y, vw(current_network, 2.5), circle_color);
    }

    // std::cout << "YOOOOOO" << std::endl;
}