#include "lossRenderer.h"
#include "../libs/raygui.h"
#include <queue>

struct Graph
{
    // Origins
    float x;
    float y;

    // Dimensions
    float width;
    float height;

    // Data
    double max_value;
};

struct Point
{
    float x;
    float y;
};

static float vw(Rectangle region, float percentage)
{
    return region.width * (percentage / 100.0f);
}

static float vw(Graph region, float percentage)
{
    return region.width * (percentage / 100.0f);
}

static float vh(Rectangle region, float percentage)
{
    return region.height * (percentage / 100.0f);
}

static float vh(Graph region, float percentage)
{
    return region.height * (percentage / 100.0f);
}

// ratio of an elements x position in relation to the whole array
float ratiox(int index, int total_elements)
{
    return static_cast<float>(index) / (total_elements - 1);
}

// ratio of an elements y position in relation to the whole array
float ratioy(double loss_value, double max_value)
{
    return static_cast<float>(loss_value / max_value);
}

void render_loss(Rectangle view_region, std::string box_title, State &state)
{
    GuiGroupBox(view_region, box_title.c_str());

    // std::cout << "WEEEEEEEEE" << std::endl;

    Graph graph = {
        view_region.x + vw(view_region, 10),
        view_region.y + vh(view_region, 7),
        vw(view_region, 80),
        vh(view_region, 86),
        0.0,
    };

    // std::cout << "WAAAAAAAAAAAAAAA" << std::endl;

    // X Bar
    Vector2 start_pos_x = {graph.x - vh(graph, 1.5), graph.y + graph.height};
    Vector2 end_pos_x = {graph.x + graph.width, graph.y + graph.height};
    DrawLineEx(start_pos_x, end_pos_x, vh(graph, 3), {60, 65, 80, 255});

    // Y Bar
    Vector2 start_pos_y = {graph.x, graph.y};
    Vector2 end_pos_y = {graph.x, graph.y + graph.height + vh(graph, 1.5)};
    DrawLineEx(start_pos_y, end_pos_y, vh(graph, 3), {60, 65, 80, 255});

    // Makes sure a loss history exists to prevent segfaulting
    if (state.loss_history.empty() || state.loss_history.size() < 2)
    {
        return;
    }

    // Get range of values from loss
    graph.max_value = state.loss_history[0];

    for (int i = 0; i < state.loss_history.size(); i++)
    {
        if (state.loss_history[i] > graph.max_value)
        {
            graph.max_value = state.loss_history[i];
        }
    }

    // Defining graph configuration stuff
    float y_margin = vh(graph, 15);
    float y_top = graph.y + y_margin;
    float y_bottom = graph.y + graph.height - vh(graph, 5);
    float y_useable = y_bottom - y_top;

    float x_margin = vw(graph, 5);
    float x_left = graph.x + x_margin;
    float x_right = graph.x + graph.width;
    float x_useable = x_right - x_left - x_margin;

    std::queue<Point> pairings;

    // std::cout << "WEEEEEEEEE" << std::endl;

    for (int i = 0; i < state.loss_history.size(); i++)
    {
        float y_coordinate = y_bottom - (ratioy(state.loss_history[i], graph.max_value) * y_useable);

        float x_coordinate = x_left + (ratiox(i, state.loss_history.size()) * x_useable);

        pairings.push({x_coordinate, y_coordinate});

        if (pairings.size() > 1)
        {
            Vector2 start_pos = {pairings.front().x, pairings.front().y};
            Vector2 end_pos = {pairings.back().x, pairings.back().y};
            DrawLineEx(start_pos, end_pos, vh(graph, 1.2), {80, 220, 160, 255});

            pairings.pop();
        }
    }

    // std::cout << "Loss Render Successful" << std::endl;
}
