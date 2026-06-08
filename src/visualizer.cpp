#include "visualizer.h"
#include "networkRenderer.h"
#include "lossRenderer.h"
#include "controlsRenderer.h"

static float vw(float percentage)
{
    return GetScreenWidth() * (percentage / 100.0f);
}

static float vh(float percentage)
{
    return GetScreenHeight() * (percentage / 100.0f);
}

Visualizer::Visualizer(State &state) : state(state) {};

void Visualizer::visualize(Rectangle &RegionB, Rectangle &RegionC)
{
    render_loss(RegionB, "Loss Curve", state);
    render_controls(RegionC, "Controls", state);
}

void Visualizer::visualize(Rectangle &RegionA)
{
    render_network(RegionA, "Neural Network", state);
}
