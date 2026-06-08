#pragma once

#include "raylib.h"
#include "utils.h"

class Visualizer
{
private:
    State &state;

public:
    Visualizer(State &state);
    void visualize(Rectangle &RegionB, Rectangle &RegionC);
    void visualize(Rectangle &RegionA);
};