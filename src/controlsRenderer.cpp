#include "controlsRenderer.h"
#include "../libs/raygui.h"
#include <cmath>
#include <algorithm>

static float slider;

struct ControlsRegion
{
    float x;
    float y;
    float width;
    float height;
};

static float vw(Rectangle region, float percentage)
{
    return region.width * (percentage / 100.0f);
}

static float vw(ControlsRegion region, float percentage)
{
    return region.width * (percentage / 100.0f);
}

static float vh(Rectangle region, float percentage)
{
    return region.height * (percentage / 100.0f);
}

static float vh(ControlsRegion region, float percentage)
{
    return region.height * (percentage / 100.0f);
}

void render_controls(Rectangle view_region, std::string box_title, State &state)
{
    GuiGroupBox(view_region, box_title.c_str());

    GuiSetStyle(DEFAULT, BACKGROUND_COLOR, 0x1C1E26FF);
    GuiSetStyle(DEFAULT, BASE_COLOR_NORMAL, 0x2A2D3AFF);
    GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, 0xCCCCCCFF);
    GuiSetStyle(DEFAULT, BORDER_COLOR_NORMAL, 0x444860FF);

    float margin_x = vw(view_region, 20);
    float margin_y = vh(view_region, 30);

    // Tick Speed Slider
    Rectangle slider_rect =
        {
            view_region.x + margin_x,
            view_region.y + margin_y,
            view_region.width - 2 * margin_x,
            vh(view_region, 12)};

    // We want the slider to map to tick speed, so that as you drag
    // to the right it gets exponentially faster instead of linearly

    const float MINIMUM_SPEED = 1;
    const float MAXIMUM_SPEED = 2000;

    // Force tick speed to fall within boundaries
    state.tick_speed = std::clamp(state.tick_speed, MINIMUM_SPEED, MAXIMUM_SPEED);

    GuiSlider(slider_rect, "Slow", "Fast", &slider, 0.0, 1);

    state.tick_speed = 1 * std::pow(MAXIMUM_SPEED, slider);

    // Stop Visualization Slider
    Rectangle button_rect =
        {
            view_region.x + margin_x,
            view_region.y + margin_y + vh(view_region, 20),
            view_region.width - 2 * margin_x,
            vh(view_region, 15)};

    if (GuiButton(button_rect, "Stop Visualizer"))
    {
        state.do_visualize = FALSE;
    }
}