#include "raylib.h"

#define RAYGUI_IMPLEMENTATION
#include "../libs/raygui.h"

#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <stdexcept>
#include "network.h"
#include "dataset.h"
#include "visualizer.h"

float vw(float percentage)
{
    return GetScreenWidth() * (percentage / 100.0f);
}

float vh(float percentage)
{
    return GetScreenHeight() * (percentage / 100.0f);
}

int main()
{
    // redirect std::out to a txt file
    std::ofstream out("console_output.txt");
    std::streambuf *cout_buffer = std::cout.rdbuf(out.rdbuf());

    // EVENTUAL IMPLEMENTATION OF NETWORK AND DATASET INITIALIZATION FROM
    // A PROVIDED JSON FILE WITH THE FIELDS BELOW:

    State state;

    std::string process_name = "XOR Problem Neural Network";
    int random_state = 52;
    double learning_rate = 0.1;
    std::initializer_list<int> topology = {2, 2, 1}; // Based on XOR Problem's 2, 2, 1
    std::string dataset_name = "XOR Problem";
    std::string csv_path = "../dataset/raw/xor.csv";
    std::string save_path = "../dataset/saved/XOR_Data.bin";
    std::string hidden_layer_activation = "RELU";
    std::string output_layer_activation = "SIGMOID";
    int epochs = 5000;
    double loss_threshold = 0.1;
    bool do_visualize = true;

    state.do_visualize = do_visualize;

    // Initialize Network
    Network network = {
        /*topology= */ topology,
        /*random_state= */ random_state,
        /*learning_rate= */ learning_rate,
        /*h_act= */ get_activation_type(hidden_layer_activation),
        /*o_act= */ get_activation_type(output_layer_activation),
        /*shared_state= */ state};

    std::cout << "Initialized Network" << std::endl;

    // Initialize Dataset
    Dataset dataset = get_dataset(
        /*dataset_name= */ dataset_name,
        /*csv_path= */ csv_path,
        /*save_path= */ save_path,
        /*input_size= */ *topology.begin(),
        /*output_size= */ *topology.end());

    std::cout << "Initialized Dataset" << std::endl;

    // Initialize Visualizer
    Visualizer visualizer = {state};

    std::cout << "Initialized Visualizer" << std::endl;

    // // Fake values fr:
    state.tick_speed = 0.2f;
    state.is_backpropping = false;

    std::vector<int> layer_sequence;
    int playhead = 0;
    Snapshot current_snapshot;

    for (int i = 0; i < state.topology.size(); i++)
    {
        layer_sequence.push_back(i);
    }

    for (int i = state.topology.size() - 1; i >= 0; i--)
    {
        layer_sequence.push_back(i);
    }

    state.active_layer_index = layer_sequence[playhead];

    float time_accumulator = 0.0;

    std::cout << "Starting Thread" << std::endl;
    std::thread training_thread(&Network::train_model, &network, epochs, dataset, loss_threshold, std::ref(state), std::ref(state.queue_mutex));
    training_thread.detach();

    std::cout << "Thread Started" << std::endl;

    if (state.do_visualize)
    {
        std::cout << "Visualizing Network" << std::endl;

        // Open Raylib window and its configs
        SetConfigFlags(FLAG_WINDOW_RESIZABLE);
        InitWindow(800, 500, process_name.c_str());
        InitAudioDevice();
        SetWindowMinSize(800, 600);
        SetTargetFPS(60);
        Sound forward_sound = LoadSound("../audio/feedforward.mp3");
        Sound backward_sound = LoadSound("../audio/backprop.mp3");

        Sound tick_sound = forward_sound;

        float tick_interval;

        while (!WindowShouldClose() && state.do_visualize)
        {
            Rectangle RegionA = {vw(2), vh(2), vw(63), vh(96)};
            Rectangle RegionB = {vw(67), vh(2), vw(31), vh(46)};
            Rectangle RegionC = {vw(67), vh(52), vw(31), vh(46)};

            // Accumulate time
            time_accumulator += GetFrameTime();

            // Flag to make sure tick speed isn't too slow
            if (state.tick_speed < 0.1f)
            {
                state.tick_speed = 0.1f;
            }

            // Set tick_interval (1 second / tick_speed)
            tick_interval = 1.0 / state.tick_speed;

            // Updates occur at desired tick_interval/tick rate
            if (time_accumulator >= tick_interval)
            {
                // Select active layer based on where playhead is currently
                state.active_layer_index = layer_sequence[playhead];

                if (playhead > 0)
                {
                    state.previous_layer_index = layer_sequence[playhead - 1];
                }
                else
                {
                    state.previous_layer_index = -1;
                }

                // Manually determine if we're in backpropagation mode and change audio
                if (playhead >= (layer_sequence.size() / 2))
                {
                    tick_sound = backward_sound;
                }
                else
                {
                    tick_sound = forward_sound;
                }

                // Play Audio
                StopSound(tick_sound);

                float dynamic_pitch = state.tick_speed * 0.1;

                // Manually clamp tick speed to the range 0.5-3.0 to make sure it don't go too fast
                if (dynamic_pitch < 0.5)
                {
                    dynamic_pitch = 0.5;
                }
                else if (dynamic_pitch > 3)
                {
                    dynamic_pitch = 3;
                }

                SetSoundPitch(tick_sound, dynamic_pitch);
                // Play all sound unless final layer of backprop
                if (!(playhead == layer_sequence.size() - 1))
                {
                    PlaySound(tick_sound);
                }

                // Increment playhead
                playhead++;

                // std::cout << "active_layer_index: " << state.active_layer_index << std::endl;
                // std::cout << "previous_layer_index: " << state.previous_layer_index << std::endl;

                // Sequence has finished, therefore we've finished animating a full epoch
                if (playhead >= layer_sequence.size())
                {
                    // Try to grab new snapshot. If we fail we try again the next loop
                    std::lock_guard<std::mutex> lock(state.queue_mutex);
                    if (!state.epoch_snapshots.empty())
                    {
                        current_snapshot = state.epoch_snapshots.front();
                        state.epoch_snapshots.pop();
                        state.weights = current_snapshot.weights;
                        state.loss_history.push_back(current_snapshot.loss);
                    }

                    playhead = 0;
                    state.is_backpropping = false;
                    // std::cout << "Backpropping should be false now" << std::endl;
                    // std::cout << (state.active_layer_index == state.previous_layer_index) << std::endl;
                }

                time_accumulator = 0;
            }

            BeginDrawing();
            ClearBackground({28, 30, 38, 255});

            // std::cout << "RayLib Started Drawing" << std::endl;

            visualizer.visualize(RegionA);
            // std::cout << "RayLib Visualized A" << std::endl;

            visualizer.visualize(RegionB, RegionC);
            // std::cout << "RayLib Visualized B and C" << std::endl;

            EndDrawing();
        }

        UnloadSound(forward_sound);
        UnloadSound(backward_sound);
    }

    CloseAudioDevice();
    CloseWindow();

    // Restore console buffer
    std::cout.rdbuf(cout_buffer);
    return 0;
}
