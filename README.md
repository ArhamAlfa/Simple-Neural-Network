# Simple Neural Network

A configurable feedforward neural network library built from scratch in C++. Define your network topology, activation functions, and training parameters through a JSON config file, point it at a dataset, and watch it train through a live raylib visualizer.

[SCREENSHOT OF VISUALIZER RUNNING]

---

## Features

- Configurable network topology at runtime via JSON, no recompilation required
- Matrix-based forward pass and backpropagation using Eigen, generalized to arbitrary network depth
- Live training visualizer built with raylib showing the network, loss curve, and controls simultaneously
- Decoupled training and visualization running on separate threads via `std::thread` and `std::mutex`
- Snapshot queue architecture allowing the network to train at full speed independent of visualizer tick rate
- Binary dataset serialization for fast dataset reuse across runs
- CSV parser with one-hot encoded label support
- Pluggable activation functions: Sigmoid, ReLU, Tanh

---

## Dependencies

All dependencies are included as git submodules and do not require separate installation.

- [Eigen 3.4](https://eigen.tuxfamily.org/) - Header-only linear algebra library
- [raylib](https://www.raylib.com/) - Lightweight graphics library
- [raygui](https://github.com/raysan5/raygui) - Immediate mode GUI extension for raylib
- [nlohmann/json](https://github.com/nlohmann/json) - Header-only JSON parser

---

## Getting Started

### Cloning the Repository

Clone the repository along with all submodules recursively:

```bash
git clone --recurse-submodules https://github.com/yourusername/simple_neural_network.git
cd simple_neural_network
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### Building

```bash
cd build
cmake ..
make
```

This produces the `neural_network` executable inside the `build` directory.

---

## Usage

### Preparing Your Dataset

Your dataset must be a CSV file with a header row. Input features should come first, followed by one-hot encoded label columns. For example, a dataset with 4 input features and 3 classes should look like:

```
x1,x2,x3,x4,y1,y2,y3
5.1,3.5,1.4,0.2,1,0,0
4.9,3.0,1.4,0.2,1,0,0
...
```

The topology you define in `config.json` must match the dataset dimensions. For the above example, a valid topology would be `[4, 8, 3]` where 4 is the input size and 3 is the output size.

If your dataset has a single label column with multiple classes (e.g. `"setosa"`, `"versicolor"`, `"virginica"`), you will need to one-hot encode it before use. This can be done with pandas:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from numpy import argmax

df = pd.read_csv("your_dataset.csv")
encoded = pd.get_dummies(df["label_column"]).astype(int)
df = pd.concat([df.drop("label_column", axis=1), encoded], axis=1)
df.to_csv("dataset/raw/your_dataset.csv", index=False)
```

Place the processed CSV in `dataset/raw/`.

### Configuring the Network

Edit `build/config.json` before running. All fields are required.

| Field | Type | Description |
|---|---|---|
| `process_name` | `string` | Name displayed in the terminal and raylib window title. |
| `random_state` | `int` | Seed value for weight matrix initialization. Identical seeds produce identical initial weights, ensuring reproducibility. |
| `learning_rate` | `double` | Step size for gradient descent. Values between `0.01` and `0.1` are recommended. Higher values converge faster but risk overshooting. |
| `topology` | `array<int>` | Defines the network architecture as a layer-by-layer node count. The first element must equal the number of input features and the last must equal the number of output classes. For example, `[4, 8, 8, 3]` defines a network with 4 inputs, two hidden layers of 8 nodes each, and 3 outputs. |
| `dataset_name` | `string` | Internal label used for display and logging purposes. |
| `csv_path` | `string` | Path to the raw CSV file relative to the project root. Example: `"dataset/raw/iris.csv"`. |
| `save_path` | `string` | Path for the serialized binary dataset file. Example: `"dataset/saved/iris.bin"`. If the file exists it will be loaded directly, skipping CSV parsing. |
| `hidden_layer_activation` | `string` | Activation function applied to all hidden layers. Accepted values: `"SIGMOID"`, `"RELU"`, `"TANH"`. |
| `output_layer_activation` | `string` | Activation function applied to the output layer. Accepted values: `"SIGMOID"`, `"RELU"`, `"TANH"`. For multi-class classification, `"SIGMOID"` is recommended. |
| `epochs` | `int` | Maximum number of full passes through the dataset during training. |
| `loss_threshold` | `double` | Early stopping threshold. Training halts if the loss drops below this value before reaching the epoch limit. Set to `0.0` to always train for the full epoch count. |

Example `config.json`:

```json
{
    "process_name": "Iris Classifier",
    "random_state": 42,
    "learning_rate": 0.1,
    "topology": [4, 8, 8, 3],
    "dataset_name": "Iris Dataset",
    "csv_path": "dataset/raw/iris.csv",
    "save_path": "dataset/saved/iris.bin",
    "hidden_layer_activation": "RELU",
    "output_layer_activation": "SIGMOID",
    "epochs": 10000,
    "loss_threshold": 0.01
}
```

### Running

```bash
./visual_test
```

[SCREENSHOT OF TERMINAL OUTPUT DURING TRAINING]

---

## Visualizer

The raylib window is divided into three regions:

**Network Visualizer (left panel)**
Displays the network architecture as a node-edge diagram. Node brightness reflects activation values for the current snapshot. Edge thickness and color reflect weight magnitude and sign, where blue indicates positive weights and red indicates negative weights. The active layer is highlighted as the forward pass progresses through each snapshot.

**Loss Curve (top right)**
Plots training loss over epochs in real time as snapshots are consumed from the queue. The curve grows leftward from the current epoch toward the origin.

**Controls (bottom right)**
Contains a tick speed slider controlling how fast the visualizer consumes snapshots from the queue, and a skip button that halts visualization and allows training to complete headless at full speed.

[SCREENSHOT OF ALL THREE REGIONS]

---

## Architecture

### Threading Model

The network and visualizer run on separate threads. The training thread produces epoch snapshots and pushes them onto a `std::queue` inside a shared `VisualState` struct. The render thread pops snapshots from the queue and animates them at its own pace. A `std::mutex` guards all queue access to prevent race conditions.

This means training speed is never bottlenecked by the visualizer's tick rate. The network runs at full CPU speed regardless of how slowly or quickly the visualizer is consuming snapshots.

### Dataset Serialization

On first run, the CSV is parsed and the resulting `Dataset` struct is serialized to a binary `.bin` file using raw byte writes via `std::ofstream`. Subsequent runs load directly from the binary file, bypassing CSV parsing entirely. The serialization format handles `Eigen::VectorXd` objects and a `std::unordered_map` of variant metadata fields (`int`, `double`, `std::string`).

### Forward Pass

Each layer stores a weight matrix of dimensions `(current_nodes x incident_nodes)` and a bias vector of size `current_nodes`. The forward pass for a single layer is:

```
z = (W * input) + bias
output = activation(z)
```

Where `activation` is whichever function is assigned to that layer. Pre-activation values `z` and post-activation values `output` are cached in the layer for use during backpropagation.

### Backpropagation

Gradients are computed using the chain rule, propagated backwards from the output layer. Each layer computes its delta as:

```
delta_output = (predicted - actual) * activation_derivative(z)
delta_hidden = (W_next^T * delta_next) * activation_derivative(z)
```

Weights and biases are updated via gradient descent:

```
W := W - learning_rate * (delta * input^T)
b := b - learning_rate * delta
```

Backpropagation is generalized to arbitrary network depth with no hardcoded layer assumptions.

---

## Project Structure

```
simple_neural_network/
    src/
        main.cpp
        network.h / network.cpp
        layer.h / layer.cpp
        dataset.h / dataset.cpp
        visualizer.h / visualizer.cpp
        activation.h / activation.cpp
        state.h
    libs/
        eigen-5.0.0/
        raylib/
        raygui.h
        json.hpp
    dataset/
        raw/
        saved/
    build/
        config.json
    CMakeLists.txt
    README.md
```

---

## Included Datasets

Two preprocessed datasets are included for immediate use.

**XOR Problem** (`dataset/raw/xor.csv`)
The canonical non-linearly separable binary classification problem. Requires at minimum one hidden layer to solve. Useful for verifying the network is training correctly.

Recommended config: topology `[2, 4, 1]`, learning rate `0.1`, epochs `5000`.

**Iris Dataset** (`dataset/raw/iris.csv`)
150 samples across 3 iris flower species classified by 4 petal and sepal measurements. A standard multi-class classification benchmark.

Recommended config: topology `[4, 8, 8, 3]`, learning rate `0.1`, epochs `10000`.

---

## License

MIT
