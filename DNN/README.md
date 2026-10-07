# Neural Network From Scratch in C++

**JUST AS GOD INTENDED**

A feed-forward neural network implemented **from scratch in C++**, trained on the **MNIST handwritten digit dataset**.

The reusable library is exposed as the CMake target `dnn::dnn`. Public headers
live under `include/dnn/`, implementations under `src/`, and classes, enums,
and functions belong to the `dnn` namespace. The sibling CNN project links to
this library directly instead of maintaining copied sources.

The goal of this project is not simply to achieve high classification accuracy, but to understand how neural networks work internally by implementing the major components without relying on machine-learning frameworks.

The project currently supports:

* Fully connected layers
* Forward propagation
* Backpropagation
* ReLU
* Softmax
* Categorical Cross-Entropy
* Sum of Squared Errors (SSE)
* Binary Cross-Entropy (BCE)
* Mini-batch training
* Adam optimization
* Gradient checking
* Xavier initialization
* Model serialization
* MNIST loading and inference

---

## Features

* Fully connected feed-forward neural network
* Forward propagation
* Backpropagation
* ReLU activation
* Softmax activation
* Categorical Cross-Entropy loss
* SSE loss
* BCE loss
* Adam optimizer
* Mini-batch gradient descent
* Numerical gradient checking
* Xavier weight initialization
* Bias parameters
* Adam first and second moments
* Model serialization and loading
* MNIST dataset loader
* Randomized training batches
* CPU-based inference
* Python tools for visualizing predictions

The implementation intentionally avoids machine-learning frameworks so that the underlying mathematics and algorithms remain visible in the code.

---

## Network Architecture

The current network used for MNIST classification is:

```text
Input
784 neurons
    |
    v
Fully Connected
128 neurons
ReLU
    |
    v
Fully Connected
64 neurons
ReLU
    |
    v
Fully Connected
10 neurons
Softmax
    |
    v
Output
10 class probabilities
```

The 784 inputs correspond to the `28 x 28` pixels of an MNIST image.

The output layer contains 10 neurons representing the digits:

```text
0 1 2 3 4 5 6 7 8 9
```

Softmax converts the final logits into a probability distribution.

---

## Mathematical Implementation

### Fully Connected Layer

For neuron `i`:

$$
z_i = \sum_j w_{ij}x_j + b_i
$$

The activation is:

$$
a_i = f(z_i)
$$

---

### ReLU

Hidden layers use ReLU:

$$
ReLU(x) = max(0,x)
$$

Its derivative is:

$$
ReLU'(x) =
\begin{cases}
1 & x > 0 \\
0 & x \leq 0
\end{cases}
$$

---

### Softmax

The output layer uses numerically stable Softmax:

$$
softmax(z_i) =
\frac{e^{z_i-z_{max}}}
{\sum_j e^{z_j-z_{max}}}
$$

where:

$$
z_{max} = max_j(z_j)
$$

Subtracting the maximum value improves numerical stability and prevents unnecessary overflow during exponentiation.

---

### Categorical Cross-Entropy

For a one-hot target vector:

$$
L = -\sum_i y_i log(\hat{y}_i)
$$

Because the target vector contains only one non-zero value:

$$
L = -log(\hat{y}_{correct})
$$

The derivative with respect to the Softmax output is:

$$
\frac{\partial L}{\partial \hat{y}_i}
=
-\frac{y_i}{\hat{y}_i}
$$

The current implementation supports the general Softmax Jacobian during backpropagation.

When Softmax and Cross-Entropy are combined, the derivative can be simplified to:

$$
\frac{\partial L}{\partial z_i}
=
\hat{y}_i - y_i
$$

This avoids explicitly calculating the Softmax Jacobian and makes the output-layer derivative much cheaper to calculate.

---

## Backpropagation

The network calculates gradients using the chain rule.

For a weight:

$$
\frac{\partial L}{\partial w_{ij}}
=
\delta_i a_j
$$

where:

* `delta_i` is the error signal for neuron `i`
* `a_j` is the activation from the previous layer

The bias gradient is:

$$
\frac{\partial L}{\partial b_i}
=
\delta_i
$$

Gradients are accumulated across a mini-batch before the optimizer updates the parameters.

---

## Adam Optimizer

Training uses the Adam optimization algorithm.

The first moment is:

$$
m_t =
\beta_1 m_{t-1}
+
(1-\beta_1)g_t
$$

The second moment is:

$$
v_t =
\beta_2 v_{t-1}
+
(1-\beta_2)g_t^2
$$

Bias correction:

$$
\hat{m}_t =
\frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t =
\frac{v_t}{1-\beta_2^t}
$$

Parameter update:

$$
\theta_t =
\theta_{t-1}
-
\alpha
\frac{\hat{m}_t}
{\sqrt{\hat{v}_t}+\epsilon}
$$

Default parameters:

```text
Learning rate:  0.001
Beta 1:         0.9
Beta 2:         0.999
Epsilon:        1e-8
```

---

## Mini-Batch Training

The network uses mini-batch gradient descent.

For each batch:

```text
1. Clear accumulated gradients
2. Forward propagate every sample
3. Backpropagate every sample
4. Accumulate gradients
5. Average gradients
6. Perform one Adam update
```

The Adam timestep is incremented **once per mini-batch**, rather than once per individual training sample.

---

## Gradient Checking

Numerical gradient checking is used to verify the backpropagation implementation.

The analytical gradient is compared against a numerical approximation using the central difference method:

$$
\frac{\partial L}{\partial w}
\approx
\frac{L(w+\epsilon)-L(w-\epsilon)}
{2\epsilon}
$$

A full gradient check produced:

```text
Total weights tested: 242304
Failed tests:         0

Worst gradient:
  Layer:              3
  Neuron:             5
  Weight:             55
  Analytical:        -0.0838815663
  Numerical:         -0.0838815671
  Absolute error:     0.0000000009
  Relative error:     0.0000000103

PASS: All weight gradients match.
```

This provides strong evidence that the implemented weight gradients are mathematically correct.

---

## Memory Layout

The network stores layer parameters using contiguous `std::vector<double>` arrays instead of individual `Neuron` objects.

For a layer containing `N` neurons and `M` inputs, the weights are stored as:

```text
[w00, w01, w02, ..., w0M,
 w10, w11, w12, ..., w1M,
 ...
 wN0, wN1, wN2, ..., wNM]
```

The weight for neuron `i` and input `j` is accessed using:

```cpp
weights[i * n_inputs + j]
```

This representation improves:

* Memory locality
* Cache utilization
* Allocation behavior
* Vectorization opportunities
* SIMD optimization opportunities

The project originally used a `Neuron` class but was later redesigned around contiguous layer-level arrays.

---

## MNIST Results

The current `784 -> 128 -> 64 -> 10` network achieves approximately:

```text
Test accuracy: 98.12%
```

on the MNIST test set.

This is a decent result for a relatively simple fully connected network implemented entirely from scratch.

A fully connected network does not explicitly exploit the spatial structure of images. Each pixel is treated as an independent input feature.

For this reason, making the network larger does not necessarily produce proportional improvements in accuracy.

---

## Why MNIST?

MNIST is useful for implementing a neural network from first principles because it is:

* Small enough to train on a CPU
* Easy to visualize
* Well understood
* Large enough to expose implementation bugs
* Simple enough to make numerical debugging practical

---

## Project Structure

The source tree is organized as follows:

```text
.
├── CMakeLists.txt
├── Doxyfile
├── Makefile
├── include/dnn/
│   ├── Adam.h
│   ├── Layer.h
│   ├── MNIST.h
│   └── Network.h
├── src/
│   ├── Adam.cpp
│   ├── Layer.cpp
│   ├── MNIST.cpp
│   ├── Network.cpp
│   └── main.cpp
├── tests/
│   ├── gradient_check.cpp
│   ├── gradient_check.h
│   ├── gradient_check_main.cpp
│   └── model_smoke.cpp
├── mnist/                 # MNIST image and label files
├── training/              # Saved network used by main
├── MNIST_Display.py       # Prediction visualizer
├── setup.sh
└── README.md
```

---

## Building and Running

The C++ program requires a compiler with **C++17** support. From `DNN/`, build
it with Make:

```bash
make
./bin/main.exe
```

Or use CMake:

```bash
cmake -S . -B build
cmake --build build
./build/bin/main.exe
```

Run these commands from `DNN/`. The executable reads the MNIST files under
`mnist/` and the saved network at `training/mnist_train.bin` using relative
paths, then starts training.

The CMake tests run numerical gradient checking and a saved-model smoke check. After building, run:

```bash
ctest --test-dir build --output-on-failure
```

To build DNN and CNN together, run the following from the repository root:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

For that build, run the MNIST executable from `DNN/` as
`../build/bin/main.exe` so its data paths resolve.

### Using the library in another project

Add the existing DNN project as a subdirectory and link its reusable target:

```cmake
add_subdirectory(path/to/ML-Projects/DNN dnn-build)
add_executable(your_app app.cpp)
target_link_libraries(your_app PRIVATE dnn::dnn)
```

The target supplies the public include path and C++17 requirement. Include
headers with their package path and qualify API names explicitly:

```cpp
#include <dnn/Network.h>

dnn::Network network(
    784,
    {{128, dnn::RELU}, {64, dnn::RELU}, {10, dnn::SOFTMAX}},
    dnn::CATEGORICAL_CROSS_ENTROPY
);
```

Other public types include `dnn::Layer`, `dnn::Adam`, `dnn::MNIST`,
`dnn::Layer_Architecture`, `dnn::Activation_Type`, and `dnn::Loss_Type`.

Linking `dnn::dnn` compiles the implementation once from `DNN/src/`; consumers
do not copy headers or implementation files. Changes to DNN are picked up by
normal CMake rebuilds. To build just the library, use
`cmake --build build --target dnn`. Set `-DBUILD_TESTING=OFF` when configuring
to omit the test targets.

### Moving or adding files

Header include paths are relative to `DNN/include/`. For example, moving
`include/dnn/Network.h` to `include/dnn/network/Network.h` requires replacing
`#include <dnn/Network.h>` with `#include <dnn/network/Network.h>` everywhere
it is referenced, including in CNN. Keep the include root at `include/` and
the namespace as `dnn`; subfolders do not change the namespace.

CMake and the native Makefile recursively discover `.cpp` files under `src/`.
Moving `src/Network.cpp` to `src/network/Network.cpp`, or adding a new library
implementation there, requires a rebuild without editing a source list. The
directory is spelled `src`, not `scr`. Keep the entrypoint at `src/main.cpp`;
moving it requires changing both `add_executable(main ...)` and the library
source exclusion in `CMakeLists.txt`. Test sources are listed explicitly in
their `add_executable` calls, so update those lists after a test move and use
`add_test` to register a new test.

After moves, refresh CMake and the compilation database used by VS Code:

```bash
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Run those commands from `DNN/`, or from the repository root to rebuild both
libraries. [CNN's file-move guide](../CNN/README.md#moving-or-adding-files)
includes examples for its nested `.hpp` headers as well.

### API Documentation

Install Doxygen to generate HTML API documentation:

```bash
doxygen Doxyfile
```

Open `docs/api/html/index.html` in a browser. With Doxygen available when CMake is configured, the same documentation can be generated through the optional `docs` target:

```bash
cmake --build build --target docs
```

### MNIST Display

`MNIST_Display.py` inspects predictions from the saved network. It requires Python, NumPy, and Pillow. A minimal virtual environment can be created with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy pillow
```

On Debian or Ubuntu, `bash setup.sh` installs the compiler and Python prerequisites and creates a virtual environment with NumPy and Pillow.

Run the display script from `DNN/` so its relative data paths resolve.

```bash
python3 MNIST_Display.py MANUAL
```

#### Manual Mode

This allows MNIST images and predictions to be inspected individually.

> **Note:** WSL does not always display GUI applications correctly. When the GUI cannot be displayed, the program generates PNG images in the `mnist_results/` directory.

#### Summary Mode

```bash
python3 MNIST_Display.py SUMMARY
```

This produces a summary of the network's performance.

---

## Dataset

The project uses the MNIST dataset.

```text
60,000 training images
10,000 test images
28 x 28 pixels
10 classes
```

Each image is represented by 784 input values.

Pixel values are normalized from:

```text
[0, 255]
```

to:

```text
[0.0, 1.0]
```

Labels are converted to one-hot vectors.

For example, label `7` becomes:

```text
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

---

## Model Serialization

The network can save its parameters to a binary file and load them later.

The serialized model contains:

```text
Network metadata
├── Version
├── Input size
├── Number of layers
├── Loss type
└── Adam timestep

For each layer
├── Number of neurons
├── Activation type
├── Number of inputs
├── Weights
├── Biases
├── Adam first moments
├── Adam second moments
├── Bias first moments
└── Bias second moments
```

Saving the Adam state allows training to resume without losing the optimizer's accumulated state.

---

## Inference

After training, the network produces a probability for each digit.

Example:

```text
0: 0.00000000
1: 0.00000000
2: 0.00000000
3: 0.00000002
4: 0.00000000
5: 0.00000000
6: 0.00000000
7: 0.99999998
8: 0.00000000
9: 0.00000000
```

The predicted class is the index with the highest probability.

```cpp
size_t predicted_label = std::distance(
    Y_HAT.begin(),
    std::max_element(Y_HAT.begin(), Y_HAT.end())
);
```

---

## Current Limitations

This project is intentionally simple and is not intended to compete with production machine-learning libraries.

Current limitations include:

* CPU-only computation
* Fully connected layers only
* No convolutional layers
* No GPU acceleration
* No SIMD implementation yet
* No multithreaded training
* No dropout
* No batch normalization
* No learning-rate scheduler
* No automatic early stopping
* Limited data augmentation

These limitations provide opportunities for future optimization and experimentation.

---

## Future Work

### Performance

* SIMD and vectorized matrix operations
* Better cache utilization
* Multithreaded training
* Fewer temporary allocations
* Improved memory layout
* Parallel batch processing
* Benchmark different layer implementations

### Training

* Learning-rate scheduling
* L2 weight decay
* Early stopping
* Better initialization strategies
* Data augmentation
* Hard-example mining
* More validation metrics

### Neural Network Features

* Convolutional layers
* Max pooling
* Leaky ReLU
* Tanh
* Sigmoid
* Additional loss functions
* Dropout
* Batch normalization

### Architecture

The eventual goal is to experiment with a more performance-oriented architecture suitable for neural-network-based chess evaluation.

In particular, this project provides a foundation for experimenting with **NNUE-style networks**, where memory layout, incremental evaluation, and CPU efficiency become especially important.

---

## Lessons Learned

This project has primarily been about understanding what happens inside a neural network instead of treating it as a black box.

Some of the most important lessons so far:

1. **Forward propagation is relatively straightforward.**
2. **Backpropagation is where indexing and mathematical mistakes become difficult to debug.**
3. **Numerical gradient checking is extremely valuable.**
4. **A mathematically correct implementation should be verified before optimizing it.**
5. **Memory layout can matter as much as the mathematical algorithm in C++.**
6. **A larger network does not automatically produce better results.**
7. **The activation function and loss function should be considered together.**
8. **Mini-batch Adam requires careful handling of gradient accumulation and optimizer timesteps.**
9. **Numerical stability matters, particularly for Softmax and Cross-Entropy.**
10. **Once correctness is established, the implementation can be optimized without changing its mathematical behavior.**

---

## Goals of the Project

The long-term goal is to progressively move from a simple educational neural network toward a high-performance neural-network implementation.

```text
Mathematical understanding
        |
        v
Basic neural network
        |
        v
Backpropagation
        |
        v
Gradient verification
        |
        v
Adam optimization
        |
        v
Mini-batch training
        |
        v
Efficient memory layout
        |
        v
SIMD / parallelization
        |
        v
High-performance inference
```

The philosophy is simple:

> **Understand it first. Optimize it second.**

> **WHO THE F NEEDS TORCH WHEN WE CAN WRITE OUR OWN NEURAL NETWORK**

---

## License

This project is intended primarily for educational and experimental purposes.

You can do whatever the f you want with it.
