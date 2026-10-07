# ML-Projects

C++ neural-network projects implemented from scratch:

- [DNN](DNN/README.md) provides the dense neural-network library, MNIST example,
  and gradient/model tests.
- [CNN](CNN/README.md) provides the convolutional library under development. It
  links directly to DNN; DNN sources are compiled once from `DNN/src/`.

Both libraries require C++17. CNN also requires OpenMP. Public C++ APIs live in
the `dnn` and `cnn` namespaces, with headers such as `<dnn/Network.h>` and
`<cnn/network/Network.hpp>`.

Build and test both projects from this directory:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The same projects can be built independently:

```sh
cmake -S DNN -B build-dnn
cmake --build build-dnn
ctest --test-dir build-dnn --output-on-failure

cmake -S CNN -B build-cnn
cmake --build build-cnn
ctest --test-dir build-cnn --output-on-failure
```

Configuring CNN also adds the sibling DNN project if its target is not already
available. In another CMake project, add this repository as a subdirectory and
link your application with `dnn::dnn` or `cnn::cnn`. The targets supply public
include paths and C++17 requirements; `cnn::cnn` also supplies its DNN and
OpenMP dependencies.

The MNIST executable uses paths relative to `DNN/`. After a root build, run it
from that directory:

```sh
cd DNN
../build/bin/main.exe
```

CNN's `Network2D` runs a scalar image pipeline through convolution, activation,
pooling, and a dense network. Backpropagation currently covers the dense stage
only; convolution kernels and multiple feature maps remain under development.
See [CNN's file-move guide](CNN/README.md#moving-or-adding-files) for the include
and build changes needed when reorganizing either project's files.
