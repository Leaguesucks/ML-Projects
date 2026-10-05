# CNN C++ library

CNN uses the dense neural-network library directly from the sibling `DNN/`
project. Its CMake target links publicly to `dnn::dnn` and
`OpenMP::OpenMP_CXX` and includes the platform thread dependency, so consumers
inherit DNN headers, its library, and concurrency requirements. There are no
copied DNN implementations under `CNN/include/`.

Public CNN headers live under `include/cnn/`, implementations under `src/`,
and C++ APIs in the `cnn` namespace. DNN APIs remain in the `dnn` namespace:

```cpp
#include <cnn/network/Network.hpp>
#include <cnn/network/2D.hpp>
#include <cnn/layer/Layer.hpp>
#include <dnn/Network.h>

void inspect_networks(cnn::Network2D& convolutional, dnn::Network& dense);
```

`cnn::Network` is the abstract base; `cnn::Network2D` implements a pipeline of
convolution, ReLU, max/average pooling, and a final DNN stage. It operates on a
single square image stored as scalar pixel values. The channel-count fields
do not implement multiple feature maps, and `back_propagation` currently
delegates only to the dense network; convolution kernels are not trained.

The current CNN layout is:

```text
CNN/
├── include/cnn/
│   ├── layer/Layer.hpp
│   └── network/
│       ├── Network.hpp
│       └── 2D.hpp
├── src/
│   ├── Network.cpp
│   ├── 2D.cpp
│   └── main.cpp
└── tests/
    └── dependency_smoke.cpp
```

## Building and testing

From `CNN/`, configure and build with a C++17 compiler and OpenMP support:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/bin/cnn_main
```

The standalone build adds `../DNN` automatically and produces `build/libcnn.a`
and the `cnn_main` example executable. `src/main.cpp` belongs to the example,
not the library.
Tests cover the CNN/DNN dependency, completed CNN operations, DNN gradients,
and loading a saved model. The model test reads its dataset and saved model
from `DNN/`, with its working directory set by CMake.

The Make wrapper delegates to the same CMake dependency graph:

```sh
make
make BUILD_DIR=build-debug CXX=clang++ CXXFLAGS='-O0 -g -std=c++17'
```

`make` preserves the default `-O2 -std=c++17` flags. It forwards `CXX`,
`CPPFLAGS`, `CXXFLAGS`, and `AR` to CMake; `CMAKE_FLAGS` accepts additional
configure arguments. `make clean` cleans compiled outputs in `BUILD_DIR` while
retaining the CMake configuration.

The default Make target builds both `cnn` and `cnn_main`. To build only the
library, use `cmake --build build --target cnn`.

To build both projects from the repository root:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## VS Code IntelliSense

The repository and CNN folder each include a `.vscode/c_cpp_properties.json`
configuration with the CNN and DNN include roots. CNN enables CMake's
compilation database by default, so IntelliSense also uses the include paths
and compiler flags selected by the build. Configure from the folder opened
in VS Code to generate or refresh `build/compile_commands.json`:

```sh
cmake -S . -B build
```

Include the public headers as `<cnn/network/2D.hpp>`,
`<cnn/network/Network.hpp>`, and `<cnn/layer/Layer.hpp>`.

## Using CNN in another CMake project

Add the CNN project and link its namespaced target:

```cmake
add_subdirectory(path/to/ML-Projects/CNN cnn-build)
add_executable(your_app app.cpp)
target_link_libraries(your_app PRIVATE cnn::cnn)
```

Alternatively, add the whole repository as a subdirectory. If `dnn::dnn` is
already available, CNN reuses it. Otherwise CNN adds the sibling DNN project.
The `cnn::cnn` target supplies its include path, C++17, DNN, and OpenMP, so
applications need no duplicated sources or manually assembled include flags.

Changing DNN sources or headers triggers the necessary recompilation when CNN
is rebuilt. A static archive alone does not bundle its dependencies; use the
CMake target to link applications with all required libraries.

## Moving or adding files

Create public headers under `include/cnn/` and include them with their package
path, for example `<cnn/network/2D.hpp>`. The compiler include roots remain
`CNN/include/` and `DNN/include/`: the text inside `#include <...>` is the
file's path relative to its project's include root.

For each move, update references as follows. The source directory is spelled
`src`, not `scr`.

| File move | What to change |
| --- | --- |
| `CNN/include/cnn/Network.hpp` → `CNN/include/cnn/network/Network.hpp` | Replace every `#include <cnn/Network.hpp>` with `#include <cnn/network/Network.hpp>`. |
| `CNN/include/cnn/2D.hpp` → `CNN/include/cnn/network/2D.hpp` | Replace every `#include <cnn/2D.hpp>` with `#include <cnn/network/2D.hpp>`. |
| `DNN/include/dnn/Network.h` → `DNN/include/dnn/network/Network.h` | Replace every `#include <dnn/Network.h>` with `#include <dnn/network/Network.h>`. DNN uses `.h`; CNN uses `.hpp`. |
| `CNN/src/Network.cpp` → `CNN/src/network/Network.cpp` | Move the file and rebuild. Recursive source discovery adds its new path automatically. The same rule applies to library `.cpp` files in `DNN/src/`. |
| `CNN/src/main.cpp` → another path | Update `add_executable(cnn_main ...)` and the source exclusion in `CNN/CMakeLists.txt`; the example's entrypoint path is explicit. Keep it at `src/main.cpp` for the current configuration. |
| A source under `CNN/tests/` or `DNN/tests/` moves or is added | Update the corresponding `add_executable` source list in that project's `CMakeLists.txt`. Register new tests with `add_test`. Tests are listed explicitly. |

After moving a header, update includes in implementation files, other headers,
tests, examples, and documentation. Moving files into subfolders does not
change their C++ namespace. Header moves within `include/` need no new CMake or
VS Code include directory.

Both libraries recursively discover `.cpp` files below `src/`, excluding their
configured `src/main.cpp` entrypoints. Public headers do not need a CMake
source-list entry. Reconfigure and rebuild after a move to refresh the source
list and the compilation database used by IntelliSense:

```sh
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Run those commands from `CNN/` for CNN alone, or from the repository root for
both projects. Use the build directory configured in VS Code if it differs
from `build`.

## Adding dependencies

Keep all CNN declarations and definitions in `namespace cnn` and qualify DNN
types as `dnn::...`.

For another external library, add or locate its CMake target and link it with
`target_link_libraries`. Keep implementations in their owning project rather
than copying `.cpp` files into the public include tree.
