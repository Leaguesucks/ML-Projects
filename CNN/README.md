# CNN C++ library

`include/cnn/` holds this project's public headers. `src/` holds this
project's `.cpp` files. Each external project gets its own directory under
`include/`; for example, DNN's headers and implementations are both in
`include/dnn/`.

Both build systems add `include/` to the compiler's header search path. Use
angle brackets with the directory name:

```cpp
#include <cnn/YourName.hpp>
#include <dnn/Network.hpp>
```

Build the static library with either CMake or Make:

```sh
cmake -S . -B build
cmake --build build
```

```sh
make
```

Both produce `build/libcnn.a`. To compile an application directly after
building the library, use `-Iinclude` and link the library:

```sh
c++ -std=c++17 -Iinclude app.cpp build/libcnn.a -o app
```

From another CMake project, use `add_subdirectory(...)` for this directory
and `target_link_libraries(your_app PRIVATE cnn)`. The `cnn` target supplies
the include path and C++17 requirement.

## Every time you add a header for this project

1. Create `include/cnn/YourName.hpp` with `#pragma once` at the top.
2. Include it as `#include <cnn/YourName.hpp>` wherever needed. Do not add
   `include/` to the directive.
3. If it declares functions that need definitions, create
   `src/YourName.cpp` and include `<cnn/YourName.hpp>` there. For a header
   with only declarations or inline definitions, this step is unnecessary.
4. Rebuild with `cmake --build build` or `make`. You do not need to edit either
   build file: both discover new `.cpp` files, and both track header changes.

## When you add another external project

1. Give it a separate directory such as `include/other/`. Keep its `.hpp`
   and `.cpp` files together there, as with `include/dnn/`.
2. Include its headers as `#include <other/YourName.hpp>` from this project's
   source or headers. Update any old include directives if you rename headers.
3. Rebuild. Both build systems discover `.cpp` files one level below
   `include/` (for example, `include/other/YourName.cpp`). Header-only
   projects need no build-file changes. If the external project needs other
   libraries, compiler flags, or a deeper source layout, add those explicitly
   to `CMakeLists.txt` and `Makefile`.
