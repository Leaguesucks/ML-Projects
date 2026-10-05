cmake_minimum_required(VERSION 3.16)

# VS Code reads this database for the CNN and sibling DNN include paths.
option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compiler commands for IDE support" ON)

project(cnn LANGUAGES CXX)

include(CTest)

# A standalone CNN build uses the same sibling DNN target as the root build.
if(NOT TARGET dnn::dnn)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../DNN" "${CMAKE_CURRENT_BINARY_DIR}/DNN")
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)

file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# Keep the executable entry point out of the reusable library.
list(REMOVE_ITEM PROJECT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

add_library(cnn STATIC ${PROJECT_SOURCES})
add_library(cnn::cnn ALIAS cnn)
target_compile_features(cnn PUBLIC cxx_std_17)
target_include_directories(cnn PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cnn PUBLIC dnn::dnn OpenMP::OpenMP_CXX Threads::Threads)

add_executable(cnn_main src/main.cpp)
target_link_libraries(cnn_main PRIVATE cnn::cnn)
set_target_properties(cnn_main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(BUILD_TESTING)
    add_executable(cnn_dependency_smoke tests/dependency_smoke.cpp)
    target_link_libraries(cnn_dependency_smoke PRIVATE cnn::cnn)
    add_test(NAME cnn_dependency_smoke COMMAND cnn_dependency_smoke)
endif()
