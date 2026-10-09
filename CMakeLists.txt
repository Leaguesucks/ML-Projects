cmake_minimum_required(VERSION 3.16)

project(MLProjects LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

add_subdirectory(DNN)
add_subdirectory(CNN)
