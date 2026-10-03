cmake_minimum_required(VERSION 3.16)
project(cnn LANGUAGES CXX)

file(GLOB PROJECT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB EXTERNAL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*/*.cpp")

add_library(cnn STATIC ${PROJECT_SOURCES} ${EXTERNAL_SOURCES})
target_compile_features(cnn PUBLIC cxx_std_17)
target_include_directories(cnn PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Consumers can link cnn and include headers from include/cnn or include/dnn.
