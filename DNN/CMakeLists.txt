cmake_minimum_required(VERSION 3.16)

project(DNN LANGUAGES CXX)

add_library(dnn
    src/Adam.cpp
    src/Layer.cpp
    src/MNIST.cpp
    src/Network.cpp
)

target_compile_features(dnn PUBLIC cxx_std_17)
target_include_directories(dnn PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE dnn)
set_target_properties(main PROPERTIES
    OUTPUT_NAME main.exe
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

include(CTest)
if(BUILD_TESTING)
    add_executable(dnn_gradient_check
        tests/gradient_check.cpp
        tests/gradient_check_main.cpp
    )
    target_link_libraries(dnn_gradient_check PRIVATE dnn)
    add_test(NAME gradient_check COMMAND dnn_gradient_check)
    set_tests_properties(gradient_check PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS: All weight gradients match"
        FAIL_REGULAR_EXPRESSION "FAIL:;Failed tests: [1-9]"
    )

    add_executable(dnn_model_smoke tests/model_smoke.cpp)
    target_link_libraries(dnn_model_smoke PRIVATE dnn)
    add_test(NAME model_smoke COMMAND dnn_model_smoke)
    set_tests_properties(model_smoke PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND "${DOXYGEN_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Generating API documentation"
        VERBATIM
    )
endif()
