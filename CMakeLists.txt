cmake_minimum_required(VERSION 4.1)

project(nnrt LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)

add_subdirectory(onnx)
add_subdirectory(test_program)
