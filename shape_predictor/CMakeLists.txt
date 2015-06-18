cmake_minimum_required(VERSION 2.6)

project(shape_predictor)
find_package(OpenCV REQUIRED)

add_library( shape_predictor shape_predictor.hpp shape_predictor.cpp)
add_library( tinyxml tinyxml.h tinyxml.cpp)
add_executable( train_model train_main.cpp)

target_link_libraries(  train_demo ${OpenCV_LIBS} shape_predictor tinyxml)

