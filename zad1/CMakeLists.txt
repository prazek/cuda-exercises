cmake_minimum_required(VERSION 3.4)
project(zad1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cu)
add_executable(zad1 ${SOURCE_FILES})