cmake_minimum_required(VERSION 3.4)
project(zad2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp reverse.h)
add_executable(zad2 ${SOURCE_FILES})