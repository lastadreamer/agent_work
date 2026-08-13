cmake_minimum_required(VERSION 3.16)
project(xiangqi_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(xiangqi_core STATIC src/board.cpp)
target_include_directories(xiangqi_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(xiangqi_core PRIVATE -O3 -DNDEBUG -Wall -Wextra -Wpedantic)

add_executable(xiangqi-perft src/perft_main.cpp)
target_link_libraries(xiangqi-perft PRIVATE xiangqi_core)
target_compile_options(xiangqi-perft PRIVATE -O3 -DNDEBUG)
