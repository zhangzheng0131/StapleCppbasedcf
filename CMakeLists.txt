cmake_minimum_required(VERSION 2.8)
project(tracker)

find_package(OpenCV REQUIRED)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

### SAK Library
add_subdirectory(./sak)
include_directories(${SAK_INCLUDE_DIR})
    
include_directories(tracker/inc)
add_subdirectory(tracker)
add_subdirectory(demo)




