### Add option
option(USE_SIMD "Use SSE in x86 and NEON on ARM" ON)
option(RUN_BENCHMARK "Turn off the threshold setting in code for benchmark testing" ON)

if (USE_SIMD)
  add_definitions(-D__USE_SIMD)
endif()

if (RUN_BENCHMARK)
  add_definitions(-D__BENCHMARK)
endif()


if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
   cmake_minimum_required(VERSION 2.8)
   project(tracker)

   ###find_package(OpenCV 2.4.11 REQUIRED)
   find_package(OpenCV REQUIRED)
   set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
   set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

   ### SAK Library
   add_subdirectory(./sak)
   include_directories(${SAK_INCLUDE_DIR})
else()
    ### Set the INCLUDE_DIR for parent scope
    set(TRACK_INCLUDE_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/ot/inc
        PARENT_SCOPE)   
endif() 

### Generate the version cpp
### Get the git commit hash code
execute_process(COMMAND git log -1 --format=%h
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE GIT_COMMIT_HASH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ot/src/version.cpp.in
    ${CMAKE_CURRENT_SOURCE_DIR}/ot/src/version.cpp)

## APP Library
include_directories(ot/inc)
add_subdirectory(ot)

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
  add_subdirectory(demo)
  add_subdirectory(benchmark)
endif()


