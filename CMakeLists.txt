if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
   cmake_minimum_required(VERSION 2.8)
   project(tracker)

   find_package(OpenCV REQUIRED)
   set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
   set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

   ### SAK Library
   add_subdirectory(./sak)
   include_directories(${SAK_INCLUDE_DIR})
else()
    ### Set the INCLUDE_DIR for parent scope
    set(TRACK_INCLUDE_DIR 
        ${CMAKE_CURRENT_SOURCE_DIR}/tracker/inc
        PARENT_SCOPE)   
endif() 

include_directories(ot/inc)
add_subdirectory(ot)

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    add_subdirectory(demo)
endif()




