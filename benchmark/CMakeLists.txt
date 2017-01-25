add_executable(tracker-benchmark ./tracker_benchmark.cpp)
target_link_libraries(tracker-benchmark sakUtils object_tracker ${OpenCV_LIBS})

add_custom_command(TARGET tracker-benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tracker-benchmark> ${CMAKE_SOURCE_DIR}/benchmark/ )

