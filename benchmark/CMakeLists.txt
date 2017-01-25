add_executable(tracker-benchmark ./tracker_benchmark.cpp)
target_link_libraries(tracker-benchmark sakUtils object_tracker ${OpenCV_LIBS})
