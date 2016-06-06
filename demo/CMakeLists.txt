add_executable(demo_video ./demo_video.cpp)
target_link_libraries(demo_video sakUtils tracker ${OpenCV_LIBS})

add_executable(demo_img_list ./demo_img_list.cpp)
target_link_libraries(demo_img_list sakUtils tracker ${OpenCV_LIBS})

add_executable(demo_img_vot ./demo_img_vot.cpp)
target_link_libraries(demo_img_vot sakUtils tracker ${OpenCV_LIBS})

add_executable(demo_vot ./demo_vot.cpp)
target_link_libraries(demo_vot sakUtils tracker ${OpenCV_LIBS})



