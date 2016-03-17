include_directories(inc)

aux_source_directory(./src src)

set(target_name "tracker")

add_library(${target_name} STATIC
            ${src})

target_link_libraries(${target_name}
                      ${OpenCV_LIBS})



