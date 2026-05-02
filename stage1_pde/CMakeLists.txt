add_executable(heat_euler heat_euler.cu)
target_compile_features(heat_euler PRIVATE cxx_std_17 cuda_std_17)

add_executable(heat_rk4 heat_rk4.cu)
target_compile_features(heat_rk4 PRIVATE cxx_std_17 cuda_std_17)

find_package(OpenCV REQUIRED)

add_executable(wave wave.cu)
target_compile_features(wave PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(wave PRIVATE argparse ${OpenCV_LIBS})
target_include_directories(wave PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(grayscott grayscott.cu)
target_compile_features(grayscott PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(grayscott PRIVATE argparse ${OpenCV_LIBS})
target_include_directories(grayscott PRIVATE ${OpenCV_INCLUDE_DIRS})
