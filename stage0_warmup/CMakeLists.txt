add_executable(vector_add vector_add.cu)
target_compile_features(vector_add PRIVATE cxx_std_17 cuda_std_17)

add_executable(image_invert image_invert.cu)
target_compile_features(image_invert PRIVATE cxx_std_17 cuda_std_17)

add_executable(sobel sobel.cu)
target_compile_features(sobel PRIVATE cxx_std_17 cuda_std_17)
