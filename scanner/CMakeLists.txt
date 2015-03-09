cmake_minimum_required( VERSION 2.8)

find_package(OpenCV REQUIRED)

add_executable( test_scanner main.cpp  )
add_library( scanner scanner.h scanner.cpp )
target_link_libraries( scanner ${OpenCV_LIBS}  opencv_libsvm chnFeature)
target_link_libraries( test_scanner ${OpenCV_LIBS}  opencv_libsvm chnFeature scanner)

