find_package(OpenCV REQUIRED)
add_library( libsvm svm.h svm.cpp)
add_library( opencv_libsvm opencv_warpper_libsvm.h opencv_warpper_libsvm.cpp)

target_link_libraries( opencv_libsvm libsvm)


