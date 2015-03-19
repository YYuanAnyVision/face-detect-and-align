cmake_minimum_required( VERSION 2.8)

find_package(OpenCV REQUIRED)

option( HAVE_SSE "enable if have sse" OFF)
if( HAVE_SSE)
add_definitions(-DHAVE_SSE)
endif(HAVE_SSE)

add_executable( test_scanner main.cpp  )
add_library( scanner scanner.h scanner.cpp )
target_link_libraries( scanner ${OpenCV_LIBS}  opencv_libsvm chnFeature nms)
target_link_libraries( test_scanner ${OpenCV_LIBS}  opencv_libsvm chnFeature scanner)

