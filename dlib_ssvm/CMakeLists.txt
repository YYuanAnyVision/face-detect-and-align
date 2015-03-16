find_package(OpenCV REQUIRED)

include(./dlib/cmake)
#add_library( dlibsource ./dlib/all/source.cpp)
#add_library( dlib_ssvm dlib_ssvm.h dlib_ssvm.cpp)
add_executable( test main.cpp)
target_link_libraries( test  ${OpenCV_LIBS}  dlib)


