add_definitions(-Wall)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)	
find_package(Boost COMPONENTS system filesystem REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable( test_misc main.cpp)
add_library( jitterImgae jitterImage.h jitterImage.cpp)
add_library( misc misc.hpp misc.cpp)
add_library(nms NonMaxSupress.cpp NonMaxSupress.h)

target_link_libraries(  test_misc ${OpenCV_LIBS}  ${Boost_LIBRARIES} jitterImgae  misc)

