add_definitions(-Wall)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)	
#find_package(Boost COMPONENTS system filesystem REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable( test_chn main.cpp  )
add_library( sseFun sseFun.h sseFun.cpp)
add_library( chnFeature Pyramid.h Pyramid.cpp )

target_link_libraries( chnFeature sseFun misc ${OpenCV_LIBS}  )
target_link_libraries(  test_chn chnFeature ${OpenCV_LIBS}    )

