find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)	
find_package(Boost COMPONENTS system filesystem  REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


add_executable( test_model test_model.cpp)
target_link_libraries( test_model chnFeature  scanner ${OpenCV_LIBS}   ${Boost_LIBRARIES}  )


