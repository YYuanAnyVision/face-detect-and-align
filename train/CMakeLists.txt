find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)	
find_package(Boost COMPONENTS system filesystem  REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable( train_fhogdet main.cpp)
add_executable( test_model test_model.cpp)
target_link_libraries(  train_fhogdet  ${OpenCV_LIBS}   ${Boost_LIBRARIES}  chnFeature opencv_libsvm scanner)
target_link_libraries( test_model  ${OpenCV_LIBS}   ${Boost_LIBRARIES}  chnFeature opencv_libsvm scanner)


