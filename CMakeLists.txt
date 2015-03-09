cmake_minimum_required(VERSION 2.6)
project( fhog_detect )

find_package(OpenCV REQUIRED)
add_subdirectory( chnfeature )
add_subdirectory( misc )
add_subdirectory( train)
add_subdirectory( svm)
add_subdirectory(scanner)