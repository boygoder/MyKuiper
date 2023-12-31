cmake_minimum_required(VERSION 3.16)
project(kuiper_course)
set(CMAKE_CXX_STANDARD 17)
include_directories(./include)
option(BUILD_DEMO "BUILD THE DEMO PROJECT")
set(BUILD_DEMO ON)

#if (BUILD_DEMO)
#    MESSAGE(STATUS "BUILD DEMO PROJECT")
#    add_subdirectory(demos)
#endif ()

find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

aux_source_directory(./source/data DIR_DATA)
aux_source_directory(./source/runtime DIR_PARSER)
aux_source_directory(./source/layer/abstract DIR_ABSTRACT_LAYER)
aux_source_directory(./source/layer/details DIR_BINOCULAR_LAYER)
aux_source_directory(./source/parser DIR_PARSER)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(link_lib glog::glog)

set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
add_library(kuiper SHARED ${DIR_DATA} ${DIR_PARSER} ${DIR_ABSTRACT_LAYER} ${DIR_BINOCULAR_LAYER} ${DIR_PARSER} )
target_link_libraries(kuiper ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)

target_include_directories(kuiper PUBLIC ${benchmark_INCLUDE_DIRS})
target_include_directories(kuiper PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(kuiper PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(kuiper PUBLIC ${Armadillo_INCLUDE_DIR})

# mathfun library defines
add_compile_definitions(SSE_MATHFUN_WITH_CODE USE_SSE_AUTO)
enable_testing()
add_subdirectory(test)

