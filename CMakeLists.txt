cmake_minimum_required(VERSION 3.16)

cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0042 NEW)
project(pointpillars) 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCMAKE_C_COMPILER=/usr/bin/gcc-6 -DCMAKE_CXX_COMPILER=/usr/bin/g++-6 ")
set(CMAKE_BUILD_TYPE Debug)
option(BUILD_SHARED_LIBS "Build as a shared library"  ON)
option(USE_DOUBLE        "Use double as kernel_type" OFF)
option(TEST              "BUILD TEST"                OFF)
option(BENCH             "BUILD BENCH"               OFF)

if(USE_DOUBLE)
    message("Use double as kernel_type")
endif()
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif ()

############################################################
# add 3th_part library
############################################################

find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})

find_package(Threads)
find_package(OpenMP REQUIRED)
add_definitions("-DELPP_NO_LOG_TO_FILE")

find_package(CUDA REQUIRED QUIET)
include_directories(${CUDA_INCLUDE_DIRS})
if (CUDA_VERSION VERSION_LESS "11.8")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Wno-deprecated-gpu-targets)
else ()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 -Wno-deprecated-gpu-targets)
endif ()

set(CMAKE_PREFIX_PATH "/home/ros/TensorRT-8.6.1.6")
find_library(NVINFER NAMES nvinfer) # 查找so动态库文件
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSERS NAMES nvonnxparser)
message(NVINFER "NVINFER path is ${NVINFER}")
message(NVPARSERS "NVPARSERS path is ${NVPARSERS}")
message(NVONNXPARSERS "nvonnxparser path is ${NVONNXPARSERS}")

include_directories("/home/ros/TensorRT-8.6.1.6/include")
#link_directories(${TENSORRT_ROOT}/lib)
set(roscpp_DIR "/opt/ros/noetic/share/roscpp/cmake/")
set(std_msgs_DIR "/opt/ros/noetic/share/std_msgs/cmake/")
# ROS
set(CMAKE_PREFIX_PATH "/opt/ros/noetic")
find_package(catkin REQUIRED COMPONENTS
  std_msgs
  rosbag             # use rosbag::Bag
  sensor_msgs
  std_msgs
  message_filters
  geometry_msgs
  nav_msgs
  visualization_msgs
)

include_directories(${catkin_INCLUDE_DIRS})
find_package(roscpp REQUIRED)
include_directories(${roscpp_INCLUDE_DIRS})
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

list(APPEND LINK_LIBRARY
            ${CUDA_cusparse_LIBRARY}
            ${NVINFER}
            ${NVONNXPARSERS}
            ${CUDA_LIBRARIES}
            ${CUDA_CUBLAS_LIBRARIES}
            ${CUDA_curand_LIBRARY}            
            ${CUDA_cusparse_LIBRARY}
            ${PCL_LIBRARIES}
            ${catkin_LIBRARIES}
            yaml-cpp
            )

# Correct error for GCC 5 and CUDA 7.5
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DYAML_BUILD_SHARED_LIBS=ON -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES ")
if (CMAKE_VERSION VERSION_LESS "3.1")
    add_compile_options("-std=c++11 -fext-numeric-literals")
else ()
    set(CMAKE_CXX_STANDARD 11)
endif ()

set(COMMON_INCLUDES ${CMAKE_CURRENT_BINARY_DIR})

############################################################
# Create a library in subdirectory
############################################################
set(PROJECT_LIB_NAME ${PROJECT_NAME})
include_directories(${COMMON_INCLUDES})
add_subdirectory(${PROJECT_SOURCE_DIR}/pointpillars)


############################################################
# Create an executable
############################################################
# if (TEST)
#     message("Building tests")
#     enable_testing()
#     add_subdirectory(test)
# endif ()

# message(STATUS "")
# message(STATUS "BUILD SUMMARY")
# message(STATUS "  CMAKE_GENERATOR      : ${CMAKE_GENERATOR}")
# message(STATUS "  Compiler ID          : ${CMAKE_CXX_COMPILER_ID}")
# message(STATUS "  Build type           : ${CMAKE_BUILD_TYPE}")
# message(STATUS "  Build shared libs    : ${BUILD_SHARED_LIBS}")
# message(STATUS "  Use double for kernel: ${USE_DOUBLE}")
# message(STATUS "  Build tests          : ${TEST}")
# message(STATUS "")


add_executable(pointpillars_test test/pointpillars_test.cc) 
target_link_libraries(pointpillars_test ${LINK_LIBRARY} ${PROJECT_LIB_NAME}) 

add_executable(pointpillars_ros test/pointpillars_ros.cc) 
target_link_libraries(pointpillars_ros ${LINK_LIBRARY} ${PROJECT_LIB_NAME}) 