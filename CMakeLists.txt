cmake_minimum_required(VERSION 3.16)
project(pointpillars_ros2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# TensorRT and CUDA
find_package(CUDA REQUIRED)
find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSERS NAMES nvonnxparser)
include_directories("/home/ros/TensorRT-8.6.1.6/include")

# jsk_recognition_msgs (assume installed via rosdep or colcon workspace)
find_package(jsk_recognition_msgs REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
)

add_subdirectory(pointpillars)

add_executable(pointpillars_node src/pointpillars_humble.cc)
target_link_libraries(pointpillars_node
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${NVINFER}
  ${NVONNXPARSERS}
  yaml-cpp
  pointpillars
)
ament_target_dependencies(pointpillars_node
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  jsk_recognition_msgs
)

install(TARGETS
  pointpillars_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
