cmake_minimum_required(VERSION 3.16)
project(pointpillars_ros2 VERSION 0.0.1 LANGUAGES C CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置编译选项
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# 设置CUDA编译选项
find_package(CUDA REQUIRED QUIET)
if (CUDA_VERSION VERSION_LESS "11.8")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Wno-deprecated-gpu-targets)
else ()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 -Wno-deprecated-gpu-targets)
endif ()

# 设置TensorRT路径和库
set(TENSORRT_ROOT "/home/ros/TensorRT-8.6.1.6")
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)

# 查找依赖库
find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSERS NAMES nvonnxparser)
message(STATUS "NVINFER path: ${NVINFER}")
message(STATUS "NVPARSERS path: ${NVPARSERS}")
message(STATUS "NVONNXPARSERS path: ${NVONNXPARSERS}")

# 查找ROS 2依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(jsk_recognition_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)

# 添加项目包含目录
include_directories(
    include
    ${PCL_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
    ${TENSORRT_ROOT}/include
)

# 定义链接库
set(POINTPILLARS_LIBRARIES
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDA_cusparse_LIBRARY}
    ${NVINFER}
    ${NVONNXPARSERS}
    ${PCL_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    OpenMP::OpenMP_CXX
    rclcpp::rclcpp
    ${sensor_msgs_TARGETS}
    ${jsk_recognition_msgs_TARGETS}
    ${pcl_conversions_TARGETS}
    ${tf2_ros_TARGETS}
    ${tf2_geometry_msgs_TARGETS}
)

# 添加子目录
add_subdirectory(pointpillars)

# 添加可执行文件
add_executable(pointpillars_ros2
    src/pointpillars_humble.cc
)

# 链接库
target_link_libraries(pointpillars_ros2
    ${POINTPILLARS_LIBRARIES}
    pointpillars  # 链接pointpillars子目录生成的库
)

# 安装规则
install(TARGETS
    pointpillars_ros2
    DESTINATION lib/${PROJECT_NAME}
)

# 安装配置文件
install(FILES
    bootstrap.yaml
    kitti_cbgs_pp_multihead.yaml
    DESTINATION share/${PROJECT_NAME}
)

# 安装模型文件
install(DIRECTORY
    model/
    DESTINATION share/${PROJECT_NAME}/model
)

# 安装rviz配置
install(DIRECTORY
    rviz/
    DESTINATION share/${PROJECT_NAME}/rviz
)

# 安装工具和文档
install(DIRECTORY
    tools/
    DESTINATION share/${PROJECT_NAME}/tools
)

install(DIRECTORY
    docs/
    DESTINATION share/${PROJECT_NAME}/docs
)

# 注册ament包
ament_package()