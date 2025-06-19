cmake_minimum_required(VERSION 3.16)

# 收集所有源文件
file(GLOB_RECURSE POINTPILLARS_SRC
  "*.cc"
  "*.cu"
)

# 构建静态库（推荐静态链接）
add_library(pointpillars STATIC ${POINTPILLARS_SRC})

# 包含头文件
target_include_directories(pointpillars PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# CUDA 设置
find_package(CUDA REQUIRED)
if (CUDA_VERSION VERSION_LESS "11.8")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Wno-deprecated-gpu-targets)
else ()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 -Wno-deprecated-gpu-targets)
endif ()

# TensorRT 链接库
find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSERS NAMES nvonnxparser)

# 添加库依赖
target_link_libraries(pointpillars
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${NVINFER}
  ${NVONNXPARSERS}
  yaml-cpp
)

# 编译选项
target_compile_features(pointpillars PUBLIC cxx_std_17)

# 可选：如果 CUDA 编译有问题，可替代使用 enable_language(CUDA)
