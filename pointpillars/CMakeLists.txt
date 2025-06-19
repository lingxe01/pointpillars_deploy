# pointpillars/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 设置C++标准和编译选项
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 添加编译定义
add_definitions(-DYAML_BUILD_SHARED_LIBS=ON)

# 收集源文件
file(GLOB_RECURSE SRC_FILES 
    *.cc 
    *.cu
)

# 添加库
add_library(pointpillars SHARED ${SRC_FILES})

# 设置CUDA编译选项
set_target_properties(pointpillars PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# 添加包含目录
target_include_directories(pointpillars PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_ROOT}/include
    ${PCL_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(pointpillars
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDA_cusparse_LIBRARY}
    ${NVINFER}
    ${NVONNXPARSERS}
    ${PCL_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    OpenMP::OpenMP_CXX
)