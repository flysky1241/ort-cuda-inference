# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.20)

project ("myDetect")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(
    STATUS
    "ENV CUDA_PATH = $ENV{CUDA_PATH}"
)

message(
    STATUS
    "ENV CUDNN_ROOT_DIR = $ENV{CUDNN_ROOT_DIR}"
)

message(
    STATUS
    "CACHE CUDNN_ROOT_DIR = ${CUDNN_ROOT_DIR}"
)

# 包含子项目。
add_subdirectory ("myDetect")
