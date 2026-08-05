# CMakeList.txt: myDetect 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.20)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

find_package(Qt5 5.15.2 REQUIRED COMPONENTS
    Core
    Gui
    Widgets
)

find_package(OpenCV REQUIRED)

find_package(onnxruntime CONFIG REQUIRED)

find_package(CUDAToolkit 12.2 REQUIRED)

find_package(nlohmann_json CONFIG REQUIRED)

get_target_property(
    ONNXRUNTIME_INCLUDE_DIRS
    onnxruntime::onnxruntime
    INTERFACE_INCLUDE_DIRECTORIES
)

get_target_property(
    NLOHMANN_JSON_INCLUDE_DIRS
    nlohmann_json::nlohmann_json
    INTERFACE_INCLUDE_DIRECTORIES
)

message(STATUS
    "CUDA compiler: ${CMAKE_CUDA_COMPILER}"
)

message(
    STATUS
    "ONNX Runtime include dirs = ${ONNXRUNTIME_INCLUDE_DIRS}"
)

message(
    STATUS
    "nlohmann_json include dirs = ${NLOHMANN_JSON_INCLUDE_DIRS}"
)

message(STATUS "Qt5_DIR = ${Qt5_DIR}")
message(STATUS "OpenCV_DIR = ${OpenCV_DIR}")

include_directories(${OpenCV_INCLUDE_DIRS})

# 将源代码添加到此项目的可执行文件。
add_executable (myDetect "src/main.cpp" 
    "src/inference_settings.cpp" 
    "src/yolov5_detector.cpp"
    "src/yoloDetectview_ui.cpp" "include/ui/yoloDetectview_ui.h"
    "src/object_detector_thread.cpp"
    "include/algo/object_detector_thread.h"
    include/algo/yolov8_detector.h
    src/yolov8_detector.cpp
    include/algo/ort_detector.h
    src/ort_detector.cpp
    include/algo/ort_gpu_runtime.h
    src/ort_gpu_runtime.cpp
    include/algo/ort_check_algo_cuda.h
    src/ort_check_algo_cuda.cpp
    include/algo/ort_detector_V2.h
    src/ort_detector_V2.cpp
    include/algo/ort_detector_V3.h
    src/ort_detector_V3.cpp
    include/algo/yolo_gpu_pipeline_V3.h
    src/yolo_gpu_pipeline_V3.cpp
    include/algo/core/yolo_gpu_pipeline_V3_Devices.cuh
    src/core/yolo_gpu_pipeline_V3_Devices.cu
    include/algo/core/yolo_gpu_pipeline_V3_kernels.cuh
    src/core/yolo_gpu_pipeline_V3_kernels.cu
    include/algo/inference_interface.h
    include/algo/core/yolo_gpu_pipeline_V3_Cubfun.h
    src/core/yolo_gpu_pipeline_V3_Cubfun.cu
    ) 

target_link_libraries(myDetect
    Qt5::Widgets
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
    CUDA::cudart
    nlohmann_json::nlohmann_json
)


target_include_directories(myDetect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# TODO: 如有需要，请添加测试并安装目标。
