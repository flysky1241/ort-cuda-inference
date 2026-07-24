# CMakeList.txt: myDetect 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.20)
project(myDetect LANGUAGES CXX)

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

get_target_property(
    ONNXRUNTIME_INCLUDE_DIRS
    onnxruntime::onnxruntime
    INTERFACE_INCLUDE_DIRECTORIES
)

message(
    STATUS
    "ONNX Runtime include dirs = ${ONNXRUNTIME_INCLUDE_DIRS}"
)

message(STATUS "Qt5_DIR = ${Qt5_DIR}")
message(STATUS "OpenCV_DIR = ${OpenCV_DIR}")

include_directories(${OpenCV_INCLUDE_DIRS})

# 将源代码添加到此项目的可执行文件。
add_executable (myDetect "src/main.cpp" 
    "src/inference_settings.cpp" 
    "src/yolov5_detector.cpp"
    "src/yolov5_detect_ui.cpp" "include/ui/yolov5_detect_ui.h"
    "src/object_detector_thread.cpp"
    "include/algo/object_detector_thread.h"
    include/algo/yolov8_detector.h
    src/yolov8_detector.cpp
    include/algo/ort_detector.h
    src/ort_detector.cpp) 

target_link_libraries(myDetect
    Qt5::Widgets
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
)


target_include_directories(myDetect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# TODO: 如有需要，请添加测试并安装目标。
