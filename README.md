[English](#english) | [简体中文](#chinese)
# ORT CUDA Inference

[English](#english) | [简体中文](#chinese)

<a id="english"></a>

## English

A C++ object detection project powered by **ONNX Runtime** and the **CUDA Execution Provider**.

This project uses ONNX Runtime to load ONNX models and perform GPU-accelerated inference on NVIDIA GPUs. It currently contains YOLOv5 and YOLOv8 object detection implementations, inference thread management, and UI-related code.

> The project is under active development and is mainly used to study C++, ONNX Runtime, CUDA inference, CMake, and computer vision engineering.

## Features

- ONNX model loading and inference with ONNX Runtime
- NVIDIA GPU acceleration through the CUDA Execution Provider
- YOLOv5 object detection
- YOLOv8 object detection
- Image preprocessing and detection result postprocessing
- Object detection worker thread
- UI-related detection code
- CMake-based project organization
- Ninja build system support
- Dependency management with vcpkg
- Custom ONNX Runtime vcpkg overlay port

## Tech Stack

- C++17
- CMake
- Ninja
- ONNX Runtime
- CUDA
- OpenCV
- Qt
- vcpkg
- MSVC

## Project Structure

```text
ort-cuda-inference/
├── myDetect/
│   ├── include/
│   │   ├── algo/                  # Detection algorithms
│   │   ├── ui/                    # UI-related headers
│   │   └── myDetect.h
│   ├── src/
│   │   ├── main.cpp
│   │   ├── myDetect.cpp
│   │   ├── inference_settings.cpp
│   │   ├── object_detector_thread.cpp
│   │   ├── ort_detector.cpp
│   │   ├── yolov5_detector.cpp
│   │   ├── yolov5_detect_ui.cpp
│   │   └── yolov8_detector.cpp
│   └── CMakeLists.txt
├── overlay-ports/
│   └── onnx/                      # Custom ONNX Runtime vcpkg port
├── rcc/                           # Qt resource files
├── script/                        # Build or helper scripts
├── CMakeLists.txt
├── CMakePresets.json
├── vcpkg.json
├── vcpkg-configuration.json
├── .gitignore
└── README.md
```

## Development Environment

The project is currently developed and tested mainly in the following environment:

- Windows
- Visual Studio 2019
- MSVC x64
- CMake 3.20 or later
- Ninja
- vcpkg
- NVIDIA GPU
- CUDA Toolkit
- ONNX Runtime with CUDA support

The versions of the following components must be compatible with each other:

- NVIDIA graphics driver
- CUDA Toolkit
- cuDNN
- ONNX Runtime GPU package

## Prerequisites

Before building the project, make sure the following tools are available:

```text
Git
CMake
Ninja
vcpkg
Visual Studio 2019 or another compatible MSVC toolchain
CUDA Toolkit
NVIDIA graphics driver
```

Check the installed versions:

```powershell
git --version
cmake --version
ninja --version
nvcc --version
```

## Dependency Management

The project uses vcpkg manifest mode.

The dependency configuration is stored in:

```text
vcpkg.json
vcpkg-configuration.json
```

The custom ONNX Runtime port is stored in:

```text
overlay-ports/onnx/
```

When CMake is configured with the vcpkg toolchain, the required dependencies can be installed automatically.

## Build

Open a terminal in the project root directory.

### Configure

```powershell
cmake --preset win-ninja-debug
```

### Build

Depending on the build presets defined in `CMakePresets.json`, run:

```powershell
cmake --build --preset win-ninja-debug
```

If only a configure preset is defined, build the generated directory directly:

```powershell
cmake --build out/build/win-ninja-debug
```

The main build output is generated under:

```text
out/build/
```

> Check `CMakePresets.json` for the actual preset names and output directories.

## Models

Large model files are not committed to this repository by default.

Prepare the required ONNX models locally, for example:

```text
models/
├── yolov5.onnx
└── yolov8.onnx
```

Then configure the correct model path in the application before running inference.

Common model formats such as the following are ignored by Git:

```text
*.onnx
*.engine
*.pt
*.pth
*.weights
```

For large models that need version control, Git LFS should be considered.

## Running

After a successful build, run the generated executable from the corresponding output directory.

Example:

```powershell
.\out\build\win-ninja-debug\myDetect\myDetect.exe
```

The actual executable path may vary depending on the configuration in `CMakeLists.txt` and `CMakePresets.json`.

## CUDA Execution Provider

The project configures ONNX Runtime with the CUDA Execution Provider.

The general initialization flow is:

```text
Create Ort::Env
        ↓
Create Ort::SessionOptions
        ↓
Create CUDA provider options
        ↓
Append CUDA Execution Provider
        ↓
Create Ort::Session
        ↓
Run model inference
```

A simplified example:

```cpp
Ort::SessionOptions session_options;

session_options.SetGraphOptimizationLevel(
    GraphOptimizationLevel::ORT_ENABLE_ALL
);

// Append the CUDA Execution Provider here.

Ort::Session session(
    env,
    model_path,
    session_options
);
```

## Git-Ignored Files

The repository does not include locally generated or machine-specific files such as:

- CMake build directories
- Visual Studio cache files
- VSCode local settings
- vcpkg-installed packages
- Executables and dynamic libraries
- Debug symbol files
- Runtime output directories
- Large ONNX model files
- Temporary files and logs

Typical ignored directories include:

```text
.vs/
.vscode/
build/
out/
vcpkg_installed/
myDetect/bin/
```

## Roadmap

- Improve the common ONNX Runtime inference wrapper
- Complete YOLOv5 and YOLOv8 preprocessing
- Improve detection result postprocessing
- Add CPU and CUDA provider switching
- Add model input and output validation
- Add inference latency statistics
- Add FPS statistics
- Improve object detection thread management
- Improve UI interaction
- Add structured logging
- Add unit tests
- Add example images and detection results
- Improve error handling and resource management

## Repository Status

This repository is currently a learning and engineering practice project.

Some interfaces, paths, model configurations, and build options may continue to change.

## Usage Notice

This project is intended for learning, research, and engineering practice. Please check the licenses of ONNX Runtime, OpenCV, Qt, CUDA, YOLO models, and other third-party dependencies before redistribution or commercial use.

---

<a id="chinese"></a>

## 简体中文

这是一个基于 **C++、ONNX Runtime 和 CUDA Execution Provider** 的目标检测项目。

项目使用 ONNX Runtime 加载 ONNX 模型，并通过 CUDA Execution Provider 在 NVIDIA GPU 上执行加速推理。目前包含 YOLOv5、YOLOv8 目标检测、推理工作线程以及检测界面相关代码。

> 项目目前仍在持续开发，主要用于学习和实践 C++、ONNX Runtime、CUDA 推理、CMake 与计算机视觉工程化。

## 项目功能

- 使用 ONNX Runtime 加载并运行 ONNX 模型
- 使用 CUDA Execution Provider 进行 NVIDIA GPU 加速
- YOLOv5 目标检测
- YOLOv8 目标检测
- 图像预处理与检测结果后处理
- 目标检测工作线程
- 检测界面相关代码
- 使用 CMake 管理项目
- 支持 Ninja 构建
- 使用 vcpkg 管理第三方依赖
- 使用自定义 ONNX Runtime vcpkg overlay port

## 技术栈

- C++17
- CMake
- Ninja
- ONNX Runtime
- CUDA
- OpenCV
- Qt
- vcpkg
- MSVC

## 项目目录

```text
ort-cuda-inference/
├── myDetect/
│   ├── include/
│   │   ├── algo/                  # 检测算法头文件
│   │   ├── ui/                    # 界面相关头文件
│   │   └── myDetect.h
│   ├── src/
│   │   ├── main.cpp
│   │   ├── myDetect.cpp
│   │   ├── inference_settings.cpp
│   │   ├── object_detector_thread.cpp
│   │   ├── ort_detector.cpp
│   │   ├── yolov5_detector.cpp
│   │   ├── yolov5_detect_ui.cpp
│   │   └── yolov8_detector.cpp
│   └── CMakeLists.txt
├── overlay-ports/
│   └── onnx/                      # 自定义 ONNX Runtime port
├── rcc/                           # Qt 资源文件
├── script/                        # 构建或辅助脚本
├── CMakeLists.txt
├── CMakePresets.json
├── vcpkg.json
├── vcpkg-configuration.json
├── .gitignore
└── README.md
```

## 开发环境

项目目前主要在以下环境中开发和测试：

- Windows
- Visual Studio 2019
- MSVC x64
- CMake 3.20 或更高版本
- Ninja
- vcpkg
- NVIDIA GPU
- CUDA Toolkit
- 支持 CUDA 的 ONNX Runtime

以下组件的版本需要相互兼容：

- NVIDIA 显卡驱动
- CUDA Toolkit
- cuDNN
- ONNX Runtime GPU 版本

## 环境要求

构建项目之前，请确保已安装：

```text
Git
CMake
Ninja
vcpkg
Visual Studio 2019 或兼容的 MSVC 工具链
CUDA Toolkit
NVIDIA 显卡驱动
```

可以使用以下命令检查：

```powershell
git --version
cmake --version
ninja --version
nvcc --version
```

## 依赖管理

项目使用 vcpkg manifest 模式管理依赖。

依赖配置位于：

```text
vcpkg.json
vcpkg-configuration.json
```

自定义的 ONNX Runtime port 位于：

```text
overlay-ports/onnx/
```

当 CMake 正确配置 vcpkg 工具链后，可以自动安装项目需要的依赖。

## 构建项目

在项目根目录打开终端。

### 配置项目

```powershell
cmake --preset win-ninja-debug
```

### 编译项目

根据 `CMakePresets.json` 中定义的构建预设执行：

```powershell
cmake --build --preset win-ninja-debug
```

如果当前只定义了 configure preset，也可以直接构建生成目录：

```powershell
cmake --build out/build/win-ninja-debug
```

主要构建结果生成在：

```text
out/build/
```

> 实际 Preset 名称和输出目录以 `CMakePresets.json` 为准。

## 模型文件

大型模型文件默认不提交到 Git 仓库。

请在本地准备所需的 ONNX 模型，例如：

```text
models/
├── yolov5.onnx
└── yolov8.onnx
```

运行推理前，需要在程序中配置正确的模型路径。

以下常见模型文件默认由 Git 忽略：

```text
*.onnx
*.engine
*.pt
*.pth
*.weights
```

需要对大型模型进行版本管理时，可以考虑 Git LFS。

## 运行项目

构建成功后，从对应输出目录运行生成的可执行文件。

示例：

```powershell
.\out\build\win-ninja-debug\myDetect\myDetect.exe
```

实际路径可能会随着 `CMakeLists.txt` 和 `CMakePresets.json` 的配置发生变化。

## CUDA Execution Provider

项目通过 CUDA Execution Provider 配置 ONNX Runtime GPU 推理。

整体初始化流程为：

```text
创建 Ort::Env
        ↓
创建 Ort::SessionOptions
        ↓
创建 CUDA Provider 配置
        ↓
添加 CUDA Execution Provider
        ↓
创建 Ort::Session
        ↓
运行模型推理
```

简化示例：

```cpp
Ort::SessionOptions session_options;

session_options.SetGraphOptimizationLevel(
    GraphOptimizationLevel::ORT_ENABLE_ALL
);

// 在此添加 CUDA Execution Provider。

Ort::Session session(
    env,
    model_path,
    session_options
);
```

## Git 忽略内容

仓库不会上传本机构建产物和机器相关文件，例如：

- CMake 构建目录
- Visual Studio 缓存
- VSCode 本地设置
- vcpkg 安装结果
- 可执行文件与动态库
- 调试符号文件
- 本地运行输出目录
- 大型 ONNX 模型
- 临时文件和日志

常见忽略目录包括：

```text
.vs/
.vscode/
build/
out/
vcpkg_installed/
myDetect/bin/
```

## 后续计划

- 完善通用 ONNX Runtime 推理封装
- 完善 YOLOv5 和 YOLOv8 图像预处理
- 完善检测结果后处理
- 增加 CPU 与 CUDA Provider 切换
- 增加模型输入输出验证
- 增加推理延迟统计
- 增加 FPS 统计
- 完善目标检测线程管理
- 完善界面交互
- 增加结构化日志
- 增加单元测试
- 增加运行截图与检测结果示例
- 完善异常处理与资源管理

## 仓库状态

本仓库目前是学习与工程实践项目。

部分接口、路径、模型配置和构建选项后续可能继续调整。

## 使用说明

本项目主要用于学习、研究和工程实践。在重新分发或商业使用之前，请检查 ONNX Runtime、OpenCV、Qt、CUDA、YOLO 模型以及其他第三方依赖的许可证。


