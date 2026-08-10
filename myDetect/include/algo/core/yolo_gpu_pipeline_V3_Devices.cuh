#ifndef YOLO_GPU_PIPELINE_V3_DEVICE_H
#define YOLO_GPU_PIPELINE_V3_DEVICE_H

#include "algo/core/yolo_gpu_pipline_V3_config.h"
#include <__clang_cuda_runtime_wrapper.h>
namespace yolo_cuda_device {

__device__ float clampFloat(float value, float min, float max);

__device__ float readModelOutput(
    const float* output,
    int attribute,
    int candidate,
    int attributes,
    int candidates,
    bool attributeFirst
);

__device__ float intersectionOverUnion(const GpuDetectionV3& left, const GpuDetectionV3& right); 

};
#endif
