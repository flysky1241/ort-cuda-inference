#ifndef YOLO_GPU_PIPELINE_V3_DEVICE_H
#define YOLO_GPU_PIPELINE_V3_DEVICE_H

#include "algo/core/yolo_gpu_pipline_V3_config.h"
namespace yolo_cuda_device {

__device__ __forceinline__ 
float clampFloat(float value, float min, float max)
{
    return fminf(fmaxf(value, min), max);
}

__device__ __forceinline__ 
float readModelOutput(
    const float* output,
    int attribute,
    int candidate,
    int attributes,
    int candidates,
    bool attributeFirst)
{
    return attributeFirst 
        ? output[attribute * candidates + candidate]
        : output[candidate * attributes + attribute];
}

__device__ __forceinline__ 
float intersectionOverUnion(const GpuDetectionV3& left, const GpuDetectionV3& right)
{
    const float intersectionLeft = fmaxf(left.x1, right.x1);
    const float intersectionTop = fmaxf(left.y1, right.y1);
    const float intersectionRight = fminf(left.x2, right.x2);
    const float intersectionBottom = fminf(left.y2, right.y2);

    const float intersectionWidth = fmaxf(0.0f, intersectionRight-intersectionLeft);
    const float intersectionHeight = fmaxf(0.0f, intersectionBottom-intersectionTop);

    const float intersectionArea = intersectionWidth * intersectionHeight;

    const float leftArea = fmaxf(0.0f, left.x2-left.x1) * fmaxf(0.0f, left.y2-left.y1);
    const float RightArea = fmaxf(0.0f, right.x2-right.x1) * fmaxf(0.0f, right.y2-right.y1);

    const float UnionArea = leftArea + RightArea - intersectionArea;
 
    return UnionArea >0.0f ? intersectionArea/UnionArea : 0.0f;
}
};
#endif
