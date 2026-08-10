#include "algo/core/yolo_gpu_pipeline_V3_Devices.cuh"

namespace yolo_cuda_device {
    
__device__ float clampFloat(float value, float min, float max)
{
    return fminf(fmaxf(value, min), max);
}

__device__ float readModelOutput(
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

__device__ float intersectionOverUnion(const GpuDetectionV3& left, const GpuDetectionV3& right)
{
    


}


};