#include "algo/core/yolo_gpu_pipeline_V3_Devices.cuh"

namespace yolo_cuda_device {
    
__device__ float clampFloat(float value, float min, float max)
{
    return fminf(fmaxf(value, min), max);
}


};