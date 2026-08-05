#ifndef YOLO_GPU_PIPELINE_V3_CUBFUN_H
#define YOLO_GPU_PIPELINE_V3_CUBFUN_H

#include "algo/core/yolo_gpu_pipline_V3_config.h"
#include <driver_types.h>

namespace yolo_cuda {

__host__
void sortDetectionsDescending(
    void* workspace,
    std::size_t workspaceBytes,
    const float* scoreInput,
    float* scoresOutput,
    const GpuDetectionV3* detectionsInput,
    GpuDetectionV3* detectionsOutput,
    int itemCount,
    cudaStream_t stream
);



};

#endif