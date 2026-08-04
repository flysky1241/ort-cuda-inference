#include "algo/yolo_gpu_pipeline_V3.h"
#include <__clang_cuda_runtime_wrapper.h>

double OrtV3StageTimings::gpuTotalMs() const noexcept
{
    return static_cast<double>(
        hostStageingMs + 
        uploadMs +
        preprocessMs +
        inferenceMs +
        gpuPostprocessMs +
        downloadMs
    );
}


