#include "algo/core/yolo_gpu_pipline_V3_config.h"

[[nodiscard]]
__host__ double OrtV3StageTimings::gpuTotalMs() const noexcept
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