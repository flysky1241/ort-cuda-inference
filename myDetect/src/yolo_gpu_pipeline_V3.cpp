#include "algo/yolo_gpu_pipeline_V3.h"
#include <stdexcept>

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

std::size_t YoloGpuPipeline_V3::elementCount(
    const std::vector<std::int64_t>& shape)
{
    if(shape.empty())
    {
        throw std::runtime_error("Tensor shape must init");
    }

    std::size_t count = 1;

    for(const int64_t& dimension: shape)
    {
        
    }
}




