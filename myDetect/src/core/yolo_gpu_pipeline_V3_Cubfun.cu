#include "algo/core/yolo_gpu_pipeline_V3_Cubfun.h"
#include "cub/device/device_radix_sort.cuh"
#include "algo/ort_check_algo_cuda.h"

namespace yolo_cuda_cub{


__host__
void sortDetectionsDescending(
    void* workspace,
    std::size_t workspaceBytes,
    const float* scoreInput,
    float* scoresOutput,
    const GpuDetectionV3* detectionsInput,
    GpuDetectionV3* detectionsOutput,
    int itemCount,
    cudaStream_t stream)
{
    if(workspace == nullptr)
    {
        std::cout<<"workspace is init";
    }

    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
        workspace,
        workspaceBytes,
        scoreInput,
        scoresOutput,

        detectionsInput,
        detectionsOutput,

        itemCount,
        0,
        32,
        stream
    ));
}





};