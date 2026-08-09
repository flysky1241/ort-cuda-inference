#ifndef YOLO_GPU_PIPELINE_V3_KERNEL_H
#define YOLO_GPU_PIPELINE_V3_KERNEL_H

#include "algo/core/yolo_gpu_pipline_V3_config.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdexcept>
constexpr int kNmsThreads = 64;
constexpr int kPreprocessBlockX = 16;
constexpr int kPreprocessBlockY = 16;
constexpr int kDecodeThreads = 256;


namespace yolo_cuda_kernel {

__global__ void preprocessBgrToRgbBNchwKernel(
    const std::uint8_t* source,
    int sourceWidth,
    int sourceHeight,
    int sourceStride,
    float* destination,
    int destinationWidth,
    int destinationHeight,
    float scale,
    int padLeft,
    int padTop
);


__host__ void launchPreprocessNchwKernel(
    const dim3 blocks,
    const dim3 grides,
    ::cudaStream_t stream,
    const std::uint8_t* source,
    int sourceWidth,
    int sourceHeight,
    int sourceStride,
    float* destination,
    int destinationWidth,
    int destinationHeight,
    float scale,
    int padLeft,
    int padTop
);


__global__ void postDecodeYoloKernel(
    const float* output,
    int attributes,
    int candidates,
    int classCount,
    LetterboxTransformV3 transform,
    float scoreThreshold,
    float* scores,
    GpuDetectionV3* detections
);


__host__ void launchPostDecodeYoloKernel(
    const int decodeBlock,
    ::cudaStream_t stream,
    const float* output,
    int attributes,
    int candidates,
    int classCount,
    LetterboxTransformV3 transform,
    float scoreThreshold,
    float* scores,
    GpuDetectionV3* detections
);





__host__ inline 
void checkKernelLaunch(const char* name)
{
    cudaError_t cu_error = ::cudaGetLastError();
    if(cu_error != cudaError_t::cudaSuccess)
    {
        throw std::runtime_error(
            std::string("name") +
            "Launch failed" + 
            ::cudaGetErrorString(cu_error)
        );
    }
}

};

#endif
