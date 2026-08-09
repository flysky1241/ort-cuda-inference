#include <__clang_cuda_builtin_vars.h>
#include "algo/core/yolo_gpu_pipeline_V3_Devices.cuh"
#include "algo/core/yolo_gpu_pipeline_V3_kernels.cuh"

namespace yolo_cuda_kernel{

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
    int padTop)
{
    const int destinationX = 
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int destinationY = 
        static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if(destinationX > destinationWidth || 
        destinationY > destinationHeight)
    {
        return;
    }

    constexpr float borderValue = 114.0f/255.0f;

    float red = borderValue;
    float green = borderValue;
    float blue = borderValue;

    const float sourceX = (static_cast<float>(destinationX - padLeft) + 0.5f) /scale - 0.5f;
    const float sourceY = (static_cast<float>(destinationY - padTop) + 0.5f) / scale -0.5f;

    if(sourceX > -0.5f && sourceX < static_cast<float>(sourceWidth)-0.5f &&
        sourceY > -0.5f && sourceY < static_cast<float>(sourceHeight)-0.5)
    {
        const float clampedX = yolo_cuda_device::clampFloat(sourceX, 0.0f, static_cast<float>(sourceWidth-1));
        const float clampedY = yolo_cuda_device::clampFloat(sourceY, 0.0f, static_cast<float>(sourceHeight-1));

        const int x0 = static_cast<int>(floorf(clampedX));
        const int y0 = static_cast<int>(floorf(clampedY));
        const int x1 = (min)(x0+1, sourceWidth-1);
        const int y1 = (min)(y0+1, sourceHeight-1);

        const float xWeight = clampedX - static_cast<float>(x0);
        const float yWeight = clampedY - static_cast<float>(y0);

        const uint8_t* row0 = source + y0 * sourceStride;
        const uint8_t* row1 = source + y1 * sourceStride;

        const int offset00 = x0 * 3U;
        const int offset01 = x1 * 3U;

        const float b00 = static_cast<float>(row0[offset00 + 0]);
        const float b01 = static_cast<float>(row0[offset01 + 0]);
        const float b10 = static_cast<float>(row1[offset00 + 0]);
        const float b11 = static_cast<float>(row1[offset01 + 0]);
        
        const float g00 = static_cast<float>(row0[offset00 + 1]);
        const float g01 = static_cast<float>(row0[offset01 + 1]);
        const float g10 = static_cast<float>(row1[offset00 + 1]);
        const float g11 = static_cast<float>(row1[offset01 + 1]);

        const float r00 = static_cast<float>(row0[offset00 + 2]);
        const float r01 = static_cast<float>(row0[offset01 + 2]);
        const float r10 = static_cast<float>(row1[offset00 + 2]);
        const float r11 = static_cast<float>(row1[offset01 + 2]);

        const float bTop = b00 + (b01 - b00) * xWeight;
        const float bBot = b10 + (b11 - b10) * xWeight;
        const float gTop = g00 + (g01 - g00) * xWeight;
        const float gBot = g10 + (g11 - g10) * xWeight;
        const float rTop = r00 + (r01 - r00) * xWeight;
        const float rBot = r10 + (r11 - r10) * xWeight;

        blue = (bTop + (bBot - bTop) * yWeight) / 255.0f;
        green = (gTop + (gBot - gTop) * yWeight) / 255.0f;
        red = (rTop + (rBot - rTop) * yWeight) / 255.0f;
    }  

    const int plane = destinationWidth * destinationHeight;
    const int pixel = destinationY * destinationWidth + destinationX;

    destination[pixel] = red;
    destination[pixel+plane] = green;
    destination[pixel*2 + plane] = blue;
}


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
    int padTop)
{
    yolo_cuda_kernel::preprocessBgrToRgbBNchwKernel<<<grides, blocks, 0, stream>>>(
        source,
        sourceWidth,
        sourceHeight,
        sourceStride,
        destination,
        destinationWidth,
        destinationHeight,
        scale,
        padLeft,
        padTop
    );
}


__global__ void postDecodeYoloKernel(
    const float* output,
    int attributes,
    int candidates,
    int classCount,
    LetterboxTransformV3 transform,
    float scoreThreshold,
    float* scores,
    GpuDetectionV3* detections)
{
    const int candidate = 
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    
    if(candidate > candidates)
    {
        return;
    }

    



}



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
    GpuDetectionV3* detections)
{
    yolo_cuda_kernel::postDecodeYoloKernel<<<decodeBlock, kDecodeThreads, 0, stream>>>(
        output,
        attributes,
        candidates,
        classCount,
        transform,
        scoreThreshold,
        scores,
        detections
    );
}



};