#include <__clang_cuda_builtin_vars.h>
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



};