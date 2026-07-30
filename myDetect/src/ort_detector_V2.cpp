#include "algo/ort_detector_V2.h"
#include "algo/ort_check_algo_cuda.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

ORTDetector_V2::ORTDetector_V2(
    OrtBackend backend, 
    int cudaDeviceId,
    OrtGpuRuntimeConfig gpuConfig)
    :m_backend_(backend),
    cudaDeviceId_(cudaDeviceId),
    m_env_(globalOrtEnvironment()),
    m_session_(nullptr),
    m_cpuMeminfo_(Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
    )),
    m_stream_(nullptr)
{
    if(this->cudaDeviceId_<0)
    {
        throw std::runtime_error("CUDA must to be init");
    }

    m_gpuConfig_.trtEpOption.deviceId = cudaDeviceId_;
    m_gpuConfig_.cudaEpOption.deviceId = cudaDeviceId_;
}


void ORTDetector_V2::localClassNames(const std::filesystem::path& filePath)
{
    m_className_.clear();

    if(!std::filesystem::is_regular_file(filePath))
    {
        throw std::invalid_argument("must be regular_file");
    }

    std::ifstream inputFile(filePath);
    if(!inputFile.is_open())
    {
        throw std::runtime_error("create ifstream stream is bad");
    }

    std::string tempStr;

    while(std::getline(inputFile, tempStr))
    {
        
    }


}
