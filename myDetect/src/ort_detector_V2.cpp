#include "algo/ort_detector_V2.h"
#include "algo/ort_check_algo_cuda.h"
#include "cuda_runtime_api.h"
#include "onnxruntime_cxx_api.h"
#include <filesystem>
#include <fstream>
#include <mutex>
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

void ORTDetector_V2::destroyGpuResources() noexcept
{
    m_ioRunner_.reset();

    m_session_ = Ort::Session{nullptr};

    if(m_stream_!=nullptr)
    {
        ::cudaStreamDestroy(m_stream_);
        m_stream_ = nullptr;
    }
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
        if(!tempStr.empty() && tempStr.back()=='\r')
        {
            tempStr.pop_back();
        }

        if(!tempStr.empty())
        {
            this->m_className_.push_back(tempStr);
        }
    }

    if(tempStr.empty())
    {
        throw std::runtime_error(
            "Class file contains no valid class names: " + 
            filePath.generic_string());
    }
}


void prepareTensorRtCacheLayout(const std::filesystem::path& modelPath)
{


}

void ORTDetector_V2::initConfig(InferenceSettings& settings) 
{
    std::lock_guard<std::mutex> lk(this->m_inferMutx_);

    m_init = false;
    this->destroyGpuResources();

    m_scoreThreshold = settings.getT_score();
    m_inputWidth_ = settings.get_input_w();
    m_inputHeight = settings.get_input_h();
    m_showScore = settings.getShow_score();
    m_showFPS = settings.getShow_fps();

    this->localClassNames(settings.getConfig_file());

    const std::filesystem::path& modelPath(settings.getWeight_file());
    if(!std::filesystem::is_regular_file(modelPath))
    {
        throw std::invalid_argument(
            "ONNX model does not exist or is not a regular file: "+
            modelPath.generic_string()
        );
    }
    
    m_inputShape_ = {
        1,
        3,
        static_cast<int64_t>(m_inputWidth_),
        static_cast<int64_t>(m_inputHeight)
    };

    if(m_backend_ == OrtBackend::TRT)
    {
        this->prepareTensorRtCacheLayout(modelPath);
    }
    
}
