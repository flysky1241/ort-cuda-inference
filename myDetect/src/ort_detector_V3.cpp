#include "algo/ort_detector_V3.h"
#include "algo/ort_check_algo_cuda.h"
#include "algo/ort_detector.h"
#include "algo/ort_gpu_runtime.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <memory>
#include <mutex>
#include <ratio>
#include <sstream>
#include <stdexcept>
#include <string>

ORTDetector_V3::ORTDetector_V3(
        OrtV3Backend backend,
        int cudaDeviceId,
        OrtGpuRuntimeConfig gpuconfig)
    :m_backend_(backend),
    cudaDeviceId_(cudaDeviceId),
    m_gpuConfig_(gpuconfig),
    m_env_(globalOrtEnvironment()),
    m_session_(nullptr)
{
    if(cudaDeviceId_<0)
    {
        throw std::runtime_error("CUDA device ID must be non-negative");
    }

    if(!m_gpuConfig_.enableIoBinding)
    {
        throw std::invalid_argument(
            "ORTDetector_V3 requires GPU I/O Binding to remain enabled"
        );
    }

    if(m_gpuConfig_.warmupRuns<0 || 
        m_gpuConfig_.nmsTopK<=0 ||
        m_gpuConfig_.maxDetections<=0
    )
    {
        throw std::invalid_argument(
            "warmupRuns must be non-negative and NMS is positive"
        );
    }

    m_gpuConfig_.cudaEpOption.deviceId = cudaDeviceId_;
    m_gpuConfig_.trtEpOption.deviceId = cudaDeviceId_; 
}



void ORTDetector_V3::validateCudaDevice()
{
    int deviceCount = 0;
    CUDA_CHECK(::cudaGetDeviceCount(std::addressof(deviceCount)));
    if(deviceCount<=0)
    {
        throw std::runtime_error(
            "No CUDA-capable device is visible"
        );
    }

    if(cudaDeviceId_>=deviceCount)
    {
        std::ostringstream inputstream;
        inputstream<<"CUDA device ID "
                   <<cudaDeviceId_
                   <<"is invalid visible device count="
                   <<deviceCount; 

        throw std::overflow_error(inputstream.str());
    }

    CUDA_CHECK(::cudaSetDevice(cudaDeviceId_));

    ::cudaDeviceProp properties{};
    CUDA_CHECK(::cudaGetDeviceProperties_v2(
        std::addressof(properties), 
        cudaDeviceId_)
    );

    std::cout<<"[ORT-V3] CUDA device "<<cudaDeviceId_
             <<": "<<properties.name
             <<", compute capability "
             <<properties.major<<"."<<properties.minor
             <<", VRAM"
             <<properties.totalGlobalMem / (1024ULL * 1024ULL)
             <<" MiB\n";
}



void ORTDetector_V3::loadClassNames(const std::filesystem::path& filepath)
{   
    if(filepath.empty())
    {
        throw std::runtime_error("filepath must not be empty");
    }

    if(!std::filesystem::is_regular_file(filepath))
    {
        throw std::invalid_argument("must be regular file");
    }

    std::ifstream inputstream{};
    inputstream.open(filepath);

    if(!inputstream.is_open())
    {
        throw std::runtime_error(
            std::string("Failed to open class file: ") +
            filepath.generic_string()
        );
    }

    m_className_.clear();
    std::string tempstr;

    while(std::getline(inputstream, tempstr))
    {
        if(!tempstr.empty() && tempstr.back() == '\r')
        {
            tempstr.pop_back();
        }

        if(!tempstr.empty())
        {
            m_className_.emplace_back(tempstr);
        }

        if(tempstr.empty())
        {
            throw std::runtime_error(
                std::string("Class file contains no valid class name") +
                filepath.generic_string()
            );
        }
    }
}



void ORTDetector_V3::configureSessionOptions(const std::filesystem::path& modelPath)
{
    m_sessionOptions_ = Ort::SessionOptions{nullptr};
    m_sessionOptions_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    m_sessionOptions_.SetExecutionMode(
        ExecutionMode::ORT_SEQUENTIAL
    );

    m_sessionOptions_.SetIntraOpNumThreads(1);
    m_sessionOptions_.SetInterOpNumThreads(1);

    if(!m_stream_)
    {
        m_stream_.create(cudaStreamNonBlocking);
    }

    OrtGpuEpConfigurator configurator(m_sessionOptions_, m_stream_.get());
     
    switch (m_backend_) 
    {
    case OrtV3Backend::CUDA:
    {
        configurator.apendCuda(m_gpuConfig_.cudaEpOption);
        std::cout<<"[Ort-V3] provider priority: CUDA->CPU\n";
        break;
    }
    case OrtV3Backend::TRT:
    {
        this->prepareTensorRtCacheLayout(modelPath);
        configurator.apendTensorRt(m_gpuConfig_.trtEpOption);
        configurator.apendCuda(m_gpuConfig_.cudaEpOption);
        std::cout<<"[Ort-V3] provider priority:TRT->CUDA->CPU\n";
        break;
    }
    default:
    {
        throw std::runtime_error(
            "UnKnown ORTDetector_V3 backend"
        );
    }
    }
}


std::string ORTDetector_V3::makeModelFingerPrint(
        const std::filesystem::path& modelPath,
        const std::array<int64_t, 4>& inputShape,
        bool fp16,
        bool int8)
{
    std::ostringstream outputStream(std::ios_base::out);
    outputStream<< std::filesystem::weakly_canonical(modelPath)
                << '|' << std::filesystem::file_size(modelPath)
                << '|' << std::filesystem::last_write_time(modelPath).time_since_epoch().count()
                << '|' << inputShape[0] << 'X'
                << inputShape[1] << 'X'
                << inputShape[2] << 'X'
                << inputShape[3]
                << "|fp1=" << fp16
                << "|int8=" << int8
                << "ort_api=" <<Ort::GetVersionString();

    std::size_t value = std::hash<std::string>{}(outputStream.str());

    outputStream.str("");
    outputStream.clear();

    outputStream << std::hex 
                 << std::setfill(0)
                 << std::setw(16)
                 << value;

    return outputStream.str();
}



std::string ORTDetector_V3::sanitizePathComponent(std::string value)
{
    for(auto& charactor : value)
    {   
        const bool allowed = 
            (charactor >= 'a' && charactor <= 'z') ||
            (charactor >= 'A' && charactor <= 'Z') ||
            (charactor >= '0' && charactor <= '9') ||
            (charactor == '_' || charactor =='-');
            
        if(!allowed)
        {
            charactor = '_';
        }
    }

    return value;
}


void ORTDetector_V3::prepareTensorRtCacheLayout(const std::filesystem::path& modelPath)
{
    CUDA_CHECK(::cudaSetDevice(cudaDeviceId_));

    cudaDeviceProp properties{};
    CUDA_CHECK(::cudaGetDeviceProperties_v2(
        std::addressof(properties), 
        cudaDeviceId_
    ));

    std::ostringstream inputStream{};
    inputStream<< properties.name
               << "_cc"
               <<properties.major
               <<properties.minor;
    
    const std::string gpuKey = this->sanitizePathComponent(inputStream.str());

    inputStream.str("");
    inputStream.clear();

    std::string modelKey = this->makeModelFingerPrint(
        modelPath, 
        inputShape_, 
        m_gpuConfig_.trtEpOption.enableFp16, 
        m_gpuConfig_.trtEpOption.enableInt8
    );

    OrtTrtEpConfig& trt = this->m_gpuConfig_.trtEpOption;
    if(trt.engineCachePath.empty())
    {
        trt.engineCachePath =
            m_gpuConfig_.cacheRoot / "engines" / modelKey / gpuKey;
    }

    if(trt.timingCachePath.empty())
    {
        trt.timingCachePath = 
            m_gpuConfig_.cacheRoot / "timing" / gpuKey;
    }

    if(trt.enginCachePrefix.empty())
    {
        trt.enginCachePrefix =
            this->sanitizePathComponent(modelPath.stem().string()) +
            "_" + modelKey;
    }

    if(trt.enableEngineCache)
    {
        std::filesystem::create_directories(trt.engineCachePath);
    }

    if(trt.enableTimingCache)
    {
        std::filesystem::create_directories(trt.timingCachePath);
    }

    std::cout<<"[ORT-V3][TRT] engine cache:"
             << trt.engineCachePath<<'\n';

    std::cout<<"[ORT-V3][TRT] timing cache"
             <<trt.timingCachePath <<'\n';
}



void ORTDetector_V3::initConfig(InferenceSettings& settings)
{
    std::lock_guard<std::mutex> lk(this->inferenceMutex_);

    if(m_init)
    {
        throw std::logic_error(
            "ORTDetector_V3 is already init create new instance"
            "to change the model or GPu config"
        );
    }

    this->m_scoreThreshold = settings.getT_score();
    m_nmsThreshold = settings.get_conf();
    m_showScore = settings.getShow_score();
    m_showFPS = settings.getShow_fps();

    m_inputWidth_ = settings.get_input_w();
    m_inputHeight = settings.get_input_h();

    if(m_inputHeight <=0 || m_inputWidth_ <=0)
    {
        throw std::invalid_argument(
            "Configured model input size must be positive"
        );
    }

    inputShape_ = {
        1,
        3,
        static_cast<std::int64_t>(m_inputHeight),
        static_cast<std::int64_t>(m_inputWidth_)
    };

    if(m_scoreThreshold <=0.0f || m_scoreThreshold >1.0f ||
        m_nmsThreshold <0.0f || m_nmsThreshold >1.0f)
    {
        throw std::invalid_argument("Score and nms Threshold must be [0,1]");
    }

    this->loadClassNames(settings.getConfig_file());

    const std::filesystem::path modelPath(settings.getWeight_file());

    if(!std::filesystem::is_regular_file(modelPath))
    {
        throw std::runtime_error(
            std::string("ONNX model not exist") +
            modelPath.generic_string()
        );
    }

    this->validateCudaDevice();
    this->configureSessionOptions(modelPath);

    auto start = std::chrono::steady_clock::now();
    m_session_ = Ort::Session(m_env_, modelPath.c_str(), m_sessionOptions_);
    auto end = std::chrono::steady_clock::now();

    auto ms = std::chrono::duration<double, std::milli>(end-start).count();
    std::cout<< "[ORT-V3] session creation took"
             << ms
             << "ms\n";

    localModelMetaData();
    createGpuPipline();

    m_init = true;
}



void ORTDetector_V3::localModelMetaData()
{
    const std::size_t inputCount = m_session_.GetInputCount();
    const std::size_t outputCount = m_session_.GetOutputCount();

    if(inputCount !=1 || outputCount!=1)
    {
        throw std::runtime_error(
            "ORTDetector_V3 currently requires one"
        );
    }

    m_inputNameStorage_.clear();
    m_outputNameStorage_.clear();

    auto localFun = [](ORTDetector_V3* p)
    {
        Ort::AllocatedStringPtr nameinput = p->m_session_.GetInputNameAllocated(0, p->m_allocator_);
        if(nameinput == nullptr)
        {
            throw std::runtime_error(
                "ORTDetector_V3 Failed to read model"
            );
        }

        p->m_inputNameStorage_.push_back(nameinput.get());

        auto nameouput = p->m_session_.GetOutputNameAllocated(0, p->m_allocator_);
        if(nameouput == nullptr)
        {
            throw std::runtime_error(
                "Failed to read model output name"
            );
        }

        p->m_outputNameStorage_.push_back(nameouput.get());
    };

    localFun(this);

    const auto inputinfo = m_session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    const auto outputinfo = m_session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();

    if(inputinfo.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || 
        outputinfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        throw std::runtime_error(
            "ORTDetector_V3 currently requires float32 model input/output"
        );
    }

    const std::



}


void ORTDetector_V3::createGpuPipline()
{

}