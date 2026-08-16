#include "algo/ort_detector_V3.h"
#include "algo/core/yolo_gpu_pipline_V3_config.h"
#include "algo/inference_settings.h"
#include "algo/ort_check_algo_cuda.h"
#include "algo/ort_detector.h"
#include "algo/ort_gpu_runtime.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "opencv2/core/fast_math.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/operations.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgproc.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <locale>
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
    m_sessionOptions_ = Ort::SessionOptions{};
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
                 << std::setfill('0')
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

    auto inputTypeInfo = m_session_.GetInputTypeInfo(0);
    auto inputinfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    auto outputTypeInfo = m_session_.GetOutputTypeInfo(0);
    auto outputinfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    if(inputinfo.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || 
        outputinfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        throw std::runtime_error(
            "ORTDetector_V3 currently requires float32 model input/output"
        );
    }

    const std::vector<std::int64_t> modelInputShape = inputinfo.GetShape();
    this->outputShape_ = outputinfo.GetShape();

    if(modelInputShape[0]!=1 ||
       modelInputShape.size()!=4 ||
       modelInputShape[1]!=3)
    {
        throw std::runtime_error(
            "Expected Model inputShape [1, 3, H, W]" +
            ::shape2String(modelInputShape)
        );
    }

    if(hasDynamicDimension(modelInputShape) || hasDynamicDimension(outputShape_))
    {
        throw std::runtime_error("must fixed shape YOLO ONNX MODEL");
    }

    this->m_inputHeight = static_cast<int>(modelInputShape.at(2));
    this->m_inputWidth_ = static_cast<int>(modelInputShape.at(3));

    this->inputShape_ = {
        1,
        3,
        static_cast<std::int64_t>(m_inputHeight),
        static_cast<std::int64_t>(m_inputWidth_)
    };

    if(outputShape_[0]!=1 || outputShape_.size()!=3)
    {
        throw  std::runtime_error(
            "Expected YoloV8 output [1, attributes, candidates] or"  
            "[1, candidates, attributes]" + 
            shape2String(outputShape_)
        );
    }

    const std::int64_t attributes = (std::min)(outputShape_[1], outputShape_[2]);
    const std::int64_t expectedAttributes = static_cast<std::int64_t>(this->m_className_.size() + 4);
    if(attributes != expectedAttributes)
    {
        throw std::runtime_error(
            "YoloV8 outputshape attributes do not match class file" +
            shape2String(outputShape_) +
            ", classes = "+ std::to_string(m_className_.size())
        );
    }

    std::cout<<"[ORT-V3] input: "<< m_inputNameStorage_.front()
             <<"[1, 3, "<<m_inputHeight<<','<<m_inputWidth_<<"]\n";
    
    std::cout<<"[ORT-V3] output: "<< m_outputNameStorage_.front()
             <<' '<<shape2String(outputShape_)<<'\n';
}


bool ORTDetector_V3::hasDynamicDimension(const std::vector<int64_t>& shape)
{
    return std::any_of(shape.begin(), shape.end(), [](int64_t temp){
        if(temp<0)
            return true;
        return false;
    });
}


void ORTDetector_V3::createGpuPipline()
{
    YoloGpuPipelineV3Config config{};
    config.nmsThrehold = m_nmsThreshold;
    config.scoreThreshold = m_scoreThreshold;
    config.disableProviderSynchronization = m_gpuConfig_.disableProviderSynchronization;
    config.maxDetections = m_gpuConfig_.maxDetections;
    config.nmsTopK = m_gpuConfig_.nmsTopK;
    config.warmupRuns = m_gpuConfig_.warmupRuns;

    this->m_gpuPipline_ = std::make_unique<YoloGpuPipeline_V3>(
        m_session_,
        m_stream_.get(),
        cudaDeviceId_,
        m_inputNameStorage_.front(),
        std::vector<std::int64_t>(inputShape_.begin(), inputShape_.end()),
        m_outputNameStorage_.front(),
        outputShape_,
        m_className_.size(),
        config
    );

    m_gpuPipline_->warmup();
}



LetterboxTransformV3 ORTDetector_V3::makeLetterBoxTransform(const cv::Mat& frame) const
{   
    if(frame.empty())
    {
        throw std::runtime_error(
            "input frame is empty"
        );
    }

    LetterboxTransformV3 transform;
    transform.originalHeight = frame.rows;
    transform.originalWidth = frame.cols;

    const float widthScale = 
        static_cast<float>(m_inputWidth_) / 
        static_cast<float>(frame.cols);

    const float heightScale = 
        static_cast<float>(m_inputHeight) / 
        static_cast<float>(frame.rows);
    
    transform.scale = (std::min)(widthScale, heightScale);

    const int resizeWidth = 
        static_cast<int>(std::round(frame.cols * transform.scale));

    const int resizeHeight = 
        static_cast<int>(std::round(frame.rows * transform.scale));
    
    transform.padLeft = (m_inputWidth_ - resizeWidth)/2;
    transform.padTop = (m_inputHeight - resizeHeight)/2;

    return transform;
}



void ORTDetector_V3::drawDetectionResults(cv::Mat& frame, const std::vector<ODResultBox>& result_boxes) const
{
    const cv::Scalar boxColor(0, 255, 0);
    const cv::Scalar textColor(0, 0, 0);

    for(const auto& result : result_boxes)
    {
        cv::rectangle(
            frame, 
            result.box, 
            boxColor,
            2,
            cv::LINE_AA
        );

        std::string label = result.name;
        if(m_showScore)
        {
            label += cv::format("%.2f", result.score);
        }

        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(
            label, 
            cv::FONT_HERSHEY_SIMPLEX, 
            0.55, 
            1, 
            std::addressof(baseline)
        );

        const int textX = std::clamp(result.box.x, 0, (std::max)(0, frame.cols-1));
        const int textY = (std::max)(textSize.height+5, result.box.y);

        const int backgroundWidth =
            std::min(textSize.width + 6, frame.cols - textX);
        const int backgroundHeight =
            textSize.height + baseline + 6;

        if (backgroundWidth > 0 && backgroundHeight > 0)
        {
            cv::rectangle(
                frame,
                cv::Rect(
                    textX,
                    std::max(0, textY - textSize.height - 5),
                    backgroundWidth,
                    backgroundHeight),
                boxColor,
                cv::FILLED);
        }

        cv::putText(
            frame,
            label,
            cv::Point(textX + 3, textY),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            textColor,
            1,
            cv::LINE_AA);

    }
}


void ORTDetector_V3::infer_frame(cv::Mat& frame, std::vector<ODResultBox>& result_boxes)
{
    std::lock_guard<std::mutex> lk(this->inferenceMutex_);

    if(m_init == false || m_gpuPipline_ == nullptr)
    {
        throw std::runtime_error("ORTDetector_V3 not init");
    }

    if(frame.empty())
    {
        throw std::runtime_error("input frame not empty");
    }

    if(frame.type() != CV_8UC3)
    {
        throw std::runtime_error("ORTDetector_V3 requires CV_8UC3 bgr");
    }

    const auto totalbegin = std::chrono::steady_clock::now();

    LetterboxTransformV3 transform = this->makeLetterBoxTransform(frame);
    
    std::vector<GpuDetectionV3> detections =
        m_gpuPipline_->run(frame, transform);
    
    
    result_boxes.clear();
    result_boxes.reserve(detections.size());

    const cv::Rect imageBound(0, 0, frame.cols, frame.rows);
    for(auto index = 0; index<detections.size(); ++index)
    {
        if(detections[index].classId <0 || detections[index].classId >= static_cast<int>(m_className_.size()))
        {
            continue;
        }

        cv::Rect rectangle(
            cvRound(detections[index].x1),
            cvRound(detections[index].y1),
            cvRound(detections[index].x2 - detections[index].x1),
            cvRound(detections[index].y2 - detections[index].y1)
        );

        rectangle &= imageBound;
        if(rectangle.empty())
        {
            continue;
        }

        ODResultBox box;
        box.box = rectangle;
        box.score = detections[index].score;
        box.name = m_className_[static_cast<int>(detections[index].classId)];
    
        result_boxes.push_back(box);
    }

    this->drawDetectionResults(frame, result_boxes);

    const auto totalEnd = std::chrono::steady_clock::now();
    const double totalMill = std::chrono::duration<double, std::milli>(totalEnd-totalbegin).count();
    if(totalMill > 0.0)
    {
        const double currentFps = 1000.0 / totalMill;   
        smoothedFps_ = currentFps;
    }

    if(m_showFPS)
    {
        const OrtV3StageTimings& timings = m_gpuPipline_->lastTimings();
        const std::string text = cv::format(
            "FPS %.1f | stage %.2f | H2D %.2f | pre %.2f | infer %.2f | post %.2f | D2H %.2f ms",
            smoothedFps_,
            timings.hostStageingMs,
            timings.uploadMs,
            timings.preprocessMs,
            timings.inferenceMs,
            timings.gpuPostprocessMs,
            timings.downloadMs);

        cv::putText(
            frame,
            text,
            cv::Point(20, 35),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(255, 0, 0),
            2,
            cv::LINE_AA);

    }

}

ORTDetector_V3::~ORTDetector_V3()
{

}