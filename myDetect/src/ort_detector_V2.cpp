#include "algo/ort_detector_V2.h"
#include "algo/ort_check_algo_cuda.h"
#include "algo/ort_gpu_runtime.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "opencv2/core/utility.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <memory>
#include <mutex>
#include <ratio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>

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




__host__ void ORTDetector_V2::prepareTensorRtCacheLayout(const std::filesystem::path& modelPath)
{
    CUDA_CHECK(::cudaSetDevice(this->cudaDeviceId_));

    ::cudaDeviceProp properties{};
    CUDA_CHECK(
        cudaGetDeviceProperties(std::addressof(properties), 
        this->cudaDeviceId_)
    );

    std::ostringstream outputStream(std::ios_base::out);
    outputStream<<std::string(properties.name)
                <<"_cc"<<std::to_string(properties.major)
                <<std::to_string(properties.minor);

    std::string gpukey = sanitizePathComponent(outputStream.str());

    std::string modelkey = this->makeModelFingerPrint(
        modelPath, 
        m_inputShape_, 
        m_gpuConfig_.trtEpOption.enableFp16, 
        m_gpuConfig_.trtEpOption.enableInt8
    );

    if(m_gpuConfig_.trtEpOption.engineCachePath.empty())
    {
        m_gpuConfig_.trtEpOption.engineCachePath = 
            m_gpuConfig_.cacheRoot /
            "engines" /
            modelkey /
            gpukey;
    }

    if(m_gpuConfig_.trtEpOption.timingCachePath.empty())
    {
        m_gpuConfig_.trtEpOption.timingCachePath = 
            m_gpuConfig_.cacheRoot /
            "timing" /
            gpukey;
    }

    if(m_gpuConfig_.trtEpOption.enginCachePrefix.empty())
    {
        m_gpuConfig_.trtEpOption.enginCachePrefix = 
            this->sanitizePathComponent(modelPath.stem().string()) +
            "_" +
            modelkey; 
    }

    std::error_code Ec;
    std::filesystem::create_directories(
        m_gpuConfig_.trtEpOption.engineCachePath, 
        Ec
    );
    if(!Ec)
    {
        throw std::runtime_error(
            "create m_gpuConfig_.trtEpOption.engineCachePath is failed" +
            Ec.message()
        );
    }

    Ec.clear();
    std::filesystem::create_directories(
        m_gpuConfig_.trtEpOption.timingCachePath, 
        Ec
    );
    if(!Ec)
    {
        throw std::runtime_error(
            "create m_gpuConfig_.trtEpOption.timingCachePath is failed" +
            Ec.message()
        );
    }

    std::cout<<"[ORT][TRT] Engine cache: "
             <<m_gpuConfig_.trtEpOption.engineCachePath<<'\n';

    std::cout<<"[ORT][TRT] timing cache: "
             <<m_gpuConfig_.trtEpOption.timingCachePath<<'\n';
}

std::string ORTDetector_V2::sanitizePathComponent(std::string value)
{
    for(char& character : value)
    {
        const bool allowed = 
            (character>='a' && character<='z') ||
            (character>='A' && character<='Z') ||
            (character>='0' && character<='9') ||
            character == '-' ||
            character == '_';
        if(!allowed)
        {
            character = '_';
        }
    }

    return value;
}

std::string makeModelFingerPrint(
    const std::filesystem::path& modelPath,
    const std::array<int64_t, 4>& inputShape,
    bool fp16,
    bool int8)
{
    std::ostringstream outputStream(std::ios_base::out);
    outputStream<<std::filesystem::weakly_canonical(modelPath).generic_string()
                <<'|'
                <<std::filesystem::file_size(modelPath)
                <<'|'
                <<std::filesystem::last_write_time(modelPath).time_since_epoch().count()
                <<'|'
                <<inputShape[0]<<'X'
                <<inputShape[1]<<'X'
                <<inputShape[2]<<'X'
                <<inputShape[3]
                <<"|fp16="<<fp16
                <<"|int8="<<int8
                <<"ort_api="<<ORT_API_VERSION;
    
    std::size_t value = std::hash<std::string>{}(outputStream.str());

    outputStream.clear();

    outputStream<<std::hex
                <<std::setfill('0')
                <<std::setw(16)
                <<value;

    return outputStream.str();
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

    this->configureSessionOptions(modelPath);

    try
    {
        auto start = std::chrono::steady_clock::now();

        this->m_session_ = Ort::Session(m_env_, modelPath.c_str(), m_sessionOption_);

        auto end = std::chrono::steady_clock::now();

        auto mssecond = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

        std::cout<<"[ORT] Session creation took"
                 << mssecond<<"ms\n";
    } 
    catch (const Ort::Exception& except) {
        throw std::runtime_error(
            std::string("Faile create ONNX Runtime Session") + 
            except.what()
        );
    }

    this->localModelMetaData();

    if(m_backend_!=OrtBackend::CPU && 
       m_gpuConfig_.enableIoBinding)
    {
        if(hasDynamicDimension(m_outputShape_))
        {
            this->resolveDynamicOutputShape();
        }

        this->createIoBindingRunner();
        m_ioRunner_->warmup(m_gpuConfig_.warmupRuns);
    }
    
    m_init = true;
}

void ORTDetector_V2::localModelMetaData()
{
    m_inputNameStorage_.clear();
    m_inputName_.clear();
    m_outputNameStorage_.clear();
    m_outputName_.clear();

    std::size_t inputCount = m_session_.GetInputCount();
    std::size_t outputCount = m_session_.GetOutputCount();

    if(inputCount!=1)
    {
        std::cout<<"ONNX Shape input Count: "<<inputCount;
    }

    if(outputCount!=1)
    {
        std::cout<<"ONNX Shape output Count: "<<outputCount;
    }

    m_inputNameStorage_.reserve(inputCount);
    for(auto index = 0; index<=inputCount; ++index)
    {
        auto name = m_session_.GetInputNameAllocated(index, this->m_allocator_);
        if(name == nullptr)
        {
            throw std::runtime_error("Failed to read model input name");
        }
        m_inputNameStorage_.emplace_back(name.get());
    }

    m_outputNameStorage_.reserve(outputCount);
    for(auto index = 0; index <= outputCount; ++index)
    {
        auto name = m_session_.GetOutputNameAllocated(index, this->m_allocator_);
        if(name==nullptr)
        {
            throw std::runtime_error("Failed to output model output name");
        }
        m_outputNameStorage_.emplace_back(name.get());
    }

    m_inputName_.reserve(inputCount);
    for(auto index = 0; index<=inputCount; ++index)
    {
        m_inputName_.push_back(m_inputNameStorage_[index].c_str());
    }

    m_outputName_.reserve(outputCount);
    for(auto index=0; index <= outputCount; ++index)
    {
        m_outputName_.push_back(m_outputNameStorage_[index].c_str());
    }

    auto inputinfo = m_session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();

    if(inputinfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        throw std::runtime_error("Only float32 model input");
    }

    std::vector<int64_t> modelInputShape = inputinfo.GetShape();
    if(modelInputShape.size()!=4)
    {
        throw std::runtime_error(
            "Excepted [N,C,W,H] model input, got "+
            ::shape2String(modelInputShape)
        );
    }

    if(modelInputShape[0]>0 && modelInputShape[0]!=1)
    {
        throw std::runtime_error(
            "batch must be one "+
            ::shape2String(modelInputShape)
        );
    }

    if(modelInputShape[1]>0 && modelInputShape[1]!=3)
    {
        throw std::runtime_error(
            "channel must be three "+
            ::shape2String(modelInputShape)
        );
    }

    if(modelInputShape[2]>0)
    {
        m_inputWidth_ = static_cast<int>(modelInputShape[2]);
    }

    if(modelInputShape[3]>0)
    {
        m_inputHeight = static_cast<int>(modelInputShape[3]);
    }

    m_inputShape_ = {
        static_cast<int64_t>(1),
        static_cast<int64_t>(3),
        static_cast<int64_t>(m_inputWidth_),
        static_cast<int64_t>(m_inputHeight)
    };

    auto outputinfo = m_session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
    if(outputinfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        throw std::runtime_error("Only float32 model output is supported");
    }

    m_outputShape_ = outputinfo.GetShape();

    std::cout<<"[ORT] Input: "
             << m_inputNameStorage_.front()
             <<" "<<::shape2String(std::initializer_list<int64_t>(m_inputShape_.data(), m_inputShape_.data()+m_inputShape_.size()));
    
    std::cout<<"[ORT] output: "
             <<m_outputName_.front()
             <<" "<<::shape2String(m_outputShape_);
};


void ORTDetector_V2::createCudaStream()
{
    if(m_stream_ != nullptr)
    {
        return;
    }

    CUDA_CHECK(::cudaSetDevice(cudaDeviceId_));
    CUDA_CHECK(::cudaStreamCreateWithFlags(
        std::addressof(m_stream_), 
        cudaStreamNonBlocking
    ));
}


void ORTDetector_V2::configureSessionOptions(const std::filesystem::path& modelPath)
{
    (void)modelPath;

    this->m_sessionOption_ = Ort::SessionOptions{};
    m_sessionOption_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );
    m_sessionOption_.SetExecutionMode(
        ExecutionMode::ORT_SEQUENTIAL
    );

    std::unique_ptr<OrtGpuEpConfigurator> config_ptr{};

    auto configFunc_ = [](ORTDetector_V2* p) mutable-> std::unique_ptr<OrtGpuEpConfigurator>
    {
        p->createCudaStream();

        p->m_sessionOption_.SetIntraOpNumThreads(1);
        p->m_sessionOption_.SetInterOpNumThreads(1);

        return 
            std::make_unique<OrtGpuEpConfigurator>(
                p->m_sessionOption_, 
                p->m_stream_
            );
    };

    switch(this->m_backend_)
    {
    case OrtBackend::CPU:
    {
        const auto hardwareThreads = std::thread::hardware_concurrency();

        const int intraOpthreads = hardwareThreads > 2U ? static_cast<int>(hardwareThreads/2U):1;

        m_sessionOption_.SetIntraOpNumThreads(intraOpthreads);
        std::cout<<"[ORT] provider priority : CPU"<<'\n';
        return;
    }
    case OrtBackend::CUDA:
    {
        config_ptr = configFunc_(this);
        config_ptr->apendCuda(m_gpuConfig_.cudaEpOption);
        std::cout<<"[ORT] provider priority: CUDAT-> CPU"<<'\n';
    
        return;   
    }
    case OrtBackend::TRT:
    {
        config_ptr = configFunc_(this);
        
        config_ptr->apendTensorRt(m_gpuConfig_.trtEpOption);
        config_ptr->apendCuda(m_gpuConfig_.cudaEpOption);

        std::cout<<"[ORT] provider priority: TensorRt->CUDA->CPU"<<'\n';
        return;
    }
    default:
    {
        throw std::runtime_error("Unknown ONNX Runtime backend");
    }
    };
}


bool ORTDetector_V2::hasDynamicDimension(const std::vector<int64_t>& shape)
{
    return std::any_of(shape.begin(), shape.end(), [](int64_t temp){
        if(temp<0)
        {
            return true;
        }
        return false;
    });
}

void ORTDetector_V2::resolveDynamicOutputShape()
{
    const std::size_t inputElems = 
        static_cast<std::size_t>(m_inputShape_[0])*
        static_cast<std::size_t>(m_inputShape_[1])*
        static_cast<std::size_t>(m_inputShape_[2])*
        static_cast<std::size_t>(m_inputShape_[3]);

    std::vector<float> zeros(inputElems, 0.0f);

    Ort::Value inputTensor = Ort::Value::CreateTensor(
        m_cpuMeminfo_,
        zeros.data(),
        inputElems,
        m_inputShape_.data(),
        m_inputShape_.size()
    );

    std::vector<Ort::Value> outputTensor = this->m_session_.Run(
        Ort::RunOptions{nullptr},
        m_inputName_.data(),
        std::addressof(inputTensor),
        1,
        m_outputName_.data(),
        1
    );

    if(outputTensor.size()!=1 && !outputTensor.front().IsTensor())
    {
        throw std::runtime_error("Failed to discover the concrete model output shape");
    }

    auto outputShape = outputTensor.front().GetTensorTypeAndShapeInfo().GetShape();
    if(hasDynamicDimension(outputShape))
    {   
        throw std::runtime_error(
            "Output shape is still dynamic after discovery run:" +
            shape2String(outputShape)
        );
    }

    std::cout << "[ORT] Resolved output shape: "
              << shape2String(m_outputShape_) << '\n';
}



void ORTDetector_V2::createIoBindingRunner()
{
    m_ioRunner_ = std::make_unique<OrtFixedShapeIoBindingRunner>(
        m_session_,
        m_stream_,
        cudaDeviceId_,
        m_inputNameStorage_.front(),
        std::vector<int64_t>(m_inputShape_.begin(), m_inputShape_.end()),
        m_outputNameStorage_.front(),
        m_outputShape_,
        m_gpuConfig_.disableProviderSynchronization
    );

    std::cout << "[ORT] Persistent GPU I/O Binding enabled\n";
}

void ORTDetector_V2::infer_frame(cv::Mat& frame, std::vector<ODResultBox>& result_boxes)
{   
    std::lock_guard<std::mutex> lk(this->m_inferMutx_);

    if(!this->m_init)
    {
        throw std::runtime_error("ORTDetector is not initialized");
    }

    if(frame.empty())
    {
        throw std::invalid_argument("frame is empty");
    }

    result_boxes.clear();

    int64_t start = cv::getTickCount();

    LetterboxInfo letterbox;

    cv::Mat inputBlob = this->preProcess(frame, letterbox);

    if(m_ioRunner_ != nullptr)
    {
        const float* outputData = this->m_ioRunner_->run(
            inputBlob.ptr<float>(), 
            inputBlob.total()
        );

        this->postProcess(
            outputData, 
            letterbox, 
            frame.size(), 
            result_boxes
        );
    }
}


cv::Mat ORTDetector_V2::preProcess(const cv::Mat& frame, LetterboxInfo& letterbox) const
{
    if(frame.empty())
    {
        throw std::invalid_argument("Cannot preprocess an empty image");
    }

    letterbox.originalHeight = frame.rows;
    letterbox.originalWidth = frame.cols;

    /*
     * 保持宽高比缩放。
     *
     * 例如：
     *
     * 原图 1280×720
     * 模型 640×640
     *
     * scale = min(640/1280, 640/720)
     *       = 0.5
     *
     * 缩放后：
     * 640×360
     */

    //这个和我们之前写的那个 是左上角，而且是原图除以模型输入尺寸，所以是取最大，而这个是取最小。 因为必须是等比例缩放，不然的话容易出问题。
    float widthScale = static_cast<float>(m_inputWidth_)/static_cast<float>(letterbox.originalWidth);
    float heightScale = static_cast<float>(m_inputHeight)/static_cast<float>(letterbox.originalHeight);

    float scale = (std::min)(widthScale, heightScale);
    letterbox.scale = scale;

    //重整之后的尺寸必须是 int类型
    int resizeWidth = static_cast<int>(std::round(frame.cols*scale));
    int resizeHeight = static_cast<int>(std::round(frame.rows*scale));

    cv::Mat resizeImage;
    //这个就是 blobFromImage的时候的 输入图形。要让其符合Yolo的输入, 来进行推理. 因为是 居中补边，所以还要进行一次补边才是blobFromImage输入
    cv::resize(frame, resizeImage, cv::Size(resizeWidth, resizeHeight));

    //用原始图的宽高减去缩放之后的宽高 得到的就是 补边多少
    int totalPaddingWidth =  m_inputWidth_ - resizeWidth;
    int totalPaddingHeight =  m_inputHeight - resizeHeight;

    //左补
    int padLeft = totalPaddingWidth/2;
    //右补
    int padRight = totalPaddingWidth - padLeft;
    //上补
    int padTop = totalPaddingHeight/2;
    //下补
    int padBottom = totalPaddingHeight - padTop;

    letterbox.padLeft = padLeft;
    letterbox.padTop = padTop;

    //创建居中补边正方形，上下左右居中填充。
    cv::Mat letterboxImage;
    cv::copyMakeBorder(
        resizeImage, 
        letterboxImage, 
        padTop, 
        padBottom, 
        padLeft, 
        padRight, 
        cv::BORDER_CONSTANT, 
        cv::Scalar(114, 114, 114)
    );

    /*
     * blobFromImage完成：
     *
     * BGR -> RGB
     * uint8 -> float32
     * /255
     * HWC -> NCHW
     *
     * 输出shape：
     * [1, 3, inputHeight, inputWidth]
     */

    cv::Mat blob = cv::dnn::blobFromImage(
        letterboxImage, 
        1.0/255, 
        cv::Size(m_inputWidth_, m_inputHeight), 
        cv::Scalar(), 
        true, 
        false
    );

    if(!blob.isContinuous())
    {
        blob = blob.clone();
    }

    return blob;
}   


void ORTDetector_V2::postProcess(
        const float* outputData,           //由Ort 加装的blob数据，类型是Ort::Value
        const LetterboxInfo& letterbox,     //因为要解析所以要还原到原始的类型大小，而letterbox里面有padLeft 和 padTop以及 缩放系数
        const cv::Size& originalImageSize,  //这个是 原生帧画面的大小
        std::vector<ODResultBox>& result_box    //ODResultBox里面就是装好经过分数筛选和NMS去除重复值之后再加上还原之后的 框的大小以及相应的分数和类别名字
    ) const
{


}