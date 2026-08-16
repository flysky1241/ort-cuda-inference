#include "algo/ort_gpu_runtime.h"
#include "algo/ort_check_algo_cuda.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

OrtGpuEpConfigurator::OrtGpuEpConfigurator(
        Ort::SessionOptions& sessionOptions,
        cudaStream_t userStream
    ):sessionOptions_(sessionOptions), 
    userStream_(userStream)
{
    if(userStream_ == nullptr)
    {
        throw std::invalid_argument("user CUDA stream cnanot be nullptr");
    }
}

void OrtGpuEpConfigurator::requireProvider(const std::string& providerName)
{
    std::vector<std::string> providers = Ort::GetAvailableProviders();

    auto iter = std::find_if(providers.begin(), providers.end(),[&providerName](std::string iter)
    {
        if(iter==providerName)
            return true;
        return false;
    });

    if(iter == providers.end())
    {
        std::ostringstream message;
        message<<"Required ONNX Runtime provider is unavailable"
               <<providerName<<". Available providers:";
        for(const std::string& provider: providers)
        {
            message<<' '<<provider;
        }
        throw std::runtime_error(message.str());
    }
}


void OrtGpuEpConfigurator::apendTensorRt(const OrtTrtEpConfig& config)
{
    this->requireProvider("TensorrtExecutionProvider");

    if(config.builderOptimizationLevel<0 || config.builderOptimizationLevel>5)
    {
        throw std::invalid_argument("builderOptimizationLevel must [0, 5]");
    }

    const bool anyProfile = !config.profileMaxShapes.empty()||
    !config.profileMinShapes.empty()||
    !config.profileOptShapes.empty();

    const bool allProfiles = !config.profileMinShapes.empty()&&
    !config.profileMaxShapes.empty()&&
    !config.profileOptShapes.empty();

    if(anyProfile && !allProfiles)
    {
        throw std::runtime_error("TensoRt min/max/opt profiles must be supplied together");
    }

    if(config.enableEngineCache)
    {
        std::filesystem::create_directories(config.engineCachePath);
    }

    if(config.enableTimingCache)
    {
        std::filesystem::create_directories(config.timingCachePath);
    }

    const OrtApi& api = Ort::GetApi();

    OrtTensorRTProviderOptionsV2* rawOptions = nullptr;

    Ort::ThrowOnError(api.CreateTensorRTProviderOptions(std::addressof(rawOptions)));

    if(rawOptions == nullptr)
    {
        throw std::runtime_error("CreateTensorRTProviderOptions return nullptr");
    }

    const auto deleter = [&api](OrtTensorRTProviderOptionsV2* options)
    {
        if(options!=nullptr)
        {
            api.ReleaseTensorRTProviderOptions(options);
        }
    };

    std::unique_ptr<OrtTensorRTProviderOptionsV2,
        decltype(deleter)> options(rawOptions, deleter);

    std::vector<std::string> keys;
    std::vector<std::string> values;

    const auto add = [&keys, &values](std::string key, std::string value)
    {
        keys.push_back(key);
        values.push_back(value);
    };

    add("device_id", std::to_string(config.deviceId));
    add("trt_max_workspace_size", std::to_string(config.maxWorkspaceBytes));
    add("trt_max_partition_iterations", std::to_string(config.maxPartitionIterations));
    add("trt_min_subgraph_size", std::to_string(config.minSubgraphSize));
    add("trt_fp16_enable", boolText(config.enableFp16));
    add("trt_int8_enable", boolText(config.enableInt8));
    add("trt_builder_optimization_level", std::to_string(config.builderOptimizationLevel));
    add("trt_context_memory_sharing_enable", boolText(config.enableContextMemorySharing));
    add("trt_cuda_graph_enable", boolText(config.enableCudaGraph));
    add("trt_detailed_build_log", boolText(config.enableDetailedBuildLog));
    add("trt_dump_subgraphs", boolText(config.dumpSubgraphs));

    if(config.enableEngineCache)
    {
        add("trt_engine_cache_enable", "1");
        add("trt_engine_cache_path", config.engineCachePath.generic_string());
        if(!config.enginCachePrefix.empty())
        {
            add("trt_engine_cache_prefix", config.enginCachePrefix);
        }
    }

    if(config.enableTimingCache)
    {
        add("trt_timing_cache_enable", "1");
        add("trt_timing_cache_path", config.timingCachePath.generic_string());
    }

    if(allProfiles)
    {
        add("trt_profile_min_shapes", config.profileMinShapes);
        add("trt_profile_max_shapes", config.profileMaxShapes);
        add("trt_profile_opt_shapes", config.profileOptShapes);
    }

    std::vector<const char*> keyView = std::move(Vstring2Char(keys));
    std::vector<const char*> valueView = std::move(Vstring2Char(values));

    if(keyView.size() != valueView.size())
    {
        throw std::invalid_argument("key size is not match value size");
    }

    Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
        options.get(),
        keyView.data(),
        valueView.data(),
        keyView.size()
    ));

    Ort::ThrowOnError(api.UpdateTensorRTProviderOptionsWithValue(
        options.get(),
        "user_compute_stream",
        reinterpret_cast<void*>(this->userStream_)
    ));

    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(
        static_cast<OrtSessionOptions*>(sessionOptions_),
        options.get()
    ));
}


void OrtGpuEpConfigurator::apendCuda(const OrtCudaEpConfig& config)
{
    this->requireProvider("CUDAExecutionProvider");

    const OrtApi& api = Ort::GetApi();

    OrtCUDAProviderOptionsV2* rawOptions = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(std::addressof(rawOptions)));

    if(rawOptions == nullptr)
    {
        throw std::runtime_error("CreateCUDAProviderOptions return nullptr");
    }

    const auto deleter = [&api](OrtCUDAProviderOptionsV2* option)
    {
        api.ReleaseCUDAProviderOptions(option);
    };

    std::unique_ptr<
        OrtCUDAProviderOptionsV2, 
        decltype(deleter)
    > cudaOption(rawOptions, deleter);



    std::vector<std::string> keysStorage
    {
        "device_id",
        "arena_extend_strategy",
        "cudnn_conv_algo_search",
        "do_copy_in_default_stream",
        "cudnn_conv_use_max_workspace",
        "use_tf32",
        "enable_cuda_graph"
    };

    std::vector<std::string> valuesStorage
    {
        std::to_string(config.deviceId),
        "kNextPowerOfTwo",
        "EXHAUSTIVE",
        "1",
        boolText(config.useMaximumCudnnWorkspace),
        boolText(config.useTf32),
        boolText(config.enableCudaGraph)
    };

    std::vector<const char*> keys = ::Vstring2Char(keysStorage);
    std::vector<const char*> values = ::Vstring2Char(valuesStorage);

    if(keys.size()!=values.size())
    {
        throw std::runtime_error("key size is not match value size");
    }

    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
        cudaOption.get(),
        keys.data(),
        values.data(),
        keys.size()
    ));

    Ort::ThrowOnError(api.UpdateCUDAProviderOptionsWithValue(
        cudaOption.get(),
        "user_compute_stream",
        reinterpret_cast<void*>(userStream_)
    ));

    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
        static_cast<OrtSessionOptions*>(sessionOptions_),
        cudaOption.get()
    ));
}


CudaEventOwner::CudaEventOwner(unsigned flags)
{
    CUDA_CHECK(::cudaEventCreateWithFlags(
        std::addressof(m_event), 
        flags)
    );
}

CudaEventOwner::~CudaEventOwner() noexcept
{
    CUDA_CHECK(::cudaEventDestroy(m_event));
}

CudaEventOwner::CudaEventOwner(CudaEventOwner&& other) noexcept
    :m_event(std::exchange(other.m_event, nullptr))
{}


CudaEventOwner& CudaEventOwner::operator=(CudaEventOwner&& other) noexcept
{
    if(this!=std::addressof(other))
    {
        this->reset();
        m_event = std::exchange(other.m_event, nullptr);
    }
    return *this;
}


void CudaEventOwner::reset() noexcept
{
    if(m_event!=nullptr)
    {
        (void)cudaEventDestroy(m_event);
        m_event = nullptr;
    }
}


void CudaEventOwner::record(cudaStream_t stream) const
{
    if(m_event == nullptr || stream == nullptr)
    {
        throw std::runtime_error("CUDA event or stream is not init");
    }
    CUDA_CHECK(::cudaEventRecord(this->m_event, stream));
}


void CudaEventOwner::synchronize() const
{
    if(m_event==nullptr)
    {
        throw std::runtime_error("event should be init");
    }

    CUDA_CHECK(::cudaEventSynchronize(m_event));
}


float CudaEventOwner::elapsedMillisecondsSince(const CudaEventOwner& start) const
{
    if(start.m_event == nullptr)
    {
        throw std::runtime_error("CUDA event start event should init");
    }

    float milliseconds = 0.0f;
    CUDA_CHECK(::cudaEventElapsedTime(
        std::addressof(milliseconds),
        start.m_event, 
        this->m_event)
    );
    return milliseconds;
}

CudaStreamOwner::CudaStreamOwner(unsigned flags)
{
    this->create(flags);
}

CudaStreamOwner::CudaStreamOwner(CudaStreamOwner&& other) noexcept
    :m_stream_(std::exchange(other.m_stream_, nullptr))
{}

CudaStreamOwner::~CudaStreamOwner() noexcept
{
    if(m_stream_!=nullptr)
    {
        CUDA_CHECK(::cudaStreamDestroy(m_stream_));
        m_stream_=nullptr;
    }
}

void CudaStreamOwner::create(unsigned flags)
{
    if(this->m_stream_ != nullptr)
    {
        return;
    }

    CUDA_CHECK(::cudaStreamCreateWithFlags(
        std::addressof(m_stream_),
        flags
    ));
}


CudaStreamOwner& CudaStreamOwner::operator=(CudaStreamOwner&& other) noexcept
{
    if(this==std::addressof(other))
    {
        return *this;
    }

    this->reset();
    this->m_stream_ = std::exchange(other.m_stream_, nullptr);

    return *this;
}


void CudaStreamOwner::synchronize() const
{
    if(this->m_stream_ == nullptr)
    {
        throw std::runtime_error("CUDA stream is not init");
    }

    CUDA_CHECK(::cudaStreamSynchronize(m_stream_));
}


void CudaStreamOwner::reset() noexcept
{
    if(m_stream_!=nullptr)
    {
        CUDA_CHECK(cudaStreamDestroy(m_stream_));
        m_stream_ = nullptr;
    }
}

    
::cudaStream_t CudaStreamOwner::get() const noexcept
{
    return m_stream_;
}

CudaStreamOwner::operator bool() const noexcept
{
    if(m_stream_!=nullptr)
        return true;   
    return false;
}


void* CudaReusableBuffer::data() noexcept
{
    return m_data;
}

const void* CudaReusableBuffer::data() const noexcept
{
    return m_data;
}

std::size_t CudaReusableBuffer::sizeBytes() const noexcept
{
    return m_bytes;
}

CudaReusableBuffer::Kind CudaReusableBuffer::kind() const noexcept
{
    return m_kind;
}

CudaReusableBuffer::CudaReusableBuffer(Kind kind) noexcept
    :m_kind(kind)
{}


CudaReusableBuffer::~CudaReusableBuffer()
{
    this->reset();
}


CudaReusableBuffer::CudaReusableBuffer(CudaReusableBuffer&& other) noexcept
    :m_kind(std::move(other.m_kind)),
    m_data(std::exchange(other.m_data, nullptr)),
    m_bytes(std::exchange(other.m_bytes, 0))
{}


CudaReusableBuffer& CudaReusableBuffer::operator=(CudaReusableBuffer&& other) noexcept
{
    if(this!=std::addressof(other))
    {
        this->reset();

        m_kind = std::move(other.m_kind);
        m_data = std::exchange(other.m_data, nullptr);
        m_bytes = std::exchange(other.m_bytes, 0);
    }
    return *this;
}


void CudaReusableBuffer::resize(std::size_t bytes)
{
    if(m_data != nullptr && bytes == m_bytes)
    {
        return;
    }

    this->reset();

    if(bytes == 0)
        return;

    if(m_kind == Kind::Device)
    {
        CUDA_CHECK(cudaMalloc(std::addressof(m_data), bytes));
    }
    else
    {
        CUDA_CHECK(cudaHostAlloc(std::addressof(m_data), bytes, cudaHostAllocPortable));
    }

    m_bytes = bytes;
}


void CudaReusableBuffer::reset() noexcept
{
    if(m_data == nullptr)
        return;

    if(m_kind == Kind::Device)
    {
        CUDA_CHECK(cudaFree(m_data));
    }
    else
    {
        CUDA_CHECK(cudaFreeHost(m_data));
    }

    m_data = nullptr;
    m_bytes = 0;
}


std::size_t OrtFixedShapeIoBindingRunner::elementCount(const std::vector<int64_t>&shape)
{
    if(shape.empty())
    {
        throw std::invalid_argument("Tensor shape is empty");
    }

    std::size_t count = 1;
    for(const int64_t& dim : shape)
    {
        if(dim<0)
        {
            throw std::invalid_argument("Persistent I/O requires fixed positive shapes");
        }

        std::size_t value = static_cast<std::size_t>(dim);
        if(count > std::numeric_limits<std::size_t>::max()/value)
        {
            throw std::overflow_error("Tensor element count overflow");
        }

        count*=value;
    }

    return count;
}




OrtFixedShapeIoBindingRunner::OrtFixedShapeIoBindingRunner(
        Ort::Session& session,
        cudaStream_t stream, 
        int deviceid,
        std::string inputName,
        std::vector<int64_t> inputShape,
        std::string outputName,
        std::vector<int64_t> outputShape,
        bool disableProviderSynchronization)
    :m_session_(session),
    m_stream_(stream),
    m_disableProviderSynchronization(disableProviderSynchronization),
    inputName_(inputName),
    inputShape_(inputShape),
    outputName_(outputName),
    outputShape_(outputShape),
    inputElements_(elementCount(inputShape_)),
    outputElements_(elementCount(outputShape_)),
    cudaMemoryInfo_(
        "Cuda",
        OrtDeviceAllocator,
        deviceid,
        OrtMemTypeDefault
    ),
    m_bind_(m_session_),
    pinnedInput_(CudaReusableBuffer::Kind::PinnedHost),
    pinnedOutput_(CudaReusableBuffer::Kind::PinnedHost),
    deviceInput_(CudaReusableBuffer::Kind::Device),
    deviceOutput_(CudaReusableBuffer::Kind::Device),
    inputValue_(),
    outputValue_()
{
    if(m_stream_ == nullptr)
    {
        throw std::runtime_error("CUDA stream is nullptr");
    }

    this->pinnedInput_.resize(inputElements_ * sizeof(float));
    this->pinnedOutput_.resize(outputElements_* sizeof(float));
    this->deviceInput_.resize(inputElements_ * sizeof(float));
    this->deviceOutput_.resize(outputElements_ * sizeof(float));

    inputValue_ = std::move(Ort::Value::CreateTensor<float>(
        cudaMemoryInfo_,
        static_cast<float*>(deviceInput_.data()),
        inputElements_,
        inputShape_.data(),
        inputShape_.size()
    ));

    outputValue_ = std::move(Ort::Value::CreateTensor<float>(
        cudaMemoryInfo_,
        static_cast<float*>(deviceOutput_.data()),
        outputElements_,
        outputShape.data(),
        outputShape.size()
    ));

    m_bind_.BindInput(inputName_.c_str(), inputValue_);
    m_bind_.BindOutput(outputName_.c_str(), outputValue_);
}



Ort::RunOptions OrtFixedShapeIoBindingRunner::makeRunOptions() const
{
    Ort::RunOptions options;
    if(this->m_disableProviderSynchronization)
    {
        options.AddConfigEntry("disable_synchronize_execution_providers",
            "1");
    }
    return options;
}



const float* OrtFixedShapeIoBindingRunner::run(
    const float* input, 
    std::size_t inputElements)
{
    if(input == nullptr)
    {
        throw std::invalid_argument("input is nullptr");
    }

    if(inputElements_ != inputElements)
    {
        throw std::invalid_argument(
            "Input element count changed after persisent I/O Binding");
    }

    const std::size_t inputBytes = inputElements_ * sizeof(float);
    const std::size_t outputBytes = outputElements_*sizeof(float);

    std::memcpy(pinnedInput_.data(), input, inputBytes);

    CUDA_CHECK(cudaMemcpyAsync(
        deviceInput_.data(), 
        pinnedInput_.data(), 
        inputBytes, 
        cudaMemcpyHostToDevice,
        m_stream_
    ));

    Ort::RunOptions option = this->makeRunOptions();

    m_session_.Run(option, m_bind_);

    CUDA_CHECK(cudaMemcpyAsync(
        pinnedOutput_.data(), 
        deviceOutput_.data(), 
        outputBytes, 
        cudaMemcpyDeviceToHost,
        m_stream_
    ));

    CUDA_CHECK(cudaStreamSynchronize(m_stream_));

    return static_cast<const float*>(pinnedOutput_.data());
}



const std::vector<int64_t>& OrtFixedShapeIoBindingRunner::outputShape() const noexcept
{
    return outputShape_;
}


std::size_t OrtFixedShapeIoBindingRunner::outputElements() const noexcept
{
    return outputElements_;
}



void OrtFixedShapeIoBindingRunner::warmup(int runs)
{
    if(runs<0)
    {
        throw std::invalid_argument(" warmup runs cannot be letter zero");
    }

    std::memset(
        pinnedInput_.data(), 
        0x00, 
        this->inputElements_*sizeof(float)
    );

    for(auto index = 0; index<runs; ++index)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            deviceInput_.data(),
            pinnedInput_.data(),
            this->inputElements_*sizeof(float),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
            m_stream_
        ));

        auto options = this->makeRunOptions();

        m_session_.Run(options, m_bind_);

        CUDA_CHECK(cudaStreamSynchronize(m_stream_));
    }
}


