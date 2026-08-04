#ifndef ORT_GPU_RUNTIME_H
#define ORT_GPU_RUNTIME_H

#include "driver_types.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <onnxruntime_cxx_api.h>
#include <filesystem>

//Trt EP挂载配置
struct OrtTrtEpConfig
{
    int deviceId = 0;

    bool enableFp16 = true;
    bool enableInt8 = false;
    bool enableEngineCache = true;
    bool enableTimingCache = true;
    bool enableContextMemorySharing = true;
    bool enableCudaGraph = true;
    bool enableDetailedBuildLog = false;
    bool dumpSubgraphs = false;

    int builderOptimizationLevel = 5;
    int minSubgraphSize = 5;
    int maxPartitionIterations = 1000;

    std::size_t maxWorkspaceBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;

    std::filesystem::path engineCachePath;
    std::filesystem::path timingCachePath;
    std::string enginCachePrefix;

    std::string profileMinShapes;
    std::string profileOptShapes;
    std::string profileMaxShapes;
};


//cuda EP 的配置，到时候挂载的时候用
struct OrtCudaEpConfig
{
    int deviceId = 0;
    std::size_t gpuMemoryLimitBytes = 0;
    bool useTf32 = true;
    bool useMaximumCudnnWorkspace = true;
    bool enableCudaGraph = true;
};


//
struct OrtGpuRuntimeConfig
{
    OrtTrtEpConfig trtEpOption;
    OrtCudaEpConfig cudaEpOption;

    bool enableIoBinding = true;
    bool disableProviderSynchronization = true;
    int warmupRuns = 2;

    //V3 GPU post-Processing 
    int nmsTopK = 1024;
    int maxDetections = 300;

    std::filesystem::path cacheRoot = "cache/ort_tensorrt";
};


class OrtGpuEpConfigurator final
{
public:
    OrtGpuEpConfigurator(
        Ort::SessionOptions& sessionOptions,
        cudaStream_t userStream
    );

    void apendTensorRt(const OrtTrtEpConfig& config);
    void apendCuda(const OrtCudaEpConfig& config);

    static void requireProvider(const std::string& providerName);

private:
    Ort::SessionOptions& sessionOptions_;
    cudaStream_t userStream_ = nullptr;
};


class CudaStreamOwner final
{
public:
    CudaStreamOwner() = default;
    explicit CudaStreamOwner(unsigned flags);
    ~CudaStreamOwner() noexcept;

    CudaStreamOwner(const CudaStreamOwner& other) = delete;
    CudaStreamOwner& operator=(const CudaStreamOwner& other) = delete;

    CudaStreamOwner(CudaStreamOwner&& other) noexcept;
    CudaStreamOwner& operator=(CudaStreamOwner&& other) noexcept;

    void create(unsigned flags = cudaStreamNonBlocking);
    void synchronize() const;
    void reset() noexcept;

    [[nodiscard]]
    ::cudaStream_t get() const noexcept;

    [[nodiscard]]
    explicit operator bool() const noexcept;

private:
    ::cudaStream_t m_stream_{nullptr};
};


//V3 GPU运转时间测速
class CudaEventOwner final
{
public:
    explicit CudaEventOwner(unsigned flags = cudaEventDefault);
    ~CudaEventOwner() noexcept; 
    
    CudaEventOwner(const CudaEventOwner&) = delete;
    CudaEventOwner& operator=(const CudaEventOwner&) = delete;

    CudaEventOwner(CudaEventOwner&& other) noexcept;
    CudaEventOwner& operator=(CudaEventOwner&& other) noexcept;

    void record(cudaStream_t stream) const;
    void synchronize() const;

    [[nodiscard]]
    float elapsedMillisecondsSince(const CudaEventOwner& start) const;

    void reset() noexcept;

private:
    ::cudaEvent_t m_event{nullptr};
};



class CudaReusableBuffer final
{
public:
    enum class Kind : unsigned
    {
        Device,
        PinnedHost
    };

    explicit CudaReusableBuffer(Kind kind) noexcept;
    ~CudaReusableBuffer();

    CudaReusableBuffer(const CudaReusableBuffer& other) = delete;
    CudaReusableBuffer& operator=(const CudaReusableBuffer& other) = delete;

    //V3 path
    CudaReusableBuffer(CudaReusableBuffer&& other) noexcept;
    CudaReusableBuffer& operator=(CudaReusableBuffer&& other) noexcept;

    void resize(std::size_t bytes);
    void reset() noexcept;

    [[nodiscard]]
    void* data() noexcept;

    [[nodiscard]]
    const void* data() const noexcept;

    [[nodiscard]]
    std::size_t sizeBytes() const noexcept;

    [[nodiscard]]
    Kind kind() const noexcept;

    template<typename Ty>
    [[nodiscard]] 
    inline Ty* dataAs() noexcept
    {
        return static_cast<Ty*>(data());
    }

    template<typename Ty>
    [[nodiscard]] 
    inline const Ty* dataAs() const noexcept
    {
        return static_cast<const Ty*>(data());
    }

private:
    Kind m_kind;
    void* m_data = nullptr;
    std::size_t m_bytes = 0;
};



class OrtFixedShapeIoBindingRunner final
{
public:
    explicit OrtFixedShapeIoBindingRunner(
        Ort::Session& session,
        cudaStream_t stream, 
        int deviceid,
        std::string inputName,
        std::vector<int64_t> inputShape,
        std::string outputName,
        std::vector<int64_t> outputShape,
        bool disableProviderSynchronization 
    );

    OrtFixedShapeIoBindingRunner(const OrtFixedShapeIoBindingRunner& other) = delete;
    OrtFixedShapeIoBindingRunner& operator=(const OrtFixedShapeIoBindingRunner& other) = delete;

    ~OrtFixedShapeIoBindingRunner() = default;

    const float* run(const float* input, std::size_t inputElements);
    void warmup(int runs);

    [[nodiscard]]
    const std::vector<int64_t>& outputShape() const noexcept;

    [[nodiscard]]
    std::size_t outputElements() const noexcept;

private:
    std::size_t elementCount(const std::vector<int64_t>&shape);
    Ort::RunOptions makeRunOptions() const;

private:
    Ort::Session& m_session_;
    cudaStream_t m_stream_ = nullptr;
    bool m_disableProviderSynchronization = true;

    std::string inputName_;
    std::string outputName_;
    std::vector<int64_t> inputShape_;
    std::vector<int64_t> outputShape_;

    std::size_t inputElements_ = 0;
    std::size_t outputElements_ = 0;

    Ort::MemoryInfo cudaMemoryInfo_;
    Ort::IoBinding m_bind_;

    CudaReusableBuffer pinnedInput_;
    CudaReusableBuffer pinnedOutput_;
    CudaReusableBuffer deviceInput_;
    CudaReusableBuffer deviceOutput_;

    Ort::Value inputValue_;
    Ort::Value outputValue_;
};

#endif


