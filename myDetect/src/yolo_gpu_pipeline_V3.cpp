#include "algo/yolo_gpu_pipeline_V3.h"
#include "algo/core/yolo_gpu_pipeline_V3_kernels.cuh"
#include "algo/ort_gpu_runtime.h"
#include "onnxruntime_cxx_api.h"
#include <cstdint>
#include <limits>
#include <stdexcept>

std::size_t YoloGpuPipeline_V3::elementCount(
    const std::vector<std::int64_t>& shape)
{
    if(shape.empty())
    {
        throw std::runtime_error("Tensor shape must init");
    }

    std::size_t count = 1;

    for(const int64_t& dimension: shape)
    {
        if(dimension<0)
        {
            throw std::invalid_argument(
                "ORTDetector_V3 requires fixed shapes"
            );
        }

        const std::size_t value = static_cast<std::size_t>(dimension);
        if(count > std::numeric_limits<std::size_t>::max()/value)
        {
            throw std::overflow_error("Tensor element count overflow");
        } 
        count*=value;   
    }

    return count;
}



YoloGpuPipeline_V3::YoloGpuPipeline_V3(
        Ort::Session& session,
        ::cudaStream_t stream,
        int deviceId,
        std::string inputName,
        std::vector<std::int64_t> inputShape,
        std::string outputName,
        std::vector<std::int64_t> outputShape,
        int classCount,
        YoloGpuPipelineV3Config config)
    :m_session_(session),
    m_stream_(stream),
    m_deviceId_(deviceId),
    inputName_(inputName),
    outputName_(outputName),
    inputShape_(inputShape),
    outputShape_(outputShape),
    classCount_(classCount),
    m_config_(config),
    m_hostRawBgr_(CudaReusableBuffer::Kind::PinnedHost),
    m_deviceRawBgr_(CudaReusableBuffer::Kind::Device),
    m_deviceInput_(CudaReusableBuffer::Kind::Device),
    m_deviceOutput_(CudaReusableBuffer::Kind::Device),
    m_scoresUnsorted_(CudaReusableBuffer::Kind::Device),
    m_scoresSorted_(CudaReusableBuffer::Kind::Device),
    m_detectionsUnSorted_(CudaReusableBuffer::Kind::Device),
    m_detectionsSorted_(CudaReusableBuffer::Kind::Device),
    m_nmsMasks_(CudaReusableBuffer::Kind::Device),
    m_nmsRemoved_(CudaReusableBuffer::Kind::Device),
    m_finalDetections_(CudaReusableBuffer::Kind::Device),
    m_finalCount_(CudaReusableBuffer::Kind::Device),
    m_sortTemporaryStorage_(CudaReusableBuffer::Kind::Device),
    m_hostFinalDetections_(CudaReusableBuffer::Kind::PinnedHost),
    m_hostFinalCount_(CudaReusableBuffer::Kind::PinnedHost),
    cudaMemoryInfo_(
        "Cuda", 
        OrtDeviceAllocator, 
        deviceId, 
        OrtMemTypeDefault  
    ),
    m_bind_(m_session_)
{
    if(m_stream_ == nullptr)
    {
        throw std::runtime_error(
            "ORTDetector_V3 expected input shape [1, 3, H, W]"
        );
    }

    if(outputShape_.size()!=3 || outputShape_[0]!=1)
    {
        throw std::invalid_argument(
            "ORTDetector_V3 outputshape must dim 3"
        );
    }

    if(m_config_.nmsThrehold<=0.0f || m_config_.scoreThreshold<=0.0f ||
        m_config_.scoreThreshold>=1.0f || m_config_.nmsThrehold>=1.0f
    )
    {
        throw std::invalid_argument(
            "ONNX Detect threshold must be in [0, 1]"
        );
    }

    inputHeight_ = static_cast<int>(inputShape_[2]);
    inputWidth_ = static_cast<int>(inputShape_[3]);

    attributesFirst_ = outputShape_[1] < outputShape_[2];
    attributes_ = static_cast<int>(
        attributesFirst_ ? outputShape_[1]:outputShape_[2]
    );

    candidates_ = static_cast<int>(
        attributesFirst_ ? outputShape_[2]:outputShape_[1]
    );

    if(attributes_!=classCount_+4)
    {
        throw std::runtime_error(
            "Yolo output attributes must equal 4+classCount"
        );
    }

    if(candidates_<=0)
    {
        throw std::invalid_argument(
            "Yolo candidate count must be positive"
        );
    }

    m_config_.nmsTopK = (std::max)(1, m_config_.nmsTopK);
    m_config_.maxDetections = (std::max)(1, m_config_.maxDetections);

    nmsTopK_ = (std::min)(m_config_.nmsTopK, candidates_);
    nmsColumBlocks_ = (nmsTopK_ + ::kNmsThreads -1) / ::kNmsThreads;

    const std::size_t inputElements = this->elementCount(inputShape_);
    const std::size_t outputElements = this->elementCount(outputShape_);
    
    m_deviceInput_.resize(inputElements * sizeof(float));
    m_deviceOutput_.resize(outputElements * sizeof(float));
    m_scoresUnsorted_.resize(static_cast<std::size_t>(candidates_) * sizeof(float));
    m_scoresSorted_.resize(static_cast<std::size_t>(candidates_) * sizeof(float));
    m_detectionsUnSorted_.resize(static_cast<std::size_t>(candidates_) * sizeof(::GpuDetectionV3));
    m_detectionsSorted_.resize(static_cast<std::size_t>(candidates_)*sizeof(GpuDetectionV3));
    m_nmsMasks_.resize(
        static_cast<std::size_t>(nmsTopK_) *
        static_cast<std::size_t>(nmsColumBlocks_) *
        sizeof(uint64_t)
    );

    m_nmsRemoved_.resize(static_cast<std::size_t>(nmsColumBlocks_)*sizeof(uint64_t));
    m_finalDetections_.resize(
        static_cast<std::size_t>(m_config_.maxDetections) * 
        sizeof(GpuDetectionV3)
    );

    m_finalCount_.resize(sizeof(int));
    m_hostFinalDetections_.resize(
        static_cast<std::size_t>(m_config_.maxDetections) * 
        sizeof(GpuDetectionV3)
    );
    m_hostFinalCount_.resize(sizeof(int));

    m_inputTensorValue_ = Ort::Value::CreateTensor<float>(
        cudaMemoryInfo_,
        m_deviceInput_.dataAs<float>(),
        inputElements,
        inputShape_.data(),
        inputShape_.size()
    );

    m_outputTensorValue_ = Ort::Value::CreateTensor<float>(
        cudaMemoryInfo_,
        m_deviceOutput_.dataAs<float>(),
        outputElements,
        outputShape_.data(),
        outputShape_.size()
    );

    m_bind_.BindInput(inputName_.c_str(), m_inputTensorValue_);
    m_bind_.BindOutput(outputName_.c_str(), m_outputTensorValue_);

    if(m_config_.disableProviderSynchronization)
    {
        m_runOptions_.AddConfigEntry(
            "disable_synchronize_execution_providers", 
            "1"
        );
    }

    
}



