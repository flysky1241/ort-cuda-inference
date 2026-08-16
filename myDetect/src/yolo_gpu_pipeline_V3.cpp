#include "algo/yolo_gpu_pipeline_V3.h"
#include "algo/core/yolo_gpu_pipeline_V3_kernels.cuh"
#include "algo/core/yolo_gpu_pipeline_V3_Cubfun.h"
#include "algo/core/yolo_gpu_pipline_V3_config.h"
#include "algo/ort_check_algo_cuda.h"
#include "algo/ort_gpu_runtime.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "onnxruntime_cxx_api.h"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
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

    yolo_cuda_cub::sortDetectionsDescending(
        nullptr, 
        sortTemporaryBytes_, 
        m_scoresUnsorted_.dataAs<float>(), 
        m_scoresSorted_.dataAs<float>(), 
        m_detectionsUnSorted_.dataAs<GpuDetectionV3>(), 
        m_detectionsSorted_.dataAs<GpuDetectionV3>(), 
        candidates_, 
        m_stream_
    );

    m_sortTemporaryStorage_.resize(sortTemporaryBytes_);
}



void YoloGpuPipeline_V3::stageHostFrame(const cv::Mat& frame)
{
    if(frame.empty())
    {
        throw std::invalid_argument(
            "Input frame is empty"
        );
    }

    if(frame.type()!=CV_8UC3)
    {
        throw std::invalid_argument(
            "ORTDetector_V3 requires CV_8UC3 BGR"
        );
    }

    const int sourceWidth = frame.cols;
    const int sourceHeight = frame.rows;

    const std::size_t stride = 
        static_cast<std::size_t>(sourceWidth) * 3U;
    const std::size_t totalBytes = 
        static_cast<std::size_t>(sourceHeight) * stride;
    

    const auto begin = std::chrono::steady_clock::now();

    if(sourceWidth != currentSourceWidth_ ||
        sourceHeight != currentSourceHeight_)
    {
        m_hostRawBgr_.resize(totalBytes);
        m_deviceRawBgr_.resize(totalBytes);

        currentSourceHeight_ = sourceHeight;
        currentSourceWidth_ = sourceWidth;
    }

    if(frame.isContinuous() && frame.step == stride)
    {
        std::memcpy(
            m_hostRawBgr_.dataAs<std::uint8_t>(),
            frame.ptr<std::uint8_t>(),
            totalBytes
        );
    }
    else
    {
        for(auto row = 0; row < sourceHeight; ++row)
        {
            std::memcpy(
                m_hostRawBgr_.dataAs<std::uint8_t>() + 
                static_cast<std::uint8_t>(row) * stride, 
                frame.ptr<std::uint8_t>(row), 
                stride
            );
        }
    }

    const auto end = std::chrono::steady_clock::now();
    this->m_lastTimings_.hostStageingMs = 
        std::chrono::duration<double, std::milli>(end-begin).count();

}



void YoloGpuPipeline_V3::launchPreprocess(const LetterboxTransformV3& transform)
{
    const dim3 block(kPreprocessBlockX, kPreprocessBlockY);
    const dim3 gride(
        static_cast<unsigned>((inputWidth_ + kPreprocessBlockX-1)/kPreprocessBlockX),
        static_cast<unsigned>((inputHeight_ + kPreprocessBlockY-1)/kPreprocessBlockY)
    );

    yolo_cuda_kernel::launchPreprocessNchwKernel(
        block,
        gride,
        m_stream_,
        m_deviceRawBgr_.dataAs<std::uint8_t>(), 
        transform.originalWidth, 
        transform.originalHeight, 
        transform.originalWidth * 3U, 
        m_deviceInput_.dataAs<float>(), 
        inputWidth_, 
        inputHeight_, 
        transform.scale, 
        transform.padLeft, 
        transform.padTop
    );

    yolo_cuda_kernel::checkKernelLaunch("launchPreprocessNchwKernel");
}



[[nodiscard]]
std::vector<GpuDetectionV3> YoloGpuPipeline_V3::run(
    const cv::Mat& bgrFrame,
    const LetterboxTransformV3& transform)
{
    if(transform.originalWidth != bgrFrame.cols || 
        transform.originalHeight != bgrFrame.rows ||
        transform.scale <= 0.0f)
    {
        throw std::runtime_error(
            "LetterboxTransformV3 does not match the input frame"
        );
    }

    this->stageHostFrame(bgrFrame);

    const std::size_t rawBytes = 
        static_cast<std::size_t>(bgrFrame.cols) *
        static_cast<std::size_t>(bgrFrame.rows) * 3U;
    
    m_stageStart_.record(m_stream_);

    CUDA_CHECK(::cudaMemcpyAsync(
        m_deviceRawBgr_.dataAs<std::uint8_t>(), 
        m_hostRawBgr_.dataAs<std::uint8_t>(), 
        rawBytes, 
        cudaMemcpyKind::cudaMemcpyHostToDevice,
        m_stream_
    ));

    m_uploadEnd_.record(m_stream_);

    this->launchPreprocess(transform);

    m_preprocessEnd_.record(m_stream_);

    m_session_.Run(m_runOptions_, m_bind_);

    m_inferenceEnd_.record(m_stream_);

    this->launchPostprocess(transform);

    m_postprocessEnd_.record(m_stream_);

    copyCompactResults2Host();
    m_downloadEnd_.record(m_stream_);
    m_downloadEnd_.synchronize();

    m_lastTimings_.uploadMs = m_uploadEnd_.elapsedMillisecondsSince(m_stageStart_);
    m_lastTimings_.preprocessMs = m_preprocessEnd_.elapsedMillisecondsSince(m_uploadEnd_);
    m_lastTimings_.inferenceMs = m_inferenceEnd_.elapsedMillisecondsSince(m_preprocessEnd_);
    m_lastTimings_.gpuPostprocessMs = m_postprocessEnd_.elapsedMillisecondsSince(m_inferenceEnd_);
    m_lastTimings_.downloadMs = m_downloadEnd_.elapsedMillisecondsSince(m_postprocessEnd_);

    const int count = std::clamp(
        m_hostFinalCount_.dataAs<int>()[0], 
        0, 
        m_config_.maxDetections
    );

    return std::vector<GpuDetectionV3>(
        m_hostFinalDetections_.dataAs<GpuDetectionV3>(),
        m_hostFinalDetections_.dataAs<GpuDetectionV3>()+count
    );
}



void YoloGpuPipeline_V3::launchPostprocess(const LetterboxTransformV3& transform)
{
    const int decodeBlocks = 
        (candidates_ + kDecodeThreads -1)/kDecodeThreads;
    
    yolo_cuda_kernel::launchPostDecodeYoloKernel(
        decodeBlocks, 
        m_stream_, 
        m_deviceOutput_.dataAs<float>(), 
        attributes_, 
        candidates_,
        classCount_, 
        attributesFirst_,
        transform, 
        m_config_.scoreThreshold, 
        m_scoresUnsorted_.dataAs<float>(), 
        m_detectionsUnSorted_.dataAs<GpuDetectionV3>()
    );

    yolo_cuda_kernel::checkKernelLaunch("launchPostDecodeYoloKernel");

    yolo_cuda_cub::sortDetectionsDescending(
        m_sortTemporaryStorage_.data(), 
        sortTemporaryBytes_, 
        m_scoresUnsorted_.dataAs<float>(), 
        m_scoresSorted_.dataAs<float>(), 
        m_detectionsUnSorted_.dataAs<GpuDetectionV3>(), 
        m_detectionsSorted_.dataAs<GpuDetectionV3>(), 
        candidates_, 
        m_stream_
    );

    const dim3 nmsBlock(kNmsThreads);
    const dim3 grides(
        static_cast<unsigned>(nmsColumBlocks_), 
        static_cast<unsigned>((nmsTopK_+kNmsThreads-1)/kNmsThreads)
    );

    yolo_cuda_kernel::launchBuildClassAwareNmsMaskKernel(
        nmsBlock, 
        grides, 
        m_stream_, 
        m_detectionsSorted_.dataAs<GpuDetectionV3>(), 
        nmsTopK_, 
        nmsColumBlocks_, 
        m_config_.scoreThreshold, 
        m_config_.nmsThrehold, 
        m_nmsMasks_.dataAs<std::uint64_t>()
    );

    yolo_cuda_kernel::checkKernelLaunch("launchBuildClassAwareNmsMaskKernel");

    yolo_cuda_kernel::launchSelectNmsResultKernel(
        m_stream_,
        m_detectionsSorted_.dataAs<GpuDetectionV3>(),
        m_nmsMasks_.dataAs<std::uint64_t>(), 
        nmsTopK_, 
        nmsColumBlocks_, 
        m_config_.scoreThreshold, 
        m_config_.maxDetections, 
        m_nmsRemoved_.dataAs<std::uint64_t>(), 
        m_finalDetections_.dataAs<GpuDetectionV3>(), 
        m_finalCount_.dataAs<int>()
    );

    yolo_cuda_kernel::checkKernelLaunch("launchSelectNmsResultKernel");
}


void YoloGpuPipeline_V3::copyCompactResults2Host()
{
    CUDA_CHECK(::cudaMemcpyAsync(
        m_hostFinalCount_.dataAs<int>(), 
        m_finalCount_.dataAs<int>(), 
        sizeof(int), 
        cudaMemcpyKind::cudaMemcpyDeviceToHost,
        m_stream_
    ));

    CUDA_CHECK(::cudaMemcpyAsync(
        m_hostFinalDetections_.dataAs<GpuDetectionV3>(), 
        m_finalDetections_.dataAs<GpuDetectionV3>(), 
        m_hostFinalDetections_.sizeBytes(), 
        cudaMemcpyKind::cudaMemcpyDeviceToHost,
        m_stream_
    ));
}


void YoloGpuPipeline_V3::warmup()
{
    if(m_config_.warmupRuns<0)
    {
        return ;
    }

    cv::Mat dummy(
        inputHeight_,
        inputWidth_,
        CV_8UC3,
        cv::Scalar(0, 0, 0)
    );

    LetterboxTransformV3 tranform;
    tranform.scale = 1.0f;
    tranform.originalHeight = inputHeight_;
    tranform.originalWidth = inputWidth_;

    for(auto index=0; index<m_config_.warmupRuns; ++index)
    {
        (void)this->run(dummy, tranform);
    }
}

const OrtV3StageTimings& YoloGpuPipeline_V3::lastTimings() const noexcept
{
    return m_lastTimings_;
}