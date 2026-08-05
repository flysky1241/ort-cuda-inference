#ifndef YOLO_GPU_PIPLINE_V3_H
#define YOLO_GPU_PIPLINE_V3_H

#include "algo/ort_gpu_runtime.h"
#include "driver_types.h"
#include "onnxruntime_cxx_api.h"
#include "ort_gpu_runtime.h"
#include <cstdint>
#include <opencv2/opencv.hpp>
#include "algo/core/yolo_gpu_pipline_V3_config.h"

// Fixed-shape YOLOv8 GPU-resident path:
// pinned raw BGR -> H2D -> CUDA resize/letterbox/normalize/NCHW
// -> ORT device I/O Binding -> TRT/CUDA EP -> GPU decode/sort/NMS
// -> compact detections D2H.

class YoloGpuPipeline_V3 final
{
public:
    YoloGpuPipeline_V3(
        Ort::Session& session,
        ::cudaStream_t stream,
        int deviceId,
        std::string inputName,
        std::vector<std::int64_t> inputShape,
        std::string outputName,
        std::vector<std::int64_t> outputShape,
        int classCount,
        YoloGpuPipelineV3Config config
    );

    YoloGpuPipeline_V3(const YoloGpuPipeline_V3& other) = delete;
    YoloGpuPipeline_V3& operator=(const YoloGpuPipeline_V3& other) = delete; 

    [[nodiscard]]
    std::vector<GpuDetectionV3> run(
        const cv::Mat& bgrFrame,
        const LetterboxTransformV3& transform
    );

    void warmup();

    [[nodiscard]]
    const OrtV3StageTimings& lastTimings() const noexcept;

private:
    static std::size_t elementCount(const std::vector<std::int64_t>& shape);
    
    //调用Kernal 核函数
    void stageHostFrame(const cv::Mat& frame);
    void launchPreprocess(const LetterboxTransformV3& transform);
    void launchPostprocess(const LetterboxTransformV3& transform);
    void copyCompactResults2Host();

private:
    Ort::Session& m_session_;
    cudaStream_t m_stream_ = nullptr;
    int m_deviceId_ = 0;

    std::string inputName_;
    std::string outputName_;
    std::vector<std::int64_t> inputShape_;
    std::vector<std::int64_t> outputShape_;

    int inputHeight_ = 0;
    int inputWidth_ = 0;
    int attributes_ = 0;
    int candidates_ = 0;
    int classCount_ = 0;
    bool attributesFirst_ = true;
    int nmsTopK_ = 0;
    int nmsColumBlocks_ = 0;

    YoloGpuPipelineV3Config m_config_;
    OrtV3StageTimings m_lastTimings_;

    CudaReusableBuffer m_hostRawBgr_;
    CudaReusableBuffer m_deviceRawBgr_;
    CudaReusableBuffer m_deviceInput_;
    CudaReusableBuffer m_deviceOutput_;

    CudaReusableBuffer m_scoresUnsorted_;
    CudaReusableBuffer m_scoresSorted_;
    CudaReusableBuffer m_detectionsUnSorted_;
    CudaReusableBuffer m_detectionsSorted_;
    CudaReusableBuffer m_nmsMasks_;
    CudaReusableBuffer m_nmsRemoved_;
    CudaReusableBuffer m_finalDetections_;
    CudaReusableBuffer m_finalCount_;
    CudaReusableBuffer m_sortTemporaryStorage_;

    CudaReusableBuffer m_hostFinalDetections_;
    CudaReusableBuffer m_hostFinalCount_;

    std::size_t sortTemporaryBytes_ = 0;
    int currentSourceWidth_ = 0;
    int currentSourceHeight_ = 0;

    Ort::MemoryInfo cudaMemoryInfo_;
    Ort::Value m_inputTensorValue_{nullptr};
    Ort::Value m_outputTensorValue_{nullptr};
    Ort::IoBinding m_bind_;
    Ort::RunOptions m_runOptions_;

    CudaEventOwner m_stageStart_;
    CudaEventOwner m_uploadEnd_;
    CudaEventOwner m_preprocessEnd_;
    CudaEventOwner m_inferenceEnd_;
    CudaEventOwner m_postprocessEnd_;
    CudaEventOwner m_downloadEnd_;
};

#endif