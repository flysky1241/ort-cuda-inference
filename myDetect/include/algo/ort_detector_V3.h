#ifndef ORT_DETECTOR_V3_H
#define ORT_DETECTOR_V3_H

#include "algo/core/yolo_gpu_pipline_V3_config.h"
#include "algo/inference_interface.h"
#include "algo/inference_settings.h"
#include "algo/ort_gpu_runtime.h"
#include "algo/yolo_gpu_pipeline_V3.h"
#include "onnxruntime_cxx_api.h"
#include <array>
#include <memory>
#include <mutex>

enum OrtV3Backend : unsigned
{
    CUDA,
    TRT
};

class ORTDetector_V3 final : public InferenceInterface
{
public:
    explicit ORTDetector_V3(
        OrtV3Backend backend = OrtV3Backend::TRT,
        int cudaDeviceId = 0,
        OrtGpuRuntimeConfig config = OrtGpuRuntimeConfig{}
    );

    ~ORTDetector_V3() override;

    void initConfig(InferenceSettings& settings) override;
    void infer_frame(cv::Mat& frame, std::vector<ODResultBox>& boxes) override;

    [[nodiscard]]
    OrtV3Backend backend() const noexcept;

    [[nodiscard]]
    bool isInitialized() const noexcept;

    [[nodiscard]]
    const OrtV3StageTimings& lastTimings() const noexcept;

private:
    void validateCudaDevice();
    void loadClassNames(const std::filesystem::path& filepath);
    void configureSessionOptions(const std::filesystem::path& modelPath);
    void prepareTensorRtCacheLayout(const std::filesystem::path& modelPath);
    void localModelMetaData();
    void createGpuPipline();

    [[nodiscard]]
    LetterboxTransformV3 makeLetterBoxTransform(const cv::Mat& frame) const;

    void drawDetectionResults(cv::Mat& frame, const std::vector<ODResultBox>& boxes) const;

    static bool hasDynamicDimension(const std::vector<int64_t>& shape);
    
    static std::string makeModelFingerPrint(
        const std::filesystem::path& modelPath,
        const std::array<int64_t, 4>& inputShape,
        bool fp16,
        bool int8
    );

    static std::string sanitizePathComponent(std::string value);

private:
    OrtV3Backend m_backend_;
    int cudaDeviceId_;
    OrtGpuRuntimeConfig m_gpuConfig_;

    Ort::Env& m_env_;
    Ort::SessionOptions m_sessionOptions_;

    CudaStreamOwner m_stream_;
    Ort::Session m_session_{nullptr};
    std::unique_ptr<YoloGpuPipeline_V3> m_gpuPipline_;

    Ort::AllocatorWithDefaultOptions m_allocator_;

    std::vector<std::string> m_inputNameStorage_;
    std::vector<std::string> m_outputNameStorage_;

    std::array<std::int64_t, 4> inputShape_{1, 3, 640, 640};
    std::vector<std::int64_t> outputShape_;
    std::vector<std::string> m_className_;

    int m_inputWidth_ = 640;
    int m_inputHeight = 640;
    float m_scoreThreshold = 0.25f;
    float m_nmsThreshold = 0.45f;
    bool m_showScore = true;
    bool m_showFPS = true;
    bool m_init = false;

    mutable std::mutex inferenceMutex_;
    double smoothedFps_{0.0};
};



#endif