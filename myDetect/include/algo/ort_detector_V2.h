#include <memory>
#include <onnxruntime_cxx_api.h>
#include "algo/inference_interface.h"
#include "algo/ort_gpu_runtime.h"
#include "driver_types.h"

enum class OrtBackend : unsigned
{
    CPU,
    CUDA,
    TRT
};


class ORTDetector_V2 final : public InferenceInterface
{
public:
    explicit ORTDetector_V2(
        OrtBackend backend = OrtBackend::CPU, 
        int cudaDeviceId = 0,
        OrtGpuRuntimeConfig gpuConfig = {}
    );

    ~ORTDetector_V2() override = default;

    void initConfig(InferenceSettings& settings) override;
    void infer_frame(cv::Mat& frame, std::vector<ODResultBox>& boxes) override;

    [[nodiscard]]
    OrtBackend backend() const noexcept;

    [[nodiscard]]
    bool isInitialized() const noexcept;

private:
    struct LetterBoxInfo
    {
        float scale;
        int padLeft;
        int padTop;
        int originalWidth;
        int originalHeight;
    };

private:
    

private:
    OrtBackend m_backend_;
    int cudaDeviceId_;
    OrtGpuRuntimeConfig m_gpuConfig_;

    Ort::Env& m_env_;
    Ort::Session m_session_;
    Ort::SessionOptions m_sessionOption_;
    Ort::MemoryInfo m_cpuMeminfo_;
    Ort::AllocatorWithDefaultOptions m_allocator_;

    cudaStream_t m_stream_;
    std::unique_ptr<OrtFixedShapeIoBindingRunner> m_ioRunner_;
    
    std::vector<std::string> m_inputNameStorage_;
    std::vector<std::string> m_outputNameStorage_;
    std::vector<const char*> m_inputName_;
    std::vector<const char*> m_outputName_;

    std::array<int64_t, 4> m_inputShape_{1, 3, 640, 640};
    std::vector<int64_t> m_outputShape_;
    std::vector<std::string> m_className_;
    
    int m_inputWidth_ = 640;
    int m_inputHeight = 640;
    float m_scoreThreshold = 0.25f;
    float m_nmsThreshold = 0.45f;
    bool m_showScore = true;
    bool m_showFPS = true;
    bool m_init = false;

    std::mutex m_inferMutx_;
};