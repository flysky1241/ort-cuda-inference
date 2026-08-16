#ifndef ORT_DETECTOR_V3_H
#define ORT_DETECTOR_V3_H

#include "algo/inference_interface.h"
#include "algo/ort_gpu_runtime.h"

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


private:
    


};



#endif