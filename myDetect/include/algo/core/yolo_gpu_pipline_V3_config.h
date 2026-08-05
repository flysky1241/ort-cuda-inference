#ifndef YOLO_GPU_PIPELINE_V3_CONFIG_H
#define YOLO_GPU_PIPELINE_V3_CONFIG_H

#include <cstdint>

struct LetterboxTransformV3
{
    float scale = 1.0f;
    int padLeft = 0;
    int padTop = 0;
    int originalWidth = 0;
    int originalHeight = 0;    
};


struct GpuDetectionV3
{
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
    std::int32_t classId = -1;
};


struct OrtV3StageTimings
{
    double hostStageingMs = 0.0;
    float uploadMs = 0.0f;
    float preprocessMs = 0.0f;
    float inferenceMs = 0.0f;
    float gpuPostprocessMs = 0.0f;
    float downloadMs = 0.0f;

    [[nodiscard]]
    double gpuTotalMs() const noexcept;
};


struct YoloGpuPipelineV3Config
{
    float scoreThreshold = 0.25f;
    float nmsThrehold = 0.45f;
    int nmsTopK = 1024;     //在执行极其耗时的 NMS 之前，最多只允许前 1024 个最高分的框进入 NMS 绞肉机。
    int maxDetections = 300;   //经过所有清理后，一张图片最多只输出 300 个目标。 算出来的坐标最终要序列化成 JSON 通过 HTTP 发送的。
    int warmupRuns = 3;
    bool disableProviderSynchronization = true;    //让ORT 解绑 调用自己的 cudaStream来管理
};


#endif