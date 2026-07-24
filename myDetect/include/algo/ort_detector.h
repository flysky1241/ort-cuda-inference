#ifndef ORT_DETECTOR_H
#define ORT_DETECTOR_H

#include "algo/inference_interface.h"
#include "algo/inference_settings.h"
#include <array>
#include <cstdint>
#include <onnxruntime_cxx_api.h>
#include <vector>

enum class OrtBackend: unsigned
{
    CPU,
    CUDA,
    TRT
};

class ORTDetector final: public InferenceInterface
{
public:
    ORTDetector(OrtBackend backend = OrtBackend::CPU, int cudaDeviceId=0);
    void initConfig(InferenceSettings& settings) override;
	void infer_frame(cv::Mat& frame, std::vector<ODResultBox>& boxes) override;

	~ORTDetector() override = default;

    [[nodiscard]]
    ::OrtBackend backend() const noexcept
    {
        return m_backend_;
    } 

    [[nodiscard]]
    bool isInitialized() const noexcept
    {
        return m_init;
    }


private:
/*
     * 保存letterbox预处理时的坐标转换参数。
     *
     * 原图：
     *      width × height
     *
     * 缩放后：
     *      resizedWidth × resizedHeight
     *
     * 最终补边到：
     *      inputWidth × inputHeight
     */
    struct LetterboxInfo
    {
        float scale = 1.0f;

        int padLeft = 0;
        int padTop = 0;

        int originalWidth = 0;
        int originalHeight = 0;
    };

    // 获取classname.txt 来得到所有的类名
    void loadClassNames(const std::string& filepath);

    // 读取模型的shape 和 输入输出节点名称，以及类型
    void loadModelMetadata();

    // 图像预处理，返回NCHW格式的blob,blob就是可以理解就是把填充图像要放置到yolo模型训练的数据
    cv::Mat preProcess(const cv::Mat& frame, LetterboxInfo& letterbox) const;

    // 解析YOLOv8原始输出
    void postProcess(
        Ort::Value& outputTensor,           //由Ort 加装的blob数据，类型是Ort::Value
        const LetterboxInfo& letterbox,     //因为要解析所以要还原到原始的类型大小，而letterbox里面有padLeft 和 padTop以及 缩放系数
        const cv::Size& originalImageSize,  //这个是 原生帧画面的大小
        std::vector<ODResultBox>& result_box    //ODResultBox里面就是装好经过分数筛选和NMS去除重复值之后再加上还原之后的 框的大小以及相应的分数和类别名字
    ) const;

    // 按类别执行NMS，防止不同类别互相抑制
    std::vector<int> classAwareNms(
        const std::vector<cv::Rect>& boxes,     //这个就是把 经过初步的不符合置信度分数的排除之后的 候选框
        const std::vector<float>& scores,       //候选框对应的 分数
        const std::vector<int>& classIds        //候选框对应的类别
    ) const;

    //这个就没啥好说的就是 画图，把经过置信度分数和NMS分类别去除重复框 和还原回去的框 画在的原生图片上
    void drawDetectionResults(          
        cv::Mat& frame, 
        const std::vector<ODResultBox>& result_box, 
        const bool showScore
    );

    void configureSessionOptions();

    void appendCudaExecutionProvider();

private:

    // 枚举体区分挂载什么样的 EP
    ::OrtBackend m_backend_;

    int m_cudaDeviceId_;

    // 整个ONNX Runtime环境
    //这是 ORT 的上帝对象，整个进程只需要、也只能有一个。它负责全局的线程池和日志调度。
    Ort::Env m_env_;

    // 模型Session配置
    //这个非常重要 是决定推理是否是用CPU 还是GPU来跑，也就是是否挂载其他 EP。
    //当选择原生CPU来跑（而不是openvino）的时候不需要专门设置EP
    Ort::SessionOptions m_sessionOptions_;

    // 真正加载后的模型Session
    Ort::Session m_session_;

    // 表示输入Tensor使用CPU内存
    // 内存的物理护照 (MemoryInfo)
    // 因为在 C++ 里，当你把一个 std::vector<float> 的指针 data() 传给 ONNX Runtime 时，ORT 是个瞎子。
    // 它不知道这个指针到底是指向你主板上的内存条，还是指向你显卡里的显存！
    // Ort::MemoryInfo 就是你为这块内存签发的“物理护照”。

    // OrtArenaAllocator：告诉 ORT 开启底层内存池（Arena）来管理，速度极快。

    // OrtMemTypeDefault：告诉 ORT，这是一块极其普通的 CPU 内存！
    Ort::MemoryInfo m_meminfo_;

    // 用于获取模型输入输出节点名称
    Ort::AllocatorWithDefaultOptions m_allocator_;

    //因为这个是封装的C库的，所以在Session::Run 中跑的是C风格字符串，所以需要两种来存储
    std::vector<std::string> m_InputNameStorage_;
    std::vector<std::string> m_OutputNameStorage_;
    
    std::vector<const char*> m_InputNames_;
    std::vector<const char*> m_OutputNames_;

    // yolov8的输入NCHW 数据大小
    std::array<int64_t, 4> m_InputShape_{1, 3, 640, 640};

    // yolo的类名 集合
    std::vector<std::string> m_ClassName_;

    int m_InputWidth;
    int m_InputHeight;

    float m_ScoreThreshold;
    float m_NmsThreshold;

    bool m_ShowScore;
    bool m_ShowFps;

    bool m_init;
};

#endif