#include "algo/ort_detector.h"
#include "algo/inference_settings.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "opencv2/core.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/operations.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <xtimec.h>

ORTDetector::ORTDetector(OrtBackend backend, int cudaDeviceId)
    :m_backend_(backend),
    m_cudaDeviceId_(cudaDeviceId),
    m_env_(ORT_LOGGING_LEVEL_WARNING, "myDetect"),     //如果嫌到时候运转的时候 终端写的log太多，可以指定log的级别只有错误才爆出来
    m_session_(nullptr),
    m_meminfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    if(m_cudaDeviceId_ < 0)
    {
        throw std::runtime_error("CUDA device id cannot be negative");
    }

    this->m_InputWidth = 640;
    this->m_InputHeight = 640;

    this->m_ScoreThreshold = 0.25f;
    this->m_NmsThreshold = 0.45f;

    this->m_ShowScore = true;
    this->m_ShowFps = true;

    this->m_init = false;
}

//读取类别文件
void ORTDetector::loadClassNames(const std::string& filepath)
{
    this->m_ClassName_.clear();

    //在 C++中 要读取文件里面的内容 就是使用这个 filesystem + ifstream，filesystem用来验证文件是否存在以及是什么类型
    //而ifstream 是真正读取内容的，把其所有内容读到流对象里面，然后通过 std::getline 来读取每一行数据

    std::ifstream inputFile(filepath);
    if(!inputFile.is_open())
    {
        throw std::runtime_error("Failed to open class file"+filepath);
    }

    std::string tempClassName;

    while(std::getline(inputFile, tempClassName))
    {
        // Windows文本文件一行可能以 \r\n 结尾。
        // getline删除了\n，但某些情况下会留下\r。
        if(!tempClassName.empty() && tempClassName.back() == '\r')
        {
            m_ClassName_.pop_back();
        }

        if(!tempClassName.empty())
        {
            std::cout<<tempClassName<<'\n';
            std::cout<<"tempClassName.size"<<tempClassName.size()<<'\n';
            m_ClassName_.push_back(tempClassName);  //这个地方必须是 push_back 不能是这个emplace_back这种，push_back是深拷贝所以没问题。
                                                    //因为是深拷贝 所以也不要用这种什么std::move这种，因为m_ClassName是跨函数栈帧的，
                                                    // 但是tempClassName 并不是所以有可能在后面用的时候 提取和找不到这个本体导致出问题
        }

        if(tempClassName.empty()){
            throw std::runtime_error("Failed class name is empty" + filepath);
        }
    }
    inputFile.close();
}


void ORTDetector::initConfig(InferenceSettings& settings) 
{
    //读取现有配置
    m_ScoreThreshold = settings.getT_score();
    m_InputHeight = settings.get_input_h();
    m_InputWidth = settings.get_input_w();
    m_ShowScore = settings.getShow_score();
    m_ShowFps = settings.getShow_fps();

    if(m_InputWidth <=0 || m_InputHeight<=0)
        throw std::runtime_error("Invalid model input size");

    this->loadClassNames(settings.getConfig_file());

    this->configureSessionOptions();

     /*
     * 4. 加载模型。
     *
     * 使用std::filesystem::path的原因是：
     *
     * Windows下ONNX Runtime模型路径使用宽字符；
     * Linux下则使用普通char。
     *
     * path::c_str()能根据平台给出对应字符类型。
     */

    const std::filesystem::path modelPath(settings.getWeight_file());

    if(!std::filesystem::exists(modelPath))
    {
        throw std::runtime_error("ONNX model does not exist: " +
            settings.getWeight_file());
    }

    try 
    {
        //初始化 模型m_session ，类似于opencv dnn 的 this->net = cv::dnn::readNetFromONNX(settings.getWeight_file());
        m_session_ = Ort::Session(this->m_env_, modelPath.c_str(), this->m_sessionOptions_);

    } catch (const Ort::Exception& exception) {
        //这些地方都用 try-catch 连接，因为Ort::Session 的构造函数 是用的 异常来搞得，所以我们创建也得用try-catch抓一哈
        /*
        inline Session::Session(const Env& env, const ORTCHAR_T* model_path, const SessionOptions& options) {
            ThrowOnError(GetApi().CreateSession(env, model_path, options, &this->p_));
        }  可以看到也是 通过throw 来搞的所以 构造的时候我们就可以在我们这一层直接 抓取一哈
        */
        throw std::runtime_error(
            std::string(
                "Failed create ONNX Runtime"
                "Session: "
            ) +
            exception.what()
        );      //不过我们抓取到 异常之后，通过标准异常继续向上抛，所以在外层只有标准异常才可以catch。
    }

    //接下来就是抓取元数据，这个元数据就是要对 我们的onnx文件进行 输入名字与个数 输出名字和个数加载，因为Ort::session::Run的时候会读取
    this->loadModelMetadata();
    this->m_init = true;
}


void ORTDetector::loadModelMetadata()
{
    m_InputNameStorage_.clear();
    m_OutputNameStorage_.clear();

    m_InputNames_.clear();
    m_OutputNames_.clear();

    // m_session已经读取这个onnx文件路径了，所以通过路径加载了模型，通过GetInputCount来得到 输入框的个数
    size_t inputCount = m_session_.GetInputCount();     
    // 通过GetOutputCount来得到 输出框的个数
    size_t OutputCount = m_session_.GetOutputCount();

    /*
    在你的思维里，YOLO 可能是一个函数：输入图片 -> YOLO -> 输出框。
    但在操作系统的物理内存里，.onnx 模型是一张极其庞大的有向无环图（DAG）。

    多门问题： 现代 AI 模型可能不止一个输入口。比如一个智能问答视觉模型，它可能有两个输入节点：
    一个是 "image"（喂图片），另一个是 "text"（喂文字）。

    各取所需： 模型的输出节点也可能有好几个。比如 YOLOv8，有些版本的模型会同时吐出三个输出："boxes"（坐标）、"scores"（置信度）、"classes"（类别）。

    ORT 的强制契约： 当你调用 session.Run() 时，
    ORT 会极其死板地问你：“老哥，你手里的这块 Tensor 数据，到底要塞进哪扇门？你要提取的结果，又是从哪扇门出来的？”

    这就是为什么我们必须在推理时，把输入输出的名字（如 "images", "output0"）明确地传给 ORT！
    
    */

    if(inputCount !=1)
    {
        throw std::runtime_error("This detector currently supports "
            "exactly one model input");
    }

    if(OutputCount == 0)
    {
        throw std::runtime_error("The ONNX model has no outputs");
    }

    
    m_InputNameStorage_.reserve(inputCount);
    for(auto index = 0; index < inputCount; ++index)
    {
        //Ort::AllocatorWithDefaultOptions m_allocator_; 是一个极其不起眼，
        // 但如果你不用它，程序就会疯狂内存泄漏（Memory Leak）的绝对核心组件！
        auto inputName = m_session_.GetInputNameAllocated(index, m_allocator_);

        /*
        灾难场景推演：
        假设你想问模型：“老哥，你的输入节点叫什么名字？”
        模型（底层 C 代码）在它的堆内存里用 malloc() 申请了一块内存，写上 "images" 这个字符串，然后把指针扔给你的 C++ 代码。
        现在问题来了：这块内存谁来释放？
        如果你在 C++ 里直接对这个指针 delete，而底层是用 malloc 分配的，跨语言内存释放规则不匹配，程序当场段错误（Segfault）暴毙！

        官方解药：Allocator（内存分配器/大管家）
        为了解决这个跨界收尸的问题，ORT 官方搞出了 AllocatorWithDefaultOptions。
        它的物理本质就是：ORT 官方指定的、跨界通用的“内存大管家”。 你把它传给 ORT，就是在告诉底层：“老哥，你有什么字符串要交给我，请用这个大管家去分配内存。等我用完了，我也会叫这个大管家去安全地释放它！”

        ⚙️ 第二重物理真相：它到底用在什么地方？

        正如你注释里写的，它 99% 的应用场景，就是在模型初始化时，动态读取模型的 Metadata（元数据：比如输入输出的名字、维度）。

        在旧版本的 ORT（v1.12 之前），获取输入名字的 API 会直接返回一个裸指针，极其危险。
        在现代 ORT（v1.13 及以后），微软强行废弃了旧 API，逼着你必须传入 m_allocator_！
        
        */

        if(inputName == nullptr)
        {
            throw std::runtime_error("Failed to read model input name");
        }

        m_InputNameStorage_.push_back(inputName.get());
    }

    m_OutputNameStorage_.reserve(OutputCount);
    for(auto index = 0; index < OutputCount; ++index)
    {
        auto OutputName = m_session_.GetOutputNameAllocated(index, m_allocator_);
        if(OutputName == nullptr)
        {
            throw std::runtime_error("Failed to read model input name");
        }

        m_OutputNameStorage_.push_back(OutputName.get());
    }

    /*
     * 注意：
     *
     * 必须等string vector全部构造完成后，
     * 再保存c_str()指针。
     *
     * 否则vector扩容可能移动string，
     * 之前保存的const char*就可能失效。
     */
    m_InputNames_.reserve(m_InputNameStorage_.size());
    for(const std::string& name : m_InputNameStorage_)
    {
        m_InputNames_.push_back(name.c_str());
    }

    m_OutputNames_.reserve(m_OutputNames_.size());
    for(const std::string& name : m_OutputNameStorage_)
    {
        m_OutputNames_.push_back(name.c_str());
    }

    /*
    * 获取输入Tensor信息。
    */
    const Ort::TypeInfo inputTypeinfo = m_session_.GetInputTypeInfo(0);
    Ort::ConstTensorTypeAndShapeInfo tensorInfo = inputTypeinfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = tensorInfo.GetElementType();

    if(inputType != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        throw std::runtime_error("This implementation only supports "
            "float32 model input");
    }

    std::vector<int64_t> modelInputShape = std::move(tensorInfo.GetShape());
    if(modelInputShape.size() != 4)
    {
        throw std::runtime_error("Excepted model input shape"
            "[N, C, H, W]");    
    }

    /*
     * YOLO一般是：
     *
     * [1, 3, 640, 640]
     *
     * 动态维度可能显示：
     *
     * [-1, 3, -1, -1]
     */
    
    if(modelInputShape[1] >0 && modelInputShape[1]!=3)
    {
        throw std::runtime_error("Expected a 3-channel model input");  
    }

    if(modelInputShape[2]>0)
    {
        m_InputHeight = static_cast<int>(modelInputShape[2]);
    }

    if(modelInputShape[3]>0)
    {
        m_InputWidth = static_cast<int>(modelInputShape[3]);
    }

    m_InputShape_ = {1, 3, static_cast<int64_t>(m_InputHeight), static_cast<int64_t>(m_InputWidth)};

    std::cout<<"ONNX input name: "
            <<m_InputNameStorage_.front()
            <<'\n';

    for(auto index =0; index<m_OutputNameStorage_.size(); ++index)
    {
        std::cout<<"ONNX Output ["
                <<index
                <<"] name:"
                <<m_OutputNameStorage_[index]
                <<'\n';
    }

    std::cout<<"ONNX input shape:"
            <<m_InputShape_[0]<<", "
            <<m_InputShape_[1]<<", "
            <<m_InputShape_[2]<<", "
            <<m_InputShape_[3]<<'\n';
}


//原图帧预处理
cv::Mat ORTDetector::preProcess(const cv::Mat& frame, LetterboxInfo& letterbox) const
{
    if(frame.empty())
    {
        throw std::invalid_argument("Cannot preprocess an empty image");
    }

    letterbox.originalHeight = frame.rows;
    letterbox.originalWidth = frame.cols;

    /*
     * 保持宽高比缩放。
     *
     * 例如：
     *
     * 原图 1280×720
     * 模型 640×640
     *
     * scale = min(640/1280, 640/720)
     *       = 0.5
     *
     * 缩放后：
     * 640×360
     */

    //这个和我们之前写的那个 是左上角，而且是原图除以模型输入尺寸，所以是取最大，而这个是取最小。 因为必须是等比例缩放，不然的话容易出问题。
    float widthScale = static_cast<float>(m_InputHeight)/static_cast<float>(letterbox.originalHeight);
    float heightScale = static_cast<float>(m_InputWidth)/static_cast<float>(letterbox.originalWidth);

    float scale = (std::min)(widthScale, heightScale);
    letterbox.scale = scale;

    //重整之后的尺寸必须是 int类型
    int resizeWidth = static_cast<int>(std::round(frame.cols*scale));
    int resizeHeight = static_cast<int>(std::round(frame.rows*scale));

    cv::Mat resizeImage;
    //这个就是 blobFromImage的时候的 输入图形。要让其符合Yolo的输入, 来进行推理. 因为是 居中补边，所以还要进行一次补边才是blobFromImage输入
    cv::resize(frame, resizeImage, cv::Size(resizeWidth, resizeHeight));

    //用原始图的宽高减去缩放之后的宽高 得到的就是 补边多少
    int totalPaddingWidth =  m_InputWidth - resizeWidth;
    int totalPaddingHeight =  m_InputHeight - resizeHeight;

    //左补
    int padLeft = totalPaddingWidth/2;
    //右补
    int padRight = totalPaddingWidth - padLeft;
    //上补
    int padTop = totalPaddingHeight/2;
    //下补
    int padBottom = totalPaddingHeight - padTop;

    letterbox.padLeft = padLeft;
    letterbox.padTop = padTop;

    //创建居中补边正方形，上下左右居中填充。
    cv::Mat letterboxImage;
    cv::copyMakeBorder(
        resizeImage, 
        letterboxImage, 
        padTop, 
        padBottom, 
        padLeft, 
        padRight, 
        cv::BORDER_CONSTANT, 
        cv::Scalar(114, 114, 114)
    );

    /*
     * blobFromImage完成：
     *
     * BGR -> RGB
     * uint8 -> float32
     * /255
     * HWC -> NCHW
     *
     * 输出shape：
     * [1, 3, inputHeight, inputWidth]
     */

    cv::Mat blob = cv::dnn::blobFromImage(
        letterboxImage, 
        1.0/255, 
        cv::Size(m_InputWidth, m_InputHeight), 
        cv::Scalar(), 
        true, 
        false
    );

    if(!blob.isContinuous())
    {
        blob = blob.clone();
    }

    return blob;
}


//开始推理
void ORTDetector::infer_frame(cv::Mat& frame, std::vector<ODResultBox>& result_boxes)
{
    //FPS 记录开始时间
    int64_t start = cv::getTickCount();

    result_boxes.clear();

    if(frame.empty())
    {
        throw std::runtime_error("frame is empty");
    }

    if(!this->m_init)
    {
        throw std::runtime_error("ORTDETECTOR is not initlization");
    }

    LetterboxInfo letterbox;

    cv::Mat inputBlob = this->preProcess(frame, letterbox);

    /*
    在工业界，OpenCV 的 blobFromImage 就是最好用的预处理工具。 
    我们用它绞好肉（cv::Mat），把指针抽出来，包上 Ort::Value 的皮，
    然后用 session.Run() 发射给 GPU（ORT 会自动帮你做显存搬运）。

    大厂架构师的绝对标准打法是：跨界白嫖！
    我们继续用 OpenCV 的 blobFromImage 来处理图片，榨出纯正的 NCHW 浮点数面条，
    然后把这坨面条的首地址指针，强行包装进 Ort::Value 里扔给 ORT！
    */
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        m_meminfo_, 
        inputBlob.ptr<float>(), 
        inputBlob.total(), 
        m_InputShape_.data(),
        m_InputShape_.size()
    );

    if(!inputTensor.IsTensor())
    {
        throw std::runtime_error("Failed to create input tensor");
    }

    std::vector<Ort::Value> outputTensor;

    /*
    物理真相：因为 ONNX Runtime 面对的不是“你这一个模型”，而是全宇宙所有的 AI 模型！

    图计算的多出口特性：
    在你眼里，YOLOv8 可能只有一个输出出口（比如叫 output0，吐出一个 1x84x8400 的张量）。
    但是！在真实的 AI 工业界：

    早期的 YOLOv5 模型，有 3 个输出出口（分别对应大、中、小三种特征图）！

    某些人脸识别模型，可能有 2 个输出（一个输出人脸坐标，另一个输出年龄性别）！

    甚至有些多模态网络，可能有 10 个输入和 5 个输出！

    ORT 的通用契约：
    为了用一个 Run 函数兼容全宇宙所有的模型，ORT 官方规定了绝对死板的契约：

    你想要几个输出？把你想要的名字装在 m_OutputNames_ 数组里告诉我。

    你想要 1 个，我就还给你一个长度为 1 的 std::vector<Ort::Value>。

    你想要 3 个，我就还给你一个长度为 3 的 std::vector<Ort::Value>。

    👑 提取结果的正确姿势：
    既然返回的是 vector，而在你的业务里你明知道只请求了 1 个输出，你只需要直接拿它的第 0 个元素即可
    */

    try
    {
        outputTensor = m_session_.Run(
            Ort::RunOptions{nullptr}, 
            m_InputNames_.data(),
            std::addressof(inputTensor),
            m_InputNames_.size(),
            m_OutputNames_.data(),
            m_OutputNames_.size()
        );
    } catch (Ort::Exception& excepted) {
        throw std::runtime_error(
            std::string("ONNX Runtime inference failed")+
            excepted.what()
        );
    }

    if(outputTensor.empty())
    {
        throw std::runtime_error("The ONNX model returned no outputs");
    }

    //开始后处理，本质上就是 通过当前得分和你自定义的置信度得分比较高于你定的置信度得分的留下作为候选，这些候选再通过NMS去除重复框
    //然后把这两个都过滤的 在还原到原图上标注出来，后处理就完成了
    this->postProcess(outputTensor.front(), letterbox, frame.size(), result_boxes);

    this->drawDetectionResults(frame, result_boxes, true);

    float t = (cv::getTickCount()-start)/static_cast<float>(cv::getTickFrequency());
    cv::putText(frame, cv::format("FPS: %.2f", 1.0/t), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 3.0f, cv::Scalar(255, 0, 0));
}


//后处理 通过分数来得到相应的 Boxes，classIDs，confidences 三个vector
void ORTDetector::postProcess(Ort::Value& outputTensor, const LetterboxInfo& letterbox, const cv::Size& originalImageSize,
        std::vector<ODResultBox>& result_box) const {
    
    //防御性编程 
    if(!outputTensor.IsTensor())
    {
        throw std::runtime_error("Model output is not a tensor");
    }           

    auto outputInfo = outputTensor.GetTensorTypeAndShapeInfo();
    if(outputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        throw std::runtime_error("Only float32 model output is supported");
    }

    //和YOLOv8 DNN 的pred 维度检测一样
    std::vector<int64_t> outputShape = outputInfo.GetShape();

    if(outputShape.size()!=3 || outputShape[0]!=1)
    {
        throw std::runtime_error(
            "Expected YOLO output shape"
            "[1, attributes, candidates] or "
            "[1, candidates, attributes]"
        );
    }

    const int64_t dimension1 = outputShape[1];
    const int64_t dimension2 = outputShape[2];

    bool attributesFirst = dimension1 < dimension2;

    const int attributes = static_cast<int>(attributesFirst? dimension1:dimension2);
    const int candidateCount = static_cast<int>(attributesFirst? dimension2:dimension1);

    if(attributes<=4)
    {
        throw std::runtime_error("Invalid YOLO output attributes");
    }

    //得到类别个数
    const int classCount = attributes - 4;
    if(classCount != this->m_ClassName_.size())
    {
        throw std::runtime_error(
            "Model class count does not match "
            "classes.txt: model=" +
            std::to_string(classCount)+
            ", files="+
            std::to_string(m_ClassName_.size())
        );
    }

    //这个是真正的 最后推导出来的 pred [1, 84, 8400]的矩阵数据
    float* outputData = outputTensor.GetTensorMutableData<float>();
    cv::Mat detectOutput;
    
    if(attributesFirst)
    {
        //把这个数据 通过cv::Mat包裹
        cv::Mat rawData(
            attributes,
            candidateCount,
            CV_32FC1,
            outputData
        );

        //然后转置，得到的就是 [8400, 84],每一行是一个框的中心点，宽高和类别的得分 [cx, cy, ow, oh, presonScore, ...]
        cv::transpose(rawData, detectOutput);
    }
    else
    {
        detectOutput = cv::Mat(candidateCount, attributes, CV_32FC1, outputData);
    }

    //候选框 和 分数 和 类别
    std::vector<cv::Rect> candidateBoxes;
    std::vector<int> candidateClassIds;
    std::vector<float> candidateScores;

    //按照最大 的 相当于每个框都能过 自定义的置信度分数
    candidateBoxes.reserve(candidateCount);
    candidateClassIds.reserve(candidateCount);
    candidateScores.reserve(candidateCount);

    //原图 rect，为的是后面还原的时候的把不在rect范围的排除掉
    cv::Rect imageBound(0, 0, originalImageSize.width, originalImageSize.height);
    for(auto candidateIndex = 0; candidateIndex<candidateCount; ++candidateIndex)
    {
        std::unique_ptr<float> candidates(detectOutput.ptr<float>(candidateIndex));
         /*
         * YOLOv8 Detect输出：
         *
         * candidate[0] = cx
         * candidate[1] = cy
         * candidate[2] = width
         * candidate[3] = height
         * candidate[4...] = 类别分数
         */

        int bestClassId = -1;
        double bestClassScore = 0.0f;
        cv::Point bestMaxPoint;
        //for(auto classId = 0; classId < classCount; ++classId)
        //{
        //    float classScore = candidates.get()[4+classId];
        //}

        cv::Mat class_score = detectOutput.row(candidateIndex).colRange(4, detectOutput.cols);
        cv::minMaxLoc(class_score, nullptr, std::addressof(bestClassScore), nullptr, std::addressof(bestMaxPoint));
        bestClassId = bestMaxPoint.x;

        if(bestClassId<0 || bestClassScore < m_ScoreThreshold)
        {
            float* raw_ptr = candidates.release();
            if((raw_ptr == nullptr) || (candidates != nullptr))
            {
                throw std::runtime_error(
                    "smart ptr is invaild, line:"+
                    std::to_string(__LINE__)+
                    "fun: "+
                    __func__
                );
            }
            continue;
        }

        //当前 类别分数的类别 对应的中心点 和 宽高
        float centerX = candidates.get()[0];
        float centerY = candidates.get()[1];
        float modelBoxWidth = candidates.get()[2];
        float modelBoxHeight = candidates.get()[3];

        /*
         * 先从模型输入坐标中减掉letterbox补边，
         * 再除以缩放比例，恢复到原图坐标。
         */

        //因为 当初是模型尺寸除以原图，所以现在要还原到原图也是要除法的
        float left = static_cast<float>(centerX - modelBoxWidth*0.5f - static_cast<float>(letterbox.padLeft))/letterbox.scale;
        float top = static_cast<float>(centerY - modelBoxHeight*0.5f - static_cast<float>(letterbox.padTop))/letterbox.scale;

        float restoreWidth = modelBoxWidth/letterbox.scale;
        float restoreHeight = modelBoxHeight/letterbox.scale;

        cv::Rect box(
            static_cast<int>(left), 
            static_cast<int>(top), 
            static_cast<int>(restoreWidth), 
            static_cast<int>(restoreHeight)
        );
        //通过这个排除掉 出了原图范围的
        box &= imageBound;

        if(box.empty())
        {
            float* raw_ptr = candidates.release();
            if((raw_ptr == nullptr) || (candidates != nullptr))
            {
                throw std::runtime_error(
                    "smart ptr is invaild, line:"+
                    std::to_string(__LINE__)+
                    "fun: "+
                    __func__
                );
            }
            continue;
        }

        candidateBoxes.push_back(box);
        candidateScores.push_back(bestClassScore);
        candidateClassIds.push_back(bestClassId);

        float* raw_ptr = candidates.release();
        if((raw_ptr == nullptr) || (candidates != nullptr))
        {
            throw std::runtime_error(
                "smart ptr is invaild, line:"+
                std::to_string(__LINE__)+
                "fun: "+
                __func__
            );
        }
    }

    //接下来就是 对不同类别进行NMS，而不是直接NMSBoxes
    std::vector<int> keptIndexes = classAwareNms(
        candidateBoxes, 
        candidateScores, 
        candidateClassIds
    );

    result_box.reserve(keptIndexes.size());
    for(const auto& keptIndex : keptIndexes)
    {
        if(keptIndex<0 || keptIndex >= static_cast<int>(candidateBoxes.size()))
        {
            continue;
        }

        const int classId = candidateClassIds[keptIndex];

        if(classId<0 || classId>= static_cast<int>(m_ClassName_.size()))
        {
            continue;
        }

        //把经历 置信度分数选取和NMS去重之后 符合的记录到std::vector<ODResultBox> 然后统一在原图frame上画框
        ODResultBox result;
        result.box = candidateBoxes[keptIndex];
        result.score = candidateScores[keptIndex];
        result.name = m_ClassName_[classId];

        result_box.push_back(std::move(result));
    }
}


//按照类别来进行 NMS
std::vector<int> ORTDetector::classAwareNms(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores,
    const std::vector<int>& classIds) const {
    
    /*
    传统的 不知道每个框属于什么类别。

    假设同一个位置同时预测：

    person 0.90
    backpack 0.82

    如果框重叠度很高，类别无关 NMS 可能把其中一个删除。

    因此可以按类别分组后分别执行 NMS。
    
    */
    if(boxes.size()!=scores.size() || classIds.size() != boxes.size())
    {
        throw std::runtime_error("NMS Boxes vector size is not match");
    }

    /*
     * key：
     *     类别编号
     *
     * value：
     *     属于这个类别的原始候选框下标
     */
    std::unordered_map<int, std::vector<int>> indexesByClass;
    
    for(auto index = 0; index<boxes.size(); ++index)
    {
        indexesByClass[classIds[index]].push_back(index);
    }

    //这个最终记录的就是 通过过滤的原始候选的下标
    std::vector<int> finalIndexes;
    for(const auto& entry : indexesByClass)
    {   
        //这个originalIndexes记载的就是 同一个类别下，不同行的下标
        const std::vector<int>& originalIndexes = entry.second;

        std::vector<cv::Rect> currentBoxes;
        std::vector<float> currentScores;

        currentBoxes.reserve(originalIndexes.size());
        currentScores.reserve(originalIndexes.size());

        for(const int originalIndex : originalIndexes)
        {
            currentBoxes.push_back(boxes[originalIndex]);
            currentScores.push_back(scores[originalIndex]);
        }
        /*
         * 这里返回的是currentClassBoxes内部下标，
         * 不是boxes的原始下标。
         */
        std::vector<int> localKeptIndexes;

        cv::dnn::NMSBoxes(currentBoxes, currentScores, m_ScoreThreshold, m_NmsThreshold, localKeptIndexes);

        for(const int localIndex : localKeptIndexes)
        {
            if(localIndex < 0 ||localIndex>= static_cast<int>(originalIndexes.size()))
            {
                continue;
            }
            //通过originalIndexes来得到原始行下标
            finalIndexes.push_back(originalIndexes[localIndex]);
        }
    }
    /*
     * 最终按照置信度从高到低排列，
     * 便于UI展示和调试。
    */

    return finalIndexes;
}


//画框
void ORTDetector::drawDetectionResults(cv::Mat& frame, const std::vector<ODResultBox>& result_box, 
    const bool showScore){
    
    if(result_box.empty())
    {
        throw std::runtime_error("result_boxes is empty");
    }

    for(const auto& result : result_box)
    {
        cv::rectangle(frame, result.box, cv::Scalar(0, 0, 255), 2);

        std::string label = result.name;
        if(showScore)
        {
            label += cv::format("%.2f", result.score);
        }
    
        cv::putText(
            frame, 
            label, 
            cv::Point(result.box.x, (std::max)(result.box.y-5, 20)), 
            cv::FONT_HERSHEY_SIMPLEX, 
            0.6, 
            cv::Scalar(0, 0, 255),
            2
        );
    }
}

void ORTDetector::configureSessionOptions()
{
    //设置 Ort的 session的属性
    //开启最高级别图优化
    m_sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    /*
     * CPU推理线程数。
     *
     * 不建议无脑设置为全部逻辑核心，
     * 因为Qt、视频解码、OpenCV也需要线程。
     */

    const unsigned hardwareThreads = std::thread::hardware_concurrency();

    const int intraOpThreads = hardwareThreads>2 ? static_cast<int>(hardwareThreads/2) : 1;

    //因为这里是 CPU推理所以我们在这里没有调用其他API函数，但是如果是CUDA推理的话，这里必须指定要挂载的EP，因为再后续构造Ort::session的时候要用
    //比如：m_seesionOptions_.AppendExecutionProvider_CUDA_V2()
    m_sessionOptions_.SetIntraOpNumThreads(intraOpThreads);

    switch (m_backend_) 
    {
    case OrtBackend::CPU:
    {
        std::cout<<"[ORT] Using CPU Execution Provider"<<'\n';
        break;
    }
    case OrtBackend::CUDA:
    {
        this->appendCudaExecutionProvider();
        std::cout<<"[ORT] Using CUDA Execution Provider"<<'\n';
        break;
    }
    default:
    {
        throw std::runtime_error("Unknown ONNX Runtime backend");
    }
    }
}


void ORTDetector::appendCudaExecutionProvider()
{
    const OrtApi& api = Ort::GetApi();

    OrtCUDAProviderOptionsV2* rawCudaOptions = nullptr;

    Ort::ThrowOnError(api.CreateCUDAProviderOptions(std::addressof(rawCudaOptions)));

    if(rawCudaOptions == nullptr)
    {
        throw std::runtime_error("CreateCUDAProviderOptions returned nullptr");
    }

    auto cudaOptionDeleter = [&api](OrtCUDAProviderOptionsV2* options)
    {
        if(options != nullptr)
        {
            api.ReleaseCUDAProviderOptions(options);
        }
    };

    std::unique_ptr<
        OrtCUDAProviderOptionsV2, 
        decltype(cudaOptionDeleter)
    > cudaOptions(
        rawCudaOptions,
        cudaOptionDeleter
    );

    const std::string deviceId = std::to_string(m_cudaDeviceId_);
    std::vector<const char*> keys {
        "device_id",
        "arena_extend_strategy",
        "cudnn_conv_algo_search",
        "do_copy_in_default_stream",
        "cudnn_conv_use_max_workspace",
        "use_tf32"
    };

    std::vector<const char*> values {
        deviceId.data(),
        "kNextPowerOfTwo",
        "DEFAULT",
        "1",
        "1",
        "1"
    };

    if(keys.size() != values.size())
    {
        throw std::runtime_error("CUDA provider option key/value size mismatch");
    }

    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
        cudaOptions.get(),
        keys.data(),
        values.data(),
        keys.size()
    ));

    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
        static_cast<OrtSessionOptions*>(this->m_sessionOptions_),
        cudaOptions.get()
    ));
}