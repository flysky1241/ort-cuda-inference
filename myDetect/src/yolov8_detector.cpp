#include "algo/yolov8_detector.h"
#include "algo/inference_settings.h"
#include "opencv2/core.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/operations.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include <fstream>
#include <memory>
#include <string>

void YOLOv8Detector::initConfig(InferenceSettings& settings)
{
    this->net = cv::dnn::readNetFromONNX(settings.getWeight_file());
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    score = settings.getT_score();
    input_h = settings.get_input_h();
    input_w = settings.get_input_w();
    show_fps = settings.getShow_fps();
    show_score = settings.getShow_score();

    std::ifstream fp(settings.getConfig_file());
    if(!fp.is_open())
    {
        return;
    }
    std::string name;
    while(std::getline(fp, name))
    {
        if(name.length()!=0)
        {
            this->class_name.push_back(name);
        }
    }
    fp.close();
}

void YOLOv8Detector::infer_frame(cv::Mat& frame, std::vector<ODResultBox>& result_boxes) 
{
    int64 start = cv::getTickCount();
    //预处理
    int w = frame.cols;
    int h = frame.rows;

    int max_ = (std::max)(w,h);
    cv::Mat image = cv::Mat::zeros(cv::Size(max_,max_), CV_8UC3);
    cv::Rect roi(0, 0, w , h);
    frame.copyTo(image(roi));

    float x_factor = image.cols/640.0f;
    float y_factor = image.rows/640.0f;

    //推理
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0/255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
    this->net.setInput(blob);
    cv::Mat pred = net.forward();

    for (int i = 0; i < pred.dims; ++i)
    {
        std::cout << "pred.size[" << i << "] = "
                << pred.size[i] << '\n';
    }


    //后处理 1*84*8400
    cv::Mat detout(pred.size[1], pred.size[2], CV_32F, pred.ptr<float>());
    //转置一哈，用reshape是不行的内存是没有变的
    cv::Mat det_output = detout.t();

    std::vector<cv::Rect> boxes;
    std::vector<int> classId;
    std::vector<float> confidences;

    for(auto i=0; i<det_output.rows; ++i)
    {
        cv::Mat class_score = det_output.row(i).colRange(4, det_output.cols);
        cv::Point classIdPoint;
        double scores;
        cv::minMaxLoc(class_score, nullptr, std::addressof(scores), nullptr, std::addressof(classIdPoint));

        if(scores > this->score)
        {
            float cx = det_output.at<float>(i, 0);
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);

            int x = static_cast<int>((cx-ow*0.5)*x_factor);
            int y = static_cast<int>((cy-oh*0.5)*y_factor);
            int width = static_cast<int>(ow*x_factor);
            int height = static_cast<int>(oh*y_factor);

            cv::Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;

            box &= roi;
            boxes.push_back(box);
            classId.push_back(classIdPoint.x);
            confidences.push_back(scores);
        }
    }

    //NMS处理
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, this->score, 0.5f, indexes);
    for(auto i=0; i<indexes.size(); ++i)
    {
        auto index = indexes[i];
        auto idx = classId[index];

        std::string classname;
        if(idx<0 || idx>=static_cast<int>(class_name.size()))
            continue;
        classname = this->class_name[idx];

        ODResultBox result;
        result.box = boxes[index];
        result.name = classname;
        result.score = confidences[index];

        result_boxes.push_back(result);

        cv::rectangle(frame, result.box, cv::Scalar(0, 0, 255), 2, 8);
		cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
			cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(255, 255, 255), -1);
		cv::putText(frame, result.name, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
    }

    float t = (cv::getTickCount()-start)/static_cast<float>(cv::getTickFrequency());
    cv::putText(frame, cv::format("FPS: %.2f", 1.0/t), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 3.0f, cv::Scalar(255, 0, 0));
}

YOLOv8Detector::~YOLOv8Detector()
{}