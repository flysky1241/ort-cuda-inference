#ifndef YOLOV5_DETECTOR_H
#define YOLOV5_DETECTOR_H

#include "algo/inference_interface.h"

class YOLOv5Detector : public InferenceInterface 
{
public:
	YOLOv5Detector() = default;
	void initConfig(InferenceSettings& settings) override;
	void infer_frame(cv::Mat& frame, std::vector<ODResultBox>& boxes) override;
	~YOLOv5Detector();
private:
	cv::dnn::Net net;
	float score;
	float conf;
	int input_w;
	int input_h;
	bool show_fps;
	bool show_score;
	std::vector<std::string> class_name;
};


#endif