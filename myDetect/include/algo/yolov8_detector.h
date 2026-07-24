#ifndef YOLOV8_DETECTOR_H
#define YOLOV8_DETECTOR_H

#include "algo/inference_interface.h"

class YOLOv8Detector : public InferenceInterface 
{
public:
	YOLOv8Detector() = default;
	void initConfig(InferenceSettings& settings) override;
	void infer_frame(cv::Mat& frame, std::vector<ODResultBox>& boxes) override;
	~YOLOv8Detector();
private:
	cv::dnn::Net net;
	float score;
	int input_w;
	int input_h;
	bool show_fps;
	bool show_score;
	std::vector<std::string> class_name;
};

#endif
