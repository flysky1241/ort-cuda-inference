#include "algo/yolov5_detector.h"
#include <iostream>
#include <fstream>

void YOLOv5Detector::initConfig(InferenceSettings& settings) 
{
	this->net = cv::dnn::readNetFromONNX(settings.getWeight_file());

	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

	this->score = settings.getT_score();
	this->conf = settings.get_conf();
	this->input_h = settings.get_input_h();
	this->input_w = settings.get_input_w();
	this->show_fps = settings.getShow_fps();
	this->show_score = settings.getShow_score();

	std::ifstream fp(settings.getConfig_file());
	if (!fp.is_open()) 
	{
		std::cout << "could not open file..." << '\n';
		return;
	}
	std::string name;
	while (std::getline(fp, name)) 
	{
		if(name.length())
		{
			this->class_name.push_back(name);
		}
	}
	fp.close();
}

void YOLOv5Detector::infer_frame(cv::Mat& frame, std::vector<ODResultBox>& result_boxes)
{
	int64 start = cv::getTickCount();

	auto w = frame.cols;
	auto h = frame.rows;

	auto _max = (std::max)(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(cv::Point(0, 0), cv::Size(w,h));
	cv::Rect frameBound(0, 0, frame.cols, frame.rows);
	frame.copyTo(image(roi));

	float x_factor = image.cols / 640.0f;
	float y_factor = image.rows / 640.0f;

	cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	cv::Mat pred = this->net.forward();

	cv::Mat det_output(pred.size[1], pred.size[2], CV_32FC1, pred.ptr<float>());
	
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;

	for(auto i=0; i<det_output.rows; ++i)
	{
		float confidence = det_output.at<float>(i, 4);
		if(confidence < this->conf)
		{
			continue;
		}

		cv::Mat class_scores = det_output.row(i).colRange(5, det_output.cols);
		cv::Point classIdPoint;
		double score;
		cv::minMaxLoc(class_scores, 0, &score, 0, &classIdPoint);
		float finalscore = score * confidence;

		if(finalscore>this->score)
		{
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);

			int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
			int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
			int width = static_cast<int>(ow * x_factor);
			int height = static_cast<int>(oh * y_factor);

			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;

			box &= frameBound;

			boxes.push_back(box);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(finalscore);
		}
	}

	//NMS
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, this->score, 0.5f, indexes);
	for(auto i=0; i<indexes.size(); ++i)
	{
		auto index = indexes[i];
		auto idx = classIds[i];

		std::string classname;

		if(idx>0 && idx< classIds.size())
		{
			classname = this->class_name[idx];
		}

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
	
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	cv::putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}

YOLOv5Detector::~YOLOv5Detector() 
{}