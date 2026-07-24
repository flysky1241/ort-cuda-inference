#ifndef INFERENCE_SETTINGS_H
#define INFERENCE_SETTINGS_H

#include <opencv2/opencv.hpp>

struct ODResultBox 
{
	cv::Rect box;
	float score;
	std::string name;
};

enum class DetectAlgoType : unsigned
{
	YOLOv5_DNN,
	YOLOv8_DNN,
	YOLOv8_OV,
	YOLOv8_OCT,
	YOLOv8_TENSOR
};

class InferenceSettings 
{
public:
	InferenceSettings(DetectAlgoType name = DetectAlgoType::YOLOv5_DNN);
	std::string getWeight_file();
	void setWeight_file(std::string weight_file);
	std::string getConfig_file();
	void setConfig_file(std::string config_file);
	float getT_score();
	void setT_score(float value);
	bool getShow_fps();
	void setShow_fps(bool value);
	bool getShow_score();
	void setShow_score(bool value);
	void loadSettings();
	void dumpSettings();
	void set_data_path(std::string file_path);
	std::string get_data_path();
	void set_conf(float conf);
	void set_input_w(int input_w);
	void set_input_h(int input_h);
	float get_conf();
	int get_input_w();
	int get_input_h();
	DetectAlgoType getName();

private:
	std::string weight_file;
	std::string config_file;
	float t_score;
	float conf;
	int input_w;
	int input_h;
	bool show_fps;
	bool show_score;
	std::string data_path;
	DetectAlgoType m_name;
};

#endif