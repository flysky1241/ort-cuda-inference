#include "algo/inference_settings.h"

InferenceSettings::InferenceSettings(DetectAlgoType name)
	:t_score(0.25),
	conf(0.25),
	input_w(640),
	input_h(640),
	show_fps(true),
	show_score(true),
	m_name(name)
{}

std::string InferenceSettings::getWeight_file() 
{
	return this->weight_file;
}

void InferenceSettings::setWeight_file(std::string weight_file) 
{
	this->weight_file = weight_file;
}

std::string InferenceSettings::getConfig_file() 
{
	return this->config_file;
}

void InferenceSettings::setConfig_file(std::string config_file) 
{
	this->config_file = config_file;
}

float InferenceSettings::getT_score() 
{
	return this->t_score;
}

void InferenceSettings::setT_score(float value) 
{
	this->t_score = value;
}

bool InferenceSettings::getShow_fps()
{
	return this->show_fps;
}

void InferenceSettings::setShow_fps(bool value)
{
	this->show_fps = value;
}

bool InferenceSettings::getShow_score()
{
	return this->show_score;
}

void InferenceSettings::setShow_score(bool value)
{
	this->show_score = value;
}

void InferenceSettings::loadSettings()
{
	cv::FileStorage fs{};
	switch (this->m_name)
	{
	case DetectAlgoType::YOLOv5_DNN:
	{
		if (!fs.open("yolov5_det_settings.yaml", cv::FileStorage::READ)) 
		{
			std::cerr << "open yolov5_det_settings.yaml is error" << '\n';
			return;
		};
		fs["weight_file"] >> this->weight_file;
		fs["config_file"] >> this->config_file;
		fs["t_score"] >> this->t_score;
		fs["conf"] >> this->conf;
		fs["show_fps"] >> this->show_fps;
		fs["show_score"] >> this->show_score;
		break;
	};
	default: 
	{
		if (!fs.open("yolov8_det_settings.yaml", cv::FileStorage::READ))
		{
			std::cerr << "open yolov8_det_settings.yaml is error" << '\n';
			return;
		};
		fs["weight_file"] >> this->weight_file;
		fs["config_file"] >> this->config_file;
		fs["t_score"] >> this->t_score;
		fs["show_fps"] >> this->show_fps;
		fs["show_score"] >> this->show_score;
		break;
	};
	}
	fs.release();
}


void InferenceSettings::dumpSettings()
{
	cv::FileStorage fs{};
	switch (this->m_name)
	{
	case DetectAlgoType::YOLOv5_DNN:
	{
		if (!fs.open("yolov5_det_settings.yaml", cv::FileStorage::WRITE))
		{
			std::cerr << "open write yolov5_det_settings.yaml is error" << '\n';
			return;
		};
		fs<<"weight_file"<< this->weight_file;
		fs<<"config_file"<< this->config_file;
		fs<<"t_score"<< this->t_score;
		fs << "conf" << this->conf;
		fs << "show_fps" << this->show_fps;
		fs << "show_score" << this->show_score;
		break;
	};
	default:
	{
		if (!fs.open("yolov8_det_settings.yaml", cv::FileStorage::WRITE))
		{
			std::cerr << "open yolov8_det_settings.yaml is error" << '\n';
			return;
		};
		fs << "weight_file" << this->weight_file;
		fs << "config_file" << this->config_file;
		fs << "t_score" << this->t_score;
		fs << "show_fps" << this->show_fps;
		fs << "show_score" << this->show_score;
		break;
	};
	}
	fs.release();
}

void InferenceSettings::set_data_path(std::string file_path)
{
	this->data_path = file_path;
}

std::string InferenceSettings::get_data_path()
{
	return this->data_path;
}

void InferenceSettings::set_conf(float conf)
{
	this->conf = conf;
}

void InferenceSettings::set_input_w(int input_w)
{
	this->input_w = input_w;
}

void InferenceSettings::set_input_h(int input_h)
{
	this->input_h = input_h;
}

float InferenceSettings::get_conf()
{
	return this->conf;
}

int InferenceSettings::get_input_w() { return this->input_w; }

int InferenceSettings::get_input_h(){ return this->input_h; }

DetectAlgoType InferenceSettings::getName(){ return this->m_name; }


