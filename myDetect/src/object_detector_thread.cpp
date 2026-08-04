#include "algo/object_detector_thread.h"
#include <QDebug>
#include <QThread>
#include <exception>
#include <iostream>
//#include "algo/ort_detector.h"
#include "algo/yolov5_detector.h"
#include "algo/yolov8_detector.h"
#include "onnxruntime_cxx_api.h"
#include "algo/ort_detector_V2.h"

ObjectDetectorThread::ObjectDetectorThread(InferenceSettings settings, QObject* parent)
	:QObject(parent)
{
	qDebug()<< "ObjectDetectorThread construct is init" << QThread::currentThreadId() << '\n';
	switch (settings.getName())
	{
	case DetectAlgoType::YOLOv5_DNN: 
	{
		this->detector = std::make_shared<YOLOv5Detector>();
		detector->initConfig(settings);
		break;
	}

	case DetectAlgoType::YOLOv8_DNN: 
	{
		this->detector = std::make_shared<YOLOv8Detector>();
		detector->initConfig(settings);
		break;
	}

	case DetectAlgoType::YOLOv8_OCT: 
	{
		try
		{
			this->detector = std::make_shared<ORTDetector_V2>(OrtBackend::TRT);
			detector->initConfig(settings);
			break;
		}
		catch(Ort::Exception& excepted)
		{
			std::cerr<<"Ort initConfig failed:"<<excepted.what()<<'\n';
		}
		catch(cv::Exception& exception)
		{
			std::cerr<<"cv initConfig failed:"<<exception.what()<<'\n';
		}
		catch(std::exception& excepted)
		{
			std::cerr<<"std initConfig failed:"<<excepted.what()<<'\n';
		}
	}

	default:
		break;
	}

	this->file_path = settings.get_data_path();

}


ObjectDetectorThread::~ObjectDetectorThread()
{}

void ObjectDetectorThread::do_work()
{
	qDebug() << "do_work is init" << QThread::currentThreadId() << '\n';
	QString path = QString::fromStdString(file_path);
	std::vector<ODResultBox> boxes;
	
	if(path.endsWith(".jpg")||path.endsWith(".png"))
	{
		try
		{
			cv::Mat img = cv::imread(file_path);
			detector->infer_frame(img, boxes);
			emit this->sendResult(img);
		}
		catch(Ort::Exception& excepted)
		{
			std::cerr<<excepted.what()<<'\n';
		}
		catch(std::exception& excepted)
		{
			std::cerr<<"std run failed:"<<excepted.what()<<'\n';
		}
	}
	else if(path.endsWith(".mp4"))
	{
		cv::VideoCapture capture(this->file_path);
		cv::Mat frame;
		while(true)
		{
			boxes.clear();
			capture >> frame;
			if (frame.empty())
				break;
			detector->infer_frame(frame, boxes);
			emit this->sendResult(frame);
		}
	}
	else
	{
		cv::Mat empty_img;
		emit sendResult(empty_img);
	}
}
