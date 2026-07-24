#ifndef OBJECT_DETECTOR_THREAD_H
#define OBJECT_DETECTOR_THREAD_H

#include <QObJect>
#include "algo/inference_interface.h"

class ObjectDetectorThread : public QObject 
{
	Q_OBJECT
public:
	ObjectDetectorThread(InferenceSettings settings, QObject* parent=nullptr);
	~ObjectDetectorThread();

public slots:
	void do_work();

signals:
	void sendResult(cv::Mat);

private:
	std::shared_ptr<InferenceInterface> detector;
	std::string file_path;
};


#endif