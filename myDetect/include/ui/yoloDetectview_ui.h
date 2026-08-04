#ifndef YOLOV5_DETECTOR_UI_H
#define YOLOV5_DETECTOR_UI_H

#include <QWidget>
#include <QLabel>
#include <QLineEdit>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QRadioButton> 
#include <QPushButton>
#include "algo/object_detector_thread.h"

class YOLODetectView : public QWidget 
{
	Q_OBJECT

public:
	YOLODetectView(QWidget* parent = nullptr);
	~YOLODetectView();
	void initUI();

private slots:
	void select_image();
	void select_weight_file();
	void select_config_file();
	void start_detect();
	void postProcess(cv::Mat);


private:
	QLabel* image_label;
	QLineEdit* weight_file_edit;
	QLineEdit* label_file_edit;

	QCheckBox* show_fps_chk;
	QCheckBox* show_score_chk;

	QDoubleSpinBox* score_spinbox;
	QDoubleSpinBox* config_spinbox;

	QRadioButton* video_file_rbt;
	QRadioButton* img_file_rbt;
	QLineEdit* data_file_edit;

	QPushButton* applyBtn;
	ObjectDetectorThread* workThread;
};


#endif