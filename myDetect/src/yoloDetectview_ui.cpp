#include "ui/yoloDetectview_ui.h"
#include <QPixmap>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QGridLayout>
#include "algo/inference_settings.h"
#include <QFileDialog>
#include <QThread>
#include <QMessageBox>
#include <QDebug>

YOLODetectView::YOLODetectView(QWidget* parent)
{
	this->initUI();
}

void YOLODetectView::initUI()
{
	QPixmap pixmap("D:\\Vsproject\\facedetect\\myDetect\\rcc\\lenna.png");
	this->image_label = new QLabel();
	image_label->setAlignment(Qt::AlignCenter);
	image_label->setPixmap(pixmap);
	image_label->setStyleSheet("background-color:rgb(0, 0, 0);color:red");
	image_label->setFixedSize(800, 600);

	this->data_file_edit = new QLineEdit;
	data_file_edit->setEnabled(false);

	QPushButton* selectButton = new QPushButton(QStringLiteral("选择图像"));
	this->img_file_rbt = new QRadioButton(QStringLiteral("图形"));
	this->video_file_rbt = new QRadioButton(QStringLiteral("视频"));
	img_file_rbt->setChecked(true);

	//布局
	QHBoxLayout* box1 = new QHBoxLayout;
	QGroupBox* panel1 = new QGroupBox("图形选择");
	box1->addWidget(img_file_rbt);
	box1->addWidget(video_file_rbt);
	box1->addWidget(data_file_edit);
	box1->addWidget(selectButton);
	panel1->setLayout(box1);

	this->weight_file_edit = new QLineEdit;
	this->label_file_edit = new QLineEdit;
	weight_file_edit->setEnabled(false);
	label_file_edit->setEnabled(false);
	QPushButton* weight_btn = new QPushButton(QStringLiteral("权重模型"));
	QPushButton* config_btn = new QPushButton(QStringLiteral("配置文件"));
	QGridLayout* box2 = new QGridLayout;
	QGroupBox* panel2 = new QGroupBox(QStringLiteral("模型选择"));
	box2->addWidget(new QLabel(u8"权重"), 0, 0);
	box2->addWidget(weight_file_edit, 0, 1);
	box2->addWidget(weight_btn, 0, 2);
	box2->addWidget(new QLabel(u8"配置"), 1, 0);
	box2->addWidget(label_file_edit, 1, 1);
	box2->addWidget(config_btn, 1, 2);
	panel2->setLayout(box2);

	this->show_fps_chk = new QCheckBox("FPS");
	this->show_score_chk = new QCheckBox(u8"分数");
	show_fps_chk->setChecked(true);
	show_score_chk->setChecked(true);
	score_spinbox = new QDoubleSpinBox;
	score_spinbox->setRange(0, 1);
	score_spinbox->setSingleStep(0.01);
	score_spinbox->setValue(0.25);

	config_spinbox = new QDoubleSpinBox;
	config_spinbox->setRange(0, 1);
	config_spinbox->setSingleStep(0.01);
	config_spinbox->setValue(0.25);

	QHBoxLayout* box3 = new QHBoxLayout;
	QGroupBox* panel3 = new QGroupBox(u8"其他参数选择");
	box3->addWidget(show_fps_chk);
	box3->addWidget(show_score_chk);
	box3->addWidget(new QLabel(u8"权重参数"));
	box3->addWidget(config_spinbox);
	box3->addWidget(new QLabel(u8"得分系数"));
	box3->addWidget(score_spinbox);
	panel3->setLayout(box3);

	QHBoxLayout* box4 = new QHBoxLayout;
	QGroupBox* panel4 = new QGroupBox;
	this->applyBtn = new QPushButton(QStringLiteral("运行"));
	box4->addStretch(1);
	box4->addWidget(applyBtn);
	panel4->setLayout(box4);

	QVBoxLayout* vbox = new QVBoxLayout;
	QWidget* paramPanel = new QWidget;
	vbox->addWidget(panel1);
	vbox->addWidget(panel2);
	vbox->addWidget(panel3);
	vbox->addWidget(panel4);
	vbox->addStretch(1);
	paramPanel->setLayout(vbox);

	QHBoxLayout* hbox = new QHBoxLayout;
	hbox->addWidget(paramPanel);
	hbox->addWidget(image_label);
	hbox->addStretch(1);
	this->setLayout(hbox);

	connect(selectButton, &QPushButton::clicked, this, &YOLODetectView::select_image);
	connect(weight_btn, &QPushButton::clicked, this, &YOLODetectView::select_weight_file);
	connect(config_btn, &QPushButton::clicked, this, &YOLODetectView::select_config_file);
	connect(applyBtn, &QPushButton::clicked, this, &YOLODetectView::start_detect);

	InferenceSettings settings(DetectAlgoType::YOLOv8_DNN);
	settings.loadSettings();
	this->weight_file_edit->setText(QString::fromStdString(settings.getWeight_file()));
	this->label_file_edit->setText(QString::fromStdString(settings.getConfig_file()));
	this->score_spinbox->setValue(settings.getT_score());
	this->config_spinbox->setValue(settings.get_conf());
	this->show_fps_chk->setChecked(settings.getShow_fps());
	this->show_score_chk->setChecked(settings.getShow_score());
}


void YOLODetectView::select_image()
{
	if(this->img_file_rbt->isChecked())
	{
		QString file_path = QFileDialog::getOpenFileName(this, "Open Image", "D:\\Vsproject\\facedetect\\myDetect", tr("Images(*.jpg *.jpeg *.png)"));
		qDebug()<<file_path<<'\n';
		if(file_path.isEmpty())
		{
			return;
		}
		this->data_file_edit->setText(file_path);
		QPixmap temp(file_path);

		//temp.scaled(QSize(800, 600), Qt::KeepAspectRatio);
		this->image_label->setPixmap(temp);
	}
	else if(this->video_file_rbt->isChecked())
	{
		QString file_path = QFileDialog::getOpenFileName(this, "Open video", "D:\\Vsproject\\facedetect\\myDetect", tr("Videos(*.mp4 *.avi)"));
		if(file_path.isEmpty())
		{
			return;
		}
		this->data_file_edit->setText(file_path);
	}
}


void YOLODetectView::select_weight_file() 
{
	QString file_path = QFileDialog::getOpenFileName(this, "cxz", "D:\\Vsproject\\facedetect\\myDetect", tr("Weights(*.onnx);"));
	if(file_path.isEmpty())
	{
		return;
	}
	this->weight_file_edit->setText(file_path);
}


void YOLODetectView::select_config_file() 
{
	QString file_path = QFileDialog::getOpenFileName(this, "cxz", "D:\\Vsproject\\facedetect\\myDetect", tr("Labels(*.txt);"));
	if (file_path.isEmpty())
	{
		return;
	}
	this->label_file_edit->setText(file_path);
}


void YOLODetectView::start_detect() 
{
	auto weight_file = this->weight_file_edit->text();
	auto config_file = this->label_file_edit->text();
	auto t_score = this->score_spinbox->value();
	bool show_fps = this->show_fps_chk->isChecked();
	bool show_score = this->show_score_chk->isChecked();
	auto data_file = this->data_file_edit->text();
	auto conf = this->config_spinbox->value();

	if(weight_file.isEmpty() || config_file.isEmpty() || data_file.isEmpty())
	{
		QMessageBox::warning(this, "weight_file", "weight_file is empty");
		return;
	}

	InferenceSettings settings(DetectAlgoType::YOLOv8_OCT);
	settings.setWeight_file(weight_file.toStdString());
	settings.setConfig_file(config_file.toStdString());
	settings.setT_score(t_score);
	settings.set_conf(conf);
	settings.setShow_fps(show_fps);
	settings.setShow_score(show_score);
	settings.dumpSettings();
	settings.set_data_path(data_file.toStdString());

	QThread* thread = new QThread(this);
	this->workThread = new ObjectDetectorThread(settings);
	workThread->moveToThread(thread);

	qRegisterMetaType<cv::Mat>("cv::Mat");
	connect(thread, &QThread::started, this->workThread, &ObjectDetectorThread::do_work, Qt::QueuedConnection);
	connect(workThread, &ObjectDetectorThread::sendResult, this, &YOLODetectView::postProcess);
	connect(thread, &QThread::finished, this->workThread, &ObjectDetectorThread::deleteLater);
	thread->start();
	this->applyBtn->setEnabled(false);
}

void YOLODetectView::postProcess(cv::Mat frame) 
{
	if(frame.empty())
	{
		this->applyBtn->setEnabled(true);
		return;
	}

	cv::Mat image;
	cv::cvtColor(frame, image, cv::COLOR_BGR2RGB);
	QImage img = QImage(image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
	img = img.scaled(QSize(800, 500), Qt::KeepAspectRatio);

	QPixmap pixmap;
	pixmap = pixmap.fromImage(img);
	this->image_label->setPixmap(pixmap);
}

YOLODetectView::~YOLODetectView() {}