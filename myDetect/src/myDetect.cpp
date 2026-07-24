// myDetect.cpp: 定义应用程序的入口点。
//

#include "myDetect.h"
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <QFile>
#include <QByteArray>
#include <windows.h>


myface_detect_panel::myface_detect_panel(QWidget* parent)
    : QWidget(parent)
{
    this->initUI();
}


myface_detect_panel::~myface_detect_panel()
{}


void myface_detect_panel::do_face_detect()
{
    //获取响应参数
    QString weight = this->weight_file->text();
    QString config = this->config_file->text();
    auto t_score = this->score_spinbox->value();
    bool show_fps_status = this->show_fps_chk->isChecked();
    bool show_score_status = this->show_score_chk->isChecked();
    QString data = this->data_file_edit->text();

    qDebug() << "data=:" << data << '\n';

    //设置是否 开始运行
    QMessageBox::StandardButton retBtn = QMessageBox::question(this, u8"选择", u8"你确定要运行人脸检测吗?",
        QMessageBox::StandardButton::Yes | QMessageBox::StandardButton::No);
    if (retBtn == QMessageBox::StandardButton::No)
    {
        qDebug() << u8"不选择" << '\n';
        return;
    }

    if (weight.isEmpty() || config.isEmpty() || data.isEmpty())
    {
        QMessageBox::warning(this, u8"警告", u8"模型参数没配置");
        return;
    }

    //把net 模型加载好，这里使用 cv::dnn::readNetTensorflow 来读取这个 老式的别人训练好的 网络参数文件 和 权重文件
    //权重文件是纯粹的数字，但是模型并不知道 那个参数是输入层的那个是输出层的，所以需要一个网络配置文件
    //在今天的发展来讲 pytorch 使用的都是 ONNX模型，只需要一个就可以了。
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(weight.toStdString(), config.toStdString());
    if (this->img_file_rbt->isChecked())
    {
        //cv::Mat img = cv::imread(this->data_file_edit->text().toStdString(), cv::IMREAD_COLOR);
        //opencv 的 imread对于直接读取这个 中文路径是支持很差的，所以当时在这里 调试了一段时间
        //解决方案：1 通过toLocal8Bit
        //解决方案2：通过QFile 来直接读取这个二进制数据，然后通过vector承载再通过 cv::imdecode来直接读取
        //不论是jpg 还是 png，其二进制数据 和 elf文件一样，开头都是带了这个 行列个数的，所以vector承载之后，cv::imdecode可以直接读取
        QFile file(this->data_file_edit->text());
        if (!file.open(QIODevice::ReadOnly))
        {
            qDebug() << "file open is error" << '\n';
            return;
        }
        QByteArray arr = file.readAll();
        file.close();

        std::vector<uchar> buffer(arr.begin(), arr.end());
        cv::Mat img = cv::imdecode(buffer, cv::IMREAD_COLOR);

        if (img.empty())
        {
            qDebug() << "img is empty" << '\n';
        }
        this->process_frame(img, t_score, show_fps_status, show_score_status, net);
    }

    if (this->video_file_rbt->isChecked())
    {
        //cv::VideoCapture cap(this->data_file_edit->text().toLocal8Bit().toStdString());

        //刚才是图像，这个是视频，这个是不能通过 imdecode来解决的，所以方案2个
        //1：通过这个 toLocal8Bit
        //2：通过 windows底层的这个 短路径，也就是windows为了适配全球的所有的 阿拉伯文也好，中文也好，整出来的 宽字符，
        //以及 任何非英文路径都适配了一个宽字符ASCII路径，通过 GetShortPathNameW 来提取，这样也是可以百分百准确读取任何非英文路径。

        // 1. 先把路径里的 / 洗成 Windows 原生的 \ 、、
        QString localStr = QDir::toNativeSeparators(this->data_file_edit->text());

        // 2. 转换成 Windows 底层 API 需要的宽字符 (UTF-16)
        std::wstring wstr = localStr.toStdWString();

        // 3. 准备接收纯英文“短路径”的缓冲区
        wchar_t shortPathBuffer[MAX_PATH];

        //4. 召唤 Windows 物理欺骗魔法！获取 8.3 纯 ASCII 短路径！
        ::GetShortPathNameW(wstr.data(), shortPathBuffer, MAX_PATH);

        // 5. 转回 std::string
        QString strShort = QString::fromStdWString(shortPathBuffer);
        cv::VideoCapture cap(strShort.toStdString());

        if (!cap.isOpened())
        {
            qDebug() << u8"错误" << '\n';
            return;
        }
        cv::Mat frame;
        while (true)
        {
            //读取每一帧，每一帧也就是一个Mat 来专门处理
            cap.read(frame);
            if (frame.empty())
            {
                break;
            }
            this->process_frame(frame, t_score, show_fps_status, show_score_status, net);
            cv::waitKey(30);
        }
    }

    QMessageBox::information(this, u8"检测", u8"人脸运行完成");
}


void myface_detect_panel::select_config_file()
{
    QString filename = QFileDialog::getOpenFileName(this, u8"选择配置文件", "D:/Vsproject/Opencv/opencv1/myface_detect_panel", tr("Config(*.pbtxt);"));
    if (filename.isEmpty())
    {
        qDebug() << "is error" << '\n';
        return;
    }
    this->config_file->setText(filename);
}


void myface_detect_panel::select_weight_file()
{
    QString filename = QFileDialog::getOpenFileName(this, u8"选择权重文件", "D:/Vsproject/Opencv/opencv1/myface_detect_panel", tr("Weights(*.pb);"));
    if (filename.isEmpty())
    {
        qDebug() << "is error" << '\n';
        return;
    }
    this->weight_file->setText(filename);
}


void myface_detect_panel::initUI()
{
    QPixmap pixmap("D:/QTWORK/vscodeOpencv/lenna.png");
    this->m_img = new QLabel;
    this->m_img->setAlignment(Qt::AlignHCenter);
    m_img->setStyleSheet("background-color:rgb(0, 0, 0);color:red");
    m_img->setPixmap(pixmap);
    m_img->setFixedSize(800, 550);

    this->data_file_edit = new QLineEdit;
    data_file_edit->setEnabled(false);
    QPushButton* selectBtn = new QPushButton;
    selectBtn->setText(u8"选择图像");

    this->img_file_rbt = new QRadioButton(u8"图像文件");
    this->video_file_rbt = new QRadioButton(u8"视频文件");

    img_file_rbt->setChecked(true);

    //这里的这个QGroupBox 可以有虚框那种展示，所以是比原生的QWidget要更好看了些
    QHBoxLayout* hBox = new QHBoxLayout;
    QGroupBox* panel1 = new QGroupBox(this);
    hBox->addWidget(img_file_rbt);
    hBox->addWidget(video_file_rbt);
    hBox->addWidget(data_file_edit);
    hBox->addWidget(selectBtn);
    panel1->setLayout(hBox);
    panel1->setTitle(u8"数据选择");

    this->weight_file = new QLineEdit;
    this->config_file = new QLineEdit;
    weight_file->setEnabled(false);
    config_file->setEnabled(false);
    QPushButton* weight_btn = new QPushButton(u8"选择");
    QPushButton* config_btn = new QPushButton(u8"选择");
    QHBoxLayout* hBox2 = new QHBoxLayout;
    QGroupBox* panel2 = new QGroupBox;
    hBox2->addWidget(new QLabel(u8"权重文件"));
    hBox2->addWidget(this->weight_file);
    hBox2->addWidget(weight_btn);
    hBox2->addWidget(new QLabel(u8"配置网络文件"));
    hBox2->addWidget(this->config_file);
    hBox2->addWidget(config_btn);
    panel2->setLayout(hBox2);

    this->show_fps_chk = new QCheckBox("FPS");
    this->show_score_chk = new QCheckBox(u8"置信度");
    this->score_spinbox = new QDoubleSpinBox;
    score_spinbox->setRange(0, 1);
    score_spinbox->setSingleStep(0.01);
    score_spinbox->setValue(0.5);
    QHBoxLayout* hBox3 = new QHBoxLayout;
    QGroupBox* panel3 = new QGroupBox;
    hBox3->addWidget(show_fps_chk);
    hBox3->addWidget(show_score_chk);
    hBox3->addWidget(new QLabel(u8"得分"));
    hBox3->addWidget(score_spinbox);
    panel3->setLayout(hBox3);

    QHBoxLayout* hBox4 = new QHBoxLayout;
    QWidget* panel4 = new QWidget;
    QPushButton* applyBtn = new QPushButton;
    applyBtn->setText(u8"运行");
    hBox4->addStretch(1);
    hBox4->addWidget(applyBtn);
    panel4->setLayout(hBox4);

    QVBoxLayout* v_box = new QVBoxLayout;
    v_box->addWidget(panel1);
    v_box->addWidget(panel2);
    v_box->addWidget(panel3);
    v_box->addWidget(panel4);
    v_box->addWidget(m_img);
    this->setLayout(v_box);

    connect(selectBtn, QOverload<bool>::of(&QPushButton::clicked), this, QOverload<>::of(&myface_detect_panel::select_image));
    connect(weight_btn, QOverload<bool>::of(&QPushButton::clicked), this, QOverload<>::of(&myface_detect_panel::select_weight_file));
    connect(config_btn, QOverload<bool>::of(&QPushButton::clicked), this, QOverload<>::of(&myface_detect_panel::select_config_file));
    connect(applyBtn, QOverload<bool>::of(&QPushButton::clicked), this, QOverload<>::of(&myface_detect_panel::do_face_detect));
}

void myface_detect_panel::process_frame(cv::Mat& frame, float t_score, bool sfps, bool sScore, cv::dnn::Net& net)
{
    //blob的本意英文是 二进制大数据，这里是 把我们的opencv 图像 HWC转化为 MLP这一类模型专门的 4D 4维 BCHW 这种，然后把他送到模型里面推理
    cv::Mat blob;
    //这个主要是来计算FPS的
    int64 startTime = cv::getTickCount();
    if (frame.empty())
    {
        qDebug() << "frame is empty" << '\n';
    }
    try
    {
        //把我们的opencv 图像 HWC转化为 MLP这一类模型专门的 4D 4维 BCHW 这种，然后把他送到模型里面推理。这就是预处理
        //这里没有采用归一化，而是均值相减，因为当年这个caffee模型训练的时候就是这么写的，所以我们做推理自然也必须遵守。
        blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104, 177, 123), false, false);
    }
    catch (std::exception& err)
    {
        qDebug() << err.what() << '\n';
    }
    //cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104, 177, 123), false, false);
    qDebug() << blob.size[0] << '\n';
    qDebug() << blob.size[1] << '\n';
    qDebug() << blob.size[2] << '\n';
    qDebug() << blob.size[3] << '\n';

    //把预处理好的 投入模型里
    net.setInput(blob);

    //前向处理输出，推理阶段
    //这个模型输出形状通常是：1 × 1 × N × 7
    //每个检测框有 7 个数字：[image_id, class_id, confidence, x1, y1, x2, y2]
    //第0列：image_id
    //第1列：class_id
    //第2列：confidence 置信度
    //第3列：x1 左上角 x
    //第4列：y1 左上角 y
    //第5列：x2 右下角 x
    //第6列：y2 右下角 y
    //不是像素坐标，而是归一化坐标，范围大概是：
    cv::Mat probs = net.forward();

    //于是 detectMat 就变成：200 行 × 7 列
    //每一行是一个检测框。每一列是这个检测框的某个属性
    cv::Mat detectMat(probs.size[2], probs.size[3], CV_32FC1, probs.data);
    for (auto row = 0; row < detectMat.rows; ++row)
    {
        //取第 2 列，也就是：置信度
        float conf = detectMat.at<float>(row, 2);
        //置信度如果是大于我们的 想要的分数的话，那就框出来
        if (conf > t_score)
        {
            //归一化坐标恢复成原图坐标
            float x1 = (std::max)(0, (std::min)(static_cast<int>(detectMat.at<float>(row, 3) * frame.cols), frame.cols - 1));
            float y1 = (std::max)(0, (std::min)(static_cast<int>(detectMat.at<float>(row, 4) * frame.rows), frame.rows - 1));
            float x2 = (std::max)(0, (std::min)(static_cast<int>(detectMat.at<float>(row, 5) * frame.cols), frame.cols - 1));
            float y2 = (std::max)(0, (std::min)(static_cast<int>(detectMat.at<float>(row, 6) * frame.rows), frame.rows - 1));

            cv::Rect rect(cv::Point(x1, y1), cv::Point(x2, y2));
            cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2);
            if (sScore)
            {
                //显示置信度
                cv::putText(frame, cv::format("%.2f", conf), cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2);
            }
        }
    }
    if (sfps)
    {
        //计算FPS，然后显示出来
        float t = (cv::getTickCount() - startTime) / cv::getTickFrequency();
        cv::putText(frame, cv::format("FPS:%.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2);
    }
    //最后就是把计算好的这个 cv::Mat 展示到这个Qt上
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    QImage img(frame.data, frame.cols, frame.rows, static_cast<int>(frame.step), QImage::Format_RGB888);
    this->m_img->setPixmap(QPixmap::fromImage(img.copy()));
}


void myface_detect_panel::select_image()
{
    //这个就没啥说的 就是对话框的选择相应的图像和视频
    if (this->img_file_rbt->isChecked())
    {
        QString filename = QFileDialog::getOpenFileName(this, "Open image", "D:/Vsproject/Opencv/opencv1/myface_detect_panel", tr("Images(*.png *.jpg);"));
        if (filename.isEmpty())
        {
            qDebug() << "is error" << '\n';
            return;
        }
        this->data_file_edit->setText(filename);
        QPixmap img(filename);
        this->m_img->setPixmap(img);
    }
    if (this->video_file_rbt->isChecked())
    {
        QString filename = QFileDialog::getOpenFileName(this, "Open image", "D:/Vsproject/Opencv/opencv1/myface_detect_panel", tr("Videos(*.mp4);"));
        if (filename.isEmpty())
        {
            qDebug() << "is error" << '\n';
            return;
        }
        this->data_file_edit->setText(filename);
    }
}

