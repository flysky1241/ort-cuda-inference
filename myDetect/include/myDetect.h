// myDetect.h: 标准系统包含文件的包含文件
// 或项目特定的包含文件。

#pragma once

#include <QWidget>
#include <QLabel>
#include <QLineEdit>
#include <QCheckBox>
#include <QSpinBox>
#include <QRadioButton>
#include "opencv2/core.hpp"
#include <opencv2/dnn/dnn.hpp>

class myface_detect_panel : public QWidget
{
    Q_OBJECT

public:
    myface_detect_panel(QWidget* parent = nullptr);
    ~myface_detect_panel();

protected slots:
    void select_image();
    void do_face_detect();
    void select_config_file();
    void select_weight_file();

private:
    void initUI();
    void process_frame(
        cv::Mat& frame,
        float t_score,
        bool sfps,
        bool sScore,
        cv::dnn::Net& net
    );

private:
    QLabel* m_img;      //显示图像的按钮
    QLineEdit* config_file;     //网络配置文件展示，在ONNX之前，Tensflow等 包括caffee框架都是输出两份文件，一个是权重文件，一个是网络参数文件
    QLineEdit* weight_file;     //权重文件展示

    QCheckBox* show_fps_chk;    //是否勾选 fps按钮
    QCheckBox* show_score_chk;      //把预测推理的这个样本的特征置信度是否绘制出来
    QDoubleSpinBox* score_spinbox;      //最终的 预测标签的特征置信度要大于我们的设置的分数，这样的话才可以把框子画在我们的图像上

    QRadioButton* img_file_rbt;     //选择是图像
    QRadioButton* video_file_rbt;   //选择还是视频
    QLineEdit* data_file_edit;     //展示出来这个 图像或者是视频的地址
};




// TODO: 在此处引用程序需要的其他标头。
