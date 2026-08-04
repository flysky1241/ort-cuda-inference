#include "ui/yoloDetectview_ui.h"
#include <QtWidgets/QApplication>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    YOLODetectView window;
    window.show();
    return app.exec();
}
