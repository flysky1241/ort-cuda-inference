from ultralytics import YOLO

# 1. 自动从云端下载官方训练好的 80 类基础模型 (n 代表 nano，最轻量最快的版本)
model = YOLO("yolov8s.pt") 

# 2. 极其狂暴的降维打击：将 PyTorch 模型强行转化为 C++ 能看懂的 ONNX 格式！
model.export(format="onnx")