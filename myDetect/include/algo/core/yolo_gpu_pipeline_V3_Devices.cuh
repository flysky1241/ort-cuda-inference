#ifndef YOLO_GPU_PIPELINE_V3_DEVICE_H
#define YOLO_GPU_PIPELINE_V3_DEVICE_H

namespace yolo_cuda_device {

__device__ float clampFloat(float value, float min, float max);

};
#endif
