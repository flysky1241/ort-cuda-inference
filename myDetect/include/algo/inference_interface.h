#ifndef INFERENCE_INTERFACE_H
#define INFERENCE_INTERFACE_H

#include <opencv2/core.hpp>
#include <vector>
#include "algo/inference_settings.h"

class InferenceInterface
{
public:
	virtual void initConfig(InferenceSettings& settings) = 0;
	virtual void infer_frame(cv::Mat& frame, std::vector<ODResultBox>& boxes) = 0;

	virtual ~InferenceInterface() = default;
};

#endif 