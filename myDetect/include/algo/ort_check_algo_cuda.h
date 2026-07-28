#ifndef ORT_CHECK_ALGO_CUDA_H
#define ORT_CHECK_ALGO_CUDA_H

#include "driver_types.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>

void checkCuda(cudaError_t status, const char* expression, const char* file, int line);

#define CUDA_CHECK(expression) \
    checkCuda((expression), #expression, __FILE__, __LINE__)

std::string boolText(bool value);

std::vector<const char*> Vstring2Char(std::vector<std::string>& other);

const char* str2Char(const std::string&);


#endif