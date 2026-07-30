#include "algo/ort_check_algo_cuda.h"
#include "cuda_runtime_api.h"
#include "onnxruntime_cxx_api.h"
#include <sstream>
#include <stdexcept>

static Ort::Env& globalOrtEnvironment()
{
    static Ort::Env environment{ORT_LOGGING_LEVEL_ERROR, "myDetect"};
    return environment;
}

void checkCuda(
    cudaError_t status, 
    const char* expression, 
    const char* file, 
    int line)
{
    if(status == cudaSuccess)
    {
        return;
    }

    std::ostringstream message;
    message<<"CUDA call failed: "<<expression
           <<", name="<<cudaGetErrorName(status)
           <<", message="<<cudaGetErrorString(status)
           <<", location="<<file<<":"<<line;
    
    throw std::runtime_error(message.str());
}

std::string boolText(bool value)
{
    return value ? "1":"0";
}


std::vector<const char*> Vstring2Char(std::vector<std::string>& other)
{
    if(other.empty())
    {
        throw std::runtime_error("std::vector<std::string> is null");
    }

    std::vector<const char*> result;
    result.reserve(other.size());

    for(const std::string& str: other)
    {
        result.push_back(str.c_str());
    }

    return result;
}

const char* str2Char(const std::string& other)
{
    const char* value = nullptr;
    value = other.c_str();
    return value;
}





