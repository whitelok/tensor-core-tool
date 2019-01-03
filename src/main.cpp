#include <iostream>
#include "cudnn_conv.h"

void* input_data_g_ = nullptr;
void* output_data_g_ = nullptr;
void* weight_data_g_ = nullptr;
void* cudnn_workspace_g_ = nullptr;


using namespace std;

void TestCudnnConv(int input_n, int input_c, int input_h, int input_w, 
                   int k_n, int k_c, int k_h, int k_w, 
                   int p_h, int p_w, 
                   int s_h, int s_w,
                   int d_h, int d_w,
                   int group,
                   cudnnDataType_t in_type = CUDNN_DATA_FLOAT,
                   cudnnDataType_t weight_type = CUDNN_DATA_FLOAT,
                   cudnnDataType_t out_type = CUDNN_DATA_FLOAT,
                   cudnnTensorFormat_t in_format = CUDNN_TENSOR_NCHW,
                   cudnnTensorFormat_t weight_format = CUDNN_TENSOR_NCHW,
                   cudnnTensorFormat_t output_format = CUDNN_TENSOR_NCHW) {
    CudnnConv conv(input_n, input_c, input_h, input_w,
                   k_n, k_c, k_h, k_w,
                   p_h, p_w, s_h, s_w, d_h, d_w,
                   group,
                   in_type, weight_type, out_type,
                   in_format, weight_format, output_format);
    conv.input_data(input_data_g_);
    conv.weight_data(weight_data_g_);
    conv.output_data(output_data_g_);
    conv.cudnn_workspace(cudnn_workspace_g_);
    conv.Run();
    
}

void InitData(cudnnDataType_t type_input = CUDNN_DATA_FLOAT,
              cudnnDataType_t type_weight = CUDNN_DATA_FLOAT,
              cudnnDataType_t type_output = CUDNN_DATA_FLOAT) {
    CHECK_EXIT(type_input != CUDNN_DATA_FLOAT, "only support float");
    CHECK_EXIT(type_weight != CUDNN_DATA_FLOAT, "only support float");
    CHECK_EXIT(type_output != CUDNN_DATA_FLOAT, "only support float");

    if (!input_data_g_) cudaMalloc(&input_data_g_, kMaxDataSize);
    if (!weight_data_g_) cudaMalloc(&weight_data_g_, kMaxWeightSize);
    if (!output_data_g_) cudaMalloc(&output_data_g_, kMaxDataSize);
    if (!cudnn_workspace_g_) cudaMalloc(&cudnn_workspace_g_, kMaxCudnnWorkspace_);

    cudnnCreate(&cudnn_handle_g);
}

void ReleaseData() {
    if (input_data_g_) cudaFree(input_data_g_);
    if (weight_data_g_) cudaFree(weight_data_g_);
    if (output_data_g_) cudaFree(output_data_g_);
    if (cudnn_workspace_g_) cudaFree(cudnn_workspace_g_);
    cudnnDestroy(cudnn_handle_g);
}

int main() {
    cout << "cudnn_test..........." << endl;
    InitData();
    TestCudnnConv(1, 32, 224, 224,
                  128, 32, 3, 3,
                  1, 1, 1, 1, 1, 1,
                  1);
    ReleaseData();
    return 0;
}
