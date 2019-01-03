#include <iostream>
#include "cudnn_conv.h"
#include "profile.h"

void* input_data_g = nullptr;
void* output_data_g = nullptr;
void* weight_data_g = nullptr;
void* cudnn_workspace_g = nullptr;

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

    //conv.input_data(input_data_g_);
    //conv.weight_data(weight_data_g_);
    //conv.output_data(output_data_g_);
    //conv.cudnn_handle(cudnn_handle_g);
    //conv.cudnn_workspace(cudnn_workspace_g_);

    conv.InitAlgo(cudnn_handle_g);

    OPT_PROFILE_TIME_START(0);
    conv.Run(input_data_g, weight_data_g, output_data_g, cudnn_workspace_g, cudnn_handle_g);
    OPT_PROFILE_TIME_STOP(0, "Run", 1, 1);
}

void InitData(cudnnDataType_t type_input = CUDNN_DATA_FLOAT,
              cudnnDataType_t type_weight = CUDNN_DATA_FLOAT,
              cudnnDataType_t type_output = CUDNN_DATA_FLOAT) {
    cout << "InitData" << endl;
    CHECK_EXIT(type_input != CUDNN_DATA_FLOAT, "only support float");
    CHECK_EXIT(type_weight != CUDNN_DATA_FLOAT, "only support float");
    CHECK_EXIT(type_output != CUDNN_DATA_FLOAT, "only support float");

    if (!input_data_g) cudaMalloc(&input_data_g, kMaxDataSize);
    if (!weight_data_g) cudaMalloc(&weight_data_g, kMaxWeightSize);
    if (!output_data_g) cudaMalloc(&output_data_g, kMaxDataSize);
    if (!cudnn_workspace_g) cudaMalloc(&cudnn_workspace_g, kMaxCudnnWorkspace);

    cudnnCreate(&cudnn_handle_g);
}

void ReleaseData() {
    cout << "ReleaseData" << endl;
    if (input_data_g) cudaFree(input_data_g);
    if (weight_data_g) cudaFree(weight_data_g);
    if (output_data_g) cudaFree(output_data_g);
    if (cudnn_workspace_g) cudaFree(cudnn_workspace_g);
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
