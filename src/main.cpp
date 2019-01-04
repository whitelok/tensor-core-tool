#include <iostream>
#include <cstring>
#include <cmath>
#include "cudnn_conv.h"
#include "profile.h"

void* input_data_g = nullptr;
void* output_data_g = nullptr;
void* weight_data_g = nullptr;
void* cudnn_workspace_g = nullptr;
void* input_data_host_g = nullptr;
void* output_data_host_g = nullptr;
void* weight_data_host_g = nullptr;

#define THRESHOLD (0.001)

void BasicConv(float* output, float* input, float* weight,
               int input_n, int input_c, int input_h, int input_w,
               int k_n, int k_c, int k_h, int k_w, 
               int p_h, int p_w,
               int s_h, int s_w,
               int d_h, int d_w,
               int group) {
     
    CudnnConv conv(input_n, input_c, input_h, input_w,
                   k_n, k_c, k_h, k_w,
                   p_h, p_w, s_h, s_w, d_h, d_w,
                   group);
    //int out_n = conv.output_n();
    int out_c = conv.output_c();
    int out_h = conv.output_h();
    int out_w = conv.output_w();

    for (int c = 0; c < out_c; c++) {
        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                float sum = 0;
                for (int kc = 0; kc < k_c; kc++ ) {
                    for (int kh = 0; kh < k_h; kh++) {
                        int ih = h + kh - p_h;
                        for (int kw = 0; kw < k_w; kw++) {
                            int iw = w + kw - p_w;
                            int src_index = kc * input_h * input_w + ih * input_w + iw;
                            int weight_index = c * k_c * k_h * k_w + kc * k_h * k_w + kh * k_w + kw;
                            float src_value;
                            if ((ih >= input_h) || (ih < 0) || (iw >= input_w) || (iw < 0)) {
                                src_value = 0;
                            } else {
                                src_value = input[src_index];
                            }
                            float weight_value = weight[weight_index];
                            sum += src_value * weight_value;
                        }
                    }
                }
                int out_index = c * out_h * out_w + h * out_w + w;
                output[out_index] = sum;
            }
        }
    }
}

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
                   cudnnTensorFormat_t output_format = CUDNN_TENSOR_NCHW,
                   bool validate = true) {
    int input_size = input_n * input_c * input_h * input_w;
    CHECK_EXIT(input_size > kMaxDataSize, "input_size > kMaxDataSize");
    int weight_size = k_n * k_c * k_h * k_w;
    CHECK_EXIT(weight_size > kMaxWeightSize, "weight_size > kMaxWeightSize");

    CudnnConv conv(input_n, input_c, input_h, input_w,
                   k_n, k_c, k_h, k_w,
                   p_h, p_w, s_h, s_w, d_h, d_w,
                   group,
                   in_type, weight_type, out_type,
                   in_format, weight_format, output_format);

    int output_size = conv.output_size();
    CHECK_EXIT(output_size > kMaxDataSize, "output_size > kMaxDataSize");

    conv.InitAlgo(cudnn_handle_g);

    OPT_PROFILE_TIME_RESET(0);
    int profile_count = 100;
    OPT_PROFILE_TIME_START(0);
    for (int i = 0; i < profile_count; i++) {
        conv.Run(input_data_g, weight_data_g, output_data_g, cudnn_workspace_g, cudnn_handle_g);
    }
    OPT_PROFILE_TIME_STOP(0, "Run", profile_count, 1);

    if (validate) {
        float diff = 0.0;
        int diff_count = 0;
        float* output_host = new float[output_size];
        float* output_host_ref = new float[output_size];
        cudaMemcpy(output_host, output_data_g, output_size * sizeof(float), cudaMemcpyDeviceToHost);
        int print_count = output_size <= 100 ? output_size : 100;
        for (int i = 0; i < print_count; i++) {
            printf("%f, ", output_host[i]);
        }
        printf("\n");

#if 1
        BasicConv((float*)output_host_ref, (float*)input_data_host_g, (float*)weight_data_host_g, 
                 input_n, input_c, input_h, input_w,
                   k_n, k_c, k_h, k_w,
                   p_h, p_w, s_h, s_w, d_h, d_w,
                   group);
        for (int i = 0; i < print_count; i++) {
            printf("%f, ", output_host_ref[i]);
        }
        printf("\n");
#endif

        for (int i = 0; i < output_size; i++) {
            diff = output_host_ref[i] - output_host[i];
            if (abs(diff) > THRESHOLD) {
                diff_count++;
                //printf("diff: %f, ", diff);
            }
        }
        printf("\n");
        printf("diff_count / total, %d / %d\n", diff_count, output_size);
        delete[] output_host;
        delete[] output_host_ref;
    }
}

void Allocdata(cudnnDataType_t type_input = CUDNN_DATA_FLOAT,
               cudnnDataType_t type_weight = CUDNN_DATA_FLOAT,
               cudnnDataType_t type_output = CUDNN_DATA_FLOAT) {
    cout << "InitData" << endl;
    CHECK_EXIT(type_input != CUDNN_DATA_FLOAT, "only support float");
    CHECK_EXIT(type_weight != CUDNN_DATA_FLOAT, "only support float");
    CHECK_EXIT(type_output != CUDNN_DATA_FLOAT, "only support float");

    if (!input_data_g) cudaMalloc(&input_data_g, kMaxDataSize * sizeof(float));
    if (!weight_data_g) cudaMalloc(&weight_data_g, kMaxWeightSize * sizeof(float));
    if (!output_data_g) cudaMalloc(&output_data_g, kMaxDataSize * sizeof(float));
    if (!cudnn_workspace_g) cudaMalloc(&cudnn_workspace_g, kMaxCudnnWorkspace);
    if (!input_data_host_g) input_data_host_g = malloc(kMaxDataSize * sizeof(float));
    if (!weight_data_host_g) weight_data_host_g = malloc(kMaxWeightSize * sizeof(float));
    if (!output_data_host_g) output_data_host_g = malloc(kMaxDataSize * sizeof(float));

    for (int i = 0; i < kMaxDataSize; i++) {
        ((float*)input_data_host_g)[i] = rand() % 10;
    }
    for (int i = 0; i < kMaxWeightSize; i++) {
        ((float*)weight_data_host_g)[i] = rand() % 10 / 10.0;
    }

    cudaMemcpy(input_data_g, input_data_host_g, kMaxDataSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data_g, weight_data_host_g, kMaxWeightSize * sizeof(float), cudaMemcpyHostToDevice);
    cudnnStatus_t sts = cudnnCreate(&cudnn_handle_g);
    CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnCreate");

}

void ReleaseData() {
    cout << "ReleaseData" << endl;
    if (input_data_g) cudaFree(input_data_g);
    if (weight_data_g) cudaFree(weight_data_g);
    if (output_data_g) cudaFree(output_data_g);
    if (cudnn_workspace_g) cudaFree(cudnn_workspace_g);
    if (input_data_host_g) free(input_data_host_g);
    if (weight_data_host_g) free(weight_data_host_g);
    if (output_data_host_g) free(output_data_host_g);
    cudnnDestroy(cudnn_handle_g);
}

int main() {
    cout << "cudnn_test..........." << endl;
    Allocdata();
    TestCudnnConv(1, 128, 512, 512,
                  64, 128, 7, 7,
                  1, 1,
                  1, 1, 1, 1,
                  1,
                  CUDNN_DATA_FLOAT,
                  CUDNN_DATA_FLOAT,
                  CUDNN_DATA_FLOAT,
                  CUDNN_TENSOR_NCHW,
                  CUDNN_TENSOR_NCHW,
                  CUDNN_TENSOR_NCHW,
                  false);

    TestCudnnConv(1, 128, 512, 512,
                  64, 128, 7, 7,
                  1, 1,
                  1, 1, 1, 1,
                  1,
                  CUDNN_DATA_HALF,
                  CUDNN_DATA_HALF,
                  CUDNN_DATA_HALF,
                  CUDNN_TENSOR_NCHW,
                  CUDNN_TENSOR_NCHW,
                  CUDNN_TENSOR_NCHW,
                  false);
    ReleaseData();
    return 0;
}
