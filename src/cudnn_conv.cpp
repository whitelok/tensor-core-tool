#include <iostream>
#include "basic.h"
#include "cudnn_conv.h"

using namespace std;

const size_t kMaxCudnnWorkspace = 1024 * 1024 * 10;
int kMaxDataSize                 = 1024 * 1024 * 1024;
int kMaxWeightSize               = 7 * 7 * 512 * 512;
cudnnHandle_t cudnn_handle_g;

CudnnConv::CudnnConv(int in_n, int in_c, int in_h, int in_w,
              int k_n, int k_c, int k_h, int k_w,
              int p_h, int p_w,
              int s_h, int s_w,
              int d_h, int d_w,
              int group,
              cudnnDataType_t in_type,
              cudnnDataType_t weight_type,
              cudnnDataType_t out_type,
              cudnnTensorFormat_t in_format,
              cudnnTensorFormat_t weight_format,
              cudnnTensorFormat_t output_format)
        : input_n_(in_n),input_c_(in_c), input_h_(in_h), input_w_(in_w),
          kernel_n_(k_n), kernel_c_(k_c), kernel_h_(k_h), kernel_w_(k_w),
          pad_h_(p_h), pad_w_(p_w),
          stride_h_(s_h), stride_w_(s_w),
          dilation_h_(d_h), dilation_w_(d_w),
          group_(group),
          input_type_(in_type), weight_type_(weight_type), output_type_(out_type),
          input_format_(in_format), weight_format_(weight_format), output_format_(output_format),
          input_data_(nullptr),
          weight_data_(nullptr),
          output_data_(nullptr),
          cudnn_workspace_(nullptr){
        CHECK_EXIT(group_ != 1, "only support group == 1 now");
        CHECK_EXIT(in_c != kernel_c_, "in_c != kernel_c_");
        // set algo 
        cudnnStatus_t sts;
        sts = cudnnCreateTensorDescriptor(&input_desc_);
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnCreateTensorDescriptor");
        sts = cudnnCreateTensorDescriptor(&output_desc_);
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnCreateTensorDescriptor");
        sts = cudnnCreateFilterDescriptor(&weight_desc_);
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnCreateTensorDescriptor");
        sts =cudnnCreateConvolutionDescriptor(&conv_desc_);
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnCreateConvolutionDescriptor");
        sts = cudnnSetTensor4dDescriptorEx(input_desc_,
                                           input_type_,
                                           input_n_,
                                           input_c_,
                                           input_h_,
                                           input_w_,
                                           input_c_ * input_h_ * input_w_,
                                           input_h_ * input_w_,
                                           input_w_,
                                           1);
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnSetTensor4dDescriptorEx");
        sts = cudnnSetTensor4dDescriptorEx(output_desc_,
                                           output_type_,
                                           output_n(),
                                           output_c(),
                                           output_h(),
                                           output_w(),
                                           output_c() * output_h() * output_w(),
                                           output_h() * output_w(),
                                           output_w(), 
                                           1);
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnSetTensor4dDescriptorEx");
        sts = cudnnSetConvolution2dDescriptor(conv_desc_, 
                                              pad_h_,
                                              pad_w_,
                                              stride_h_,
                                              stride_w_,
                                              dilation_h_,
                                              dilation_w_,
                                              CUDNN_CROSS_CORRELATION,
                                              conv_type());
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnSetConvolution2dDescriptor");
        sts = cudnnSetFilter4dDescriptor(weight_desc_, 
                                         weight_type_,
                                         weight_format_,
                                         kernel_n_,
                                         kernel_c_,
                                         kernel_h_,
                                         kernel_w_);


}

cudnnDataType_t CudnnConv::conv_type() {
    if ((input_type_ == CUDNN_DATA_FLOAT) &&
        (output_type_ == CUDNN_DATA_FLOAT) &&
        (weight_type_ == CUDNN_DATA_FLOAT)) {
        return CUDNN_DATA_FLOAT;
    } else {
        CHECK_EXIT(true, "conv_type not support");
    }
}

void CudnnConv::InitAlgo(cudnnHandle_t handle) {
    cudnnStatus_t sts;
    sts = cudnnGetConvolutionForwardAlgorithm(handle, 
                                              input_desc_,
                                              weight_desc_,
                                              conv_desc_,
                                              output_desc_,
                                              CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                              0,
                                              &algo_);
    CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnGetConvolutionForwardAlgorithm");
    sts = cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                  input_desc_,
                                                  weight_desc_,
                                                  conv_desc_,
                                                  output_desc_,
                                                  algo_,
                                                  &cudnn_workspace_size_);
    CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnGetConvolutionForwardWorkspaceSize");
}

void CudnnConv::Run(void* input,
                    void* weight,
                    void* output,
                    void* cudnn_workspace,
                    cudnnHandle_t handle) {
    float alpha = 1.0f;
    float beta = 1.0f;
    cudnnStatus_t sts;
    sts = cudnnConvolutionForward(handle,
                                  &alpha,
                                  input_desc_,
                                  input,
                                  weight_desc_,
                                  weight,
                                  conv_desc_,
                                  algo_,
                                  cudnn_workspace,
                                  cudnn_workspace_size_,
                                  &beta,
                                  output_desc_,
                                  output);
    CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnConvolutionForward");
}
