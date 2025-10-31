// poolinglayer.cuh
#pragma once
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void maxpool_nchw_kernel(const float* input, float* output,
                                    int B, int C, int H, int W, int K)
{
    int out_w = W / K;
    int out_h = H / K;
    int linear_idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int c = blockIdx.y;
    int b = blockIdx.z;
    if (linear_idx >= out_h * out_w) return;

    int oy = linear_idx / out_w;
    int ox = linear_idx % out_w;

    float best = -1e9f;
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            int in_y = oy * K + ky;
            int in_x = ox * K + kx;
            int in_idx = ((b * C + c) * H + in_y) * W + in_x;
            float v = input[in_idx];
            if (v > best) best = v;
        }
    }

    int out_idx = ((b * C + c) * out_h + oy) * out_w + ox;
    output[out_idx] = best;
}

// =====================================================
// Wrapper Function for Pooling (callable from pipeline)
// =====================================================
inline void run_pooling_layer(const float* d_input, float* d_output,
                              int B, int C, int H, int W, int K,
                              cudaStream_t stream = 0)
{
    int out_h = H / K;
    int out_w = W / K;
    int out_size = out_h * out_w;

    dim3 block(256);
    dim3 grid((out_size + block.x - 1) / block.x, C, B);

    maxpool_nchw_kernel<<<grid, block, 0, stream>>>(d_input, d_output, B, C, H, W, K);
    cudaCheckError(cudaGetLastError());
}
