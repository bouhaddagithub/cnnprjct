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
    int total_out = B * C * out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    int b = idx / (C * out_h * out_w);
    int rem = idx % (C * out_h * out_w);
    int c = rem / (out_h * out_w);
    int oy = (rem % (out_h * out_w)) / out_w;
    int ox = (rem % (out_h * out_w)) % out_w;

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

    output[idx] = best;
}

// =====================================================
// Wrapper Function for Pooling (callable from pipeline)
// =====================================================
inline void run_pooling_layer(const float* d_input, float* d_output,
                              int B, int C, int H, int W, int K,
                              cudaStream_t stream = 0)
{
    int total_out = B * C * (H / K) * (W / K);
    dim3 block(256);
    dim3 grid((total_out + block.x - 1) / block.x);

    maxpool_nchw_kernel<<<grid, block, 0, stream>>>(d_input, d_output, B, C, H, W, K);
    cudaCheckError(cudaGetLastError());
}
