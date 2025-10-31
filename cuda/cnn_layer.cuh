// cnn_layer.cuh - OPTIMIZED VERSION
#pragma once
#include "cuda_utils.h"
#include <cuda_runtime.h>

#define TILE_SIZE 16

// ======================================================================
// Optimized CNN kernel with shared memory
// ======================================================================
__global__ void conv_relu_kernel_shared_batched(
    const float* __restrict__ d_input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ d_conv_out,
    int C_in, int C_out, int H, int W, int K,
    int batch_stride_in, int batch_stride_conv_out)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ox = blockIdx.x * TILE_SIZE + tx;
    int oy = blockIdx.y * TILE_SIZE + ty;
    int bz = blockIdx.z;
    int batch_idx = bz / C_out;
    int oc = bz % C_out;
    
    int out_h = H - K + 1;
    int out_w = W - K + 1;
    
    if (ox >= out_w || oy >= out_h) return;
    
    // Shared memory for input tile
    __shared__ float tile[TILE_SIZE + 4][TILE_SIZE + 4];
    
    float val = bias[oc];
    const float* in_base = d_input + batch_idx * batch_stride_in;
    
    for (int ic = 0; ic < C_in; ++ic) {
        const float* in_chan = in_base + ic * H * W;
        const float* w_ptr = weight + ((oc * C_in + ic) * K * K);
        
        // Load input tile into shared memory
        for (int i = ty; i < TILE_SIZE + K - 1; i += TILE_SIZE) {
            for (int j = tx; j < TILE_SIZE + K - 1; j += TILE_SIZE) {
                int in_y = blockIdx.y * TILE_SIZE + i;
                int in_x = blockIdx.x * TILE_SIZE + j;
                
                if (in_x < W && in_y < H) {
                    tile[i][j] = in_chan[in_y * W + in_x];
                } else {
                    tile[i][j] = 0.0f;
                }
            }
        }
        __syncthreads();
        
        // Convolution using shared memory
        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                if (ty + ky < TILE_SIZE + K - 1 && tx + kx < TILE_SIZE + K - 1) {
                    val += tile[ty + ky][tx + kx] * w_ptr[ky * K + kx];
                }
            }
        }
        __syncthreads();
    }
    
    // ReLU activation
    if (val < 0.0f) val = 0.0f;
    
    // Output indexing
    int output_idx = batch_idx * batch_stride_conv_out + 
                    oc * (out_h * out_w) + 
                    oy * out_w + ox;
    d_conv_out[output_idx] = val;
}

// ======================================================================
// Wrapper function for launching the CNN kernel
// ======================================================================
inline void run_cnn_layer(float* d_input,
                          float* d_conv_out,
                          const float* d_conv_w,
                          const float* d_conv_b,
                          int B, int C_in, int C_out,
                          int H, int W, int K,
                          cudaStream_t stream = 0)
{
    int out_h = H - K + 1;
    int out_w = W - K + 1;
    int batch_stride_in = C_in * H * W;
    int batch_stride_conv_out = C_out * out_h * out_w;
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((out_w + TILE_SIZE - 1) / TILE_SIZE,
                (out_h + TILE_SIZE - 1) / TILE_SIZE,
                B * C_out);
    
    conv_relu_kernel_shared_batched<<<blocks, threads, 0, stream>>>(
        d_input, d_conv_w, d_conv_b, d_conv_out,
        C_in, C_out, H, W, K,
        batch_stride_in, batch_stride_conv_out
    );
    cudaCheckError(cudaGetLastError());
}