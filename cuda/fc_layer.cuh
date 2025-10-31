// fclayer.cuh
#pragma once
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <iostream>

// =============================================================
// Matrix Multiplication Kernel (Batched FC)
// =============================================================
__global__ void matmul_batch_kernel(const float* X, const float* Wt, const float* b,
                                    float* Y, int B, int K, int N)
{
    // X: (B x K), Wt: (K x N), b: (N), Y: (B x N)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B || col >= N) return;

    float acc = b ? b[col] : 0.0f;
    for (int k = 0; k < K; ++k)
        acc += X[row * K + k] * Wt[k * N + col];

    // ReLU activation integrated (optional)
    if (acc < 0.0f) acc = 0.0f;
    Y[row * N + col] = acc;
}

// =============================================================
// Wrapper Function to Launch FC Layer
// =============================================================
inline void run_fc_layer(const float* d_input, const float* d_Wt, const float* d_b,
                         float* d_output, int B, int K, int N,
                         cudaStream_t stream = 0)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (B + block.y - 1) / block.y);

    matmul_batch_kernel<<<grid, block, 0, stream>>>(d_input, d_Wt, d_b, d_output, B, K, N);
    cudaCheckError(cudaGetLastError());
}
