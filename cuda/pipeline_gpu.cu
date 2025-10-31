#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define BATCH_SIZE 64     // Tune: 32–128 is ideal for MNIST
#define THREADS 256       // for FC layer
#define CHECK(call) { cudaError_t e = call; if(e != cudaSuccess){printf("CUDA Error: %s\n", cudaGetErrorString(e)); exit(1);} }

// ======================================================================
// Convolution Kernel (Simplified, shared memory optimized)
// ======================================================================
__global__ void conv_forward_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weights,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int C_in, int C_out,
                                    int H, int W, int K) {
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_c = blockIdx.z;

    int out_h = H - K + 1;
    int out_w = W - K + 1;
    if (out_y >= out_h || out_x >= out_w) return;

    float sum = 0.0f;
    int kernel_size = K * K;
    for (int c = 0; c < C_in; ++c) {
        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                int in_idx = ((c * H) + (out_y + ky)) * W + (out_x + kx);
                int w_idx = ((out_c * C_in + c) * K + ky) * K + kx;
                sum += input[in_idx] * weights[w_idx];
            }
        }
    }
    output[(out_c * out_h + out_y) * out_w + out_x] = sum + bias[out_c];
}

// ======================================================================
// FC Layer Kernel (Matrix Multiply style)
// ======================================================================
__global__ void fc_forward_kernel(const float* __restrict__ input,
                                  const float* __restrict__ W,
                                  const float* __restrict__ b,
                                  float* __restrict__ output,
                                  int in_size, int out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;
    float sum = b[idx];
    for (int i = 0; i < in_size; ++i)
        sum += input[i] * W[idx * in_size + i];
    output[idx] = sum;
}

// ======================================================================
// Helper: Normalization of MNIST bytes to float
// ======================================================================
void normalize_image(const unsigned char* src, float* dst, int size) {
    for (int i = 0; i < size; ++i) dst[i] = src[i] / 255.0f;
}

// ======================================================================
// MAIN GPU PIPELINE
// ======================================================================
int main() {
    try {
        std::cout << "\n🚀 Optimized GPU Pipeline (Batched + Async)\n";

        // Load MNIST test data
        int img_count = 0, lbl_count = 0;
        auto imgs_raw = load_mnist_images("../../data/t10k-images-idx3-ubyte", img_count);
        auto labels = load_mnist_labels("../../data/t10k-labels-idx1-ubyte", lbl_count);
        int H = 28, W = 28, C_in = 1;
        img_count = std::min(img_count, lbl_count);

        // Load pretrained parameters
        std::vector<int> s1, s2, s3, s4;
        auto conv_w = load_csv_weights("../../exports/conv_weights.csv", s1);
        auto conv_b = load_csv_weights("../../exports/conv_bias.csv", s2);
        auto fc_w   = load_csv_weights("../../exports/fc_weights.csv", s3);
        auto fc_b   = load_csv_weights("../../exports/fc_bias.csv", s4);

        int C_out = s1[0]; // number of filters
        int K = s1[2];
        int out_h = H - K + 1, out_w = W - K + 1;
        int fc_in = C_out * out_h * out_w;
        int fc_out = s3[0];

        // Device buffers
        float *d_input, *d_conv_w, *d_conv_b, *d_conv_out;
        float *d_fc_w, *d_fc_b, *d_fc_out;
        CHECK(cudaMalloc(&d_conv_w, conv_w.size() * sizeof(float)));
        CHECK(cudaMalloc(&d_conv_b, conv_b.size() * sizeof(float)));
        CHECK(cudaMalloc(&d_fc_w, fc_w.size() * sizeof(float)));
        CHECK(cudaMalloc(&d_fc_b, fc_b.size() * sizeof(float)));

        CHECK(cudaMemcpy(d_conv_w, conv_w.data(), conv_w.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_conv_b, conv_b.data(), conv_b.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_fc_w, fc_w.data(), fc_w.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_fc_b, fc_b.data(), fc_b.size() * sizeof(float), cudaMemcpyHostToDevice));

        // Allocate device work buffers
        CHECK(cudaMalloc(&d_input, BATCH_SIZE * C_in * H * W * sizeof(float)));
        CHECK(cudaMalloc(&d_conv_out, BATCH_SIZE * C_out * out_h * out_w * sizeof(float)));
        CHECK(cudaMalloc(&d_fc_out, BATCH_SIZE * fc_out * sizeof(float)));

        // Pinned host memory for DMA
        float* h_input;
        CHECK(cudaHostAlloc(&h_input, BATCH_SIZE * C_in * H * W * sizeof(float), cudaHostAllocDefault));
        float* h_output = new float[BATCH_SIZE * fc_out];

        // Streams
        const int NUM_STREAMS = 4;
        cudaStream_t streams[NUM_STREAMS];
        for (int i = 0; i < NUM_STREAMS; ++i) cudaStreamCreate(&streams[i]);

        // Timer
        auto start = std::chrono::high_resolution_clock::now();

        // Process MNIST test set
        int correct = 0;
        for (int base = 0; base < img_count; base += BATCH_SIZE) {
            int current_batch = std::min(BATCH_SIZE, img_count - base);
            int sid = base / BATCH_SIZE % NUM_STREAMS;
            cudaStream_t s = streams[sid];

            // Prepare batch input (pinned memory)
            for (int i = 0; i < current_batch; ++i)
                normalize_image(&imgs_raw[(base + i) * H * W],
                                &h_input[i * H * W], H * W);

            // Async transfer to device
            CHECK(cudaMemcpyAsync(d_input, h_input,
                                  current_batch * H * W * sizeof(float),
                                  cudaMemcpyHostToDevice, s));

            // Launch convolution
            dim3 threads(16, 16);
            dim3 blocks((out_w + 15) / 16, (out_h + 15) / 16, C_out);
            conv_forward_kernel<<<blocks, threads, 0, s>>>(d_input, d_conv_w, d_conv_b,
                                                           d_conv_out, C_in, C_out, H, W, K);

            // Flatten conv_out & FC
            int grid_fc = (fc_out + THREADS - 1) / THREADS;
            fc_forward_kernel<<<grid_fc, THREADS, 0, s>>>(d_conv_out, d_fc_w, d_fc_b, d_fc_out, fc_in, fc_out);

            // Async copy result back
            CHECK(cudaMemcpyAsync(h_output, d_fc_out,
                                  current_batch * fc_out * sizeof(float),
                                  cudaMemcpyDeviceToHost, s));

            // Compute predictions (CPU side, after stream sync)
            cudaStreamSynchronize(s);
            for (int i = 0; i < current_batch; ++i) {
                int best = 0; float maxv = h_output[i * fc_out];
                for (int j = 1; j < fc_out; ++j) {
                    float v = h_output[i * fc_out + j];
                    if (v > maxv) { maxv = v; best = j; }
                }
                if (best == labels[base + i]) correct++;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "\n✅ GPU Accuracy: " << (correct * 100.0 / img_count) << "%";
        std::cout << "\n⏱️ Runtime: " << time_ms / 1000.0 << " seconds\n";

        // Cleanup
        for (int i = 0; i < NUM_STREAMS; ++i) cudaStreamDestroy(streams[i]);
        cudaFreeHost(h_input);
        delete[] h_output;
        cudaFree(d_input);
        cudaFree(d_conv_out);
        cudaFree(d_fc_out);
        cudaFree(d_conv_w);
        cudaFree(d_conv_b);
        cudaFree(d_fc_w);
        cudaFree(d_fc_b);
    }
    catch (std::exception &e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
