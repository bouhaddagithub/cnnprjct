// cnn_gpu_benchmark.cu
#include "cuda_utils.h"
#include "cnn_layer.cuh"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

#ifndef BATCH_SIZE
#define BATCH_SIZE 128
#endif

int main() {
    try {
        int n_images;
        auto images_u8 = load_mnist_images("data/t10k-images-idx3-ubyte", n_images);
        if (images_u8.empty()) {
            std::cerr << "MNIST test images missing.\n";
            return 1;
        }

        // Load CNN weights
        std::vector<int> conv_shape, conv_shape_b;
        auto conv_w = load_csv_weights("exports/cnn_only/conv_weight.csv", conv_shape);
        auto conv_b = load_csv_weights("exports/cnn_only/conv_bias.csv", conv_shape_b);

        int C_out = conv_shape[0];
        int C_in  = conv_shape[1];
        int K     = conv_shape[2];
        int H = 28, W = 28;
        int out_h = H - K + 1;
        int out_w = W - K + 1;

        size_t per_image_in  = C_in * H * W;
        size_t per_image_out = C_out * out_h * out_w;

        // Allocate device memory
        float *d_input, *d_conv_out, *d_conv_w, *d_conv_b;
        cudaCheckError(cudaMalloc(&d_input, BATCH_SIZE * per_image_in * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_conv_out, BATCH_SIZE * per_image_out * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_conv_w, conv_w.size() * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_conv_b, conv_b.size() * sizeof(float)));

        cudaCheckError(cudaMemcpy(d_conv_w, conv_w.data(), conv_w.size() * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_conv_b, conv_b.data(), conv_b.size() * sizeof(float), cudaMemcpyHostToDevice));

        // Pinned host buffer for input
        float* h_input_pinned = nullptr;
        cudaCheckError(cudaHostAlloc(&h_input_pinned, BATCH_SIZE * per_image_in * sizeof(float), cudaHostAllocDefault));

        // Events for timing
        cudaEvent_t ev_start, ev_end;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_end);

        cudaEventRecord(ev_start);

        for (int base = 0; base < n_images; base += BATCH_SIZE) {
            int curr = std::min(BATCH_SIZE, n_images - base);

            // Copy & normalize batch into pinned memory
            for (int b = 0; b < curr; ++b) {
                const unsigned char* src = images_u8.data() + (base + b) * H * W;
                float* dst = h_input_pinned + b * per_image_in;
                for (int i = 0; i < H * W; ++i) dst[i] = src[i] / 255.0f;
            }

            // H2D
            cudaCheckError(cudaMemcpy(d_input, h_input_pinned, curr * per_image_in * sizeof(float), cudaMemcpyHostToDevice));

            // Run CNN only
            run_cnn_layer(d_input, d_conv_out, d_conv_w, d_conv_b,
                          curr, C_in, C_out, H, W, K);
        }

        cudaEventRecord(ev_end);
        cudaEventSynchronize(ev_end);
        float total_ms = 0.0f;
        cudaEventElapsedTime(&total_ms, ev_start, ev_end);

        std::cout << "✅ CNN GPU Benchmark Complete.\n";
        std::cout << "⏱️ Total time for " << n_images << " images: " << total_ms << " ms\n";
        std::cout << "   Avg per image: " << total_ms / n_images << " ms\n";

        std::ofstream perf("finalresults/cnn_gpu_benchmark.csv");
        perf << "total_ms,n_images,avg_per_image_ms\n";
        perf << total_ms << "," << n_images << "," << total_ms / n_images << "\n";
        perf.close();

        // Cleanup
        cudaFreeHost(h_input_pinned);
        cudaFree(d_input);
        cudaFree(d_conv_out);
        cudaFree(d_conv_w);
        cudaFree(d_conv_b);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_end);

    } catch (const std::exception &e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
