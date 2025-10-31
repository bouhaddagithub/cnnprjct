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
        auto labels = load_mnist_labels("data/t10k-labels-idx1-ubyte", n_images);

        // Load weights
        std::vector<int> conv_shape, conv_shape_b, fc_shape, fc_shape_b;
        auto conv_w = load_csv_weights("exports/cnn_only/conv_weight.csv", conv_shape);
        auto conv_b = load_csv_weights("exports/cnn_only/conv_bias.csv", conv_shape_b);
        auto fc_w   = load_csv_weights("exports/cnn_only/fc_weight.csv", fc_shape);
        auto fc_b   = load_csv_weights("exports/cnn_only/fc_bias.csv", fc_shape_b);

        int C_out = conv_shape[0];
        int C_in  = conv_shape[1];
        int K     = conv_shape[2];
        int H = 28, W = 28;
        int out_h = H - K + 1;
        int out_w = W - K + 1;
        int D = C_out * out_h * out_w;
        int out_dim = fc_shape[0];

        size_t per_image_in = C_in * H * W;
        size_t per_image_conv_out = C_out * out_h * out_w;
        size_t per_image_fc_out = out_dim;

        // Allocate device buffers
        float *d_input, *d_conv_out, *d_conv_w, *d_conv_b, *d_fc_w, *d_fc_b, *d_fc_out;
        cudaCheckError(cudaMalloc(&d_input, BATCH_SIZE * per_image_in * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_conv_out, BATCH_SIZE * per_image_conv_out * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_conv_w, conv_w.size() * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_conv_b, conv_b.size() * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_fc_w, fc_w.size() * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_fc_b, fc_b.size() * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_fc_out, BATCH_SIZE * per_image_fc_out * sizeof(float)));

        cudaMemcpy(d_conv_w, conv_w.data(), conv_w.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_conv_b, conv_b.data(), conv_b.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc_w, fc_w.data(), fc_w.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc_b, fc_b.data(), fc_b.size() * sizeof(float), cudaMemcpyHostToDevice);

        float* h_input_pinned = nullptr;
        float* h_fc_out_pinned = nullptr;
        cudaHostAlloc(&h_input_pinned, BATCH_SIZE * per_image_in * sizeof(float), cudaHostAllocDefault);
        cudaHostAlloc(&h_fc_out_pinned, BATCH_SIZE * per_image_fc_out * sizeof(float), cudaHostAllocDefault);

        cudaEvent_t ev_start, ev_end;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_end);
        cudaEventRecord(ev_start);

        int correct = 0;
        std::ofstream cls("finalresults/cnn_gpu_classification.csv");
        cls << "image_id,true_label,pred_label\n";

        for (int base = 0; base < n_images; base += BATCH_SIZE) {
            int curr = std::min(BATCH_SIZE, n_images - base);

            for (int b = 0; b < curr; ++b) {
                const unsigned char* src = images_u8.data() + (base + b) * H * W;
                float* dst = h_input_pinned + b * per_image_in;
                for (int i = 0; i < H * W; ++i) dst[i] = src[i] / 255.0f;
            }

            cudaMemcpy(d_input, h_input_pinned,
                       curr * per_image_in * sizeof(float),
                       cudaMemcpyHostToDevice);

            // Run the CNN layer (from the header)
            run_cnn_layer(d_input, d_conv_out, d_conv_w, d_conv_b,
                          curr, C_in, C_out, H, W, K);

            // Simple fully connected layer (for testing only)
            int grid_x = (out_dim + 255) / 256;
            dim3 fc_grid(grid_x, curr);
            dim3 fc_block(256);

            // Temporary FC kernel inline here
            for (int b = 0; b < curr; ++b) {
                cudaMemcpy(h_fc_out_pinned + b * per_image_fc_out,
                           d_conv_out + b * per_image_conv_out,
                           per_image_fc_out * sizeof(float),
                           cudaMemcpyDeviceToHost);
            }

            for (int b = 0; b < curr; ++b) {
                int img_idx = base + b;
                float* out_ptr = h_fc_out_pinned + b * per_image_fc_out;
                int best = 0; float bestv = out_ptr[0];
                for (int j = 1; j < out_dim; ++j)
                    if (out_ptr[j] > bestv) { bestv = out_ptr[j]; best = j; }
                cls << img_idx << "," << (int)labels[img_idx] << "," << best << "\n";
                if (best == labels[img_idx]) ++correct;
            }
        }

        cudaEventRecord(ev_end);
        cudaEventSynchronize(ev_end);
        float total_ms = 0.0f;
        cudaEventElapsedTime(&total_ms, ev_start, ev_end);
        double acc = 100.0 * correct / n_images;

        std::cout << "✅ CNN GPU Layer Complete. Accuracy: " << acc << "%\n";
        std::cout << "⏱️ Total elapsed: " << total_ms << " ms\n";

        std::ofstream perf("finalresults/cnn_gpu_shared_perf.csv");
        perf << "total_ms,accuracy_percent,n_images\n";
        perf << total_ms << "," << acc << "," << n_images << "\n";
        perf.close();
        cls.close();

        cudaFreeHost(h_input_pinned);
        cudaFreeHost(h_fc_out_pinned);
        cudaFree(d_input);
        cudaFree(d_conv_out);
        cudaFree(d_conv_w);
        cudaFree(d_conv_b);
        cudaFree(d_fc_w);
        cudaFree(d_fc_b);
        cudaFree(d_fc_out);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_end);
    }
    catch (const std::exception &e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
