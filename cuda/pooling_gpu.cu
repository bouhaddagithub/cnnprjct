// ============================================================
// pooling_gpu.cu
// Runs the pooling-only layer on GPU and measures performance.
// Loads pooling meta (kernel size) from exports/pooling_only/.
// ============================================================

#include "cuda_utils.h"
#include "pooling_layer.cuh"

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <algorithm>

// ------------------------------------------------------------
// Helper to read pooling.meta.json to extract pool_k
// Simple parser for {"kernel_size": N} format
// ------------------------------------------------------------
int read_pool_meta(const std::string &meta_path) {
    std::ifstream f(meta_path);
    if (!f.is_open()) {
        std::cerr << "⚠️ Could not open " << meta_path << ", using default pool_k=2\n";
        return 2;
    }
    
    std::string line;
    int kernel_size = 2; // default
    
    while (std::getline(f, line)) {
        // Look for "kernel_size": followed by a number
        size_t pos = line.find("\"kernel_size\"");
        if (pos != std::string::npos) {
            size_t colon_pos = line.find(":", pos);
            if (colon_pos != std::string::npos) {
                // Extract number after colon
                std::string num_str;
                for (size_t i = colon_pos + 1; i < line.length(); ++i) {
                    if (isdigit(line[i])) {
                        num_str += line[i];
                    } else if (!num_str.empty()) {
                        break;
                    }
                }
                if (!num_str.empty()) {
                    kernel_size = std::stoi(num_str);
                    break;
                }
            }
        }
    }
    
    f.close();
    return kernel_size;
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main() {
    try {
        // --- Load MNIST test set ---
        int n_images;
        auto images_u8 = load_mnist_images("data/t10k-images-idx3-ubyte", n_images);
        auto labels = load_mnist_labels("data/t10k-labels-idx1-ubyte", n_images);

        if (images_u8.empty() || labels.empty()) {
            std::cerr << "❌ MNIST test data missing.\n";
            return 1;
        }

        // --- Load pooling metadata ---
        int pool_k = read_pool_meta("exports/pooling_only/pooling.meta.json");
        std::cout << "ℹ️ Using pooling kernel size K = " << pool_k << "\n";

        // --- Dimensions ---
        int B = n_images;
        int C = 1;
        int H = 28, W = 28;
        int out_h = H / pool_k;
        int out_w = W / pool_k;
        int D = C * out_h * out_w;

        // --- Allocate device memory for all images ---
        float *d_input = nullptr;
        float *d_output = nullptr;
        size_t input_bytes = (size_t)B * C * H * W * sizeof(float);
        size_t output_bytes = (size_t)B * C * out_h * out_w * sizeof(float);

        cudaCheckError(cudaMalloc(&d_input, input_bytes));
        cudaCheckError(cudaMalloc(&d_output, output_bytes));

        // Normalize all images
        std::vector<float> h_input((size_t)B * C * H * W);
        for (int i = 0; i < B; ++i) {
            for (int p = 0; p < H * W; ++p)
                h_input[(size_t)i * H * W + p] = images_u8[i * H * W + p] / 255.0f;
        }

        cudaEvent_t ev_start, ev_end;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_end);

        cudaEventRecord(ev_start);

        // Copy all input to GPU
        cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice);

        // --- Run pooling on all images ---
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0);

        run_pooling_layer(d_input, d_output, B, C, H, W, pool_k);

        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float pool_ms = 0.0f;
        cudaEventElapsedTime(&pool_ms, t0, t1);

        cudaEventRecord(ev_end);
        cudaEventSynchronize(ev_end);
        float total_ms = 0.0f;
        cudaEventElapsedTime(&total_ms, ev_start, ev_end);

        int processed = B;

        std::cout << "✅ Pooling GPU layer completed.\n"
                  << "   Images processed: " << processed << "\n"
                  << "   Kernel size: " << pool_k << "\n"
                  << "   Total time: " << total_ms << " ms\n";

        // --- Write performance CSV ---
        std::ofstream perf("finalresults/pooling_gpu_perf.csv");
        perf << "total_pool_ms,images_processed,kernel_size\n";
        perf << total_ms << "," << processed << "," << pool_k << "\n";
        perf.close();

        // --- Cleanup ---
        cudaFree(d_input);
        cudaFree(d_output);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_end);

        return 0;

    } catch (std::exception &e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}