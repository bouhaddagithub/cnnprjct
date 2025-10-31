#include "cuda_utils.h"
#include "fc_layer.cuh"
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

int main() {
    try {
        int n_images;
        auto images_u8 = load_mnist_images("data/t10k-images-idx3-ubyte", n_images);
        auto labels = load_mnist_labels("data/t10k-labels-idx1-ubyte", n_images);

        std::vector<int> s_fc1w, s_fc1b, s_fc2w, s_fc2b;
        auto fc1_w = load_csv_weights("exports/fc_only/fc1_weight.csv", s_fc1w);
        auto fc1_b = load_csv_weights("exports/fc_only/fc1_bias.csv", s_fc1b);
        auto fc2_w = load_csv_weights("exports/fc_only/fc2_weight.csv", s_fc2w);
        auto fc2_b = load_csv_weights("exports/fc_only/fc2_bias.csv", s_fc2b);

        // Validate shapes
        if (s_fc1w.size() < 2 || s_fc2w.size() < 2)
            throw std::runtime_error("Invalid FC weight shapes.");
        if (s_fc1b.empty() || s_fc2b.empty())
            throw std::runtime_error("Invalid FC bias shapes.");

        int hidden = s_fc1w[0];
        int K = s_fc1w[1];
        int out_dim = s_fc2w[0];
        int B = n_images;

        // Prepare host input (normalized)
        std::vector<float> h_X((size_t)B * K);
        for (int i = 0; i < B; ++i)
            for (int p = 0; p < K; ++p)
                h_X[i * K + p] = images_u8[i * 28 * 28 + p] / 255.0f;

        // Transpose weights for GPU (so Wt = K x N)
        std::vector<float> fc1_wt(K * hidden), fc2_wt(hidden * out_dim);
        for (int i = 0; i < hidden; ++i)
            for (int j = 0; j < K; ++j)
                fc1_wt[j * hidden + i] = fc1_w[i * K + j];

        for (int i = 0; i < out_dim; ++i)
            for (int j = 0; j < hidden; ++j)
                fc2_wt[j * out_dim + i] = fc2_w[i * hidden + j];

        // Device allocations
        float *d_X, *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b, *d_hidden, *d_out;
        cudaCheckError(cudaMalloc(&d_X, B * K * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_fc1_w, K * hidden * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_fc1_b, hidden * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_fc2_w, hidden * out_dim * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_fc2_b, out_dim * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_hidden, B * hidden * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_out, B * out_dim * sizeof(float)));

        // Copy weights to GPU
        cudaCheckError(cudaMemcpy(d_X, h_X.data(), B * K * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_fc1_w, fc1_wt.data(), K * hidden * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_fc1_b, fc1_b.data(), hidden * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_fc2_w, fc2_wt.data(), hidden * out_dim * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_fc2_b, fc2_b.data(), out_dim * sizeof(float), cudaMemcpyHostToDevice));

        // Timers
        cudaEvent_t ev_start, ev_end;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_end);

        cudaEventRecord(ev_start);

        // FC1 Forward
        run_fc_layer(d_X, d_fc1_w, d_fc1_b, d_hidden, B, K, hidden);

        // FC2 Forward (after ReLU integrated)
        run_fc_layer(d_hidden, d_fc2_w, d_fc2_b, d_out, B, hidden, out_dim);

        cudaEventRecord(ev_end);
        cudaEventSynchronize(ev_end);
        float total_ms = 0.0f;
        cudaEventElapsedTime(&total_ms, ev_start, ev_end);

        // Accuracy computation
        std::vector<float> h_out(B * out_dim);
        cudaMemcpy(h_out.data(), d_out, B * out_dim * sizeof(float), cudaMemcpyDeviceToHost);

        int correct = 0;
        for (int i = 0; i < B; ++i) {
            int best = 0;
            float bestv = h_out[i * out_dim];
            for (int j = 1; j < out_dim; ++j)
                if (h_out[i * out_dim + j] > bestv) { bestv = h_out[i * out_dim + j]; best = j; }
            if (best == labels[i]) correct++;
        }
        double acc = 100.0 * correct / B;

        // Save perf results
        std::ofstream perf("finalresults/fc_perf.csv");
        perf << "total_ms,accuracy_percent,B,K,hidden,out_dim\n";
        perf << total_ms << "," << acc << "," << B << "," << K << "," << hidden << "," << out_dim << "\n";
        perf.close();

        std::cout << "✅ FC GPU Completed. Accuracy: " << acc << "%, Time: " << total_ms << " ms\n";

        cudaFree(d_X); cudaFree(d_fc1_w); cudaFree(d_fc1_b);
        cudaFree(d_fc2_w); cudaFree(d_fc2_b);
        cudaFree(d_hidden); cudaFree(d_out);
        cudaEventDestroy(ev_start); cudaEventDestroy(ev_end);
    }
    catch (const std::exception &e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
