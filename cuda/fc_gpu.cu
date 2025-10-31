// fc_gpu.cu


#include "cuda_utils.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

__global__ void matmul_batch(const float* X, const float* Wt, const float* b, float* Y, int B, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B || col >= N) return;

    float acc = b[col];
    for (int k = 0; k < K; ++k)
        acc += X[row * K + k] * Wt[k * N + col];
    Y[row * N + col] = acc;
}

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

       
        if (s_fc1w.size() < 2 || s_fc2w.size() < 2)
            throw std::runtime_error("Invalid FC weight shapes.");
        if (s_fc1b.size() < 1 || s_fc2b.size() < 1)
            throw std::runtime_error("Invalid FC bias shapes.");

        int hidden = s_fc1w[0];
        int K = s_fc1w[1];
        int out_dim = s_fc2w[0];
        int B = n_images;

        std::vector<float> h_X((size_t)B * K);
        for (int i = 0; i < B; ++i)
            for (int p = 0; p < K; ++p)
                h_X[i * K + p] = images_u8[i * 28 * 28 + p] / 255.0f;

        // Transpose weights for GPU efficiency
        std::vector<float> fc1_wt(K * hidden), fc2_wt(hidden * out_dim);
        for (int i = 0; i < hidden; ++i)
            for (int j = 0; j < K; ++j)
                fc1_wt[j * hidden + i] = fc1_w[i * K + j];

        for (int i = 0; i < out_dim; ++i)
            for (int j = 0; j < hidden; ++j)
                fc2_wt[j * out_dim + i] = fc2_w[i * hidden + j];

        float *d_X, *d_hidden, *d_out, *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b;
        cudaMalloc(&d_X, B * K * sizeof(float));
        cudaMalloc(&d_hidden, B * hidden * sizeof(float));
        cudaMalloc(&d_out, B * out_dim * sizeof(float));
        cudaMalloc(&d_fc1_w, K * hidden * sizeof(float));
        cudaMalloc(&d_fc1_b, hidden * sizeof(float));
        cudaMalloc(&d_fc2_w, hidden * out_dim * sizeof(float));
        cudaMalloc(&d_fc2_b, out_dim * sizeof(float));

        // Measure performance
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        // Transfer timing
        cudaEventRecord(t0);
        cudaMemcpy(d_X, h_X.data(), B * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc1_w, fc1_wt.data(), K * hidden * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc1_b, fc1_b.data(), hidden * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc2_w, fc2_wt.data(), hidden * out_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc2_b, fc2_b.data(), out_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms_transfer = 0;
        cudaEventElapsedTime(&ms_transfer, t0, t1);

        // FC1 forward
        dim3 block(16, 16);
        dim3 grid((hidden + block.x - 1) / block.x, (B + block.y - 1) / block.y);
        cudaEventRecord(t0);
        matmul_batch<<<grid, block>>>(d_X, d_fc1_w, d_fc1_b, d_hidden, B, K, hidden);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms_fc1 = 0;
        cudaEventElapsedTime(&ms_fc1, t0, t1);

        // ReLU (host-based)
        std::vector<float> h_hidden(B * hidden);
        cudaMemcpy(h_hidden.data(), d_hidden, B * hidden * sizeof(float), cudaMemcpyDeviceToHost);
        auto t_relu_start = std::chrono::high_resolution_clock::now();
        for (float &v : h_hidden) if (v < 0) v = 0;
        auto t_relu_end = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_hidden, h_hidden.data(), B * hidden * sizeof(float), cudaMemcpyHostToDevice);
        double ms_relu = std::chrono::duration<double, std::milli>(t_relu_end - t_relu_start).count();

        // FC2 forward
        dim3 grid2((out_dim + block.x - 1) / block.x, (B + block.y - 1) / block.y);
        cudaEventRecord(t0);
        matmul_batch<<<grid2, block>>>(d_hidden, d_fc2_w, d_fc2_b, d_out, B, hidden, out_dim);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms_fc2 = 0;
        cudaEventElapsedTime(&ms_fc2, t0, t1);

        // Accuracy
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

        std::ofstream perf("fc_perf.csv");
        perf << "transfer_ms,fc1_ms,relu_ms,fc2_ms,accuracy_percent,B,K,hidden,out\n";
        perf << ms_transfer << "," << ms_fc1 << "," << ms_relu << "," << ms_fc2 << "," << acc << "," << B << "," << K << "," << hidden << "," << out_dim << "\n";
        perf.close();

        std::cout << "FC GPU Performance:\n"
                  << "Transfer: " << ms_transfer << " ms\n"
                  << "FC1: " << ms_fc1 << " ms\n"
                  << "ReLU: " << ms_relu << " ms\n"
                  << "FC2: " << ms_fc2 << " ms\n"
                  << "Accuracy: " << acc << "%\n";

        cudaFree(d_X); cudaFree(d_hidden); cudaFree(d_out);
        cudaFree(d_fc1_w); cudaFree(d_fc1_b); cudaFree(d_fc2_w); cudaFree(d_fc2_b);
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
