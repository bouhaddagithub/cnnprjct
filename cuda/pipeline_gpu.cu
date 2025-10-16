// pipeline_gpu.cu
// Enhanced full GPU pipeline with detailed performance and classification export.
// Build: nvcc -std=c++14 -O2 -arch=sm_61 -o pipeline_gpu pipeline_gpu.cu

#include "cuda_utils.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

__global__ void conv_naive(const float* input, const float* weight, const float* bias,
                           float* output, int Kin, int Kout, int H, int W, int K) {
    int out_h = H - K + 1;
    int out_w = W - K + 1;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;
    if (ox >= out_w || oy >= out_h) return;

    float val = bias[oc];
    for (int ic = 0; ic < Kin; ++ic)
        for (int ky = 0; ky < K; ++ky)
            for (int kx = 0; kx < K; ++kx) {
                int in_y = oy + ky;
                int in_x = ox + kx;
                int in_idx = (ic * H + in_y) * W + in_x;
                int w_idx = ((oc * Kin + ic) * K + ky) * K + kx;
                val += input[in_idx] * weight[w_idx];
            }
    output[(oc * out_h + oy) * out_w + ox] = fmaxf(val, 0.0f);
}

__global__ void maxpool_sample(const float* input, float* output, int C, int H, int W, int K) {
    int out_w = W / K, out_h = H / K;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y;
    if (idx >= out_h * out_w) return;
    int oy = idx / out_w, ox = idx % out_w;
    float best = -1e9f;
    for (int ky = 0; ky < K; ++ky)
        for (int kx = 0; kx < K; ++kx) {
            int in_y = oy * K + ky, in_x = ox * K + kx;
            float v = input[(c * H + in_y) * W + in_x];
            if (v > best) best = v;
        }
    output[(c * out_h + oy) * out_w + ox] = best;
}

int main() {
    try {
        int n_images;
        auto images_u8 = load_mnist_images("../data/t10k-images-idx3-ubyte", n_images);
        auto labels = load_mnist_labels("../data/t10k-labels-idx1-ubyte", n_images);

        std::vector<int> conv_meta, fc_meta;
        auto conv_w = load_csv_weights("../exports/pipeline/conv_weight.csv", conv_meta);
        auto conv_b = load_csv_weights("../exports/pipeline/conv_bias.csv", conv_meta);
        auto fc_w = load_csv_weights("../exports/pipeline/fc_weight.csv", fc_meta);
        auto fc_b = load_csv_weights("../exports/pipeline/fc_bias.csv", fc_meta);

        int Kout = conv_meta[0], Kin = conv_meta[1], K = conv_meta[2];
        int H = 28, W = 28;
        int out_h = H - K + 1, out_w = W - K + 1;
        int pool_k = 2, pool_h = out_h / pool_k, pool_w = out_w / pool_k;
        int D = Kout * pool_h * pool_w, out_dim = fc_meta[0];

        float *d_input, *d_conv_out, *d_pool, *d_conv_w, *d_conv_b;
        cudaMalloc(&d_input, Kin * H * W * sizeof(float));
        cudaMalloc(&d_conv_out, Kout * out_h * out_w * sizeof(float));
        cudaMalloc(&d_pool, Kout * pool_h * pool_w * sizeof(float));
        cudaMalloc(&d_conv_w, conv_w.size() * sizeof(float));
        cudaMalloc(&d_conv_b, conv_b.size() * sizeof(float));

        cudaMemcpy(d_conv_w, conv_w.data(), conv_w.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_conv_b, conv_b.data(), conv_b.size() * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0);

        float t_conv = 0, t_pool = 0, t_fc = 0, t_h2d = 0, t_d2h = 0;
        int correct = 0;
        std::ofstream cls("classification_results.csv");
        cls << "image_id,true_label,pred_label,confidence\n";

        for (int i = 0; i < n_images; ++i) {
            std::vector<float> in(Kin * H * W);
            for (int p = 0; p < H * W; ++p) in[p] = images_u8[i * H * W + p] / 255.0f;

            cudaEvent_t eh2d0, eh2d1;
            cudaEventCreate(&eh2d0); cudaEventCreate(&eh2d1);
            cudaEventRecord(eh2d0);
            cudaMemcpy(d_input, in.data(), Kin * H * W * sizeof(float), cudaMemcpyHostToDevice);
            cudaEventRecord(eh2d1);
            cudaEventSynchronize(eh2d1);
            float ms;
            cudaEventElapsedTime(&ms, eh2d0, eh2d1);
            t_h2d += ms;

            // ---- Convolution ----
            cudaEvent_t ec0, ec1;
            cudaEventCreate(&ec0); cudaEventCreate(&ec1);
            cudaEventRecord(ec0);
            dim3 threads(16, 16);
            dim3 blocks((out_w + 15) / 16, (out_h + 15) / 16, Kout);
            conv_naive<<<blocks, threads>>>(d_input, d_conv_w, d_conv_b, d_conv_out, Kin, Kout, H, W, K);
            cudaDeviceSynchronize();
            cudaEventRecord(ec1);
            cudaEventSynchronize(ec1);
            cudaEventElapsedTime(&ms, ec0, ec1);
            t_conv += ms;

            // ---- Pooling ----
            cudaEvent_t ep0, ep1;
            cudaEventCreate(&ep0); cudaEventCreate(&ep1);
            cudaEventRecord(ep0);
            dim3 blockPool(256);
            dim3 gridPool((pool_h * pool_w + 255) / 256, Kout);
            maxpool_sample<<<gridPool, blockPool>>>(d_conv_out, d_pool, Kout, out_h, out_w, pool_k);
            cudaDeviceSynchronize();
            cudaEventRecord(ep1);
            cudaEventSynchronize(ep1);
            cudaEventElapsedTime(&ms, ep0, ep1);
            t_pool += ms;

            // ---- Copy pooled output to host ----
            cudaEvent_t ed2h0, ed2h1;
            cudaEventCreate(&ed2h0); cudaEventCreate(&ed2h1);
            cudaEventRecord(ed2h0);
            std::vector<float> h_pool(D);
            cudaMemcpy(h_pool.data(), d_pool, D * sizeof(float), cudaMemcpyDeviceToHost);
            cudaEventRecord(ed2h1);
            cudaEventSynchronize(ed2h1);
            cudaEventElapsedTime(&ms, ed2h0, ed2h1);
            t_d2h += ms;

            // ---- Fully Connected ----
            auto t_fc_start = std::chrono::high_resolution_clock::now();
            std::vector<float> outv(out_dim);
            for (int o = 0; o < out_dim; ++o) {
                float s = fc_b[o];
                for (int d = 0; d < D; ++d) s += fc_w[o * D + d] * h_pool[d];
                outv[o] = s;
            }
            auto t_fc_end = std::chrono::high_resolution_clock::now();
            t_fc += std::chrono::duration<float, std::milli>(t_fc_end - t_fc_start).count();

            // ---- Classification ----
            auto best_it = std::max_element(outv.begin(), outv.end());
            int best = std::distance(outv.begin(), best_it);
            float confidence = *best_it;
            cls << i << "," << labels[i] << "," << best << "," << confidence << "\n";
            if (best == labels[i]) correct++;
        }

        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float total_ms = 0;
        cudaEventElapsedTime(&total_ms, t0, t1);
        double acc = 100.0 * correct / n_images;

        std::ofstream perf("pipeline2_perf.csv");
        perf << "total_ms,conv_ms,pool_ms,fc_ms,h2d_ms,d2h_ms,accuracy_percent,n_images,D,out_dim\n";
        perf << total_ms << "," << t_conv/n_images << "," << t_pool/n_images << "," 
             << t_fc/n_images << "," << t_h2d/n_images << "," << t_d2h/n_images << ","
             << acc << "," << n_images << "," << D << "," << out_dim << "\n";
        perf.close();
        cls.close();

        std::cout << "Pipeline GPU completed!\n";
        std::cout << "Accuracy: " << acc << "%\n";
        std::cout << "Avg Conv: " << t_conv/n_images << " ms | Avg Pool: " << t_pool/n_images
                  << " ms | Avg FC: " << t_fc/n_images << " ms\n";
        std::cout << "Avg H2D: " << t_h2d/n_images << " ms | Avg D2H: " << t_d2h/n_images << " ms\n";

        cudaFree(d_input); cudaFree(d_conv_out); cudaFree(d_pool);
        cudaFree(d_conv_w); cudaFree(d_conv_b);
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
