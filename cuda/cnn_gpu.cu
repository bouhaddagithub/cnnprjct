// cnn_gpu.cu

#include "cuda_utils.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>


// ====================================================
// CUDA kernel: naive convolution with ReLU activation
// ====================================================
__global__ void conv_relu_kernel(const float* input, const float* weight, const float* bias,
                                 float* output, int C_in, int C_out, int H, int W, int K) {
    int out_h = H - K + 1;
    int out_w = W - K + 1;
    int oc = blockIdx.z;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    if (oy >= out_h || ox >= out_w) return;

    float val = bias[oc];
    for (int ic = 0; ic < C_in; ++ic) {
        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                int in_y = oy + ky;
                int in_x = ox + kx;
                float v = input[(ic * H + in_y) * W + in_x];
                float w = weight[((oc * C_in + ic) * K + ky) * K + kx];
                val += v * w;
            }
        }
    }
    if (val < 0.0f) val = 0.0f;  // ReLU
    output[(oc * out_h + oy) * out_w + ox] = val;
}

int main() {
    try {
        int n_images;
        auto images_u8 = load_mnist_images("../data/t10k-images-idx3-ubyte", n_images);
        auto labels = load_mnist_labels("../data/t10k-labels-idx1-ubyte", n_images);

        // Load parameters
        std::vector<int> conv_shape_w, conv_shape_b, fc_shape_w, fc_shape_b;
        auto conv_w = load_csv_weights("../exports/cnn_only/conv_weight.csv", conv_shape_w);
        auto conv_b = load_csv_weights("../exports/cnn_only/conv_bias.csv", conv_shape_b);
        auto fc_w = load_csv_weights("../exports/cnn_only/fc_weight.csv", fc_shape_w);
        auto fc_b = load_csv_weights("../exports/cnn_only/fc_bias.csv", fc_shape_b);

        if (conv_shape_w.size() < 4)
            throw std::runtime_error("Invalid conv weight shape");

        int C_out = conv_shape_w[0];
        int C_in  = conv_shape_w[1];
        int K     = conv_shape_w[2];
        int H = 28, W = 28;
        int out_h = H - K + 1;
        int out_w = W - K + 1;
        int D = C_out * out_h * out_w;
        int out_dim = fc_shape_w[0];

        // Allocate device memory
        float *d_input, *d_conv_out, *d_conv_w, *d_conv_b;
        cudaCheckError(cudaMalloc(&d_input, C_in * H * W * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_conv_out, C_out * out_h * out_w * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_conv_w, conv_w.size() * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_conv_b, conv_b.size() * sizeof(float)));

        cudaCheckError(cudaMemcpy(d_conv_w, conv_w.data(), conv_w.size() * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_conv_b, conv_b.data(), conv_b.size() * sizeof(float), cudaMemcpyHostToDevice));

        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

        int correct = 0;
        cudaEventRecord(t0);
        for (int i = 0; i < n_images; ++i) {
            // Normalize image to [0,1]
            std::vector<float> in_sample(H * W);
            for (int p = 0; p < H * W; ++p)
                in_sample[p] = images_u8[i * H * W + p] / 255.0f;

            cudaCheckError(cudaMemcpy(d_input, in_sample.data(), C_in * H * W * sizeof(float), cudaMemcpyHostToDevice));

            // --- Convolution + ReLU ---
            dim3 threads(16, 16);
            dim3 blocks((out_w + 15)/16, (out_h + 15)/16, C_out);
            conv_relu_kernel<<<blocks, threads>>>(d_input, d_conv_w, d_conv_b, d_conv_out,
                                                  C_in, C_out, H, W, K);
            cudaDeviceSynchronize();

            // --- Copy result back ---
            std::vector<float> h_conv(D);
            cudaMemcpy(h_conv.data(), d_conv_out, D * sizeof(float), cudaMemcpyDeviceToHost);

            // --- Fully Connected (CPU only) ---
            std::vector<float> outv(out_dim);
            for (int o = 0; o < out_dim; ++o) {
                float s = fc_b[o];
                for (int d = 0; d < D; ++d)
                    s += fc_w[o * D + d] * h_conv[d];
                outv[o] = s;
            }

            // --- Argmax ---
            int best = 0; float bestv = outv[0];
            for (int o = 1; o < out_dim; ++o)
                if (outv[o] > bestv) { bestv = outv[o]; best = o; }

            if (best == labels[i]) ++correct;
        }
        cudaEventRecord(t1); cudaEventSynchronize(t1);

        float ms_total = 0;
        cudaEventElapsedTime(&ms_total, t0, t1);
        double acc = 100.0 * correct / n_images;

        // Save performance
        std::ofstream perf("cnn_perf.csv");
        perf << "total_ms,accuracy_percent,n_images,D,out_dim\n";
        perf << ms_total << "," << acc << "," << n_images << "," << D << "," << out_dim << "\n";
        perf.close();

        std::cout << "CNN GPU Performance:\n";
        std::cout << "Total time: " << ms_total << " ms\n";
        std::cout << "Accuracy: " << acc << "%\n";

        cudaFree(d_input); cudaFree(d_conv_out); cudaFree(d_conv_w); cudaFree(d_conv_b);

    } catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
