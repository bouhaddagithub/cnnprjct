// pooling_gpu.cu


#include "cuda_utils.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

__global__ void maxpool_nchw_kernel(const float* input, float* output,
                                    int B, int C, int H, int W, int K) {
    // input layout: (B, C, H, W) flattened row-major
    // output layout: (B, C, H/K, W/K)
    int out_w = W / K;
    int out_h = H / K;
    int linear_idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int c = blockIdx.y;
    int b = blockIdx.z;
    if (linear_idx >= out_h * out_w) return;
    int oy = linear_idx / out_w;
    int ox = linear_idx % out_w;
    float best = -1e9f;
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            int in_y = oy * K + ky;
            int in_x = ox * K + kx;
            int in_idx = ((b * C + c) * H + in_y) * W + in_x;
            float v = input[in_idx];
            if (v > best) best = v;
        }
    }
    int out_idx = ((b * C + c) * out_h + oy) * out_w + ox;
    output[out_idx] = best;
}

__global__ void fc_batch_kernel(const float* W, const float* b, const float* X, float* Y,
                                int B, int D, int out_dim) {
    // X: B x D ; W: out_dim x D ; b: out_dim ; Y: B x out_dim
    int row = blockIdx.y * blockDim.y + threadIdx.y; // sample index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output index
    if (row >= B || col >= out_dim) return;
    float acc = b ? b[col] : 0.0f;
    const float* wrow = W + (size_t)col * D;
    const float* xptr = X + (size_t)row * D;
    for (int k = 0; k < D; ++k) acc += wrow[k] * xptr[k];
    Y[row * out_dim + col] = acc;
}

int main() {
    try {
        int n_images;
        auto images_u8 = load_mnist_images("data/t10k-images-idx3-ubyte", n_images);
        auto labels = load_mnist_labels("data/t10k-labels-idx1-ubyte", n_images);
        if (images_u8.empty() || labels.empty()) {
            std::cerr << "MNIST test files not found in ../data/. Place t10k-images-idx3-ubyte and t10k-labels-idx1-ubyte there.\n";
            return 1;
        }

        // pooling params
        int pool_k = 2; 

        // Load FC parameters exported by pooling_only.py
        std::vector<int> fc_shape;
        auto fc_w = load_csv_weights("exports/pooling_only/fc_weight.csv", fc_shape);
        auto fc_b = load_csv_weights("exports/pooling_only/fc_bias.csv", fc_shape);

        int out_dim = (fc_shape.size() >= 1) ? fc_shape[0] : 10;
      
        int C = 1, H = 28, W = 28;
        int out_h = H / pool_k;
        int out_w = W / pool_k;
        int D = C * out_h * out_w;


        int fc_in_dim = (fc_shape.size() >= 2) ? fc_shape[1] : D;
        if (fc_in_dim != D) {
           
            std::cout << "Note: fc meta in_dim != computed D. Using fc meta in_dim = " << fc_in_dim << "\n";
            D = fc_in_dim;
        }

       
        if ((int)fc_w.size() < out_dim * D) fc_w.assign((size_t)out_dim * D, 0.0f);
        if ((int)fc_b.size() < out_dim) fc_b.assign(out_dim, 0.0f);

        
        const int CHUNK = 256; // adjust if needed (lower to reduce memory)
        int total = n_images;
        int chunks = (total + CHUNK - 1) / CHUNK;

        // Device buffers for chunking (allocate max chunk size)
        float *d_images = nullptr, *d_pool_out = nullptr, *d_fc_W = nullptr, *d_fc_b = nullptr, *d_preds = nullptr;
        size_t max_chunk = std::min(CHUNK, total);
        size_t images_bytes = max_chunk * C * H * W * sizeof(float);
        size_t pool_out_bytes = max_chunk * C * out_h * out_w * sizeof(float);
        size_t W_bytes = (size_t)out_dim * D * sizeof(float);
        size_t b_bytes = (size_t)out_dim * sizeof(float);
        size_t preds_bytes = max_chunk * out_dim * sizeof(float);

        cudaCheckError(cudaMalloc(&d_images, images_bytes));
        cudaCheckError(cudaMalloc(&d_pool_out, pool_out_bytes));
        cudaCheckError(cudaMalloc(&d_fc_W, W_bytes));
        cudaCheckError(cudaMalloc(&d_fc_b, b_bytes));
        cudaCheckError(cudaMalloc(&d_preds, preds_bytes));

        // copy FC params once
        cudaCheckError(cudaMemcpy(d_fc_W, fc_w.data(), W_bytes, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_fc_b, fc_b.data(), b_bytes, cudaMemcpyHostToDevice));

        // host buffer for chunk images normalized
        std::vector<float> h_chunk((size_t)max_chunk * C * H * W);

        // perf accumulators
        double total_transfer_ms = 0.0, total_pool_ms = 0.0, total_fc_ms = 0.0;
        int correct = 0;
        int processed = 0;

        for (int ch = 0; ch < chunks; ++ch) {
            int start_idx = ch * CHUNK;
            int chunk_size = std::min(CHUNK, total - start_idx);
            if (chunk_size <= 0) break;

            // prepare host chunk
            for (int i = 0; i < chunk_size; ++i) {
                int img_idx = start_idx + i;
                for (int p = 0; p < H * W; ++p) {
                    h_chunk[(size_t)i * H * W + p] = images_u8[img_idx * H * W + p] / 255.0f;
                }
            }

            // Transfer to device and measure
            cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
            cudaEventRecord(t0);
            cudaCheckError(cudaMemcpy(d_images, h_chunk.data(), (size_t)chunk_size * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            float ms_transfer = 0.0f; cudaEventElapsedTime(&ms_transfer, t0, t1);
            total_transfer_ms += ms_transfer;

            // Pool kernel launch: compute out_size = out_h * out_w
            int out_size = out_h * out_w;
          
            dim3 blockPool(256);
            dim3 gridPool((out_size + blockPool.x - 1) / blockPool.x, C, chunk_size);
            cudaEventRecord(t0);
            maxpool_nchw_kernel<<<gridPool, blockPool>>>(d_images, d_pool_out, chunk_size, C, H, W, pool_k);
            cudaCheckError(cudaGetLastError());
            cudaCheckError(cudaDeviceSynchronize());
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            float ms_pool = 0.0f; cudaEventElapsedTime(&ms_pool, t0, t1);
            total_pool_ms += ms_pool;

            
            // Launch fc_batch_kernel with block(16,16) as earlier
            dim3 blockFC(16, 16);
            dim3 gridFC((out_dim + blockFC.x - 1) / blockFC.x, (chunk_size + blockFC.y - 1) / blockFC.y);
            cudaEventRecord(t0);
            fc_batch_kernel<<<gridFC, blockFC>>>(d_fc_W, d_fc_b, d_pool_out, d_preds, chunk_size, D, out_dim);
            cudaCheckError(cudaGetLastError());
            cudaCheckError(cudaDeviceSynchronize());
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            float ms_fc = 0.0f; cudaEventElapsedTime(&ms_fc, t0, t1);
            total_fc_ms += ms_fc;

            // copy predictions back
            std::vector<float> h_preds((size_t)chunk_size * out_dim);
            cudaCheckError(cudaMemcpy(h_preds.data(), d_preds, (size_t)chunk_size * out_dim * sizeof(float), cudaMemcpyDeviceToHost));

            // compute accuracy for this chunk
            for (int i = 0; i < chunk_size; ++i) {
                int best = 0;
                float bestv = h_preds[(size_t)i * out_dim + 0];
                for (int j = 1; j < out_dim; ++j) {
                    float v = h_preds[(size_t)i * out_dim + j];
                    if (v > bestv) { bestv = v; best = j; }
                }
                int true_label = labels[start_idx + i];
                if (best == true_label) ++correct;
            }

            processed += chunk_size;
        }

        double accuracy = 100.0 * correct / processed;
        // average times per chunk (ms) or total times
        std::ofstream perf("pooling_perf.csv");
        perf << "transfer_ms_total,pool_ms_total,fc_ms_total,accuracy_percent,processed,B_chunk,D,out_dim\n";
        perf << total_transfer_ms << "," << total_pool_ms << "," << total_fc_ms << "," << accuracy << "," << processed << "," << CHUNK << "," << D << "," << out_dim << "\n";
        perf.close();

        std::cout << "Pooling: transfer=" << total_transfer_ms << " ms pool=" << total_pool_ms << " ms fc=" << total_fc_ms
                  << " ms acc=" << accuracy << "% processed=" << processed << "\n";

        // cleanup
        cudaFree(d_images);
        cudaFree(d_pool_out);
        cudaFree(d_fc_W);
        cudaFree(d_fc_b);
        cudaFree(d_preds);

    } catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}