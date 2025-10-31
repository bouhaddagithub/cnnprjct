// cnn_gpu_optimized_batched.cu
// Batched CNN: pinned host memory + async H2D/D2H + streams.
// Keeps your kernel logic, but runs in batches for throughput.

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

// Use the cudaCheckError macro from your header (cuda_utils.h)

#ifndef BATCH_SIZE
#define BATCH_SIZE 64   // tune: 32 / 64 / 128 depending on GPU memory
#endif

// Keep the same conv_relu_kernel and fc_kernel logic but with __restrict__ and small adjustments
__global__ void conv_relu_kernel_batched(const float* __restrict__ d_input, // [BATCH, C_in, H, W] flattened per image
                                         const float* __restrict__ weight,
                                         const float* __restrict__ bias,
                                         float* __restrict__ d_conv_out,   // [BATCH, C_out, out_h, out_w]
                                         int C_in, int C_out, int H, int W, int K,
                                         int batch_stride_in, int batch_stride_conv_out)
{
    // Identify per-image coordinates
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z % C_out;                  // filter index
    int batch_idx = blockIdx.z / C_out;           // which image in batch

    int out_h = H - K + 1;
    int out_w = W - K + 1;
    if (ox >= out_w || oy >= out_h) return;

    // pointers for this batch
    const float* in_base = d_input + batch_idx * batch_stride_in;
    float val = bias[oc];

    for (int ic = 0; ic < C_in; ++ic) {
        const float* in_chan = in_base + ic * (H * W);
        const float* w_ptr = weight + ((oc * C_in + ic) * K * K);
        for (int ky = 0; ky < K; ++ky) {
            const float* in_row = in_chan + (oy + ky) * W;
            for (int kx = 0; kx < K; ++kx) {
                val += in_row[ox + kx] * w_ptr[ky * K + kx];
            }
        }
    }
    // ReLU
    if (val < 0.0f) val = 0.0f;
    // store in per-batch conv_out
    d_conv_out[ batch_idx * batch_stride_conv_out + (oc * out_h + oy) * out_w + ox ] = val;
}

__global__ void fc_kernel_batched(const float* __restrict__ d_conv_out, // [BATCH, D]
                                  const float* __restrict__ weight,
                                  const float* __restrict__ bias,
                                  float* __restrict__ d_fc_out, // [BATCH, out_dim]
                                  int D, int out_dim, int batch_stride_conv_out)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y; // batch index
    if (o >= out_dim) return;

    const float* pool_ptr = d_conv_out + b * batch_stride_conv_out;
    const float* w_row = weight + o * D;
    float sum = bias[o];
    for (int d = 0; d < D; ++d) sum += w_row[d] * pool_ptr[d];
    d_fc_out[b * out_dim + o] = sum;
}

int main() {
    try {
        int n_images;
        auto images_u8 = load_mnist_images("data/t10k-images-idx3-ubyte", n_images);
        auto labels = load_mnist_labels("data/t10k-labels-idx1-ubyte", n_images);

        // Load parameters (same paths as your current setup)
        std::vector<int> conv_shape, conv_shape_b, fc_shape, fc_shape_b;
        auto conv_w = load_csv_weights("exports/cnn_only/conv_weight.csv", conv_shape);
        auto conv_b = load_csv_weights("exports/cnn_only/conv_bias.csv", conv_shape_b);
        auto fc_w   = load_csv_weights("exports/cnn_only/fc_weight.csv", fc_shape);
        auto fc_b   = load_csv_weights("exports/cnn_only/fc_bias.csv", fc_shape_b);

        if (conv_shape.size() < 3 || fc_shape.size() < 1) {
            throw std::runtime_error("Invalid weight shapes from CSVs");
        }

        int C_out = conv_shape[0];
        int C_in  = conv_shape[1];
        int K     = conv_shape[2];

        int H = 28, W = 28;
        int out_h = H - K + 1;
        int out_w = W - K + 1;
        int D = C_out * out_h * out_w;
        int out_dim = fc_shape[0];

        // Device allocations (sized for BATCH_SIZE)
        size_t per_image_in = C_in * H * W;
        size_t per_image_conv_out = C_out * out_h * out_w;
        size_t per_image_fc_out = out_dim;

        // device pointers
        float *d_input = nullptr;
        float *d_conv_out = nullptr;
        float *d_conv_w = nullptr;
        float *d_conv_b = nullptr;
        float *d_fc_w = nullptr;
        float *d_fc_b = nullptr;
        float *d_fc_out = nullptr;
        int   *d_pred = nullptr;

        // allocate max-size device buffers for a batch
        cudaCheckError(cudaMalloc(&d_input, BATCH_SIZE * per_image_in * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_conv_out, BATCH_SIZE * per_image_conv_out * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_conv_w, conv_w.size() * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_conv_b, conv_b.size() * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_fc_w, fc_w.size() * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_fc_b, fc_b.size() * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_fc_out, BATCH_SIZE * per_image_fc_out * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_pred, sizeof(int))); // used per-image after argmax kernel

        // Copy weights once
        cudaCheckError(cudaMemcpy(d_conv_w, conv_w.data(), conv_w.size() * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_conv_b, conv_b.data(), conv_b.size() * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_fc_w, fc_w.data(), fc_w.size() * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_fc_b, fc_b.data(), fc_b.size() * sizeof(float), cudaMemcpyHostToDevice));

        // Create pinned host buffers for input batch and fc output
        float* h_input_pinned = nullptr;
        float* h_fc_out_pinned = nullptr;
        size_t input_pinned_bytes = BATCH_SIZE * per_image_in * sizeof(float);
        size_t fc_out_pinned_bytes = BATCH_SIZE * per_image_fc_out * sizeof(float);
        cudaCheckError(cudaHostAlloc(&h_input_pinned, input_pinned_bytes, cudaHostAllocDefault));
        cudaCheckError(cudaHostAlloc(&h_fc_out_pinned, fc_out_pinned_bytes, cudaHostAllocDefault));

        // Streams: use a few streams to pipeline multiple batches
        const int NUM_STREAMS = 4;
        cudaStream_t streams[NUM_STREAMS];
        for (int s = 0; s < NUM_STREAMS; ++s) cudaCheckError(cudaStreamCreate(&streams[s]));

        // Timing/events
        cudaEvent_t ev_start, ev_end;
        cudaEventCreate(&ev_start); cudaEventCreate(&ev_end);
        cudaEventRecord(ev_start);

        int correct = 0;
        std::ofstream cls("finalresults/cnn_gpu_classification.csv");
        cls << "image_id,true_label,pred_label\n";

        // Batch loop: fill pinned buffer, async-copy to device, launch kernels on stream
        for (int base = 0; base < n_images; base += BATCH_SIZE) {
            int curr = std::min(BATCH_SIZE, n_images - base);
            int stream_id = (base / BATCH_SIZE) % NUM_STREAMS;
            cudaStream_t s = streams[stream_id];

            // pack curr images into pinned buffer (float normalized)
            for (int b = 0; b < curr; ++b) {
                int img_idx = base + b;
                float* dst = h_input_pinned + b * per_image_in;
                const unsigned char* src = images_u8.data() + img_idx * (H * W);
                for (int p = 0; p < H * W; ++p) dst[p] = src[p] / 255.0f;
                // if C_in > 1, replicate channels appropriately (MNIST has C_in == 1)
            }

            // async host->device (only curr images worth)
            cudaCheckError(cudaMemcpyAsync(d_input, h_input_pinned,
                                           curr * per_image_in * sizeof(float),
                                           cudaMemcpyHostToDevice, s));

            // compute grid dims
            dim3 threads(16, 16);
            dim3 blocks((out_w + 15) / 16, (out_h + 15) / 16, C_out * curr);
            // careful: blockIdx.z encodes (batch * C_out) in kernel; but kernel uses batch and oc separate
            // We'll launch in a loop over batch elements to keep kernel logic simple and safe:
            for (int b = 0; b < curr; ++b) {
                // pointers offset for this batch element
                float* d_input_img = d_input + b * per_image_in;
                float* d_conv_out_img = d_conv_out + b * per_image_conv_out;

                dim3 blocks_img((out_w + 15) / 16, (out_h + 15) / 16, C_out);
                conv_relu_kernel_batched<<<blocks_img, threads, 0, s>>>(
                    d_input_img, d_conv_w, d_conv_b, d_conv_out_img,
                    C_in, C_out, H, W, K,
                    /*batch stride in*/ per_image_in, /*batch stride conv out*/ per_image_conv_out
                );

                // Note: we intentionally keep FC on device but using a single-block-per-output-per-batch pattern
                // flatten conv_out to D and call FC kernel per batch image
                int grid_fc_x = (out_dim + 255) / 256;
                dim3 grid_fc(grid_fc_x, 1, 1);
                dim3 block_fc(256, 1, 1);
                // but fc_kernel_batched expects batch dim in blockIdx.y -> we can launch with blockIdx.y = b
                // simpler: call fc_kernel_batched with blockIdx.y = 0 but pass pointer to current conv_out flattened region
                const float* d_fc_input_for_img = d_conv_out_img; // expected layout matches D
                float* d_fc_out_for_img = d_fc_out + b * per_image_fc_out;
                // launch fc_kernel_batched treating batch dim implicitly by using blockIdx.y=0 and passing pointers
                // We'll use a small wrapper kernel call that treats pointer as single image input
                fc_kernel_batched<<<grid_fc, block_fc, 0, s>>>(
                    d_fc_input_for_img, d_fc_w, d_fc_b, d_fc_out_for_img, D, out_dim, /*batch_stride_conv_out*/ D
                );
            }

            // async device->host copy of fc outputs for curr images
            cudaCheckError(cudaMemcpyAsync(h_fc_out_pinned, d_fc_out,
                                           curr * per_image_fc_out * sizeof(float),
                                           cudaMemcpyDeviceToHost, s));

            // synchronize this stream to ensure data is ready for CPU classification
            cudaCheckError(cudaStreamSynchronize(s));

            // Do classification on host using the returned fc outputs
            for (int b = 0; b < curr; ++b) {
                int img_idx = base + b;
                float* out_ptr = h_fc_out_pinned + b * per_image_fc_out;
                // find argmax
                int best = 0;
                float bestv = out_ptr[0];
                for (int o = 1; o < out_dim; ++o) {
                    if (out_ptr[o] > bestv) { bestv = out_ptr[o]; best = o; }
                }
                cls << img_idx << "," << (int)labels[img_idx] << "," << best << "\n";
                if (best == labels[img_idx]) ++correct;
            }
        } // end batch loop

        cudaEventRecord(ev_end);
        cudaEventSynchronize(ev_end);
        float total_ms = 0.0f;
        cudaEventElapsedTime(&total_ms, ev_start, ev_end);

        double acc = 100.0 * correct / n_images;
        std::cout << "CNN GPU batched complete. Accuracy: " << acc << "%\n";
        std::cout << "Total elapsed (ms): " << total_ms << "\n";

        // write summary file (optional)
        std::ofstream perf("finalresults/cnn_gpu_batched_perf.csv");
        perf << "total_ms,accuracy_percent,n_images\n";
        perf << total_ms << "," << acc << "," << n_images << "\n";
        perf.close();
        cls.close();

        // cleanup
        for (int s = 0; s < NUM_STREAMS; ++s) cudaCheckError(cudaStreamDestroy(streams[s]));
        cudaCheckError(cudaFreeHost(h_input_pinned));
        cudaCheckError(cudaFreeHost(h_fc_out_pinned));

        cudaCheckError(cudaFree(d_input));
        cudaCheckError(cudaFree(d_conv_out));
        cudaCheckError(cudaFree(d_conv_w));
        cudaCheckError(cudaFree(d_conv_b));
        cudaCheckError(cudaFree(d_fc_w));
        cudaCheckError(cudaFree(d_fc_b));
        cudaCheckError(cudaFree(d_fc_out));
        cudaCheckError(cudaFree(d_pred));

        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_end);
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
