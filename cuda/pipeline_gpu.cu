// pipeline_gpu_fixed.cu
#include "cuda_utils.h"
#include "cnn_layer.cuh"
#include "pooling_layer.cuh"
#include "fc_layer.cuh"

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

#ifndef BATCH_SIZE
#define BATCH_SIZE 128
#endif

// Debug: Print first few values to verify correctness
void debug_print(const std::string& name, const float* data, int count) {
    std::cout << name << " first " << std::min(5, count) << " values: ";
    for (int i = 0; i < std::min(5, count); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    try {
        // --- Load MNIST test data ---
        int n_images;
        auto images_u8 = load_mnist_images("data/t10k-images-idx3-ubyte", n_images);
        auto labels    = load_mnist_labels("data/t10k-labels-idx1-ubyte", n_images);
        
        std::cout << "Loaded " << n_images << " test images" << std::endl;

        // --- Load exported pipeline weights ---
        std::vector<int> conv_shape, conv_shape_b, fc_shape, fc_shape_b;
        auto conv_w = load_csv_weights("exports/pipeline/conv_weight.csv", conv_shape);
        auto conv_b = load_csv_weights("exports/pipeline/conv_bias.csv",  conv_shape_b);
        auto fc_w   = load_csv_weights("exports/pipeline/fc_weight.csv",   fc_shape);
        auto fc_b   = load_csv_weights("exports/pipeline/fc_bias.csv",    fc_shape_b);

        std::cout << "Conv weight shape: " << conv_shape[0] << "x" << conv_shape[1] << "x" << conv_shape[2] << std::endl;
        std::cout << "FC weight shape: " << fc_shape[0] << "x" << fc_shape[1] << std::endl;

        // Debug: Print first few weights
        debug_print("Conv_w", conv_w.data(), 5);
        debug_print("Conv_b", conv_b.data(), 5);
        debug_print("FC_w", fc_w.data(), 5);
        debug_print("FC_b", fc_b.data(), 5);

        // --- Model geometry ---
        int C_out = conv_shape[0];
        int C_in  = conv_shape[1];
        int K_conv = conv_shape[2];
        int H = 28, W = 28;
        int out_h = H - K_conv + 1;
        int out_w = W - K_conv + 1;

        int pool_k = 2;
        int pool_h = out_h / pool_k;
        int pool_w = out_w / pool_k;

        int D = C_out * pool_h * pool_w; // FC input dim
        int out_dim = fc_shape[0];

        std::cout << "Model geometry: C_in=" << C_in << ", C_out=" << C_out 
                  << ", K_conv=" << K_conv << ", D=" << D << ", out_dim=" << out_dim << std::endl;

        // Verify FC weight dimensions match expected
        if (fc_shape[1] != D) {
            std::cerr << "ERROR: FC weight dimension mismatch! Expected " << D << " got " << fc_shape[1] << std::endl;
            return 1;
        }

        // --- TRANSPOSE FC WEIGHTS  ---
        std::vector<float> fc_w_transposed(D * out_dim);
        for (int i = 0; i < out_dim; ++i) {
            for (int j = 0; j < D; ++j) {
                fc_w_transposed[j * out_dim + i] = fc_w[i * D + j];
            }
        }

        size_t per_image_in = (size_t)C_in * H * W;
        size_t per_image_conv_out = (size_t)C_out * out_h * out_w;
        size_t per_image_pool_out = (size_t)C_out * pool_h * pool_w;
        size_t per_image_fc_out = (size_t)out_dim;

        // --- Allocate device memory ---
        float *d_input, *d_conv_out, *d_pool_out, *d_fc_out;
        float *d_conv_w, *d_conv_b, *d_fc_w, *d_fc_b;

        cudaCheckError(cudaMalloc(&d_input,  BATCH_SIZE * per_image_in * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_conv_out, BATCH_SIZE * per_image_conv_out * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_pool_out, BATCH_SIZE * per_image_pool_out * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_fc_out, BATCH_SIZE * per_image_fc_out * sizeof(float)));

        cudaCheckError(cudaMalloc(&d_conv_w, conv_w.size() * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_conv_b, conv_b.size() * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_fc_w,   fc_w_transposed.size() * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_fc_b,   fc_b.size() * sizeof(float)));

        // Copy weights to device
        cudaCheckError(cudaMemcpy(d_conv_w, conv_w.data(), conv_w.size() * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_conv_b, conv_b.data(), conv_b.size() * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_fc_w, fc_w_transposed.data(), fc_w_transposed.size() * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_fc_b, fc_b.data(), fc_b.size() * sizeof(float), cudaMemcpyHostToDevice));

        // Pinned host buffers
        float *h_input_pinned, *h_fc_out_pinned;
        cudaCheckError(cudaHostAlloc(&h_input_pinned, BATCH_SIZE * per_image_in * sizeof(float), cudaHostAllocDefault));
        cudaCheckError(cudaHostAlloc(&h_fc_out_pinned, BATCH_SIZE * per_image_fc_out * sizeof(float), cudaHostAllocDefault));

        cudaStream_t stream = 0;

        // --- Performance Measurement Events ---
        cudaEvent_t ev_total_start, ev_total_end;
        cudaEvent_t ev_h2d_start, ev_h2d_end;
        cudaEvent_t ev_cnn_start, ev_cnn_end;
        cudaEvent_t ev_pool_start, ev_pool_end;
        cudaEvent_t ev_fc_start, ev_fc_end;
        cudaEvent_t ev_d2h_start, ev_d2h_end;

        cudaEventCreate(&ev_total_start);
        cudaEventCreate(&ev_total_end);
        cudaEventCreate(&ev_h2d_start);
        cudaEventCreate(&ev_h2d_end);
        cudaEventCreate(&ev_cnn_start);
        cudaEventCreate(&ev_cnn_end);
        cudaEventCreate(&ev_pool_start);
        cudaEventCreate(&ev_pool_end);
        cudaEventCreate(&ev_fc_start);
        cudaEventCreate(&ev_fc_end);
        cudaEventCreate(&ev_d2h_start);
        cudaEventCreate(&ev_d2h_end);

        // Output files
        std::ofstream perf("finalresults/pipeline_perf_fixed.csv");
        perf << "batch_idx,images_in_batch,h2d_ms,cnn_ms,pool_ms,fc_ms,d2h_ms,total_batch_ms\n";

        std::ofstream cls("finalresults/pipeline_classification_fixed.csv");
        cls << "image_id,true_label,pred_label\n";

        // Start total timer
        cudaEventRecord(ev_total_start, stream);

        int correct = 0;
        int batch_idx = 0;
        
        for (int base = 0; base < n_images; base += BATCH_SIZE) {
            int curr = std::min(BATCH_SIZE, n_images - base);
            
            cudaEvent_t ev_batch_start, ev_batch_end;
            cudaEventCreate(&ev_batch_start);
            cudaEventCreate(&ev_batch_end);
            cudaEventRecord(ev_batch_start, stream);

            // --- Prepare batch ---
            for (int b = 0; b < curr; ++b) {
                const unsigned char* src = images_u8.data() + (base + b) * H * W;
                float* dst = h_input_pinned + (size_t)b * per_image_in;
                for (int i = 0; i < H * W; ++i) dst[i] = src[i] / 255.0f;
            }

            // --- H2D Transfer ---
            cudaEventRecord(ev_h2d_start, stream);
            cudaCheckError(cudaMemcpyAsync(d_input, h_input_pinned,
                                           curr * per_image_in * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
            cudaEventRecord(ev_h2d_end, stream);

            // --- CNN Layer ---
            cudaEventRecord(ev_cnn_start, stream);
            run_cnn_layer(d_input, d_conv_out, d_conv_w, d_conv_b,
                         curr, C_in, C_out, H, W, K_conv, stream);
            cudaEventRecord(ev_cnn_end, stream);
            cudaCheckError(cudaStreamSynchronize(stream));

            // --- Pooling Layer ---  
            cudaEventRecord(ev_pool_start, stream);
            run_pooling_layer(d_conv_out, d_pool_out, curr, C_out, out_h, out_w, pool_k, stream);
            cudaEventRecord(ev_pool_end, stream);
            cudaCheckError(cudaStreamSynchronize(stream));

            // --- FC Layer ---
            cudaEventRecord(ev_fc_start, stream);
            run_fc_layer(d_pool_out, d_fc_w, d_fc_b, d_fc_out, curr, D, out_dim, stream);
            cudaEventRecord(ev_fc_end, stream);
            cudaCheckError(cudaStreamSynchronize(stream));

            // --- D2H Transfer ---
            cudaEventRecord(ev_d2h_start, stream);
            cudaCheckError(cudaMemcpyAsync(h_fc_out_pinned, d_fc_out,
                                           curr * per_image_fc_out * sizeof(float),
                                           cudaMemcpyDeviceToHost, stream));
            cudaEventRecord(ev_d2h_end, stream);
            cudaCheckError(cudaStreamSynchronize(stream));

            // Record batch end time
            cudaEventRecord(ev_batch_end, stream);
            cudaCheckError(cudaStreamSynchronize(stream));

            // --- Measure Times ---
            float h2d_ms = 0.0f, cnn_ms = 0.0f, pool_ms = 0.0f, fc_ms = 0.0f, d2h_ms = 0.0f, total_batch_ms = 0.0f;
            
            cudaEventElapsedTime(&h2d_ms, ev_h2d_start, ev_h2d_end);
            cudaEventElapsedTime(&cnn_ms, ev_cnn_start, ev_cnn_end);
            cudaEventElapsedTime(&pool_ms, ev_pool_start, ev_pool_end);
            cudaEventElapsedTime(&fc_ms, ev_fc_start, ev_fc_end);
            cudaEventElapsedTime(&d2h_ms, ev_d2h_start, ev_d2h_end);
            cudaEventElapsedTime(&total_batch_ms, ev_batch_start, ev_batch_end);

            // --- Classification ---
            for (int b = 0; b < curr; ++b) {
                int img_idx = base + b;
                float* logits = h_fc_out_pinned + (size_t)b * per_image_fc_out;
                
                // Debug first image
                if (img_idx == 0) {
                    std::cout << "First image logits: ";
                    for (int j = 0; j < out_dim; ++j) {
                        std::cout << logits[j] << " ";
                    }
                    std::cout << std::endl;
                }
                
                int best = 0;
                float bestv = logits[0];
                for (int j = 1; j < out_dim; ++j) {
                    if (logits[j] > bestv) { 
                        bestv = logits[j]; 
                        best = j; 
                    }
                }
                cls << img_idx << "," << (int)labels[img_idx] << "," << best << "\n";
                if (best == labels[img_idx]) ++correct;
            }

            // --- Write per-batch performance ---
            perf << batch_idx << "," << curr << "," << h2d_ms << "," << cnn_ms << "," 
                 << pool_ms << "," << fc_ms << "," << d2h_ms << "," << total_batch_ms << "\n";
            
            std::cout << "Batch " << batch_idx << ": " << curr << " images, "
                      << "H2D: " << h2d_ms << "ms, CNN: " << cnn_ms << "ms, "
                      << "Pool: " << pool_ms << "ms, FC: " << fc_ms << "ms, "
                      << "D2H: " << d2h_ms << "ms, Total: " << total_batch_ms << "ms" << std::endl;

            batch_idx++;
            cudaEventDestroy(ev_batch_start);
            cudaEventDestroy(ev_batch_end);
        }

        // --- Total Time Measurement ---
        cudaEventRecord(ev_total_end, stream);
        cudaCheckError(cudaStreamSynchronize(stream));
        
        float total_ms = 0.0f;
        cudaEventElapsedTime(&total_ms, ev_total_start, ev_total_end);

        double accuracy = 100.0 * correct / n_images;

        // --- Write Summary Files ---
        std::ofstream perf_summary("finalresults/pipeline_perf_summary_fixed.csv");
        perf_summary << "total_ms,accuracy_percent,n_images,batch_size\n";
        perf_summary << total_ms << "," << accuracy << "," << n_images << "," << BATCH_SIZE << "\n";
        perf_summary.close();

        std::cout << "✅ Fixed Pipeline complete. Accuracy: " << accuracy << "%" << std::endl;
        std::cout << "⏱️  Total elapsed time: " << total_ms << " ms" << std::endl;
        std::cout << "📊 Performance data saved to finalresults/pipeline_perf_fixed.csv" << std::endl;
        std::cout << "📈 Summary saved to finalresults/pipeline_perf_summary_fixed.csv" << std::endl;

        // --- Cleanup ---
        perf.close();
        cls.close();
        
        cudaFree(d_input); 
        cudaFree(d_conv_out); 
        cudaFree(d_pool_out); 
        cudaFree(d_fc_out);
        cudaFree(d_conv_w); 
        cudaFree(d_conv_b); 
        cudaFree(d_fc_w); 
        cudaFree(d_fc_b);
        cudaFreeHost(h_input_pinned); 
        cudaFreeHost(h_fc_out_pinned);

        // Destroy events
        cudaEventDestroy(ev_total_start);
        cudaEventDestroy(ev_total_end);
        cudaEventDestroy(ev_h2d_start);
        cudaEventDestroy(ev_h2d_end);
        cudaEventDestroy(ev_cnn_start);
        cudaEventDestroy(ev_cnn_end);
        cudaEventDestroy(ev_pool_start);
        cudaEventDestroy(ev_pool_end);
        cudaEventDestroy(ev_fc_start);
        cudaEventDestroy(ev_fc_end);
        cudaEventDestroy(ev_d2h_start);
        cudaEventDestroy(ev_d2h_end);

    } catch (const std::exception &e) {
        std::cerr << "❌ Pipeline error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
