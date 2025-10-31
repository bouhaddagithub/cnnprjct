// pooling_cpu.cpp
#include "pooling_cpu.h"
#include <fstream>
#include <sstream>

// ============================================================
// Load Pooling Metadata (kernel size from pooling.meta.json)
// ============================================================
PoolingLayerCPU load_pooling_meta_cpu(const std::string &meta_path,
                                      int C, int H, int W)
{
    PoolingLayerCPU pool;
    pool.C = C;
    pool.H = H;
    pool.W = W;
    pool.K = 2; // default kernel size

    // Simple JSON parser for {"kernel_size": N}
    std::ifstream f(meta_path);
    if (f.is_open()) {
        std::string line;
        while (std::getline(f, line)) {
            size_t pos = line.find("\"kernel_size\"");
            if (pos != std::string::npos) {
                size_t colon_pos = line.find(":", pos);
                if (colon_pos != std::string::npos) {
                    std::string num_str;
                    for (size_t i = colon_pos + 1; i < line.length(); ++i) {
                        if (isdigit(line[i])) {
                            num_str += line[i];
                        } else if (!num_str.empty()) {
                            break;
                        }
                    }
                    if (!num_str.empty()) {
                        pool.K = std::stoi(num_str);
                        break;
                    }
                }
            }
        }
        f.close();
    } else {
        std::cerr << "⚠️ Could not open " << meta_path << ", using default kernel_size=2\n";
    }

    pool.out_h = H / pool.K;
    pool.out_w = W / pool.K;

    return pool;
}

// ============================================================
// Pooling Forward Pass (MaxPool only)
// ============================================================
std::vector<float> pooling_forward_cpu(const std::vector<float> &in,
                                       const PoolingLayerCPU &pool,
                                       float &t_pool_ms)
{
    TimerCPU timer;
    timer.start();

    std::vector<float> pool_out((size_t)pool.C * pool.out_h * pool.out_w, 0.0f);

    for (int c = 0; c < pool.C; ++c) {
        for (int oy = 0; oy < pool.out_h; ++oy) {
            for (int ox = 0; ox < pool.out_w; ++ox) {
                float max_val = -1e9f;
                
                for (int ky = 0; ky < pool.K; ++ky) {
                    for (int kx = 0; kx < pool.K; ++kx) {
                        int in_y = oy * pool.K + ky;
                        int in_x = ox * pool.K + kx;
                        int in_idx = (c * pool.H + in_y) * pool.W + in_x;
                        
                        if (in[in_idx] > max_val) {
                            max_val = in[in_idx];
                        }
                    }
                }
                
                pool_out[(c * pool.out_h + oy) * pool.out_w + ox] = max_val;
            }
        }
    }

    t_pool_ms = timer.stop_ms();
    return pool_out;
}

// ============================================================
// Standalone Pooling Test for timing and export
// ============================================================
int run_pooling_cpu_standalone()
{
    try {
        int n_images, rows, cols;
        auto images = load_mnist_images("data/t10k-images-idx3-ubyte", n_images, rows, cols);
        int n_labels;
        auto labels = load_mnist_labels("data/t10k-labels-idx1-ubyte", n_labels);
        
        if (n_images != n_labels)
            std::cerr << "Warning: image/label counts differ\n";

        // Load pooling metadata
        auto pool = load_pooling_meta_cpu("exports/pooling_only/pooling.meta.json",
                                          1, rows, cols);

        std::cout << "ℹ️ Using pooling kernel size K = " << pool.K << "\n";

        TimerCPU timer_total;
        timer_total.start();
        float t_pool_sum = 0.0f;

        // Run pooling on subset of images
        int test_count = std::min(n_images, 10); // limit for perf test
        for (int i = 0; i < test_count; ++i) {
            // Normalize input
            std::vector<float> in((size_t)pool.H * pool.W);
            for (int p = 0; p < pool.H * pool.W; ++p)
                in[p] = images[(size_t)i * pool.H * pool.W + p] / 255.0f;

            // Run pooling
            float t_pool_ms;
            auto pooled = pooling_forward_cpu(in, pool, t_pool_ms);
            t_pool_sum += t_pool_ms;

            // Save first pooled output for debug
            if (i == 0) {
                std::vector<std::vector<float>> mat;
                for (int c = 0; c < pool.C; ++c) {
                    std::vector<float> one_map;
                    for (int oy = 0; oy < pool.out_h; ++oy)
                        for (int ox = 0; ox < pool.out_w; ++ox)
                            one_map.push_back(pooled[(c * pool.out_h + oy) * pool.out_w + ox]);
                    mat.push_back(one_map);
                }
                write_csv_matrix("finalresults/pooling_cpu_features_sample.csv", mat, {});
            }
        }

        float total_ms = timer_total.stop_ms();
        auto mem = get_memory_usage_bytes();

        write_perf_csv("finalresults/pooling_cpu_perf.csv",
            {"total_ms", "pool_ms_sum", "n_images", "C", "K", "out_h", "out_w", 
             "mem_current_bytes", "mem_peak_bytes"},
            {total_ms, t_pool_sum, (float)test_count, (float)pool.C, (float)pool.K,
             (float)pool.out_h, (float)pool.out_w, (float)mem.first, (float)mem.second});

        std::cout << "✅ Pooling CPU completed.\n"
                  << "   Total time: " << total_ms << " ms\n"
                  << "   Avg per image: " << (t_pool_sum / test_count) << " ms\n"
                  << "   Images processed: " << test_count << "\n"
                  << "   Kernel size: " << pool.K << "\n"
                  << "   Output dims: " << pool.out_h << "x" << pool.out_w << "\n";

        return 0;
    }
    catch (std::exception &e) {
        std::cerr << "❌ Error: " << e.what() << "\n";
        return 1;
    }
}

// Main entry for standalone testing
int main() {
    return run_pooling_cpu_standalone();
}