// pipeline_cpu.cpp


#include "utils_cpu.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

int main() {
    try {
        // --- load MNIST test set ---
        int n_images, rows, cols;
        auto images = load_mnist_images("../data/t10k-images-idx3-ubyte", n_images, rows, cols);
        int n_labels;
        auto labels = load_mnist_labels("../data/t10k-labels-idx1-ubyte", n_labels);
        int N = std::min(n_images, n_labels);

        // --- load weights from exports/pipeline ---
        std::vector<int> conv_meta, conv_meta_b, fc_meta;
        auto conv_w = load_csv_weights("../exports/pipeline/conv_weight.csv", conv_meta);
        auto conv_b = load_csv_weights("../exports/pipeline/conv_bias.csv", conv_meta_b);
        auto fc_w   = load_csv_weights("../exports/pipeline/fc_weight.csv", fc_meta);
        auto fc_b   = load_csv_weights("../exports/pipeline/fc_bias.csv", fc_meta);

        // infer conv meta if not provided
        int Kout = (conv_meta.size() >= 1 ? conv_meta[0] : 8);
        int Kin  = (conv_meta.size() >= 2 ? conv_meta[1] : 1);
        int K    = (conv_meta.size() >= 3 ? conv_meta[2] : 5);

        int H = rows, W = cols;
        int out_h = H - K + 1;
        int out_w = W - K + 1;

        int pool_k = 2;
        int pool_h = out_h / pool_k;
        int pool_w = out_w / pool_k;

        int D = Kout * pool_h * pool_w;
        int out_dim = (fc_meta.size() >= 1 ? fc_meta[0] : 10);

        
        if ((int)conv_b.size() < Kout) {
            std::cerr << "Warning: conv_bias length < Kout; using zeros for missing bias\n";
            conv_b.resize(Kout, 0.0f);
        }
        if ((int)fc_b.size() < out_dim) {
            std::cerr << "Warning: fc_bias length < out_dim; using zeros for missing bias\n";
            fc_b.resize(out_dim, 0.0f);
        }

        
        TimerCPU total_timer; total_timer.start();
        float t_conv_sum = 0.0f, t_pool_sum = 0.0f, t_fc_sum = 0.0f;

        int correct = 0;
        std::vector<std::vector<float>> classification_rows;
        classification_rows.reserve(N);

       
        for (int i = 0; i < N; ++i) {
            // normalize input
            std::vector<float> in((size_t)Kin * H * W, 0.0f);
            
            for (int p = 0; p < H * W; ++p) in[p] = images[(size_t)i * H * W + p] / 255.0f;

            // --- CONV + ReLU ---
            TimerCPU t1; t1.start();
            int conv_out_h = out_h, conv_out_w = out_w;
            std::vector<float> conv_out((size_t)Kout * conv_out_h * conv_out_w, 0.0f);
            for (int oc = 0; oc < Kout; ++oc) {
                for (int oy = 0; oy < conv_out_h; ++oy) {
                    for (int ox = 0; ox < conv_out_w; ++ox) {
                        float s = conv_b[oc];
                        for (int ic = 0; ic < Kin; ++ic) {
                            for (int ky = 0; ky < K; ++ky) {
                                for (int kx = 0; kx < K; ++kx) {
                                    int in_y = oy + ky;
                                    int in_x = ox + kx;
                                    int in_idx = (ic * H + in_y) * W + in_x;
                                    int w_idx = ((oc * Kin + ic) * K + ky) * K + kx;
                                   
                                    float wv = (w_idx < (int)conv_w.size()) ? conv_w[w_idx] : 0.0f;
                                    s += in[in_idx] * wv;
                                }
                            }
                        }
                        conv_out[(oc * conv_out_h + oy) * conv_out_w + ox] = std::max(0.0f, s);
                    }
                }
            }
            t_conv_sum += t1.stop_ms();

            // --- POOL ---
            TimerCPU t2; t2.start();
            std::vector<float> pooled((size_t)Kout * pool_h * pool_w, 0.0f);
            for (int c = 0; c < Kout; ++c) {
                for (int oy = 0; oy < pool_h; ++oy) {
                    for (int ox = 0; ox < pool_w; ++ox) {
                        float best = -1e9f;
                        for (int ky = 0; ky < pool_k; ++ky) {
                            for (int kx = 0; kx < pool_k; ++kx) {
                                int in_y = oy * pool_k + ky;
                                int in_x = ox * pool_k + kx;
                                int idx = (c * conv_out_h + in_y) * conv_out_w + in_x;
                                float v = conv_out[idx];
                                if (v > best) best = v;
                            }
                        }
                        pooled[(c * pool_h + oy) * pool_w + ox] = best;
                    }
                }
            }
            t_pool_sum += t2.stop_ms();

            // --- FC (on CPU) ---
            TimerCPU t3; t3.start();
            
            std::vector<float> outv((size_t)out_dim, 0.0f);
            for (int o = 0; o < out_dim; ++o) {
                float s = (o < (int)fc_b.size() ? fc_b[o] : 0.0f);
                for (int d = 0; d < D; ++d) {
                    int widx = o * D + d;
                    float wv = (widx < (int)fc_w.size()) ? fc_w[widx] : 0.0f;
                    s += wv * pooled[d];
                }
                outv[o] = s;
            }
            t_fc_sum += t3.stop_ms();

            // record prediction
            int pred = argmax(outv);
            classification_rows.push_back({(float)i, (float)labels[i], (float)pred});
            if (pred == labels[i]) ++correct;
        }

        float total_ms = total_timer.stop_ms();
        double accuracy = 100.0 * correct / N;
        auto mem = get_memory_usage_bytes();

        // --- write performance CSV to finalresults ---
        write_perf_csv("../finalresults/pipeline_cpu_perf.csv",
            {"total_ms","conv_ms_sum","pool_ms_sum","fc_ms_sum","accuracy_percent","n_images","D","out_dim","mem_current_bytes","mem_peak_bytes"},
            { total_ms, t_conv_sum, t_pool_sum, t_fc_sum, (float)accuracy, (float)N, (float)D, (float)out_dim, (float)mem.first, (float)mem.second });

        // --- write classification CSV to finalresults ---
        write_csv_matrix("../finalresults/pipeline_cpu_classification.csv", classification_rows, {"index","label","prediction"});

        // print summary
        std::cout << "Pipeline CPU:\n";
        std::cout << "Total: " << total_ms << " ms\n";
        std::cout << "Accuracy: " << accuracy << " %\n";
        std::cout << "Written: ../finalresults/pipeline_cpu_perf.csv and ../finalresults/pipeline_cpu_classification.csv\n";

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
