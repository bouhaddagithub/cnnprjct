// cnn_cpu.cpp

#include "utils_cpu.h"
#include <iostream>
#include <vector>
#include <algorithm>

int main(){
    try {
        int n_images, rows, cols;
        auto images = load_mnist_images("data/t10k-images-idx3-ubyte", n_images, rows, cols);
        int n_labels; auto labels = load_mnist_labels("data/t10k-labels-idx1-ubyte", n_labels);

        std::vector<int> conv_w_shape, conv_b_shape;
        auto conv_w = load_csv_weights("exports/cnn_only/conv_weight.csv", conv_w_shape);
        auto conv_b = load_csv_weights("exports/cnn_only/conv_bias.csv", conv_b_shape);

        int outC = (conv_w_shape.size()>=1 ? conv_w_shape[0] : 8);
        int inC = (conv_w_shape.size()>=2 ? conv_w_shape[1] : 1);
        int K = (conv_w_shape.size()>=3 ? conv_w_shape[2] : 5);
        int H = rows, W = cols;
        int out_h = H - K + 1;
        int out_w = W - K + 1;

        TimerCPU t_total; t_total.start();
        float t_conv_sum=0.0f;
        int correct=0;
        std::vector<std::vector<float>> classifications;

        for(int i=0;i<std::min(n_images,n_labels);++i){
            std::vector<float> in((size_t)inC * H * W, 0.0f);
            for(int p=0;p<H*W;++p) in[p] = images[(size_t)i*H*W + p] / 255.0f;

            TimerCPU t1; t1.start();
            std::vector<float> conv_out((size_t)outC * out_h * out_w, 0.0f);
            for(int oc=0; oc<outC; ++oc){
                for(int oy=0; oy<out_h; ++oy){
                    for(int ox=0; ox<out_w; ++ox){
                        float s = conv_b[oc];
                        for(int ic=0; ic<inC; ++ic){
                            for(int ky=0; ky<K; ++ky){
                                for(int kx=0; kx<K; ++kx){
                                    int in_y = oy + ky;
                                    int in_x = ox + kx;
                                    int in_idx = (ic * H + in_y) * W + in_x;
                                    int w_idx = ((oc * inC + ic) * K + ky) * K + kx;
                                    float wv = (w_idx < (int)conv_w.size()) ? conv_w[w_idx] : 0.0f;
                                    s += in[in_idx] * wv;
                                }
                            }
                        }
                        conv_out[(oc * out_h + oy) * out_w + ox] = std::max(0.0f, s);
                    }
                }
            }
            t_conv_sum += t1.stop_ms();

            // flatten and argmax (no FC, just for demo)
            int pred = argmax(conv_out);
            classifications.push_back({(float)i, (float)labels[i], (float)pred});
            if(pred == labels[i]) ++correct;
        }

        float total_ms = t_total.stop_ms();
        double acc = 100.0 * correct / std::min(n_images,n_labels);
        auto mem = get_memory_usage_bytes();

        write_perf_csv("finalresults/cnn_cpu_benchmark.csv",
            {"total_ms","conv_ms_sum","accuracy_percent","n_images","outC","inC","K","out_h","out_w","mem_current_bytes","mem_peak_bytes"},
            { total_ms, t_conv_sum, (float)acc, (float)std::min(n_images,n_labels), (float)outC, (float)inC, (float)K, (float)out_h, (float)out_w, (float)mem.first, (float)mem.second });

        write_csv_matrix("finalresults/cnn_cpu_test_preds.csv", classifications, {"index","label","prediction"});
        std::cout<<"CNN CPU: total_ms="<<total_ms<<" acc="<<acc<<"%\n";
        return 0;
    } catch(std::exception &e){
        std::cerr<<"Error: "<<e.what()<<"\n";
        return 1;
    }
}


