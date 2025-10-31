// pooling_cpu.cpp

#include "utils_cpu.h"
#include <iostream>
#include <vector>
#include <algorithm>

int main(){
    try {
        int n_images, rows, cols;
        auto images = load_mnist_images("data/t10k-images-idx3-ubyte", n_images, rows, cols);
        int n_labels; auto labels = load_mnist_labels("data/t10k-labels-idx1-ubyte", n_labels);

        int pool_k = 2;
        int H = rows, W = cols;
        int pool_h = H / pool_k;
        int pool_w = W / pool_k;

        TimerCPU t_total; t_total.start();
        float t_pool_sum=0.0f;
        int correct=0;
        std::vector<std::vector<float>> classifications;

        for(int i=0;i<std::min(n_images,n_labels);++i){
            std::vector<float> in((size_t)H * W);
            for(int p=0;p<H*W;++p) in[p] = images[(size_t)i*H*W + p] / 255.0f;

            TimerCPU t1; t1.start();
            std::vector<float> pooled((size_t)pool_h * pool_w, 0.0f);
            for(int oy=0; oy<pool_h; ++oy){
                for(int ox=0; ox<pool_w; ++ox){
                    float best = -1e9f;
                    for(int ky=0; ky<pool_k; ++ky){
                        for(int kx=0; kx<pool_k; ++kx){
                            int in_y = oy * pool_k + ky;
                            int in_x = ox * pool_k + kx;
                            int idx = in_y * W + in_x;
                            float v = in[idx];
                            if(v > best) best = v;
                        }
                    }
                    pooled[oy * pool_w + ox] = best;
                }
            }
            t_pool_sum += t1.stop_ms();

            // flatten and argmax (no FC, just for demo)
            int pred = argmax(pooled);
            classifications.push_back({(float)i, (float)labels[i], (float)pred});
            if(pred == labels[i]) ++correct;
        }

        float total_ms = t_total.stop_ms();
        double acc = 100.0 * correct / std::min(n_images,n_labels);
        auto mem = get_memory_usage_bytes();

        write_perf_csv("finalresults/pooling_cpu_perf.csv",
            {"total_ms","pool_ms_sum","accuracy_percent","n_images","K","out_h","out_w","mem_current_bytes","mem_peak_bytes"},
            { total_ms, t_pool_sum, (float)acc, (float)std::min(n_images,n_labels), (float)pool_k, (float)pool_h, (float)pool_w, (float)mem.first, (float)mem.second });

        write_csv_matrix("finalresults/pooling_cpu_test_preds.csv", classifications, {"index","label","prediction"});
        std::cout<<"Pooling CPU: total_ms="<<total_ms<<" acc="<<acc<<"%\n";
        return 0;
    } catch(std::exception &e){
        std::cerr<<"Error: "<<e.what()<<"\n";
        return 1;
    }
}
