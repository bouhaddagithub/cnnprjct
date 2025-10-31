// fc_cpu.cpp

#include "utils_cpu.h"
#include <iostream>
#include <vector>
#include <algorithm>

int main(){
    try {
        int n_images, rows, cols;
        auto images = load_mnist_images("data/t10k-images-idx3-ubyte", n_images, rows, cols);
        int n_labels; auto labels = load_mnist_labels("data/t10k-labels-idx1-ubyte", n_labels);

        std::vector<int> s1, s2;
        auto fc1_w = load_csv_weights("exports/fc_only/fc1_weight.csv", s1); // [hidden, 784]
        auto fc1_b = load_csv_weights("exports/fc_only/fc1_bias.csv", s1);
        auto fc2_w = load_csv_weights("exports/fc_only/fc2_weight.csv", s2); // [10, hidden]
        auto fc2_b = load_csv_weights("exports/fc_only/fc2_bias.csv", s2);

        int hidden = (s1.size()>=1 ? s1[0] : 128);
        int K = (s1.size()>=2 ? s1[1] : 784);
        int out_dim = (s2.size()>=1 ? s2[0] : 10);

        TimerCPU t_total; t_total.start();
        float t_fc1_sum=0.0f, t_relu_sum=0.0f, t_fc2_sum=0.0f;
        int correct=0;
        std::vector<std::vector<float>> classifications;

        for(int i=0;i<std::min(n_images,n_labels);++i){
            
            std::vector<float> X((size_t)K);
            for(int p=0;p<K;++p) X[p] = images[(size_t)i*rows*cols + p] / 255.0f;

          
            TimerCPU t1; t1.start();
            std::vector<float> hidden_v((size_t)hidden);
            for(int h=0; h<hidden; ++h){
                float s = fc1_b.size()>h ? fc1_b[h] : 0.0f;
                for(int k=0;k<K;++k) s += fc1_w[h*K + k] * X[k];
                hidden_v[h] = s;
            }
            t_fc1_sum += t1.stop_ms();

            // relu
            TimerCPU t2; t2.start();
            for(auto &v: hidden_v) if(v<0) v=0;
            t_relu_sum += t2.stop_ms();

            
            TimerCPU t3; t3.start();
            std::vector<float> outv(out_dim,0.0f);
            for(int o=0;o<out_dim;++o){
                float s = fc2_b.size()>o ? fc2_b[o] : 0.0f;
                for(int h=0;h<hidden;++h) s += fc2_w[o*hidden + h] * hidden_v[h];
                outv[o] = s;
            }
            t_fc2_sum += t3.stop_ms();

            int pred = argmax(outv);
            classifications.push_back({(float)i, (float)labels[i], (float)pred});
            if(pred == labels[i]) ++correct;
        }

        float total_ms = t_total.stop_ms();
        double acc = 100.0 * correct / std::min(n_images,n_labels);
        auto mem = get_memory_usage_bytes();

        write_perf_csv("finalresults/fc_cpu_perf.csv",
            {"total_ms","fc1_ms_sum","relu_ms_sum","fc2_ms_sum","accuracy_percent","n_images","K","hidden","out_dim","mem_current_bytes","mem_peak_bytes"},
            { total_ms, t_fc1_sum, t_relu_sum, t_fc2_sum, (float)acc, (float)std::min(n_images,n_labels), (float)K, (float)hidden, (float)out_dim, (float)mem.first, (float)mem.second });

        write_csv_matrix("finalresults/fc_cpu_classification.csv", classifications, {"index","label","prediction"});
        std::cout<<"FC CPU: total_ms="<<total_ms<<" acc="<<acc<<"%\n";
        return 0;
    } catch(std::exception &e){
        std::cerr<<"Error: "<<e.what()<<"\n";
        return 1;
    }
}