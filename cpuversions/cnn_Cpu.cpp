// cnn_cpu.cpp

#include "utils_cpu.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

int main(){
    try {
     
        int n_images, rows, cols;
        auto images = load_mnist_images("data/t10k-images-idx3-ubyte", n_images, rows, cols);
        int n_labels;
        auto labels = load_mnist_labels("data/t10k-labels-idx1-ubyte", n_labels);
        if(n_images != n_labels) std::cerr<<"Warning: image/label counts differ\n";

      
        std::vector<int> conv_shape, conv_bias_shape, fc_shape, fc_bias_shape;
        auto conv_w = load_csv_weights("exports/cnn_only/conv_weight.csv", conv_shape); 
        auto conv_b = load_csv_weights("exports/cnn_only/conv_bias.csv", conv_bias_shape); 
        auto fc_w   = load_csv_weights("exports/cnn_only/fc_weight.csv", fc_shape); 
        auto fc_b   = load_csv_weights("exports/cnn_only/fc_bias.csv", fc_bias_shape); 

        // infer shapes if meta missing
        int outC = (conv_shape.size()>=1 ? conv_shape[0] : 8);
        int inC  = (conv_shape.size()>=2 ? conv_shape[1] : 1);
        int K = (conv_shape.size()>=3 ? conv_shape[2] : 5);
        int H = rows, W = cols;
        int out_h = H - K + 1, out_w = W - K + 1;
        int D = outC * out_h * out_w;
        int out_dim = (fc_shape.size()>=1 ? fc_shape[0] : 10);

        TimerCPU timer_total; timer_total.start();
        float t_conv_sum = 0.0f, t_fc_sum = 0.0f;
        std::vector<std::vector<float>> classifications;
        int correct=0;

        for(int i=0;i<std::min(n_images,n_labels);++i){
         
            std::vector<float> in((size_t)H*W);
            for(int p=0;p<H*W;++p) in[p] = images[(size_t)i*H*W + p] / 255.0f;

            // conv
            TimerCPU t; t.start();
            std::vector<float> conv_out((size_t)outC * out_h * out_w, 0.0f);
            for(int oc=0; oc<outC; ++oc){
                for(int oy=0; oy<out_h; ++oy){
                    for(int ox=0; ox<out_w; ++ox){
                        float s = conv_b.size()>oc ? conv_b[oc] : 0.0f;
                        for(int ic=0; ic<inC; ++ic){
                            for(int ky=0; ky<K; ++ky){
                                for(int kx=0; kx<K; ++kx){
                                    int in_y = oy+ky;
                                    int in_x = ox+kx;
                                    int in_idx = (ic*H + in_y)*W + in_x;
                                    int w_idx = ((oc*inC + ic)*K + ky)*K + kx;
                                    s += in[in_idx] * conv_w[w_idx];
                                }
                            }
                        }
                        conv_out[(oc*out_h + oy)*out_w + ox] = std::max(0.0f, s);
                    }
                }
            }
            t_conv_sum += t.stop_ms();

            // FC (flatten conv_out -> fc_w)
            TimerCPU t2; t2.start();
            // conv_out size D
            std::vector<float> outv(out_dim, 0.0f);
            for(int o=0;o<out_dim;++o){
                float s = fc_b.size()>o ? fc_b[o] : 0.0f;
                for(int d=0; d<D; ++d){
                    s += fc_w[o*D + d] * conv_out[d];
                }
                outv[o] = s;
            }
            t_fc_sum += t2.stop_ms();

            int pred = argmax(outv);
            classifications.push_back({(float)i, (float)labels[i], (float)pred});
            if(pred == labels[i]) ++correct;
        }

        float total_ms = timer_total.stop_ms();
        double acc = 100.0 * correct / std::min(n_images,n_labels);
        auto mem = get_memory_usage_bytes();

        // write perf
        write_perf_csv("finalresults/cnn_cpu_perf.csv",
            {"total_ms","conv_ms_sum","fc_ms_sum","accuracy_percent","n_images","D","out_dim","mem_current_bytes","mem_peak_bytes"},
            { total_ms, t_conv_sum, t_fc_sum, (float)acc, (float)std::min(n_images,n_labels), (float)D, (float)out_dim, (float)mem.first, (float)mem.second });

        // write classification
        write_csv_matrix("finalresults/cnn_cpu_classification.csv", classifications, {"index","label","prediction"});

        std::cout<<"CNN CPU: total_ms="<<total_ms<<" acc="<<acc<<"%\n";
        return 0;
    } catch(std::exception &e){
        std::cerr<<"Error: "<<e.what()<<"\n";
        return 1;
    }
}