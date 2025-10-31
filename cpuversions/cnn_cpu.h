// cnn_cpu.h
#pragma once
#include "utils_cpu.h"
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>

// ============================================================
// CNN Layer CPU Interface (Conv + ReLU only)
// ============================================================

struct CNNLayerCPU {
    std::vector<float> conv_w;
    std::vector<float> conv_b;
    int outC;
    int inC;
    int K;
    int H;
    int W;
    int out_h;
    int out_w;
};

// Load convolution weights and infer shapes
CNNLayerCPU load_cnn_weights_cpu(const std::string &conv_w_path,
                                 const std::string &conv_b_path,
                                 int H, int W);

// Run forward convolution + ReLU (no pooling, no FC)
std::vector<float> cnn_forward_cpu(const std::vector<float> &in,
                                   const CNNLayerCPU &net,
                                   float &t_conv_ms);

// Standalone runner for performance testing
int run_cnn_cpu_standalone();
