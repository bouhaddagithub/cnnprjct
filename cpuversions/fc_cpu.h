// fc_cpu.h
#pragma once
#include "utils_cpu.h"
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>

// ============================================================
// Fully Connected Layer CPU Interface
// ============================================================
struct FCLayerCPU {
    std::vector<float> weight;
    std::vector<float> bias;
    int in_dim;
    int out_dim;
};

// Load FC layer weights from CSV files
FCLayerCPU load_fc_weights_cpu(const std::string &weight_path,
                                const std::string &bias_path);

// Run forward pass through FC layer (matrix-vector multiply + bias)
std::vector<float> fc_forward_cpu(const std::vector<float> &in,
                                   const FCLayerCPU &fc,
                                   float &t_fc_ms);

// Apply ReLU activation in-place
void relu_forward_cpu(std::vector<float> &data, float &t_relu_ms);

// Standalone runner for performance testing
int run_fc_cpu_standalone();