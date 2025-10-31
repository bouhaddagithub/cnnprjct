// pooling_cpu.h
#pragma once
#include "utils_cpu.h"
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>

// ============================================================
// Pooling Layer CPU Interface (MaxPool only)
// ============================================================
struct PoolingLayerCPU {
    int K;          // kernel size
    int C;          // channels
    int H;          // input height
    int W;          // input width
    int out_h;      // output height
    int out_w;      // output width
};

// Load pooling metadata (kernel size) from pooling.meta.json
PoolingLayerCPU load_pooling_meta_cpu(const std::string &meta_path,
                                      int C, int H, int W);

// Run forward max pooling
std::vector<float> pooling_forward_cpu(const std::vector<float> &in,
                                       const PoolingLayerCPU &pool,
                                       float &t_pool_ms);

// Standalone runner for performance testing
int run_pooling_cpu_standalone();