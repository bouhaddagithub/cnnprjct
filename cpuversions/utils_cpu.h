#ifndef UTILS_CPU_H
#define UTILS_CPU_H

#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <tuple>

// Timer
struct TimerCPU {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    float stop_ms() {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float,std::milli> d = t1 - t0;
        return d.count();
    }
};

// MNIST loader (ubyte)
std::vector<unsigned char> load_mnist_images(const std::string &path, int &count, int &rows, int &cols);
std::vector<unsigned char> load_mnist_labels(const std::string &path, int &count);

// CSV/Csv-meta loader
// Loads flat float vector and optionally fills 'shape' if meta found
std::vector<float> load_csv_weights(const std::string &path, std::vector<int> &shape);

// helper to parse meta JSON-like file that contains "shape":[a,b,...]
bool read_shape_from_meta(const std::string &meta_path, std::vector<int> &shape);

// CSV writer (matrix of floats)
void write_csv_matrix(const std::string &path, const std::vector<std::vector<float>> &rows, const std::vector<std::string> &headers = {});

// write perf single-row CSV with headers
void write_perf_csv(const std::string &path, const std::vector<std::string> &headers, const std::vector<float> &values);

// memory (current & peak) in bytes
std::pair<size_t, size_t> get_memory_usage_bytes();

// argmax
int argmax(const std::vector<float> &v);

// softmax
std::vector<float> softmax(const std::vector<float> &x);

#endif 
