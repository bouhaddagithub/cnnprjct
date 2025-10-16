#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

#ifdef _MSC_VER
    #include <intrin.h>
    #define bswap32 _byteswap_ulong
#else
    #define bswap32 __builtin_bswap32
#endif

// Error check
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// MNIST loader (reads big-endian ints correctly)
std::vector<unsigned char> load_mnist_images(const std::string &path, int &count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open MNIST image file " + path);

    int magic = 0, rows = 0, cols = 0;
    file.read(reinterpret_cast<char*>(&magic), 4); magic = bswap32(magic);
    file.read(reinterpret_cast<char*>(&count), 4); count = bswap32(count);
    file.read(reinterpret_cast<char*>(&rows), 4); rows = bswap32(rows);
    file.read(reinterpret_cast<char*>(&cols), 4); cols = bswap32(cols);

    if (count <= 0 || rows <= 0 || cols <= 0) throw std::runtime_error("MNIST image header corrupted: " + path);

    std::vector<unsigned char> images((size_t)count * rows * cols);
    file.read(reinterpret_cast<char*>(images.data()), images.size());
    if (!file) throw std::runtime_error("Failed to read MNIST image data from " + path);
    return images;
}

std::vector<unsigned char> load_mnist_labels(const std::string &path, int &count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open MNIST label file " + path);
    int magic = 0;
    file.read(reinterpret_cast<char*>(&magic), 4); magic = bswap32(magic);
    file.read(reinterpret_cast<char*>(&count), 4); count = bswap32(count);
    if (count <= 0) throw std::runtime_error("MNIST label header corrupted: " + path);
    std::vector<unsigned char> labels(count);
    file.read(reinterpret_cast<char*>(labels.data()), labels.size());
    if (!file) throw std::runtime_error("Failed to read MNIST label data from " + path);
    return labels;
}

// Robust CSV loader.
// - If the first non-empty line contains a "shape":[r,c] JSON-like entry, it will be parsed.
// - All remaining numeric values (possibly across multiple lines) are parsed as floats.
// - If no meta shape provided, the function will infer rows/cols by counting columns in each data row.
std::vector<float> load_csv_weights(const std::string &path, std::vector<int> &shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open CSV file: " + path);

    shape.clear();
    std::vector<std::string> lines;
    std::string line;

    // --- Read all lines and trim whitespace/BOM ---
    while (std::getline(f, line)) {
        // Remove UTF-8 BOM if present
        if (!line.empty() && (unsigned char)line[0] == 0xEF) {
            if (line.size() >= 3 &&
                (unsigned char)line[1] == 0xBB &&
                (unsigned char)line[2] == 0xBF) {
                line = line.substr(3);
            }
        }
        // Trim spaces and skip empties
        size_t a = line.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        size_t b = line.find_last_not_of(" \t\r\n");
        lines.push_back(line.substr(a, b - a + 1));
    }
    if (lines.empty()) throw std::runtime_error("CSV file is empty: " + path);

    // --- Parse optional shape metadata ---
    bool meta_parsed = false;
    if (!lines.empty()) {
        std::string first = lines[0];
        // remove enclosing braces if any { ... }
        if (!first.empty() && first.front() == '{' && first.back() == '}')
            first = first.substr(1, first.size() - 2);

        size_t pos = first.find("\"shape\"");
        if (pos == std::string::npos) pos = first.find("shape:");
        if (pos != std::string::npos) {
            size_t start = first.find("[", pos);
            size_t end = first.find("]", start);
            if (start != std::string::npos && end != std::string::npos && end > start) {
                std::string inner = first.substr(start + 1, end - start - 1);
                std::stringstream ss(inner);
                std::string tok;
                while (std::getline(ss, tok, ',')) {
                    size_t aa = tok.find_first_not_of(" \t\r\n");
                    size_t bb = tok.find_last_not_of(" \t\r\n");
                    if (aa != std::string::npos) tok = tok.substr(aa, bb - aa + 1);
                    if (!tok.empty()) shape.push_back(std::stoi(tok));
                }
                meta_parsed = !shape.empty();
            }
        }
    }

    // --- Determine where data starts ---
    size_t data_start = (meta_parsed ? 1 : 0);

    // --- Parse numeric matrix data ---
    std::vector<std::vector<float>> rows_data;
    for (size_t i = data_start; i < lines.size(); ++i) {
        std::string &L = lines[i];
        if (L.empty() || L[0] == '{' || L[0] == '#' || L.find("shape") != std::string::npos)
            continue;

        std::stringstream ss(L);
        std::string tok;
        std::vector<float> rowvals;
        while (std::getline(ss, tok, ',')) {
            size_t aa = tok.find_first_not_of(" \t\r\n");
            if (aa == std::string::npos) continue;
            size_t bb = tok.find_last_not_of(" \t\r\n");
            std::string tv = tok.substr(aa, bb - aa + 1);
            if (tv.empty()) continue;
            try {
                rowvals.push_back(std::stof(tv));
            } catch (...) {
                // ignore invalid tokens
            }
        }
        if (!rowvals.empty()) rows_data.push_back(std::move(rowvals));
    }

    if (rows_data.empty())
        throw std::runtime_error("No numeric data found in CSV: " + path);

    // --- Infer shape if not given ---
    if (!meta_parsed) {
        size_t rows = rows_data.size();
        size_t cols = rows_data[0].size();
        bool consistent = true;
        for (auto &r : rows_data)
            if (r.size() != cols) { consistent = false; break; }

        if (rows == 1) shape = {1, (int)cols};
        else if (consistent) shape = {(int)rows, (int)cols};
        else {
            int total = 0;
            for (auto &r : rows_data) total += (int)r.size();
            shape = {total, 1};
        }
    }

    // --- Flatten ---
    std::vector<float> data;
    for (auto &r : rows_data)
        for (float v : r)
            data.push_back(v);

    return data;
}


#endif
