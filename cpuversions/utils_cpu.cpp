#include "utils_cpu.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <psapi.h>
#else
  #include <sys/resource.h>
  #include <unistd.h>
#endif

// --- MNIST loader ---
static inline uint32_t bswap32_u(uint32_t x){
    return ((x>>24)&0xff) | ((x<<8)&0xff0000) | ((x>>8)&0xff00) | ((x<<24)&0xff000000);
}

std::vector<unsigned char> load_mnist_images(const std::string &path, int &count, int &rows, int &cols) {
    std::ifstream f(path, std::ios::binary);
    if(!f) throw std::runtime_error("Cannot open MNIST images: "+path);
    uint32_t magic=0, n=0, r=0, c=0;
    f.read(reinterpret_cast<char*>(&magic),4); magic = bswap32_u(magic);
    f.read(reinterpret_cast<char*>(&n),4); n = bswap32_u(n);
    f.read(reinterpret_cast<char*>(&r),4); r = bswap32_u(r);
    f.read(reinterpret_cast<char*>(&c),4); c = bswap32_u(c);
    count = (int)n; rows=(int)r; cols=(int)c;
    std::vector<unsigned char> data((size_t)count * rows * cols);
    f.read(reinterpret_cast<char*>(data.data()), data.size());
    return data;
}

std::vector<unsigned char> load_mnist_labels(const std::string &path, int &count) {
    std::ifstream f(path, std::ios::binary);
    if(!f) throw std::runtime_error("Cannot open MNIST labels: "+path);
    uint32_t magic=0, n=0;
    f.read(reinterpret_cast<char*>(&magic),4); magic = bswap32_u(magic);
    f.read(reinterpret_cast<char*>(&n),4); n = bswap32_u(n);
    count = (int)n;
    std::vector<unsigned char> data((size_t)count);
    f.read(reinterpret_cast<char*>(data.data()), data.size());
    return data;
}

// --- read shape from meta file (if exists) ---
bool read_shape_from_meta(const std::string &meta_path, std::vector<int> &shape) {
    std::ifstream f(meta_path);
    if(!f) return false;
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::smatch m;
    std::regex re("\"shape\"\\s*:\\s*\\[([^\\]]+)\\]");
    if(std::regex_search(s, m, re)) {
        std::string inner = m[1].str();
        std::stringstream ss(inner);
        std::string tok;
        shape.clear();
        while(std::getline(ss, tok, ',')) {
            // trim
            size_t a = tok.find_first_not_of(" \t\r\n");
            size_t b = tok.find_last_not_of(" \t\r\n");
            if(a==std::string::npos) continue;
            std::string num = tok.substr(a, b-a+1);
            shape.push_back(std::stoi(num));
        }
        return !shape.empty();
    }
    return false;
}

// --- load csv weights robustly ---
std::vector<float> load_csv_weights(const std::string &path, std::vector<int> &shape) {
    std::ifstream f(path);
    if(!f) throw std::runtime_error("Cannot open CSV: "+path);
    std::vector<std::string> lines;
    std::string line;
    while(std::getline(f,line)) {
        // trim
        size_t a = line.find_first_not_of(" \t\r\n");
        if(a==std::string::npos) continue;
        size_t b = line.find_last_not_of(" \t\r\n");
        lines.push_back(line.substr(a, b-a+1));
    }
    // check first line for meta
    bool meta=false;
    if(!lines.empty()) {
        std::string first = lines[0];
        std::smatch m;
        std::regex re("\"shape\"\\s*:\\s*\\[([^\\]]+)\\]");
        if(std::regex_search(first, m, re)) {
            std::string inner = m[1].str();
            std::stringstream ss(inner);
            std::string tok;
            shape.clear();
            while(std::getline(ss, tok, ',')) {
                size_t a = tok.find_first_not_of(" \t\r\n");
                size_t b = tok.find_last_not_of(" \t\r\n");
                if(a==std::string::npos) continue;
                shape.push_back(std::stoi(tok.substr(a, b-a+1)));
            }
            meta=true;
        }
    }
    size_t start = meta ? 1 : 0;
    std::vector<float> data;
    for(size_t i=start;i<lines.size();++i){
        std::stringstream ss(lines[i]);
        std::string tok;
        while(std::getline(ss, tok, ',')) {
            size_t a = tok.find_first_not_of(" \t\r\n");
            if(a==std::string::npos) continue;
            size_t b = tok.find_last_not_of(" \t\r\n");
            std::string num = tok.substr(a, b-a+1);
            try { data.push_back(std::stof(num)); } catch(...) {}
        }
    }
    // if no meta infer shape: if one row -> shape = [1, N] else [rows, cols] can't know rows for this loader
    if(!meta && !data.empty() && shape.empty()) {
        // leave shape empty; callers can set expected shape
    }
    return data;
}

// --- CSV writer ---
void write_csv_matrix(const std::string &path, const std::vector<std::vector<float>> &rows, const std::vector<std::string> &headers) {
    std::ofstream f(path);
    if(!f) { std::cerr<<"Cannot write "<<path<<"\n"; return; }
    if(!headers.empty()){
        for(size_t i=0;i<headers.size();++i){
            f<<headers[i];
            if(i+1<headers.size()) f<<",";
        }
        f<<"\n";
    }
    for(auto &r: rows){
        for(size_t i=0;i<r.size();++i){
            f<<r[i];
            if(i+1<r.size()) f<<",";
        }
        f<<"\n";
    }
}

// perf csv writer
void write_perf_csv(const std::string &path, const std::vector<std::string> &headers, const std::vector<float> &values) {
    std::ofstream f(path);
    if(!f) { std::cerr<<"Cannot write "<<path<<"\n"; return; }
    for(size_t i=0;i<headers.size();++i){
        f<<headers[i];
        if(i+1<headers.size()) f<<",";
    }
    f<<"\n";
    for(size_t i=0;i<values.size();++i){
        f<<values[i];
        if(i+1<values.size()) f<<",";
    }
    f<<"\n";
}

// --- memory usage ---
std::pair<size_t,size_t> get_memory_usage_bytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        SIZE_T cur = pmc.WorkingSetSize;
        SIZE_T peak = pmc.PeakWorkingSetSize;
        return { (size_t)cur, (size_t)peak };
    }
    return {0,0};
#else
    struct rusage usage;
    if(getrusage(RUSAGE_SELF, &usage) == 0) {
        // ru_maxrss in kilobytes on Linux
        size_t peak = (size_t)usage.ru_maxrss * 1024;
        // current RSS: parse /proc/self/statm or /proc/self/status
        size_t current = 0;
        std::ifstream f("/proc/self/statm");
        if(f) {
            size_t size,resident,share,trs,drs,total;
            if(f>>size>>resident>>share>>trs>>drs>>total) {
                long page = sysconf(_SC_PAGESIZE);
                current = resident * page;
            }
        }
        return {current, peak};
    }
    return {0,0};
#endif
}

// argmax
int argmax(const std::vector<float> &v) {
    if(v.empty()) return 0;
    return (int)(std::max_element(v.begin(), v.end()) - v.begin());
}

// softmax
std::vector<float> softmax(const std::vector<float> &x){
    std::vector<float> out(x.size());
    float m = *std::max_element(x.begin(), x.end());
    double s=0.0;
    for(size_t i=0;i<x.size();++i){ out[i]=std::exp(x[i]-m); s+=out[i]; }
    for(size_t i=0;i<out.size();++i) out[i]/=(float)s;
    return out;
}
