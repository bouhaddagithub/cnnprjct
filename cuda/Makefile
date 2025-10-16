# Makefile for cuda kernels
NVCC = nvcc
NVCCFLAGS = -std=c++14 -O2 -Xcompiler -fPIC

all: pooling_gpu fc_gpu pipeline_gpu

pooling_gpu: pooling_gpu.cu cuda_utils.h
	$(NVCC) $(NVCCFLAGS) -o pooling_gpu pooling_gpu.cu

fc_gpu: fc_gpu.cu cuda_utils.h
	$(NVCC) $(NVCCFLAGS) -o fc_gpu fc_gpu.cu

pipeline_gpu: pipeline_gpu.cu cuda_utils.h
	$(NVCC) $(NVCCFLAGS) -o pipeline_gpu pipeline_gpu.cu

clean:
	rm -f pooling_gpu fc_gpu pipeline_gpu *.o *.csv
