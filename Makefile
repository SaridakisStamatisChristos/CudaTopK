CXX ?= c++
CXXFLAGS ?= -O3 -Wall -Wextra -Wpedantic
BUILD ?= build

CPPFLAGS += -Iinclude
HOST_FLAGS = $(CXXFLAGS) -std=c++17

.PHONY: all test clean
all: $(BUILD)/libcudatopk.a
test: $(BUILD)/test_cpu $(BUILD)/test_merge
	$(BUILD)/test_cpu
	$(BUILD)/test_merge

$(BUILD):
	mkdir -p $@

$(BUILD)/cpu.o: src/cpu.cpp include/cudatopk/topk.hpp src/key.hpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(HOST_FLAGS) -c $< -o $@

$(BUILD)/libcudatopk.a: $(BUILD)/cpu.o
	$(AR) rcs $@ $^

$(BUILD)/test_cpu: tests/test_cpu.cpp tests/cases.hpp $(BUILD)/libcudatopk.a
	$(CXX) $(CPPFLAGS) $(HOST_FLAGS) $< $(BUILD)/libcudatopk.a $(LDFLAGS) -o $@

$(BUILD)/test_merge: tests/test_merge.cpp tests/cases.hpp src/tiled.hpp include/cudatopk/topk.hpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(HOST_FLAGS) $< $(LDFLAGS) -o $@

clean:
	$(RM) -- $(addprefix $(BUILD)/,cpu.o cuda.o libcudatopk.a libcudatopk_cuda.a test_cpu test_merge test_cuda benchmark_cpu benchmark_cuda)

NVCC ?= nvcc
NVCCFLAGS ?= -O3 -lineinfo
CUDA_ARCH ?= 86
CUDA_FLAGS = $(NVCCFLAGS) -std=c++17 -gencode arch=compute_$(CUDA_ARCH),code=[sm_$(CUDA_ARCH),compute_$(CUDA_ARCH)]

.PHONY: cuda test-gpu
cuda: $(BUILD)/libcudatopk_cuda.a $(BUILD)/test_cuda
test-gpu: cuda
	$(BUILD)/test_cuda

$(BUILD)/cuda.o: src/cuda.cu src/key.hpp src/tiled.hpp include/cudatopk/cuda.hpp include/cudatopk/topk.hpp | $(BUILD)
	$(NVCC) $(CPPFLAGS) $(CUDA_FLAGS) -c $< -o $@

$(BUILD)/libcudatopk_cuda.a: $(BUILD)/cuda.o
	$(AR) rcs $@ $^

$(BUILD)/test_cuda: tests/test_cuda.cu tests/cases.hpp tests/cuda_support.hpp $(BUILD)/libcudatopk_cuda.a
	$(NVCC) $(CPPFLAGS) $(CUDA_FLAGS) $< $(BUILD)/libcudatopk_cuda.a -o $@

.PHONY: bench bench-cuda
bench: $(BUILD)/benchmark_cpu
bench-cuda: $(BUILD)/benchmark_cuda

$(BUILD)/benchmark_cpu: bench/benchmark.cu tests/cases.hpp $(BUILD)/libcudatopk.a
	$(CXX) $(CPPFLAGS) $(HOST_FLAGS) -x c++ $< -x none $(BUILD)/libcudatopk.a $(LDFLAGS) -o $@

$(BUILD)/benchmark_cuda: bench/benchmark.cu tests/cases.hpp tests/cuda_support.hpp $(BUILD)/libcudatopk.a $(BUILD)/libcudatopk_cuda.a
	$(NVCC) $(CPPFLAGS) $(CUDA_FLAGS) -DCUDATOPK_BENCH_CUDA $< $(BUILD)/libcudatopk.a $(BUILD)/libcudatopk_cuda.a -o $@
