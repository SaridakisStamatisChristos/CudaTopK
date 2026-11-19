#pragma once

#include "cudatopk/topk.hpp"

#include <cuda_runtime_api.h>

namespace cudatopk {

// Tiled accepts k <= 256 and reduction accepts k <= 1. Automatic uses
// reduction for k = 1, tiled for other small k, and radix sort otherwise.
enum class GpuAlgorithm { automatic, radix_sort, tiled, reduction };

class GpuPlan {
public:
    std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }
    GpuAlgorithm algorithm() const noexcept { return algorithm_; }

private:
    int n_ = 0;
    int k_ = 0;
    GpuAlgorithm algorithm_ = GpuAlgorithm::radix_sort;
    std::size_t workspace_bytes_ = 0;
    std::size_t second_offset_ = 0;

    friend cudaError_t make_gpu_plan(std::size_t, std::size_t, GpuAlgorithm, GpuPlan&) noexcept;
    friend cudaError_t select_gpu(const GpuPlan&, const float*, Entry*, void*, std::size_t,
                                  cudaStream_t) noexcept;
};

// Prepare the shape on the CUDA device that will execute it. On error, plan
// retains its previous value. A default plan is equivalent to n = k = 0.
cudaError_t make_gpu_plan(std::size_t n, std::size_t k, GpuAlgorithm algorithm,
                          GpuPlan& plan) noexcept;

// Enqueue selection with the same ordering as select_cpu, without allocation
// or synchronization. Execute on the device used to prepare the plan.
// For k > 0, input and output must be disjoint device
// accessible buffers, with n and k elements respectively. If needed, workspace
// must be disjoint, at least workspace_bytes() bytes, and 256-byte aligned
// (as provided by cudaMalloc). Keep buffers alive until stream completion.
// Use separate buffers for overlapping executions. Observe asynchronous
// execution failures by synchronizing the stream.
cudaError_t select_gpu(const GpuPlan& plan, const float* input, Entry* output,
                       void* workspace, std::size_t workspace_bytes,
                       cudaStream_t stream) noexcept;

}  // namespace cudatopk
