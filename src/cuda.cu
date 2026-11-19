#include "cudatopk/cuda.hpp"

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>

#include "key.hpp"

#include <algorithm>
#include <limits>

namespace cudatopk {
namespace {

using Key = std::uint64_t;
constexpr int threads = 256;
constexpr std::size_t workspace_alignment = 256;

int grid_for(int n) noexcept {
    return std::min(65535, 1 + (n - 1) / threads);
}

__global__ void pack_keys(const float* input, Key* keys, int n) {
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < static_cast<std::size_t>(n); i += stride) {
        keys[i] = detail::rank_key(__float_as_uint(input[i]), static_cast<std::uint32_t>(i));
    }
}

__device__ void write_entry(const float* input, Entry* output, Key key) {
    const auto index = detail::key_index(key);
    output->index = index;
    output->value = input[index];
}

__global__ void gather(const float* input, const Key* keys, Entry* output, int k) {
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < static_cast<std::size_t>(k); i += stride) {
        write_entry(input, output + i, keys[i]);
    }
}

}  // namespace

cudaError_t make_gpu_plan(std::size_t n, std::size_t k, GpuAlgorithm algorithm,
                          GpuPlan& plan) noexcept {
    static_assert(sizeof(std::size_t) >= 8, "CudaTopK CUDA requires a 64-bit host");
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        k > n || algorithm != GpuAlgorithm::radix_sort) {
        return cudaErrorInvalidValue;
    }

    GpuPlan next;
    next.n_ = static_cast<int>(n);
    next.k_ = static_cast<int>(k);
    next.algorithm_ = algorithm;
    if (k != 0) {
        std::size_t sort_bytes = 0;
        const auto status = cub::DeviceRadixSort::SortKeysDescending(
            nullptr, sort_bytes, static_cast<const Key*>(nullptr),
            static_cast<Key*>(nullptr), next.n_);
        if (status != cudaSuccess) {
            return status;
        }
        const auto arrays_bytes = 2 * n * sizeof(Key);
        next.sort_offset_ = (arrays_bytes + workspace_alignment - 1) & ~(workspace_alignment - 1);
        next.workspace_bytes_ = next.sort_offset_ + sort_bytes;
    }
    plan = next;
    return cudaSuccess;
}

cudaError_t select_gpu(const GpuPlan& plan, const float* input, Entry* output,
                       void* workspace, std::size_t workspace_bytes, cudaStream_t stream) noexcept {
    if (plan.k_ == 0) {
        return cudaSuccess;
    }
    if (input == nullptr || output == nullptr || workspace == nullptr ||
        workspace_bytes < plan.workspace_bytes_ ||
        reinterpret_cast<std::uintptr_t>(workspace) % workspace_alignment != 0) {
        return cudaErrorInvalidValue;
    }

    auto* keys = static_cast<Key*>(workspace);
    auto* sorted = keys + plan.n_;
    pack_keys<<<grid_for(plan.n_), threads, 0, stream>>>(input, keys, plan.n_);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    auto* temp = static_cast<unsigned char*>(workspace) + plan.sort_offset_;
    auto temp_bytes = plan.workspace_bytes_ - plan.sort_offset_;
    status = cub::DeviceRadixSort::SortKeysDescending(
        temp, temp_bytes, keys, sorted, plan.n_, 0, 64, stream);
    if (status != cudaSuccess) {
        return status;
    }
    gather<<<grid_for(plan.k_), threads, 0, stream>>>(input, sorted, output, plan.k_);
    return cudaGetLastError();
}

}  // namespace cudatopk
