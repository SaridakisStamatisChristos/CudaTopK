#include "cudatopk/cuda.hpp"

#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>

#include "key.hpp"
#include "tiled.hpp"

#include <algorithm>
#include <limits>

namespace cudatopk {
namespace {

using Key = std::uint64_t;
constexpr int threads = 256;
constexpr int tile_threads = 128;
constexpr int keys_per_thread = detail::tile_items / tile_threads;
constexpr std::size_t workspace_alignment = 256;

int capped_grid(int blocks) noexcept {
    return std::min(65535, blocks);
}

int reduction_grid(int n) noexcept {
    return std::min(1024, detail::ceil_div(n, detail::tile_items));
}

int grid_for(int n) noexcept {
    return capped_grid(detail::ceil_div(n, threads));
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

// Compare packed keys so even exceptional floats have a deterministic winner.
struct MaxKey {
    __device__ Key operator()(Key a, Key b) const { return a > b ? a : b; }
};

using BlockMax = cub::BlockReduce<Key, threads>;

__global__ void reduce_input(const float* input, int n, Key* partials, Entry* output) {
    __shared__ BlockMax::TempStorage storage;
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    Key best = 0;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < static_cast<std::size_t>(n); i += stride) {
        best = MaxKey{}(best, detail::rank_key(__float_as_uint(input[i]), static_cast<std::uint32_t>(i)));
    }
    const Key winner = BlockMax(storage).Reduce(best, MaxKey{});
    if (threadIdx.x == 0) {
        if (partials) {
            partials[blockIdx.x] = winner;
        } else {
            write_entry(input, output, winner);
        }
    }
}

__global__ void reduce_partials(const float* input, const Key* partials, int count, Entry* output) {
    __shared__ BlockMax::TempStorage storage;
    Key best = 0;
    for (int i = threadIdx.x; i < count; i += threads) {
        best = MaxKey{}(best, partials[i]);
    }
    const Key winner = BlockMax(storage).Reduce(best, MaxKey{});
    if (threadIdx.x == 0) {
        write_entry(input, output, winner);
    }
}

cudaError_t select_one(const float* input, Entry* output, int n, Key* scratch,
                       cudaStream_t stream) {
    const int blocks = reduction_grid(n);
    reduce_input<<<blocks, threads, 0, stream>>>(input, n, blocks == 1 ? nullptr : scratch, output);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess || blocks == 1) {
        return status;
    }
    reduce_partials<<<1, threads, 0, stream>>>(input, scratch, blocks, output);
    return cudaGetLastError();
}

// Block radix sort starts from blocked registers and ends in striped order.
template <bool DirectOutput>
__global__ void select_tiles(const float* input, int n, int k, int tiles,
                             Key* runs, Entry* output) {
    using Sort = cub::BlockRadixSort<Key, tile_threads, keys_per_thread>;
    __shared__ typename Sort::TempStorage storage;
    for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const auto start = static_cast<std::size_t>(tile) * detail::tile_items;
        Key local[keys_per_thread];
        for (int i = 0; i < keys_per_thread; ++i) {
            // Threads read striped positions for coalesced input. Any initial
            // register permutation is valid since the packed keys are unique.
            const auto index = start + threadIdx.x + i * tile_threads;
            local[i] = index < static_cast<std::size_t>(n)
                ? detail::rank_key(__float_as_uint(input[index]), static_cast<std::uint32_t>(index))
                : 0;
        }
        Sort(storage).SortDescendingBlockedToStriped(local);
        for (int i = 0; i < keys_per_thread; ++i) {
            const int rank = threadIdx.x + i * tile_threads;
            if (rank < k && local[i] != 0) {
                if constexpr (DirectOutput) {
                    write_entry(input, output + rank, local[i]);
                } else {
                    runs[static_cast<std::size_t>(tile) * k + rank] = local[i];
                }
            }
        }
        __syncthreads();
    }
}

__global__ void merge_runs(const Key* input, Key* output, int n, int k,
                           std::size_t span, int groups) {
    for (int group = blockIdx.x; group < groups; group += gridDim.x) {
        const auto left_start = 2 * static_cast<std::size_t>(group) * span;
        const int left_count = detail::run_length(n, k, span, left_start);
        const int right_count = detail::run_length(n, k, span, left_start + span);
        const auto base = static_cast<std::size_t>(group) * k;
        const auto* left = input + 2 * base;
        const auto* right = left + k;
        const int rank = threadIdx.x;
        if (rank < left_count) {
            const Key key = left[rank];
            const int merged = rank + detail::count_better(right, right_count, key);
            if (merged < k) {
                output[base + merged] = key;
            }
        }
        if (rank < right_count) {
            const Key key = right[rank];
            const int merged = rank + detail::count_better(left, left_count, key);
            if (merged < k) {
                output[base + merged] = key;
            }
        }
    }
}

cudaError_t select_tiled(const float* input, Entry* output, int n, int k,
                         Key* scratch, std::size_t second_offset, cudaStream_t stream) {
    int runs = detail::ceil_div(n, detail::tile_items);
    if (runs == 1) {
        select_tiles<true><<<1, tile_threads, 0, stream>>>(input, n, k, runs, nullptr, output);
        return cudaGetLastError();
    }

    auto* current = scratch;
    auto* next = scratch + second_offset / sizeof(Key);
    select_tiles<false><<<capped_grid(runs), tile_threads, 0, stream>>>(
        input, n, k, runs, current, nullptr);
    auto status = cudaGetLastError();
    std::size_t span = detail::tile_items;
    while (status == cudaSuccess && runs > 1) {
        runs = detail::ceil_div(runs, 2);
        merge_runs<<<capped_grid(runs), threads, 0, stream>>>(current, next, n, k, span, runs);
        status = cudaGetLastError();
        std::swap(current, next);
        span *= 2;
    }
    if (status != cudaSuccess) {
        return status;
    }
    gather<<<grid_for(k), threads, 0, stream>>>(input, current, output, k);
    return cudaGetLastError();
}

}  // namespace

cudaError_t make_gpu_plan(std::size_t n, std::size_t k, GpuAlgorithm algorithm,
                          GpuPlan& plan) noexcept {
    static_assert(sizeof(std::size_t) >= 8, "CudaTopK CUDA requires a 64-bit host");
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        k > n) {
        return cudaErrorInvalidValue;
    }

    if (algorithm == GpuAlgorithm::automatic) {
        if (k == 1) {
            algorithm = GpuAlgorithm::reduction;
        } else {
            algorithm = k <= detail::tiled_max_k ? GpuAlgorithm::tiled : GpuAlgorithm::radix_sort;
        }
    }
    switch (algorithm) {
    case GpuAlgorithm::reduction:
        if (k > 1) return cudaErrorInvalidValue;
        break;
    case GpuAlgorithm::tiled:
        if (k > detail::tiled_max_k) return cudaErrorInvalidValue;
        break;
    case GpuAlgorithm::radix_sort:
        break;
    default:
        return cudaErrorInvalidValue;
    }

    GpuPlan next;
    next.n_ = static_cast<int>(n);
    next.k_ = static_cast<int>(k);
    next.algorithm_ = algorithm;
    if (algorithm == GpuAlgorithm::reduction && k != 0) {
        const int blocks = reduction_grid(next.n_);
        next.workspace_bytes_ = blocks > 1 ? blocks * sizeof(Key) : 0;
    }
    if (algorithm == GpuAlgorithm::tiled && k != 0) {
        const auto tiles = detail::ceil_div(next.n_, detail::tile_items);
        if (tiles > 1) {
            next.second_offset_ = static_cast<std::size_t>(tiles) * k * sizeof(Key);
            next.workspace_bytes_ = next.second_offset_ + detail::ceil_div(tiles, 2) * k * sizeof(Key);
        }
    }
    if (algorithm == GpuAlgorithm::radix_sort && k != 0) {
        std::size_t sort_bytes = 0;
        const auto status = cub::DeviceRadixSort::SortKeysDescending(
            nullptr, sort_bytes, static_cast<const Key*>(nullptr),
            static_cast<Key*>(nullptr), next.n_);
        if (status != cudaSuccess) {
            return status;
        }
        const auto arrays_bytes = 2 * n * sizeof(Key);
        next.second_offset_ = (arrays_bytes + workspace_alignment - 1) & ~(workspace_alignment - 1);
        next.workspace_bytes_ = next.second_offset_ + sort_bytes;
    }
    plan = next;
    return cudaSuccess;
}

cudaError_t select_gpu(const GpuPlan& plan, const float* input, Entry* output,
                       void* workspace, std::size_t workspace_bytes, cudaStream_t stream) noexcept {
    if (plan.k_ == 0) {
        return cudaSuccess;
    }
    if (input == nullptr || output == nullptr ||
        (plan.workspace_bytes_ != 0 &&
         (workspace == nullptr || workspace_bytes < plan.workspace_bytes_ ||
          reinterpret_cast<std::uintptr_t>(workspace) % workspace_alignment != 0))) {
        return cudaErrorInvalidValue;
    }

    auto* keys = static_cast<Key*>(workspace);
    if (plan.algorithm_ == GpuAlgorithm::reduction) {
        return select_one(input, output, plan.n_, keys, stream);
    }
    if (plan.algorithm_ == GpuAlgorithm::tiled) {
        return select_tiled(input, output, plan.n_, plan.k_, keys, plan.second_offset_, stream);
    }
    auto* sorted = keys + plan.n_;
    pack_keys<<<grid_for(plan.n_), threads, 0, stream>>>(input, keys, plan.n_);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    auto* temp = static_cast<unsigned char*>(workspace) + plan.second_offset_;
    auto temp_bytes = plan.workspace_bytes_ - plan.second_offset_;
    status = cub::DeviceRadixSort::SortKeysDescending(
        temp, temp_bytes, keys, sorted, plan.n_, 0, 64, stream);
    if (status != cudaSuccess) {
        return status;
    }
    gather<<<grid_for(plan.k_), threads, 0, stream>>>(input, sorted, output, plan.k_);
    return cudaGetLastError();
}

}  // namespace cudatopk
