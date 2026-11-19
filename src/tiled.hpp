#pragma once

#include <cstddef>
#include <cstdint>

namespace cudatopk::detail {

inline constexpr int tile_items = 1024;
inline constexpr int tiled_max_k = 256;

#ifdef __CUDACC__
#define CUDATOPK_HD __host__ __device__
#else
#define CUDATOPK_HD
#endif

CUDATOPK_HD constexpr int ceil_div(int n, int d) noexcept {
    return n == 0 ? 0 : 1 + (n - 1) / d;
}

CUDATOPK_HD constexpr int run_length(int n, int k, std::size_t span,
                                     std::size_t start) noexcept {
    if (start >= static_cast<std::size_t>(n)) {
        return 0;
    }
    auto length = static_cast<std::size_t>(n) - start;
    length = length < span ? length : span;
    return length < static_cast<std::size_t>(k) ? static_cast<int>(length) : k;
}

// Unique descending integer keys: a key's merged rank is its local rank plus
// the number of keys in the other run that sort ahead of it.
CUDATOPK_HD inline int count_better(const std::uint64_t* run, int size,
                                   std::uint64_t key) noexcept {
    int first = 0;
    while (first < size) {
        const int middle = first + (size - first) / 2;
        if (run[middle] > key) {
            first = middle + 1;
        } else {
            size = middle;
        }
    }
    return first;
}

#undef CUDATOPK_HD

}  // namespace cudatopk::detail
