#pragma once

#include <cstdint>

namespace cudatopk::detail {

#ifdef __CUDACC__
__host__ __device__
#endif
constexpr std::uint64_t rank_key(std::uint32_t bits, std::uint32_t index) noexcept {
    const std::uint32_t magnitude = bits & 0x7fffffffU;
    std::uint32_t score = 0;  // Every NaN sorts below negative infinity.
    if (magnitude <= 0x7f800000U) {
        if (magnitude == 0) {
            bits = 0;  // Canonicalize both zeros for comparison only.
        }
        score = (bits & 0x80000000U) ? ~bits : (bits ^ 0x80000000U);
    }
    return (static_cast<std::uint64_t>(score) << 32) | (0xffffffffU - index);
}

#ifdef __CUDACC__
__host__ __device__
#endif
constexpr std::uint32_t key_index(std::uint64_t key) noexcept {
    return 0xffffffffU - static_cast<std::uint32_t>(key);
}

}  // namespace cudatopk::detail
