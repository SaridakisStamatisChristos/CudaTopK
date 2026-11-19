#pragma once

#include "cudatopk/topk.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace test {

inline void require(bool condition, const char* description) {
    if (!condition) {
        throw std::runtime_error(description);
    }
}

inline std::uint32_t bits(const float& value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

inline std::vector<float> from_bits(std::initializer_list<std::uint32_t> values) {
    std::vector<float> result(values.size());
    std::size_t i = 0;
    for (const auto value : values) {
        std::memcpy(&result[i++], &value, sizeof(value));
    }
    return result;
}

// Independently sort original indices using numeric comparisons.
inline std::vector<std::uint32_t> oracle(const std::vector<float>& input) {
    std::vector<std::uint32_t> indices(input.size());
    std::iota(indices.begin(), indices.end(), 0U);
    std::stable_sort(indices.begin(), indices.end(), [&](auto a, auto b) {
        const bool a_nan = std::isnan(input[a]);
        const bool b_nan = std::isnan(input[b]);
        return !a_nan && (b_nan || input[a] > input[b]);
    });
    return indices;
}

inline void check(const cudatopk::Entry* actual, const std::vector<float>& input,
                  const std::vector<std::uint32_t>& expected, std::size_t k) {
    for (std::size_t i = 0; i < k; ++i) {
        require(actual[i].index == expected[i], "wrong selected index or order");
        require(bits(actual[i].value) == bits(input[expected[i]]), "value bits changed");
    }
}

template <class Verify>
void exercise(Verify verify) {
    const auto run = [&](const std::vector<float>& input, std::size_t k) {
        verify(input, k, oracle(input));
    };
    run({}, 0);
    const auto special = from_bits({0x7fc00001, 0xff800000, 0x80000000, 0,
                                    0x7f800000, 0xff800024, 0xffffffff,
                                    0x80000001, 1, 0x7f800000, 0xff7fffff,
                                    0x7f7fffff, 0x7fa54321, 0x007fffff});
    require(oracle(special) == std::vector<std::uint32_t>{
        4, 9, 11, 13, 8, 2, 3, 7, 10, 1, 0, 5, 6, 12}, "invalid test oracle");
    for (std::size_t k = 0; k <= special.size(); ++k) {
        run(special, k);
    }

    std::mt19937 rng(423901);
    for (const auto n : {1U, 31U, 32U, 255U, 256U, 1023U, 1024U, 1025U,
                         2048U, 2049U, 4097U, 32769U}) {
        std::vector<float> input(n);
        for (const auto mode : {0, 1, 2, 3, 4, 5}) {
            for (std::uint32_t i = 0; i < n; ++i) {
                switch (mode) {
                case 0: {
                    const std::uint32_t raw = rng();
                    std::memcpy(&input[i], &raw, sizeof(raw));
                    break;
                }
                case 1: input[i] = static_cast<float>(rng() % 13); break;
                case 2: input[i] = 7.0f; break;
                case 3: input[i] = static_cast<float>(i); break;
                case 4: input[i] = static_cast<float>(n - i); break;
                case 5: {
                    const std::uint32_t raw = 0x7fc00000U | i;
                    std::memcpy(&input[i], &raw, sizeof(raw));
                    break;
                }
                }
            }
            const auto expected = oracle(input);
            for (const auto k : {1U, 17U, 32U, 127U, 128U, 255U, 256U, 257U, n / 2, n}) {
                if (k <= n) {
                    verify(input, k, expected);
                }
            }
        }
    }
}

}  // namespace test
