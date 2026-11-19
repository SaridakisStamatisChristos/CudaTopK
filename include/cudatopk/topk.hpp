#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cudatopk {

struct Entry {
    float value;
    std::uint32_t index;
};

// Select exactly k indices in descending numeric order. NaNs come last.
// Equal scores (including signed zeros and NaNs) use ascending input index.
// Each returned value retains the input's IEEE binary32 representation.
// Throws std::invalid_argument for k > n or missing input when k > 0.
// Throws std::length_error for n > INT32_MAX.
std::vector<Entry> select_cpu(const float* input, std::size_t n, std::size_t k);

}  // namespace cudatopk
