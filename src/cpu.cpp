#include "cudatopk/topk.hpp"

#include "key.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>

namespace cudatopk {

static_assert(sizeof(float) == sizeof(std::uint32_t) &&
              std::numeric_limits<float>::is_iec559 &&
              std::numeric_limits<float>::digits == 24);

std::vector<Entry> select_cpu(const float* input, std::size_t n, std::size_t k) {
    if (n > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::length_error("CudaTopK supports at most INT32_MAX elements");
    }
    if (k > n || (k != 0 && input == nullptr)) {
        throw std::invalid_argument("CudaTopK requires k <= n and input for k > 0");
    }
    if (k == 0) {
        return {};
    }

    std::vector<std::uint64_t> keys(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, input + i, sizeof(bits));
        keys[i] = detail::rank_key(bits, static_cast<std::uint32_t>(i));
    }
    const auto end = keys.begin() + k;
    const auto better = std::greater<std::uint64_t>{};
    if (k < n) {
        std::nth_element(keys.begin(), end, keys.end(), better);
    }
    std::sort(keys.begin(), end, better);

    std::vector<Entry> result(k);
    for (std::size_t i = 0; i < k; ++i) {
        const auto index = detail::key_index(keys[i]);
        result[i].index = index;
        std::memcpy(&result[i].value, input + index, sizeof(float));
    }
    return result;
}

}  // namespace cudatopk
