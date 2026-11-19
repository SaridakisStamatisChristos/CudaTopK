#include "cases.hpp"

#include <iostream>
#include <limits>

int main() {
    try {
        std::size_t cases = 0;
        test::exercise([&](const auto& input, auto k, const auto& expected) {
            const auto selected = cudatopk::select_cpu(input.data(), input.size(), k);
            test::require(selected.size() == k, "wrong output length");
            test::check(selected.data(), input, expected, k);
            ++cases;
        });
        test::require(cudatopk::select_cpu(nullptr, 99, 0).empty(), "k=0 read input");
        try {
            cudatopk::select_cpu(nullptr, 10, 1);
            throw std::runtime_error("null input accepted");
        } catch (const std::invalid_argument&) {}
        const float input = 0;
        try {
            cudatopk::select_cpu(&input, 1, 2);
            throw std::runtime_error("k > n accepted");
        } catch (const std::invalid_argument&) {}
        try {
            cudatopk::select_cpu(nullptr, std::size_t{INT32_MAX} + 1, 0);
            throw std::runtime_error("oversized n accepted");
        } catch (const std::length_error&) {}
        std::cout << "CPU: " << cases << " oracle comparisons and 4 API checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
