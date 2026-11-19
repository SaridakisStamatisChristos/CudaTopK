#include "cases.hpp"
#include "../src/tiled.hpp"

#include <functional>
#include <iostream>

int main() {
    try {
        using namespace cudatopk::detail;
        std::size_t cases = 0;
        // Exhaustively assign unique ranks to either side of an unequal merge,
        // then truncate each side as it would be in the device pipeline.
        for (unsigned assignment = 0; assignment < (1U << 10); ++assignment) {
            for (int k = 1; k <= 10; ++k) {
                std::vector<std::uint64_t> side[2];
                for (int i = 0; i < 10; ++i) {
                    auto& run = side[(assignment >> i) & 1U];
                    if (run.size() < static_cast<std::size_t>(k)) {
                        run.push_back(10 - i);
                    }
                }
                std::vector<std::uint64_t> merged(k);
                for (int s = 0; s < 2; ++s) {
                    const auto& run = side[s];
                    const auto& other = side[1 - s];
                    for (std::size_t i = 0; i < run.size(); ++i) {
                        const auto rank = i + count_better(other.data(),
                                                          static_cast<int>(other.size()), run[i]);
                        if (rank < merged.size()) {
                            test::require(merged[rank] == 0, "duplicate merge destination");
                            merged[rank] = run[i];
                        }
                    }
                }
                for (int i = 0; i < k; ++i) {
                    test::require(merged[i] == static_cast<unsigned>(10 - i), "incorrect merge");
                }
                ++cases;
            }
        }

        for (const int n : {1, 1023, 1024, 1025, 2049, 4097, INT32_MAX}) {
            for (const int k : {1, 127, tiled_max_k}) {
                const int tiles = ceil_div(n, tile_items);
                for (int runs = tiles, span_tiles = 1; runs > 1;
                     runs = ceil_div(runs, 2), span_tiles *= 2) {
                    const auto span = static_cast<std::size_t>(span_tiles) * tile_items;
                    const auto last = static_cast<std::size_t>(runs - 1) * span;
                    test::require(last < static_cast<std::size_t>(n), "extra run");
                    test::require(last + span >= static_cast<std::size_t>(n), "missing run");
                    test::require(run_length(n, k, span, last) ==
                                      std::min(k, static_cast<int>(n - last)), "wrong tail length");
                    test::require(run_length(n, k, span, last + span) == 0, "missing right run");
                }
            }
        }
        std::cout << "Tiled merge: " << cases << " exhaustive comparisons and shape checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
