#include "cudatopk/cuda.hpp"
#include "cases.hpp"
#include "cuda_support.hpp"

#include <iostream>

int main() {
    try {
        using cudatopk::Entry;
        using cudatopk::GpuAlgorithm;
        using cudatopk::GpuPlan;
        GpuPlan plan;
        constexpr auto radix = GpuAlgorithm::radix_sort;
        cuda_check(cudatopk::select_gpu(plan, nullptr, nullptr, nullptr, 0, nullptr), "default plan");
        cuda_check(cudatopk::make_gpu_plan(99, 0, radix, plan), "plan empty output");
        test::require(plan.workspace_bytes() == 0, "empty output requires workspace");
        test::require(cudatopk::make_gpu_plan(4, 5, radix, plan) == cudaErrorInvalidValue,
                      "k > n accepted");
        test::require(cudatopk::make_gpu_plan(std::size_t{INT32_MAX} + 1, 0, radix, plan)
                          == cudaErrorInvalidValue, "oversized n accepted");
        test::require(cudatopk::make_gpu_plan(0, 0, static_cast<GpuAlgorithm>(99), plan)
                          == cudaErrorInvalidValue, "invalid algorithm accepted");

        cuda_check(cudatopk::make_gpu_plan(1024, 256, GpuAlgorithm::automatic, plan),
                   "plan automatic single tile");
        test::require(plan.algorithm() == GpuAlgorithm::tiled && plan.workspace_bytes() == 0,
                      "single tile should use no workspace");
        test::require(cudatopk::make_gpu_plan(1024, 257, GpuAlgorithm::tiled, plan)
                          == cudaErrorInvalidValue, "tiled accepted oversized k");
        test::require(plan.algorithm() == GpuAlgorithm::tiled && plan.workspace_bytes() == 0,
                      "failed plan preparation changed the previous plan");
        cuda_check(cudatopk::make_gpu_plan(1025, 256, GpuAlgorithm::tiled, plan),
                   "plan two tiles");
        test::require(plan.workspace_bytes() == 3 * 256 * sizeof(std::uint64_t),
                      "unexpected two-tile workspace");

        int devices = 0;
        const auto probe = cudaGetDeviceCount(&devices);
        if (probe != cudaSuccess || devices == 0) {
            std::cerr << "GPU execution unavailable: "
                      << (probe == cudaSuccess ? "no devices" : cudaGetErrorString(probe))
                      << '\n';
            return 77;
        }

        const Stream stream;
        cuda_check(cudatopk::make_gpu_plan(2, 1, radix, plan), "plan API checks");
        const DeviceBuffer<float> sample(2);
        const DeviceBuffer<Entry> selected(2);
        const DeviceBuffer<unsigned char> scratch(plan.workspace_bytes() + 1);
        const auto select = [&](const float* in, Entry* out, void* work, std::size_t bytes) {
            return cudatopk::select_gpu(plan, in, out, work, bytes, stream.get());
        };
        test::require(select(nullptr, selected.get(), scratch.get(), plan.workspace_bytes())
                          == cudaErrorInvalidValue, "null input accepted");
        test::require(select(sample.get(), nullptr, scratch.get(), plan.workspace_bytes())
                          == cudaErrorInvalidValue, "null output accepted");
        test::require(select(sample.get(), selected.get(), nullptr, plan.workspace_bytes())
                          == cudaErrorInvalidValue, "missing workspace accepted");
        test::require(select(sample.get(), selected.get(), scratch.get(), plan.workspace_bytes() - 1)
                          == cudaErrorInvalidValue, "short workspace accepted");
        test::require(select(sample.get(), selected.get(), scratch.get() + 1, plan.workspace_bytes())
                          == cudaErrorInvalidValue, "misaligned workspace accepted");

        std::size_t cases = 0;
        test::exercise([&](const auto& input, auto k, const auto& expected) {
            for (const auto algorithm : {radix, GpuAlgorithm::tiled, GpuAlgorithm::automatic}) {
                if (algorithm == GpuAlgorithm::tiled && k > 256) {
                    continue;
                }
                GpuPlan shape;
                cuda_check(cudatopk::make_gpu_plan(input.size(), k, algorithm, shape), "plan shape");
                DeviceBuffer<float> values(k ? input.size() : 0);
                DeviceBuffer<Entry> output(k);
                DeviceBuffer<unsigned char> workspace(shape.workspace_bytes());
                std::vector<Entry> actual(k);
                if (k) {
                    cuda_check(cudaMemcpyAsync(values.get(), input.data(), input.size() * sizeof(float),
                                               cudaMemcpyHostToDevice, stream.get()), "copy input");
                }
                cuda_check(cudatopk::select_gpu(shape, values.get(), output.get(), workspace.get(),
                                                 shape.workspace_bytes(), stream.get()), "enqueue selection");
                if (k) {
                    cuda_check(cudaMemcpyAsync(actual.data(), output.get(), k * sizeof(Entry),
                                               cudaMemcpyDeviceToHost, stream.get()), "copy output");
                }
                cuda_check(cudaStreamSynchronize(stream.get()), "complete selection");
                test::check(actual.data(), input, expected, k);
                ++cases;
            }
        });
        std::cout << "CUDA: " << cases << " oracle comparisons and API checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
