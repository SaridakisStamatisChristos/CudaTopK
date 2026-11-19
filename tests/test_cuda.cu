#include "cudatopk/cuda.hpp"
#include "cases.hpp"
#include "cuda_support.hpp"

#include <iostream>
#include <limits>
#include <string_view>

namespace {

void check_concurrent_reduction() {
    using cudatopk::Entry;
    using cudatopk::GpuAlgorithm;
    using cudatopk::GpuPlan;
    // This shape exceeds a full grid stride in the capped reduction grid.
    constexpr std::size_t n = 1048581;
    const float infinity = std::numeric_limits<float>::infinity();
    std::vector<float> a(n, 0.0f);
    std::vector<float> b(n, -1.0f);
    a[n - 1] = infinity;
    a[n - 2] = infinity;
    b[n - 1] = 9.0f;

    GpuPlan plan;
    cuda_check(cudatopk::make_gpu_plan(n, 1, GpuAlgorithm::automatic, plan), "plan concurrent shape");
    const Stream stream_a;
    const Stream stream_b;
    const DeviceBuffer<float> input_a(n);
    const DeviceBuffer<float> input_b(n);
    const DeviceBuffer<Entry> output_a(1);
    const DeviceBuffer<Entry> output_b(1);
    const DeviceBuffer<unsigned char> work_a(plan.workspace_bytes());
    const DeviceBuffer<unsigned char> work_b(plan.workspace_bytes());
    Entry result_a{};
    Entry result_b{};

    const auto upload = [&](const auto& input, const auto& device, const auto& stream) {
        cuda_check(cudaMemcpyAsync(device.get(), input.data(), n * sizeof(float),
                                   cudaMemcpyHostToDevice, stream.get()), "upload concurrent input");
    };
    upload(a, input_a, stream_a);
    upload(b, input_b, stream_b);
    for (int repeat = 0; repeat < 3; ++repeat) {
        cuda_check(cudatopk::select_gpu(plan, input_a.get(), output_a.get(), work_a.get(),
                                         plan.workspace_bytes(), stream_a.get()), "enqueue first stream");
        cuda_check(cudatopk::select_gpu(plan, input_b.get(), output_b.get(), work_b.get(),
                                         plan.workspace_bytes(), stream_b.get()), "enqueue second stream");
    }
    cuda_check(cudaMemcpyAsync(&result_a, output_a.get(), sizeof(result_a),
                               cudaMemcpyDeviceToHost, stream_a.get()), "read first stream");
    cuda_check(cudaMemcpyAsync(&result_b, output_b.get(), sizeof(result_b),
                               cudaMemcpyDeviceToHost, stream_b.get()), "read second stream");
    cuda_check(cudaStreamSynchronize(stream_a.get()), "complete first stream");
    cuda_check(cudaStreamSynchronize(stream_b.get()), "complete second stream");
    test::check(&result_a, a, {n - 2}, 1);
    test::check(&result_b, b, {n - 1}, 1);
}

}  // namespace

int main(int argc, char** argv) {
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

        constexpr auto reduction = GpuAlgorithm::reduction;
        cuda_check(cudatopk::make_gpu_plan(1024, 1, GpuAlgorithm::automatic, plan),
                   "plan one-block reduction");
        test::require(plan.algorithm() == reduction && plan.workspace_bytes() == 0,
                      "incorrect one-block reduction plan");
        cuda_check(cudatopk::make_gpu_plan(INT32_MAX, 1, reduction, plan),
                   "plan bounded reduction workspace");
        test::require(plan.workspace_bytes() == 1024 * sizeof(std::uint64_t),
                      "incorrect reduction workspace bound");
        test::require(cudatopk::make_gpu_plan(3, 2, reduction, plan)
                          == cudaErrorInvalidValue, "reduction accepted k > 1");

        if (argc == 2 && std::string_view(argv[1]) == "--plan-only") {
            std::cout << "CUDA: pre-device plan and argument checks passed\n";
            return 0;
        }
        test::require(argc == 1, "usage: test_cuda [--plan-only]");

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
            for (const auto algorithm : {radix, GpuAlgorithm::tiled, reduction, GpuAlgorithm::automatic}) {
                if ((algorithm == GpuAlgorithm::tiled && k > 256) ||
                    (algorithm == reduction && k > 1)) {
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
        check_concurrent_reduction();
        std::cout << "CUDA: " << cases
                  << " oracle comparisons, API checks and two-stream reuse passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
