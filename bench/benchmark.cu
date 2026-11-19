#include "../tests/cases.hpp"

#ifdef CUDATOPK_BENCH_CUDA
#include "../tests/cuda_support.hpp"
#include "cudatopk/cuda.hpp"
#endif

#include <array>
#include <charconv>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <system_error>

namespace {

using cudatopk::Entry;
using Clock = std::chrono::steady_clock;
constexpr int warmups = 5;

enum class Pattern { uniform, ties, ascending, special };

struct Workload {
    const char* name;
    std::size_t n;
    std::size_t k;
    Pattern pattern;
};

constexpr std::array workloads{
    Workload{"small_uniform", 1024, 32, Pattern::uniform},
    Workload{"large_argmax", 1048576, 1, Pattern::uniform},
    Workload{"large_uniform", 1048576, 32, Pattern::uniform},
    Workload{"tail_ties", 1048583, 256, Pattern::ties},
    Workload{"large_k", 1048576, 16384, Pattern::uniform},
    Workload{"ascending", 262145, 32, Pattern::ascending},
    Workload{"special_values", 131073, 128, Pattern::special},
    Workload{"full_selection", 65536, 65536, Pattern::ties},
};

std::vector<float> make_input(const Workload& workload, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<float> values(workload.n);
    for (std::size_t i = 0; i < values.size(); ++i) {
        switch (workload.pattern) {
        case Pattern::uniform:
            values[i] = static_cast<float>(static_cast<int>(rng() & 0xffffffU) - 0x800000);
            break;
        case Pattern::ties:
            values[i] = static_cast<float>(rng() & 7U);
            break;
        case Pattern::ascending:
            values[i] = static_cast<float>(i);
            break;
        case Pattern::special: {
            constexpr std::uint32_t words[] = {
                0x7fc00001U, 0xff800021U, 0x80000000U, 0U,
                0x7f800000U, 0xff800000U, 0x80000001U, 1U
            };
            const auto word = words[rng() & 7U];
            std::memcpy(&values[i], &word, sizeof(word));
            break;
        }
        }
    }
    return values;
}

std::uint64_t fingerprint(const std::vector<Entry>& result) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto& entry : result) {
        hash = (hash ^ entry.index) * 1099511628211ULL;
        hash = (hash ^ test::bits(entry.value)) * 1099511628211ULL;
    }
    return hash;
}

void report(const Workload& work, const char* method, std::size_t scratch,
            std::vector<double>& times, std::uint64_t signature) {
    std::sort(times.begin(), times.end());
    const auto size = times.size();
    const double median = (times[(size - 1) / 2] + times[size / 2]) / 2;
    const double p95 = times[size - 1 - size / 20];
    std::cout << work.name << ',' << work.n << ',' << work.k << ',' << method << ','
              << scratch << ',' << std::fixed << std::setprecision(3) << median << ',' << p95
              << ',' << std::hex << signature << std::dec << '\n';
}

void run_cpu(const Workload& work, const std::vector<float>& input,
             const std::vector<std::uint32_t>& expected, int samples) {
    std::vector<double> times;
    times.reserve(samples);
    std::uint64_t signature = 0;
    for (int i = -warmups; i < samples; ++i) {
        const auto start = Clock::now();
        const auto result = cudatopk::select_cpu(input.data(), input.size(), work.k);
        const auto end = Clock::now();
        if (i == -warmups || i == samples - 1) {
            test::require(result.size() == work.k, "wrong benchmark result length");
            test::check(result.data(), input, expected, work.k);
            signature = fingerprint(result);
        }
        if (i >= 0) {
            times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }
    }
    report(work, "cpu", work.n * sizeof(std::uint64_t), times, signature);
}

#ifdef CUDATOPK_BENCH_CUDA
class Event {
public:
    Event() { cuda_check(cudaEventCreate(&value_), "create event"); }
    ~Event() { cudaEventDestroy(value_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const noexcept { return value_; }

private:
    cudaEvent_t value_ = nullptr;
};

void run_gpu(const Workload& work, const std::vector<float>& input,
             const std::vector<std::uint32_t>& expected, int samples,
             cudatopk::GpuAlgorithm algorithm, const char* name) {
    cudatopk::GpuPlan plan;
    cuda_check(cudatopk::make_gpu_plan(work.n, work.k, algorithm, plan), "prepare plan");
    const Stream stream;
    const DeviceBuffer<float> values(work.n);
    const DeviceBuffer<Entry> output(work.k);
    const DeviceBuffer<unsigned char> scratch(plan.workspace_bytes());
    const Event start;
    const Event end;
    std::vector<Entry> host_result(work.k);
    std::vector<double> times;
    times.reserve(samples);

    cuda_check(cudaMemcpyAsync(values.get(), input.data(), work.n * sizeof(float),
                               cudaMemcpyHostToDevice, stream.get()), "copy input");
    const auto enqueue = [&] {
        cuda_check(cudatopk::select_gpu(plan, values.get(), output.get(), scratch.get(),
                                         plan.workspace_bytes(), stream.get()), "enqueue GPU selection");
    };
    const auto verify = [&] {
        cuda_check(cudaMemcpyAsync(host_result.data(), output.get(), work.k * sizeof(Entry),
                                   cudaMemcpyDeviceToHost, stream.get()), "read output");
        cuda_check(cudaStreamSynchronize(stream.get()), "complete GPU selection");
        test::check(host_result.data(), input, expected, work.k);
    };
    enqueue();
    verify();
    for (int i = 1; i < warmups; ++i) {
        enqueue();
    }
    cuda_check(cudaStreamSynchronize(stream.get()), "complete warm-up");

    for (int i = 0; i < samples; ++i) {
        cuda_check(cudaEventRecord(start.get(), stream.get()), "record start event");
        enqueue();
        cuda_check(cudaEventRecord(end.get(), stream.get()), "record end event");
        cuda_check(cudaEventSynchronize(end.get()), "finish timed selection");
        float milliseconds = 0;
        cuda_check(cudaEventElapsedTime(&milliseconds, start.get(), end.get()), "measure event interval");
        times.push_back(milliseconds * 1000.0);
    }
    verify();
    report(work, name, plan.workspace_bytes(), times, fingerprint(host_result));
}

bool print_gpu_metadata() {
    int count = 0;
    const auto probe = cudaGetDeviceCount(&count);
    if (probe != cudaSuccess || count == 0) {
        std::cerr << "GPU benchmark unavailable: "
                  << (probe == cudaSuccess ? "no devices" : cudaGetErrorString(probe)) << '\n';
        return false;
    }
    int device = 0;
    int driver = 0;
    int runtime = 0;
    cudaDeviceProp properties{};
    cuda_check(cudaGetDevice(&device), "get current device");
    cuda_check(cudaDriverGetVersion(&driver), "get driver version");
    cuda_check(cudaRuntimeGetVersion(&runtime), "get runtime version");
    cuda_check(cudaGetDeviceProperties(&properties, device), "get device properties");
    std::cout << "# GPU=" << properties.name << " sm=" << properties.major << properties.minor
              << " driver=" << driver << " runtime=" << runtime
              << " nvcc=" << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__
              << '.' << __CUDACC_VER_BUILD__ << '\n';
    return true;
}
#endif

std::uint32_t parse(std::string_view arg) {
    std::uint32_t value = 0;
    const auto result = std::from_chars(arg.data(), arg.data() + arg.size(), value);
    if (result.ec != std::errc{} || result.ptr != arg.data() + arg.size()) {
        throw std::invalid_argument("expected a decimal unsigned integer");
    }
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto samples = argc >= 2 ? parse(argv[1]) : 21U;
        const auto seed = argc >= 3 ? parse(argv[2]) : 423901U;
        if (argc > 3 || samples < 1 || samples > 10000) {
            throw std::invalid_argument("usage: benchmark [samples (1-10000) [seed]]");
        }
#ifdef CUDATOPK_BENCH_CUDA
        if (!print_gpu_metadata()) {
            return 77;
        }
#endif
        std::cout << "# host_compiler=" << __VERSION__ << '\n'
                  << "# samples=" << samples << " warmups=" << warmups << " seed=" << seed << '\n'
                  << "workload,n,k,engine,scratch_bytes,median_us,p95_us,output_fingerprint\n";
        for (const auto& work : workloads) {
            const auto input = make_input(work, seed);
            const auto expected = test::oracle(input);
            run_cpu(work, input, expected, static_cast<int>(samples));
#ifdef CUDATOPK_BENCH_CUDA
            using Algorithm = cudatopk::GpuAlgorithm;
            run_gpu(work, input, expected, samples, Algorithm::radix_sort, "radix_sort");
            if (work.k <= 256) {
                run_gpu(work, input, expected, samples, Algorithm::tiled, "tiled");
            }
            if (work.k == 1) {
                run_gpu(work, input, expected, samples, Algorithm::reduction, "reduction");
            }
#endif
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
