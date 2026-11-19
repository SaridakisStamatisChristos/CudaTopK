# CudaTopK

CudaTopK selects the highest-scoring entries from a contiguous array of
IEEE binary32 values. It provides a C++17 CPU reference and a stream-ordered
CUDA API using reusable caller-owned scratch space.

## Selection contract

- Return exactly `k` entries in descending numeric order, each with its original
  zero-based index.
- Break equal scores by increasing index. This includes both signs of zero and
  equal infinities. All NaNs tie and rank below negative infinity.
- Copy the original float representation into the output, including NaN
  payloads and the sign of zero.
- Accept `0 <= k <= n <= INT32_MAX`. With `k = 0`, no buffers are accessed.

The implementation packs an ordered score and the complemented index in a
unique 64-bit key. Key zero is reserved for padding. GPU algorithms gather the
original float after selection.

| CUDA algorithm | Supported `k` | Implementation |
| --- | --- | --- |
| `reduction` | 0–1 | Block maxima followed by a final reduction for multi-block inputs. Uses at most 8 KiB of scratch. |
| `tiled` | 0–256 | Sorts 1,024-item tiles, keeps at most `k` keys per tile, and merges truncated runs. Single-tile execution writes results directly. |
| `radix_sort` | 0–`n` | CUB device radix sort over all packed keys followed by gather. This is the general GPU baseline. |
| `automatic` | 0–`n` | Uses reduction for `k = 1`, tiled through `k = 256`, and radix sort above that. These cutoffs are heuristics pending hardware timings. |

Truncating a run to `k` is exact because any discarded key already has at least
`k` keys ahead of it in that run. At merge time, each key adds its rank within
its own run to the count of better keys in the other run. Unique keys give each
writer a distinct destination.

## Build and test

Linux x86-64, GNU Make and a C++17 compiler form the tested host target. The
tested CUDA compilation target is NVCC 13.0.88 with CCCL 13.0.85 and an
`sm_86` cubin plus PTX. Set `CUDA_ARCH` to the target GPU and use a CUDA toolkit
and driver that support it. Use a distinct `BUILD` directory when changing
compiler options or architectures.

```sh
make -j2 all test bench
build/benchmark_cpu 21 423901
```

CPU callers include `cudatopk/topk.hpp` and link `libcudatopk.a`.

```cpp
#include "cudatopk/topk.hpp"

int main() {
    const float scores[] = {2.0f, 9.0f, 9.0f};
    const auto top = cudatopk::select_cpu(scores, 3, 2);
    return (top[0].index == 1 && top[1].index == 2) ? 0 : 1;
}
```

```sh
# RTX 4060 example
make -j2 BUILD=build-ada CUDA_ARCH=89 test-gpu bench-cuda
build-ada/benchmark_cuda 21 423901
compute-sanitizer --tool memcheck --error-exitcode=1 build-ada/test_cuda
compute-sanitizer --tool racecheck --error-exitcode=1 build-ada/test_cuda
```

CUDA callers include `cudatopk/cuda.hpp`, link `libcudatopk_cuda.a` with NVCC or
the CUDA runtime, and use the following sequence.

1. Set the CUDA device and call `make_gpu_plan(n, k, algorithm, plan)`.
2. Allocate device input and output arrays of `n` floats and `k` entries.
   Allocate `plan.workspace_bytes()` bytes using `cudaMalloc` when that count
   is nonzero. Keep all buffers disjoint.
3. Call `select_gpu(plan, input, output, scratch, scratch_bytes, stream)`.
   Check each returned `cudaError_t`, then check stream completion before
   accessing results on the host or releasing any buffers.

A plan can be reused on its original device. Calls queued on the same stream
can reuse buffers in stream order. Use separate workspaces and output buffers
for concurrent calls and keep shared input read-only until completion.
`build/test_cuda --plan-only` runs the driver-independent API checks. A full
GPU test or benchmark exits with code 77 if the runtime cannot access a device.

## Benchmark and evidence

Each workload uses the specified MT19937 seed. The CSV reports five warm-ups,
the sample count, median, nearest-rank p95, scratch bytes, and a fingerprint of
the ordered indices and original float bits. An independent stable numeric sort
checks each algorithm before and after timing. The CPU timing includes
allocation and creation of the result. GPU events measure the entire
device-resident selection pipeline with allocation, plan creation, and host
transfers outside the timed interval. Compare CUDA paths against one another
under the same run. The CPU and GPU columns represent different timing scopes.

Observed CPU run on an AMD EPYC 9V74 host with GCC 14.2, `-O3 -std=c++17`,
seed `423901`, and 21 timed samples. A separate run reproduced all fingerprints.

| Workload | `n` | `k` | Median µs | p95 µs |
| --- | ---: | ---: | ---: | ---: |
| Small uniform | 1,024 | 32 | 1.923 | 2.123 |
| Large argmax | 1,048,576 | 1 | 10,257.804 | 17,824.681 |
| Large uniform | 1,048,576 | 32 | 7,737.996 | 9,288.270 |
| Tie-heavy short tail | 1,048,583 | 256 | 2,216.070 | 3,720.273 |
| Large `k` | 1,048,576 | 16,384 | 8,863.909 | 9,609.125 |
| Ascending | 262,145 | 32 | 435.258 | 2,498.567 |
| Special values | 131,073 | 128 | 437.081 | 500.016 |
| Full selection | 65,536 | 65,536 | 2,985.362 | 3,387.931 |

| Requirement | Status | Evidence |
| --- | --- | --- |
| R1 CPU reference and exact order | Verified | 610 oracle comparisons and four API checks passed. ASan and UBSan passed with leak detection disabled because the environment runs under ptrace. |
| R2 CUDA engines and exact order | Partial | CUDA compilation, linkage, and driver-independent plan checks passed for `sm_86`. Device parity remains open. |
| R3 Reproducible benchmarks | Partial | Eight CPU workloads ran with stable fingerprints. The CUDA benchmark compiled and linked. GPU timing remains open. |
| R4 Adversarial tests | Partial | CPU and host merge tests cover exceptional float bits, tails, duplicate scores, and invalid arguments. The host merge test checks 10,240 exhaustive cases and shape bounds. Device tests await hardware. |
| R5 Atomic commits | Verified | Local Git history records the implementation and observed checks per milestone. |

The build workspace had no usable NVIDIA driver. GPU performance and
device-side memory and race checks require a CUDA hardware run.
