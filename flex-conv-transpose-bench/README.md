# Flex ConvTranspose measurements

This change reduces the column buffer and runs col2im across output channels. It applies to the shared 1D, 2D, and 3D implementation. The public API, arithmetic count, and GEMM kernels are unchanged.

| Version | Commit |
|---|---|
| Main | `3a93fbfc0f6517c9165031f8b2fbbe8523ff5c29` |
| Input tiles only | `2a12122dd578b1959cbdf901811a89fd2ba8aba0` |
| Input tiles and channel work | `99d11a0688d0662518942cbd13e8c14b5eee627c` |

## Implementation

- Reuse an 8 MiB column buffer per active batch/group. One input position is the minimum tile and can exceed this limit. GEMM work buffers are separate.
- Keep the original GEMM input row stride. Process tiles from the last input position to the first. This keeps the col2im addition order. Buffers at or below the limit do not need tile bounds.
- Use Rayon for output channels when there are fewer batch/group tasks than threads. There must be at least two output channels and 262,144 column elements in the current tile. Small tail tiles use the serial loop. Each task owns a separate output channel, so the addition order within that channel is unchanged.
- Keep one-thread calls and calls without Rayon on the serial path. The channel change also applies below the 8 MiB limit. It does not allocate more column buffers.

For each batch/group, the column allocation changes from `Cout * K * N` elements to `Cout * K * min(N, tile_size)` elements. GEMM still performs `Cin * Cout * K * N` multiply-adds. The col2im work remains proportional to `Cout * K * N`. Smaller GEMM calls can increase packing and scheduling costs. The byte limit is not a claim of the best value for every CPU.

## Method

- Intel Core Ultra 7 255H; Linux under WSL2; Rust 1.97.1. Default Flex features: `std`, `simd`, `rayon`. Release builds; no extra CPU target flags.
- CPU 0 for one thread; CPUs 0–3 for four threads. `RAYON_NUM_THREADS` sets the pool size.
- Five separate process runs per version, case, and thread count. The version order alternates. Each process runs one case, warms the operator, then records 3–25 complete calls. Tables show the median of the five process medians. Raw samples and ranges are in `parallel_fold/final.json` and `parallel_fold/final_summary.json`.
- Fixed, generated nonzero F32 inputs and weights. Exact shapes are in `probe.rs`. Timing includes output allocation, GEMM, and col2im. These are operator measurements. No model inference, accuracy, ARM, or GPU result is claimed.
- A global allocator counts live requested bytes. The peak is the maximum increase during a call after warm-up. It includes output and temporary buffers, but excludes existing inputs, weights, retained allocations, allocator overhead, and process RSS.

## Time

A negative change means less time. All values are milliseconds. The last column compares the final version with the tile-only version.

### 1 thread

| Case | Main | Tiles only | Final | Change from main | Change from tiles |
|---|---:|---:|---:|---:|---:|
| image_368_k2s2 | 196.320 | 130.417 | 132.514 | -32.5% | +1.6% |
| image_256_k4s2 | 91.498 | 48.888 | 51.427 | -43.8% | +5.2% |
| image_128_k3s1 | 28.185 | 13.230 | 14.141 | -49.8% | +6.9% |
| volume_32_k3s2 | 33.669 | 13.072 | 12.405 | -63.2% | -5.1% |
| volume_24_k2s2 | 7.930 | 6.420 | 6.532 | -17.6% | +1.7% |
| audio_l8192_k16s8 | 5.592 | 5.629 | 5.116 | -8.5% | -9.1% |
| audio_l2048_k8s4 | 1.490 | 1.478 | 1.516 | +1.8% | +2.6% |
| groups4_batch2 | 16.290 | 18.117 | 19.484 | +19.6% | +7.5% |
| dilation2_tail | 8.431 | 6.709 | 6.585 | -21.9% | -1.8% |
| small_7_k4s2 | 0.124 | 0.111 | 0.101 | -18.4% | -8.6% |
| small_14_k4s2 | 0.951 | 0.929 | 0.905 | -4.9% | -2.7% |
| depthwise_256 | 9.165 | 8.833 | 8.737 | -4.7% | -1.1% |

### 4 threads

| Case | Main | Tiles only | Final | Change from main | Change from tiles |
|---|---:|---:|---:|---:|---:|
| image_368_k2s2 | 196.412 | 171.964 | 85.909 | -56.3% | -50.0% |
| image_256_k4s2 | 79.318 | 64.769 | 32.580 | -58.9% | -49.7% |
| image_128_k3s1 | 16.359 | 11.341 | 5.998 | -63.3% | -47.1% |
| volume_32_k3s2 | 22.001 | 16.294 | 8.013 | -63.6% | -50.8% |
| volume_24_k2s2 | 19.661 | 6.632 | 3.671 | -81.3% | -44.6% |
| audio_l8192_k16s8 | 5.602 | 13.253 | 3.060 | -45.4% | -76.9% |
| audio_l2048_k8s4 | 1.353 | 1.311 | 1.037 | -23.3% | -20.9% |
| groups4_batch2 | 6.750 | 6.928 | 7.166 | +6.2% | +3.4% |
| dilation2_tail | 7.578 | 7.216 | 3.512 | -53.7% | -51.3% |
| small_7_k4s2 | 0.095 | 0.096 | 0.096 | +0.7% | -0.8% |
| small_14_k4s2 | 0.853 | 0.853 | 0.798 | -6.4% | -6.5% |
| depthwise_256 | 3.528 | 3.560 | 3.392 | -3.9% | -4.7% |

## Allocation and variation

The four-thread peak values are below. The channel change keeps the same column allocation as the tile-only version. Small differences in the raw peaks can come from work buffers first used by a thread.

| Case | Main peak MiB | Final peak MiB |
|---|---:|---:|
| image_368_k2s2 | 264.56 | 140.31 |
| image_256_k4s2 | 160.06 | 40.06 |
| image_128_k3s1 | 40.14 | 12.14 |
| volume_32_k3s2 | 69.29 | 23.29 |
| volume_24_k2s2 | 30.03 | 21.53 |
| audio_l8192_k16s8 | 27.06 | 16.06 |
| audio_l2048_k8s4 | 6.13 | 6.13 |
| groups4_batch2 | 48.01 | 48.01 |
| dilation2_tail | 34.49 | 17.84 |
| small_7_k4s2 | 0.51 | 0.51 |
| small_14_k4s2 | 1.56 | 1.56 |
| depthwise_256 | 24.94 | 24.94 |

The four large 2D/3D cases use 47.1–50.8% less time than tiles alone at four threads. A second comparison ran all 12 cases in each process, as in the first tile study. It found 49.6–59.1% less time for the same four cases. Those samples and ranges are in `parallel_fold/sequence.json` and `parallel_fold/sequence_summary.json`.

Small cases, one-thread calls, and grouped controls did not give stable small differences on this shared WSL2 host. For example, the grouped control was 19.6% slower than main at one thread and 6.2% slower at four threads in the primary run. In the sequence run it was 7.1% and 0.6% faster. The small 7x7 case also varied strongly at one thread. These measurements do not establish a stable regression or gain for these controls. They do not rule out a regression on another host.

Allocation history also matters. The isolated `audio_l8192_k16s8` run has much higher system time for the tile-only version than the sequence run. Do not treat its 76.9% additional reduction as a stable estimate. The raw isolated data includes process user time, system time, and page faults. A previous whole-sequence run (`parallel_fold/paired.json`) had severe changes in host load and is not used for the tables.

The probe text section grew by 18,536 bytes (1.0%) from the tile-only version, and by 41,504 bytes (2.2%) from main. This is the probe executable size, not a size estimate for every application.

## Correctness and checks

- All 12 complete F32 outputs match main bit for bit: 261,104,288 output bytes. `parallel_fold/outputs.json` records the hashes. The check uses a pipe, so it does not save large output tensors.
- The new tests compare serial and four-thread results for F16, F32, and F64. They cover tiled and untiled calls, short tail tiles, nested batch/channel work, and enough groups to fill the pool.
- Existing tile tests compare against a direct scalar reference. They cover row and plane boundaries, tails, padding, output padding, dilation, groups, bias, empty channels, cancellation, and multiple input channels.
- `cargo run-checks`: 3,994 passed, 43 ignored. Release Flex unit tests: 415 passed. All-target Clippy with warnings denied: passed. Public ConvTranspose benchmark smoke tests: 18 passed. Formatting and diff checks passed.

Commands, hashes, and counts are in `metadata.json` and `parallel_fold/checks.json`. Check logs are in `parallel_fold/checks.tar.xz`. The benchmark executable was rebuilt after the final test changes; its hash was unchanged.

## Reproduce

For each commit in the table, copy `probe.rs` to `crates/burn-flex/examples/conv_transpose_probe.rs` and build:

```sh
CARGO_BUILD_JOBS=6 CARGO_INCREMENTAL=0 cargo build --release -p burn-flex --example conv_transpose_probe
```

Copy `target/release/examples/conv_transpose_probe` into `parallel_fold/` as `base`, `tiled`, or `preserved` for the three respective commits. From this report directory, run:

```sh
python3 parallel_fold/bench.py base tiled preserved --pairs 5 --output final.json
python3 bench.py parallel_fold/base parallel_fold/tiled parallel_fold/preserved --pairs 5 --output parallel_fold/sequence.json
python3 parallel_fold/check_outputs.py
```

The scripts require Linux and four available logical CPUs. Adjust affinity for the host if needed. Do not run builds during timing. The original `paired.json`, `paired_summary.json`, and `checks.tar.xz` in this directory record the first tile-only study; they are retained for comparison.

The public tensor benchmarks can also be run with:

```sh
BURN_DEVICE=flex cargo bench -p burn-backend-tests --bench conv_transpose_ops --no-default-features --features std,flex-simd,flex-rayon
```
