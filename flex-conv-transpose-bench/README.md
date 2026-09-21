# Flex ConvTranspose measurements

This change limits the column matrix used by Flex ConvTranspose. It uses the existing GEMM kernels and processes input tiles in reverse order. Small buffers keep the original loop.

The base is `3a93fbfc0f6517c9165031f8b2fbbe8523ff5c29`. The candidate commit is recorded in `metadata.json`.

## Method

- Intel Core Ultra 7 255H; Linux under WSL2; Rust 1.97.1.
- Release builds with the default `burn-flex` features (`std`, `simd`, `rayon`). No extra CPU target flags.
- CPU 0 for one thread; CPUs 0–3 for four threads. `RAYON_NUM_THREADS` sets the pool size.
- Five separate process runs per version and thread count. The order alternates between base/candidate and candidate/base.
- Each process warms each case, then records 3–25 complete operator calls. The table shows the median of the five process medians. Raw samples are in `paired.json`.
- Inputs and weights contain fixed, generated nonzero F32 values. The probe calls the Flex operator, including output allocation, GEMM, and col2im. There is no model-level or accuracy result.
- A global allocator counts live requested bytes. The peak is the maximum increase during a call after warm-up. It includes the output and temporary buffers. It excludes existing inputs, weights, retained allocations, allocator overhead, and process RSS.
- The column limit is 8 MiB per active batch/group. One input position is the minimum tile and can exceed this limit. GEMM work buffers are separate.

## Results

Time change is `(candidate / base - 1) * 100`. A negative value means less time.

| Threads | Case | Base ms | Candidate ms | Time change | Peak MiB, base → candidate |
|---:|---|---:|---:|---:|---:|
| 1 | image_368_k2s2 | 202.037 | 137.170 | -32.1% | 264.56 → 140.31 |
| 1 | image_256_k4s2 | 105.922 | 57.559 | -45.7% | 160.06 → 40.06 |
| 1 | image_128_k3s1 | 30.805 | 14.399 | -53.3% | 40.14 → 12.14 |
| 1 | volume_32_k3s2 | 37.207 | 12.881 | -65.4% | 69.29 → 23.29 |
| 1 | volume_24_k2s2 | 6.252 | 5.452 | -12.8% | 27.03 → 21.53 |
| 1 | audio_l8192_k16s8 | 6.074 | 4.967 | -18.2% | 24.06 → 16.06 |
| 1 | audio_l2048_k8s4 | 1.647 | 1.651 | +0.3% | 6.13 → 6.13 |
| 1 | groups4_batch2 | 16.464 | 16.438 | -0.2% | 24.00 → 24.00 |
| 1 | dilation2_tail | 8.213 | 5.954 | -27.5% | 34.49 → 17.84 |
| 1 | small_7_k4s2 | 0.100 | 0.101 | +1.1% | 0.51 → 0.51 |
| 1 | small_14_k4s2 | 0.539 | 0.514 | -4.5% | 1.56 → 1.56 |
| 1 | depthwise_256 | 7.455 | 7.463 | +0.1% | 18.19 → 18.19 |
| 4 | image_368_k2s2 | 165.495 | 147.344 | -11.0% | 264.56 → 140.31 |
| 4 | image_256_k4s2 | 70.771 | 56.564 | -20.1% | 160.06 → 40.06 |
| 4 | image_128_k3s1 | 14.932 | 9.621 | -35.6% | 40.14 → 12.14 |
| 4 | volume_32_k3s2 | 21.637 | 12.847 | -40.6% | 69.29 → 23.29 |
| 4 | volume_24_k2s2 | 5.091 | 4.786 | -6.0% | 27.03 → 21.53 |
| 4 | audio_l8192_k16s8 | 4.700 | 4.207 | -10.5% | 24.06 → 16.06 |
| 4 | audio_l2048_k8s4 | 1.134 | 1.129 | -0.4% | 6.13 → 6.13 |
| 4 | groups4_batch2 | 7.626 | 7.645 | +0.2% | 48.01 → 48.01 |
| 4 | dilation2_tail | 7.058 | 5.981 | -15.3% | 34.49 → 17.84 |
| 4 | small_7_k4s2 | 0.103 | 0.097 | -6.2% | 0.51 → 0.51 |
| 4 | small_14_k4s2 | 0.453 | 0.467 | +3.1% | 1.56 → 1.56 |
| 4 | depthwise_256 | 3.088 | 3.024 | -2.1% | 24.94 → 24.94 |

Exact shapes, strides, padding, dilation, and group counts are in `probe.rs`.

The arithmetic count is unchanged. For each batch/group, the column allocation changes from `Cout * K * N` elements to `Cout * K * min(N, tile_size)` elements. GEMM still performs `Cin * Cout * K * N` multiply-adds for that group. Smaller calls can increase GEMM packing and scheduling costs. The limit is a tradeoff, not a claim that 8 MiB is best on every CPU.

## Correctness and checks

All 12 full F32 outputs match the base bit for bit (261,104,288 output bytes in total). The output stream was hashed without saving large tensor files. Hashes and byte counts are in `output_final.json`.

The committed tests compare tiles with a direct scalar reference. They cover row and plane boundaries, tails, padding, output padding, dilation, groups, bias, and empty channels. F16/F32/F64 tests use cancellation and multiple input channels to check both addition order and the input row stride.

Final check results are recorded in `metadata.json`; logs are in `checks.tar.xz`.

## Reproduce

Use the base and candidate commits from `metadata.json`. For each revision, copy `probe.rs` to `crates/burn-flex/examples/conv_transpose_probe.rs`, then build:

```sh
CARGO_BUILD_JOBS=6 CARGO_INCREMENTAL=0 cargo build --release -p burn-flex --example conv_transpose_probe
```

Copy the two executables from `target/release/examples/conv_transpose_probe` into this directory as `baseline` and `candidate`. Then run:

```sh
python3 bench.py baseline candidate --pairs 5 --output paired.json
```

The script requires Linux and four available logical CPUs. Adjust CPU affinity for the host if needed. Do not run builds or other CPU work during the measurements.

The committed benchmark cases can also be run through the public tensor API:

```sh
BURN_DEVICE=flex cargo bench -p burn-backend-tests --bench conv_transpose_ops --no-default-features --features std,flex-simd,flex-rayon
```
