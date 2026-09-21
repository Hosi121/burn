# Flex ConvTranspose channel work limits

This follow-up lowers the col2im work limit when GEMM already uses Rayon. It compares the previous PR commit `99d11a0688d0662518942cbd13e8c14b5eee627c` with `824853843aca885abe73b45b876211fb2f7df28f`. The [earlier report](../README.md) compares input tiles and channel work with main.

## Selected conditions

The normal limit stays at 262,144 column elements. A smaller limit requires at least two output channels per thread. The smaller limit is the maximum of:

- 131,072 column elements;
- 32,768 column elements per thread;
- `ceil(192^3 / input_channels_per_group)`, so GEMM already meets its Rayon work limit.

The result is capped at the normal limit. Thus, eight or more threads keep the old limit. The test uses the actual column count for each tile, including the tail. There must still be fewer batch/group tasks than threads. One-thread calls stay serial.

The change adds no column buffers and does not change the arithmetic count, GEMM kernels, or addition order. The rule uses element counts for F16, F32, and F64. The new unit test uses 128K-element tiles and a short serial tail for all three types.

## Measurements

Intel Core Ultra 7 255H, WSL2, Rust 1.97.1, release build, default Flex features. Each process uses CPUs `0..threads-1`. Two binaries use the same [probe source](production_probe.rs), without the temporary policy controls used during exploration.

Each case has five alternating process pairs. Each process makes four warm-up calls, then seven timed blocks of 3–100 complete calls. The result is the median of the block medians, then the median across processes. Timing includes output allocation, GEMM, and col2im. These are generated-input operator tests, not model measurements.

The table shows four threads. Batch and group counts are one. All 1D cases use K8/S2/P3; the 2D case uses K4/S2/P1 in both dimensions.

| Type | Input, CI → CO | Previous ms | Revised ms | Time change |
|---|---|---:|---:|---:|
| F32 | L3072, 64 → 8 | 0.351 | 0.282 | -19.7% |
| F32 | L768, 64 → 32 | 0.292 | 0.207 | -29.1% |
| F32 | L480, 64 → 64 | 0.477 | 0.406 | -14.8% |
| F32 | L512, 64 → 32 | 0.138 | 0.151 | +9.7% |
| F32 | 14x14, 128 → 64 | 0.817 | 0.852 | +4.4% |
| F16 | L3072, 64 → 8 | 0.826 | 0.499 | -39.6% |
| F16 | L512, 64 → 32 | 0.601 | 0.349 | -42.0% |
| F16 | 14x14, 128 → 64 | 1.094 | 0.767 | -29.9% |
| F64 | L3072, 64 → 8 | 0.616 | 0.479 | -22.3% |
| F64 | L512, 64 → 32 | 0.351 | 0.320 | -8.9% |
| F64 | 14x14, 128 → 64 | 0.573 | 0.495 | -13.6% |

The F32 L512 case changed from +13.3% to +0.1% at two threads and from +9.7% to -40.5% at four threads in a second run with 12 pairs. Its process ranges overlap strongly. Do not claim a stable change for this case. The F32 14x14 case stayed about 4–5% slower at four threads; at two threads it was 41.8% faster in the second run. This is not a claim that every shape improves.

At six threads, the CI512/CO32/L768 case was 14.0% slower in the policy probe. A 12-pair comparison of the production builds found 0.617 → 0.636 ms (+3.1%), with overlapping ranges. `confirm6*.json` retains this additional check.

All timing samples, ranges, and controls are in `production*.json` and `confirm*.json`. F32 uses 1, 2, 4, 6, and 8 threads; F16 and F64 use 2 and 4. Peak live requested bytes were unchanged in the measured relaxed cases. The peak covers one warmed call and excludes existing inputs, weights, retained allocations, allocator overhead, and process RSS. Some unchanged controls had small allocation differences; the source adds no buffers.

## Rejected changes

- An unconditional 64K limit increased one small 1D case from 0.042 to 0.074 ms.
- Removing the group-count guard increased a four-group case from 0.638 to 0.807 ms. In another run, it also increased peak live allocations for an eight-group case from 20.01 to 24.01 MiB.
- Entering a Rayon scope once, or assigning several channels to each task, did not give consistent gains.
- A 64K total floor was too small at two threads. A 128K floor without a work-per-thread limit also caused regressions at eight threads.

`exploration.tar.xz` contains the raw policy tests, probe sources, and patches against `99d11a0`. These temporary controls are absent from the PR. Exploration results guided the final rule; the table above comes from separate builds of the actual implementations.

Apply `experiment.patch` for `probe.rs`, `pool.patch` for `pool_probe.rs`, or `grain.patch` for the other exploration probes. Each patch starts from `99d11a0`. Copy the selected probe into the Flex examples directory and build it in release mode. Copy its executable here as `probe`, then use `run.py`. `exploration.json` lists the source, patch, thread count, and round count for each data file.

## Checks

- All 18 full output streams matched the previous commit bit for bit: 12 F32, 3 F16, and 3 F64 cases, 7,584,768 bytes. See `outputs.json`.
- Release Flex unit tests: 415 passed, including the new tile-boundary checks for all three types.
- `cargo run-checks`: 3,994 passed, 43 ignored. All-target Clippy passed with warnings denied. Commands and exit codes are in `checks.json`; logs are in `checks.tar.xz`.
- All 19 public ConvTranspose benchmark smoke cases passed. Formatting and diff checks passed.

## Reproduce

For each commit, copy `production_probe.rs` to `crates/burn-flex/examples/conv_transpose_dispatch.rs`, then build:

```sh
CARGO_BUILD_JOBS=6 CARGO_INCREMENTAL=0 cargo build --release -p burn-flex --example conv_transpose_dispatch
```

Copy the two executables into this directory as `baseline` and `candidate`. Run:

```sh
python3 production_run.py
python3 production_run.py --threads 2 4 --dtype f16 f64 --cases gemm_1d_co8_ci64_work196608 gemm_1d_co32_ci64_work131072 gemm_2d_co64_ci128_side14 --name production_types
python3 production_run.py --threads 2 4 --pairs 12 --cases gemm_1d_co32_ci64_work131072 gemm_2d_co64_ci128_side14 gemm_1d_co8_ci64_work196608 --name confirm
python3 production_run.py --threads 6 --pairs 12 --cases gemm_1d_co32_ci512_work196608 --name confirm6
python3 check_outputs.py
```

Do not run builds during timing. Adjust CPU affinity for the host as needed. The raw results use the affinity above. Remove the temporary example before project checks. Set `BURN_REPO` to the repository path and run `python3 checks.py`.
