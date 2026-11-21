# GateRV3 kernel tuning and full-FP16 experiment

The original GateRV3 paths and commands below document historical
runs. See the [current runtime guide](gaterv3-python.md) for `.ummodel`
and `.umplan` usage.

The control is the validated mixed-precision CUDA Graph runner. Keep the
checkpoint, all 12 spatial-attention blocks and the 128×128 tile geometry.
Use batch 1 for individual tile latency and batch 16 for page throughput.

- [x] Reprofile the fixed-shape control and identify the largest kernel costs.
- [x] Tune dense 1×1 and 3×3 library plans on the RTX 3050. Measure the
  optimization separately from precision changes.
- [x] Fuse adjacent layout and residual operations where the schedule can
  retain the existing FP16 rounding points.
- [x] Add an opt-in mode with FP16 RGB I/O, FP16 dense and attention
  accumulators, and FP16 pointwise and reduction arithmetic. Scale the
  RMS reduction to stay within FP16 range.
- [x] Verify raw output against untouched Torch FP32 on all three pages
  and the deterministic stress tiles. Report PSNR and SSIMULACRA2 separately.
- [x] Compare the control, optimized mixed, and full FP16 on the same GPU,
  including every actual batch-1 tile and batch-16 per-page throughput.
  Compare with the prepared Torch and TensorRT-RTX baselines.
- [x] Measure isolated VRAM usage at both batch sizes. Report the runner's
  allocation sizes separately from the GPU-wide usage change with the
  model and captured graph resident.

The previous matched run recorded 5.420 ms mean batch-1 latency and
3.441 MPx/s batch-16 page throughput for native mixed precision. The
largest operation classes in its instrumented batch-16 replay were the
1×1 projections and 3×3 convolutions. See the
[original implementation and results](gaterv3-native.md).

## Implementation

- Select cuBLASLt and cuDNN forward plans by timing compatible tactics once
  per shape at startup. Resize the shared workspace to the largest selected
  tactic or reduction before CUDA Graph capture. The temporary tuning
  workspace is capped at 64 MiB.
- Merge decoder pixel shuffle and skip concatenation. Merge adjacent residual
  updates and RMSNorm, preserving the FP16 rounding point and keeping residual
  inputs live until both outputs are written.
- Compose the latent input projection's final branch with QKV in FP32 during
  export. A single projection emits the gate, identity and QKV channels.
  This removes 12 matrix multiplications and approximately 6 MiB of packed
  weights. The depthwise filter, attention, and output projection remain.
- Use a separate full-FP16 execution path with binary16 I/O and arithmetic,
  FP16 cuBLASLt and cuDNN compute descriptors, and WMMA half accumulators.
  The half-math intrinsics return FP16 values for nonlinear operations.
  RMSNorm scales activations before squaring to preserve FP16 range.
  [cuBLAS compute modes](https://docs.nvidia.com/cuda/cublas/index.html)
  [cuDNN convolution configurations](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.21.1/api/cudnn-cnn-library.html)
  [CUDA half2 math](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__FUNCTIONS.html)

The schedule has 400 operations before dispatch fusion, down from 416.
Another 51 residual and RMSNorm pairs share a kernel launch.
The first decoder-fusion prototype launched for half the intended number of
output pairs and failed the FP32 comparison. Its timings were discarded.
The fixed version launches one thread per output pair across the entire batch.

## Reproduce

Use the system interpreter and the headers from the
[original build instructions](gaterv3-native.md#build-and-run). The old
bundle at `output/native` is the control in the commands below.

```powershell
$py = 'C:\Python314\python.exe'
$bundle = 'output/native-qkv'
& $py scripts/build_native.py --output $bundle
& $py scripts/export_native.py --output $bundle
& $py scripts/benchmark_native.py --suite quality --bundle $bundle --output output/native-qkv-quality
& $py scripts/benchmark_native.py --suite quality --precision fp16 --bundle $bundle --output output/native-qkv-fp16-quality
& $py scripts/benchmark_native.py --suite memory --bundle $bundle --control-bundle output/native --compare-fp16 --output output/native-qkv-memory
foreach ($batch in 1,16) {
    & $py scripts/benchmark_native.py --suite speed --bundle $bundle --control-bundle output/native --compare-fp16 --batch $batch --profile --rounds 4 --repeats 6 --output "output/native-qkv-speed-b$batch"
}
```

Use `gt_create_fp16` and `gt_run_fp16` to pass binary16 buffers through
the DLL. The standalone executable takes `--fp16` after the output path.
Both modes support contiguous RGB NCHW 128×128 tiles at batch 1 or 16.

```powershell
& .\output\native-qkv\gaterv3.exe $bundle 1 .\input.f16 .\restored.f16 --fp16
```

The memory suite creates one runner at a time and warms eight calls.
It reports exact native weights, activations, workspace and RGB buffer
sizes alongside CUDA device-wide used memory before initialization, after
capture, after warmup and after release. Startup samples also cover the
temporary tuning workspace. Device-wide deltas include driver and CUDA
library allocations and can vary with other GPU activity. Sampled startup
usage is a lower bound on the maximum between samples.
The suite also records initialization wall time before the eight calls.

## Fidelity

| Image | Mixed PSNR (dB) | Full FP16 PSNR (dB) | Mixed SSIMULACRA2 | Full FP16 SSIMULACRA2 |
| --- | ---: | ---: | ---: | ---: |
| page-001.jpg | 83.80 | 73.27 | 94.47 | 94.02 |
| page-005.jpg | 88.84 | 71.77 | 91.11 | 90.52 |
| page-009.jpg | 85.39 | 71.43 | 91.48 | 91.07 |
| Aggregate PSNR / mean SSIMULACRA2 | **85.55** | **72.09** | **92.35** | **91.87** |

The previous mixed runner measured 85.55 dB aggregate raw PSNR. The prepared
TensorRT-RTX FP16 result was 72.13 dB, with mean and minimum SSIMULACRA2
of 91.92 and 90.59. The optimized mixed runner remains the default. All
five batch-1 stress inputs stayed finite. The lowest batch-1 stress PSNR was
79.01 dB for mixed on the seeded random tile and 69.22 dB for full FP16 on
the checkerboard tile.

The standalone executable ran both precision modes at both batch sizes
with finite outputs and the expected file sizes. On seeded random inputs
it measured 78.78–78.97 dB with FP32 RGB I/O and 70.49 dB with FP16 RGB I/O.

## VRAM and initialization

Measured with the graphs resident on the RTX 3050 Laptop GPU. All values are
MiB. The older native binary exposes only the device-wide measurements.

| Runner | Batch | Runner buffers | Warmed device increase | Sampled startup increase |
| --- | ---: | ---: | ---: | ---: |
| Previous native mixed | 1 | — | 156 | — |
| Optimized mixed | 1 | 83.94 | 102 | 158 |
| Optimized full FP16 | 1 | 81.37 | 90 | 154 |
| Previous native mixed | 16 | — | 316 | — |
| Optimized mixed | 16 | 240.44 | 246 | 310 |
| Optimized full FP16 | 16 | 237.37 | 240 | 304 |

At batch 16 the optimized mixed runner used 70.31 MiB of weights, 164 MiB
of activations, 0.125 MiB of shared workspace and 6 MiB of RGB buffers.
The full-FP16 runner used 0.063 MiB of workspace and 3 MiB of RGB buffers
with the same weight and activation sizes. The warmed device increase
fell by 70 MiB for mixed and 76 MiB for full FP16 against the previous
mixed runner.

In a separate isolated setup pass, batch-1 initialization took 0.16 s for
the previous runner, 0.51 s for optimized mixed and 0.62 s for full FP16.
Batch-16 initialization took 0.18 s, 1.04 s and 5.68 s respectively.
Initialization includes tactic tuning and graph capture. The performance
tables measure warmed inference.

## Batch-16 throughput

The same three pages used 33 batch-16 forward calls per runner. Page times
include batching and synchronous host I/O. Image decoding, tiling, stitching,
and encoding were outside the timed region. The RTX profiler reported
`NvTensorRTRTXExecutionProvider` for the executed partitions.

| Runner | Total (s) | MPx/s | page-001 s / MPx/s | page-005 s / MPx/s | page-009 s / MPx/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous native mixed | 2.269 | 3.712 | 0.748 / 3.752 | 0.745 / 3.767 | 0.776 / 3.621 |
| Optimized mixed | 2.161 | 3.899 | 0.754 / 3.726 | 0.702 / 4.003 | 0.706 / 3.979 |
| Optimized full FP16 | 1.840 | 4.578 | 0.614 / 4.576 | 0.616 / 4.556 | 0.610 / 4.603 |
| Torch eager FP16 | 5.714 | 1.474 | 1.916 / 1.466 | 1.851 / 1.517 | 1.948 / 1.442 |
| TorchInductor FP16 | 7.543 | 1.117 | 2.553 / 1.100 | 2.534 / 1.108 | 2.457 / 1.143 |
| ORT TensorRT-RTX FP16 | 2.659 | 3.168 | 0.934 / 3.006 | 0.859 / 3.268 | 0.866 / 3.244 |

Optimized mixed improved total throughput by 5.0% over the preceding native
runner. Full FP16 improved throughput by 23.3%. Repeated GPU-resident
batch-16 medians were 67.036 ms for the previous native runner,
61.941 ms for optimized mixed, 57.229 ms for full FP16 and 70.141 ms
for TensorRT-RTX. Dense kernels remained the largest instrumented
classes in optimized mixed at 18.35 ms for 1×1 projections and 14.39 ms
for 3×3 convolutions.

## Batch-1 latency

The initial whole-runner sweep showed drift between pages. The paired rerun
alternated runner order after every 16 source tiles. Each number below is
the mean of individually synchronized calls with host I/O. Each runner
measured all 528 tiles. TensorRT-RTX used its separate batch-1 graph and
reported `NvTensorRTRTXExecutionProvider` for the executed partition.

| Runner | Overall (ms) | page-001 (ms) | page-005 (ms) | page-009 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Previous native mixed | 5.889 | 6.007 | 5.764 | 5.897 |
| Optimized mixed | 5.759 | 5.711 | 5.719 | 5.847 |
| Optimized full FP16 | 5.335 | 5.733 | 5.092 | 5.179 |
| Torch eager FP16 | 61.819 | 62.154 | 66.014 | 57.290 |
| TorchInductor FP16 | 20.016 | 19.981 | 20.322 | 19.744 |
| ORT TensorRT-RTX FP16 | 6.977 | 7.094 | 7.051 | 6.786 |

Optimized mixed reduced mean tile latency by 2.2% compared with the
previous native runner. Full FP16 reduced it by 9.4%. The repeated
GPU-resident batch-1 medians were 4.943 ms for the previous native runner,
4.849 ms for optimized mixed, 4.266 ms for full FP16, and 5.656 ms for
TensorRT-RTX.

The local combined record is `output/native-qkv/verified-report.json`.
It includes benchmark and checkpoint hashes, paired per-tile times, batch-16
per-image times, profiles, VRAM and initialization data, and standalone
verification.

## Planned improvements

The verified `output/native-qkv` bundle is the control for this work. Keep
128×128 at batches 1 and 16 as regression cases for both precisions. The
[runtime refactor plan](gaterv3-native-refactor.md) describes how the same
optimizations carry over to variable input shapes and the single model file.

### 1. Repack the activation arena

- [x] Replace append-order allocation with an offline packing pass that
  places larger tensors first and assigns offsets using their lifetimes.
  Treat fused launches as simultaneous reads and writes of all participating
  tensors, including the two outputs of residual plus RMSNorm.
- [x] Validate each offset's alignment and confirm that overlapping lifetimes
  never share memory. Run both precisions against the existing bundle at the
  same shape and precision, with the same kernel plans when checking bitwise
  parity. Remeasure exact runner buffers and warmed device usage separately.

The new exporter reached **8 MiB per tile** and **128 MiB at batch 16**,
down from 164 MiB. With the same runner build and verified cache hits,
the original and repacked schedules produced bitwise-identical results
on all 528 tiles and four stress inputs for each precision and batch size.
The weights and logical operation list matched.

Compared with the preceding measured bundle, warmed batch-16 device usage
increased by **210 MiB** for repacked mixed and **204 MiB** for repacked full
FP16, down from **246 MiB** and **240 MiB** respectively. Exact runner buffer
totals were 204.44 MiB for mixed and 201.37 MiB for full FP16. Device-wide
readings include CUDA library and graph overhead.

### 2. Persist selected kernel plans

- [x] Add an optional cache directory to `gt_create_ex` and a `--cache-dir` CLI
  flag. Save the selected cuDNN and cuBLASLt choices and workspace
  requirements. Use the compatibility key and miss behavior in the
  [refactor plan](gaterv3-native-refactor.md#configurable-kernel-plan-cache).
- [x] Benchmark a first load into an empty cache and a fresh process with
  a populated cache. Record tune time, graph capture time, total time to first
  output, warmed latency and output parity. Include mismatched and corrupt
  cache tests.

The two benchmark processes below each started with a fresh CUDA context
and ran batch 1 followed by batch 16. The first process populated the
directory. The second loaded and validated the saved plans and avoided
allocating the temporary 64 MiB tuning workspace on compatible hits.

| Mode | Batch | Initialization empty / populated (ms) | First output empty / populated (ms) |
| --- | ---: | ---: | ---: |
| Mixed | 1 | 686 / 472 | 692 / 478 |
| Full FP16 | 1 | 741 / 211 | 747 / 216 |
| Mixed | 16 | 1043 / 225 | 1106 / 292 |
| Full FP16 | 16 | 5688 / 191 | 5741 / 245 |

The compatibility key includes the native build, CUDA module, logical
schedule, weights, shape, precision, device UUID, driver DLL version,
backend versions, and workspace cap. The same aligned logical schedule
can reuse its plans with different arena offsets. The standalone binary
and DLL accept a caller-specified cache directory. Cache tests covered
hits with bitwise parity, separate keys for mixed and full FP16, a payload
from a different key, corruption, an unavailable directory, and concurrent
standalone writers followed by a fresh reader. Graph instances are captured
for each session.

### 3. Prototype convolution and SiLU fusion

- [x] Start with the repeated 128×128, 32-channel convolutions in the SISR
  branch. Prototype a convolution with a SiLU epilogue using a compatible
  cuDNN graph plan on the RTX 3050. Keep the current separate kernels as a
  fallback, and check that fused intermediates preserve the intended rounding
  or meet the same-precision fidelity criteria.
- [x] Compare the fused and split paths with batch 1 and 16, both precisions,
  and identical inputs. Include workspace and startup cost alongside warm
  page throughput. Extend to other shapes after the first candidate is measured.

The batch-16 full-FP16 profile attributed **7.01 ms** to 19 full-resolution
32-to-32 convolutions and **2.98 ms** to 12 SiLU dispatches. Only the
convolutions with a following SiLU are fusion candidates. These are
instrumented operation times. The paired measurements below quantify the
end-to-end difference.
[cuDNN graph fusion documentation](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html)

The prototype uses the cuDNN convolution–SiLU operation graph at 12 sites
in mixed precision. The tested cuDNN heuristics returned no compatible
full-FP16 graph plan on this workstation. The split fallback produced
bitwise-identical full-FP16 output with and without the fusion flag at
both batch sizes. The fusion flag is opt-in and the disk cache stores the
split-path plans.

| Repacked runner | Aggregate raw PSNR vs Torch FP32 (dB) | Mean SSIMULACRA2 | Minimum SSIMULACRA2 |
| --- | ---: | ---: | ---: |
| Mixed, separate kernels | 85.5470 | 92.35 | 91.11 |
| Mixed, convolution–SiLU | 85.5471 | 92.29 | 91.10 |
| Full FP16, separate kernels | 72.0891 | 91.87 | 90.52 |

All three modes passed finite-output checks on black, white, checkerboard,
seeded random, and real-image batch-1 inputs. The full set of per-image
metrics and stress results is under `output/native-next-validation/`.

### Build and exercise this pass

```powershell
$py = 'C:\Python314\python.exe'
$bundle = 'output/native-arena'
$results = 'output/native-next-validation'
$cache = "$results/cache"
& $py scripts/build_native.py --output $bundle
& $py scripts/export_native.py --output $bundle
& $py scripts/benchmark_native.py --suite memory --bundle $bundle --cache-dir $cache --compare-fp16 --compare-silu --output "$results/memory"
& $py scripts/benchmark_native.py --suite speed --bundle $bundle --cache-dir $cache --compare-fp16 --compare-silu --rounds 4 --repeats 4 --output "$results/speed"
```

Use a new cache directory for an empty-cache measurement. Repeat the memory
command in a fresh process to measure a populated cache. For the standalone
binary, append `--cache-dir <directory>` after the raw output path. The
optional `--fuse-silu` flag requests the mixed fusion prototype. C callers
can use `gt_create_ex` with `GT_CREATE_GRAPH`, optional `GT_CREATE_FP16`
or `GT_CREATE_FUSE_SILU`, and the UTF-8 cache directory. The existing
entry points retain their original signatures.

```powershell
& .\output\native-arena\gaterv3.exe $bundle 16 .\tiles.f16 .\restored.f16 --fp16 --cache-dir $cache
```

### Measured latency

On the RTX 3050 Laptop GPU, each runner timed all 528 tiles as individual
synchronized batch-1 calls, alternating runner order every 16 tiles. All
native cases used the populated disk cache. TensorRT-RTX executed the
separate batch-1 graph through its profiled RTX provider.

| Runner | Mean tile (ms) | page-001 (ms) | page-005 (ms) | page-009 (ms) | GPU-resident median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed, separate kernels | 5.422 | 5.417 | 5.476 | 5.374 | 4.656 |
| Mixed, convolution–SiLU | 5.404 | 5.391 | 5.619 | 5.201 | 4.557 |
| Full FP16, separate kernels | 5.094 | 5.037 | 5.229 | 5.018 | 4.286 |
| Full FP16 with fusion requested, split fallback | 5.173 | 5.395 | 5.204 | 4.920 | 4.266 |
| Torch eager FP16 | 50.589 | 51.733 | 52.294 | 47.740 | 43.839 |
| TorchInductor FP16 | 18.450 | 18.014 | 19.047 | 18.288 | 16.624 |
| ORT TensorRT-RTX FP16 | 6.353 | 6.265 | 6.537 | 6.256 | 5.292 |

### Measured batch-16 throughput

Page timings include batch assembly and synchronous host I/O. Each runner
processed the same three pages in 33 batch-16 calls. ORT profiling confirmed
RTX placement for the fixed batch-16 graph.

| Runner | Total (s) | MPx/s | page-001 s / MPx/s | page-005 s / MPx/s | page-009 s / MPx/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed, separate kernels | 2.138 | 3.941 | 0.711 / 3.950 | 0.704 / 3.986 | 0.722 / 3.888 |
| Mixed, convolution–SiLU | 2.129 | 3.957 | 0.717 / 3.917 | 0.719 / 3.907 | 0.694 / 4.048 |
| Full FP16, separate kernels | 1.816 | 4.638 | 0.608 / 4.621 | 0.598 / 4.696 | 0.611 / 4.596 |
| Full FP16 with fusion requested, split fallback | 1.802 | 4.674 | 0.598 / 4.696 | 0.603 / 4.654 | 0.601 / 4.671 |
| Torch eager FP16 | 5.515 | 1.527 | 1.883 / 1.491 | 1.814 / 1.548 | 1.818 / 1.545 |
| TorchInductor FP16 | 7.424 | 1.135 | 2.489 / 1.128 | 2.440 / 1.151 | 2.495 / 1.126 |
| ORT TensorRT-RTX FP16 | 2.603 | 3.236 | 0.854 / 3.288 | 0.907 / 3.095 | 0.842 / 3.335 |

The mixed fusion improved the GPU-resident batch-16 median from
61.319 ms to 60.588 ms, while the repeated host-I/O median went from
62.764 ms to 65.827 ms. Page throughput differed by 0.4% and the fused
graph added about 101 ms of initialization in the populated-cache pass.
Keep fusion opt-in based on these measurements. The full-FP16 fallback
row times the same split execution path as the regular full-FP16 row.

The combined report is `output/native-next-validation/verified-report.json`.
It records bundle hashes, output parity, cache cases, VRAM, per-image
quality, all tile timings, and per-page throughput.

## Projection and gate–pooling optimization plan

Use the verified `output/native-arena` bundle as the control. Keep the
full checkpoint, the repacked arena, and the 128×128 tile geometry. The
latest batch-16 profile attributed 14.02 ms to 1×1 projections in full
FP16 and 18.55 ms in mixed precision. Measure the changes below independently
before combining any winners.

1. [x] **Replay-based projection tuning.** Widen the cuBLASLt candidate
   search for the latent projections and the busiest high-resolution
   matrix shapes. Warm and time candidates under CUDA Graph replay.
   Keep mode and search policy in the plan-cache key. Compare warm
   latency, throughput, startup with a populated cache, and VRAM against
   the existing eight-candidate tuner.
2. [x] **Grouped gate and pooling.** Have the grouped convolution and gate
   write its FP16 output and partial spatial sums together. Keep the
   channel projection and its input rounding. Compare with the existing
   gate followed by a separate pooling pass. Start with the 128×128,
   32-channel blocks, where the gate reads and writes the largest
   feature maps. Compare register pressure, GPU time, and fidelity in
   both precisions before extending the fusion to smaller shapes.
3. [x] **Selective FP16 accumulation.** Start with the 12 folded QKV
   projections in the mixed runner. Request FP16 accumulation for those
   projections and keep its existing FP32 normalization reductions,
   final RGB skip arithmetic, and FP32 RGB I/O. Cache this plan separately
   and compare its latency and fidelity with both existing precision modes.

Validate through the native DLL and compare with untouched Torch CUDA FP32
across all three pages and the existing stress inputs. Save raw-float
PSNR, per-image SSIMULACRA2 and maximum error, all 528 synchronized
individual batch-1 tile times, batch-16 page throughput, and isolated
VRAM. Collect the Torch and TensorRT-RTX controls at their existing
validated shapes. Retain a candidate as an opt-in experiment when its
speed or quality benefit is inconclusive.

### Prototype and cache checks

The DLL exposes the independent experiments through `gt_create_ex`.
The CLI and Python benchmark accept `--tune-projections`,
`--fuse-gate-pool` and `--fp16-qkv`. The QKV switch applies to mixed
precision. The original flags still select the existing execution paths.

Replay tuning requests up to 24 cuBLASLt candidates for the targeted
shapes. It times eight launches of each candidate in a graph, then
compares the median of three CUDA-event measurements. If no candidate
can be timed that way, it retries the original search over up to eight
candidates.
[cuBLAS CUDA Graph support](https://docs.nvidia.com/cuda/cublas/index.html#cuda-graphs-support)

On the RTX 3050 Laptop GPU, a cold replay-tuning run selected plans for
12 shapes at each batch size in each precision. The gate fused at four
sites, and the mixed QKV switch selected 12 projections. Populated cache
runs skipped replay tuning. Gate fusion reused the split cuBLASLt plans.
The fused gate used 35 registers per thread in mixed precision and 38 in
full FP16. The split gate used 30 and 40, respectively.

Batch-16 warmed device usage was about 210 MiB above the CUDA context in
mixed precision and 204–206 MiB in full FP16. Cold plan selection sampled
274 MiB and 268 MiB, respectively. For replay tuning, mixed batch-16
time to first output was 1.90 s cold and 0.41 s with a populated cache.
Full FP16 took 6.52 s cold and 0.25 s with a populated cache. These are
single measurements and include model loading, graph capture, and the
first host call.

### Output fidelity

The rebuilt control and prototypes used the same checkpoint, packed
weights, and schedule as `output/native-arena`. For both rebuilt split
controls, aggregate PSNR and all six rendered PNGs matched the previous
validation run. All configurations produced finite output on the three
pages and five batch-1 stress tiles.
PSNR uses raw floats, peak 1, and untouched Torch CUDA FP32 as the
reference. Each page cell shows raw PSNR in dB followed by SSIMULACRA2
on the rendered PNG. Maximum error is over the raw page output.

| Runner | Aggregate PSNR (dB) | page-001 | page-005 | page-009 | Max page error | Minimum stress PSNR (dB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed, split | 85.547 | 83.800 / 94.469 | 88.843 / 91.111 | 85.391 / 91.475 | 0.001902 | 78.988 |
| Mixed, replay tuning | 85.547 | 83.800 / 94.469 | 88.843 / 91.111 | 85.391 / 91.475 | 0.001902 | 78.961 |
| Mixed, fused gate–pool | 85.547 | 83.800 / 94.465 | 88.842 / 90.969 | 85.391 / 91.436 | 0.001902 | 78.988 |
| Mixed, FP16 QKV | 85.547 | 83.799 / 94.392 | 88.848 / 91.059 | 85.391 / 91.563 | 0.001898 | 78.998 |
| Full FP16, split | 72.089 | 73.272 / 94.019 | 71.774 / 90.522 | 71.433 / 91.066 | 0.001923 | 68.218 |
| Full FP16, replay tuning | 72.089 | 73.272 / 94.019 | 71.774 / 90.522 | 71.433 / 91.066 | 0.001923 | 69.132 |
| Full FP16, fused gate–pool | 72.101 | 73.258 / 94.169 | 71.811 / 90.484 | 71.439 / 91.096 | 0.002005 | 68.117 |

### Individual batch-1 tile latency

Every runner made 528 synchronized calls with one tile each, 176 per
page. The overall and per-page columns show arithmetic means of the
individual calls with host I/O. The final column comes from a separate
repeated batch test with input and output on the GPU. Torch and RTX
rows below come from the mixed comparison. The full-FP16 comparison
also ran these baselines and saved their timings. ORT profiling verified
RTX placement for the fixed batch-1 graph in both comparisons.

| Runner | Mean tile (ms) | page-001 (ms) | page-005 (ms) | page-009 (ms) | Repeated device median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed, split | 5.679 | 5.807 | 5.621 | 5.608 | 5.751 |
| Mixed, replay tuning | 5.467 | 5.695 | 5.358 | 5.349 | 4.824 |
| Mixed, fused gate–pool | 5.728 | 5.984 | 5.566 | 5.634 | 5.033 |
| Mixed, FP16 QKV | 5.523 | 5.619 | 5.468 | 5.481 | 4.987 |
| Full FP16, split | 5.085 | 5.156 | 5.058 | 5.040 | 4.608 |
| Full FP16, replay tuning | 4.946 | 4.933 | 5.082 | 4.824 | 4.339 |
| Full FP16, fused gate–pool | 5.078 | 4.986 | 5.214 | 5.033 | 4.496 |
| Torch eager FP16 | 52.167 | 59.483 | 48.482 | 48.537 | 45.404 |
| TorchInductor FP16 | 18.319 | 19.860 | 17.537 | 17.559 | 16.935 |
| ORT TensorRT-RTX FP16 | 6.491 | 6.972 | 6.199 | 6.304 | 5.478 |

### Batch-16 page throughput

Total and per-page wall times include batch assembly and synchronous host
I/O. Each runner made 33 batch-16 calls across three pages. The last
column shows the separate repeated test with device-resident I/O. ORT
profiling verified RTX placement for the fixed batch-16 graph. The
baseline rows below are from the mixed comparison. The matched RTX
row in the full-FP16 comparison measured 3.311 MPx/s.

| Runner | Total (s) | MPx/s | page-001 s / MPx/s | page-005 s / MPx/s | page-009 s / MPx/s | Repeated device median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed, split | 2.081 | 4.047 | 0.690 / 4.070 | 0.693 / 4.054 | 0.699 / 4.018 | 60.345 |
| Mixed, replay tuning | 2.115 | 3.983 | 0.716 / 3.922 | 0.702 / 4.002 | 0.697 / 4.026 | 60.159 |
| Mixed, fused gate–pool | 2.123 | 3.968 | 0.722 / 3.889 | 0.695 / 4.038 | 0.706 / 3.980 | 59.766 |
| Mixed, FP16 QKV | 2.079 | 4.053 | 0.698 / 4.024 | 0.693 / 4.055 | 0.688 / 4.080 | 60.514 |
| Full FP16, split | 1.959 | 4.301 | 0.621 / 4.523 | 0.658 / 4.271 | 0.680 / 4.126 | 52.700 |
| Full FP16, replay tuning | 1.876 | 4.490 | 0.616 / 4.555 | 0.649 / 4.328 | 0.611 / 4.597 | 52.281 |
| Full FP16, fused gate–pool | 1.832 | 4.598 | 0.602 / 4.666 | 0.604 / 4.650 | 0.626 / 4.482 | 51.563 |
| Torch eager FP16 | 5.580 | 1.510 | 1.890 / 1.486 | 1.839 / 1.527 | 1.851 / 1.517 | 163.712 |
| TorchInductor FP16 | 7.479 | 1.126 | 2.559 / 1.097 | 2.494 / 1.126 | 2.426 / 1.158 | 217.301 |
| ORT TensorRT-RTX FP16 | 2.648 | 3.182 | 0.876 / 3.204 | 0.907 / 3.096 | 0.864 / 3.248 | 76.084 |

The instrumented batch-16 profile put the four mixed gate and pooling
sites at 2.99 ms split and 2.25 ms fused. For full FP16, that measured
2.87 ms split and 2.88 ms fused. The 12 folded QKV projections in the
mixed runner went from 2.76 ms to 1.55 ms with FP16 accumulation.
These are per-operation sums from an instrumented replay. Mixed page
throughput stayed within 2% of the split control across the three
experiments.

### Full-FP16 follow-up

The follow-up used the same replay-tuned cuBLASLt plans on both sides.
Adding gate–pool fusion improved batch-16 page throughput by 4.6%.
The batch-16 device median fell from 54.044 ms to 52.075 ms. At batch 1,
the combination's repeated device median fell from 4.414 ms to
4.340 ms, and its mean of 528 individual host calls rose from 4.834 ms
to 5.218 ms.

| Runner | Mean tile (ms) | page-001 (ms) | page-005 (ms) | page-009 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Full FP16, replay tuning | 4.834 | 4.732 | 4.839 | 4.930 |
| Full FP16, both switches | 5.218 | 5.352 | 5.358 | 4.944 |

| Runner | Total (s) | MPx/s | page-001 s / MPx/s | page-005 s / MPx/s | page-009 s / MPx/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full FP16, replay tuning | 1.852 | 4.549 | 0.613 / 4.582 | 0.604 / 4.646 | 0.635 / 4.424 |
| Full FP16, both switches | 1.771 | 4.758 | 0.591 / 4.750 | 0.591 / 4.749 | 0.588 / 4.775 |

The combination reproduced the gate-only batch-16 page quality
metrics. Aggregate raw PSNR was 72.101 dB with mean SSIMULACRA2
91.916 and minimum 90.484. Batch-1 checkerboard PSNR was 65.516 dB
with both switches, 69.132 dB with replay tuning alone, and 68.218 dB
with the split control.

With a populated cache, both switches used 204 MiB of warmed device
memory above the context baseline at batch 16. The runner reported
201.56 MiB in its own weights, activations, scratch, and RGB buffers.
Time to first output was 254 ms.

The standalone EXE and the DLL produced bit-identical outputs on a
seeded batch-1 input with the combined flags in each precision when
both loaded the same CUDA libraries. The standalone NVIDIA wheel path
and the Torch process loaded different cuBLASLt builds and used
separate plan keys. The maximum raw RGB difference across those
environments was 0.000366 in mixed precision and 0.000977 in full FP16.
The benchmark tables use the DLL inside the system Python process.

Keep all three switches opt-in. Use replay tuning alone when testing
single-tile latency. Both switches are useful for batch-16 full-FP16
throughput on this GPU. Mixed precision remains the higher-fidelity
path. Its selective QKV switch speeds up those projections with
little change to page throughput.

The combined report is
`output/native-projection-validation/verified-report.json`. It links
the raw reports, including every tile time, per-page quality and
throughput, measured VRAM and startup, and the individual operation
profiles.

## Packed-session rerun on 21 November 2026

The run used the packed-session DLL and the same GateRV3 checkpoint on
the RTX 3050 Laptop GPU with Torch 2.15.0+cu132. The original 18 reports matched
the DLL, packed model and checkpoint hashes. Every timed runner used
the same 528 tiles across three pages. The host tables use end-to-end
inputs and outputs. The repeated device medians include the packed
session's device-buffer copies. [Raw benchmark reports](../output/session-bench-20261121-135646/)

Bold highlights the best aggregate results within native mixed,
native full FP16, and the external FP16 baselines. Values are ranked
before rounding. Exact ties share highlighting.

### Output fidelity

PSNR uses raw floats against untouched Torch CUDA FP32 with peak 1.
Each page cell shows PSNR in dB followed by SSIMULACRA2 on the rendered
PNG. The aggregate PSNR pools pixel errors across all three pages, and
mean SSIMULACRA2 averages their three perceptual scores. The ten native
quality reports and the follow-up for Torch eager, TorchInductor, and
TensorRT-RTX used byte-identical FP32 reference PNGs. The follow-up used
the same checkpoint and fixed batch-16 RTX graph and verified provider
placement. [Baseline fidelity report](../output/session-bench-20261121-135646/baseline-fidelity-b16/report.json)

| Runner | Aggregate PSNR (dB) | Mean SSIMULACRA2 | page-001 | page-005 | page-009 | Max page error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed, split | 85.547 | **92.352** | 83.800 / 94.469 | 88.843 / 91.111 | 85.391 / 91.475 | 0.001902 |
| Mixed, replay tuning | 85.547 | **92.352** | 83.800 / 94.469 | 88.843 / 91.111 | 85.391 / 91.475 | 0.001902 |
| Mixed, fused gate–pool | 85.547 | 92.290 | 83.800 / 94.465 | 88.842 / 90.969 | 85.391 / 91.436 | 0.001902 |
| Mixed, FP16 QKV | **85.547** | 92.338 | 83.799 / 94.392 | 88.848 / 91.059 | 85.391 / 91.563 | 0.001898 |
| Mixed, fused convolution–SiLU | 85.547 | 92.286 | 83.800 / 94.329 | 88.843 / 91.096 | 85.390 / 91.434 | 0.001902 |
| Full FP16, split | 72.089 | 91.869 | 73.272 / 94.019 | 71.774 / 90.522 | 71.433 / 91.066 | 0.001923 |
| Full FP16, replay tuning | 72.089 | 91.869 | 73.272 / 94.019 | 71.774 / 90.522 | 71.433 / 91.066 | 0.001923 |
| Full FP16, fused gate–pool | **72.101** | **91.916** | 73.258 / 94.169 | 71.811 / 90.484 | 71.439 / 91.096 | 0.002005 |
| Full FP16, SiLU requested | 72.089 | 91.869 | 73.272 / 94.019 | 71.774 / 90.522 | 71.433 / 91.066 | 0.001923 |
| Full FP16, replay tuning + fused gate–pool | **72.101** | **91.916** | 73.258 / 94.169 | 71.811 / 90.484 | 71.439 / 91.096 | 0.002005 |
| Torch eager FP16 | **72.436** | 91.943 | 74.190 / 94.126 | 71.889 / 90.568 | 71.652 / 91.134 | 0.002005 |
| TorchInductor FP16 | 72.409 | **91.997** | 73.700 / 94.244 | 71.974 / 90.660 | 71.795 / 91.088 | 0.002005 |
| ORT TensorRT-RTX FP16 | 72.133 | 91.899 | 73.818 / 94.070 | 71.707 / 90.548 | 71.275 / 91.080 | 0.001923 |

Full FP16 gate–pool fusion lowered batch-1 checkerboard PSNR from
69.347 to 65.404 dB. Replay tuning plus fusion gave the same checkerboard
score and batch-16 page metrics as gate–pool fusion alone.

### Individual batch-1 tile latency

The means use 528 synchronized, individual batch-1 calls per runner.
Each report records the image, tile index, coordinates and time for
all 528 calls. The TensorRT-RTX row uses the separate fixed batch-1
graph. Profiling confirmed RTX placement.

| Runner | Mean tile (ms) | page-001 (ms) | page-005 (ms) | page-009 (ms) | Repeated device median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed, split | 4.901 | 4.960 | 4.874 | 4.868 | 4.677 |
| Mixed, replay tuning | 4.853 | 4.908 | 4.835 | 4.816 | 4.672 |
| Mixed, fused gate–pool | **4.819** | 4.837 | 4.827 | 4.793 | 4.607 |
| Mixed, FP16 QKV | 4.844 | 4.873 | 4.829 | 4.829 | 4.631 |
| Full FP16, split | 4.461 | 4.464 | 4.458 | 4.462 | 4.254 |
| Full FP16, replay tuning | 4.495 | 4.497 | 4.497 | 4.493 | 4.323 |
| Full FP16, fused gate–pool | **4.421** | 4.430 | 4.404 | 4.430 | 4.219 |
| Torch eager FP16 | 42.459 | 42.574 | 42.251 | 42.552 | 39.407 |
| TorchInductor FP16 | 16.225 | 16.329 | 16.252 | 16.094 | 15.601 |
| ORT TensorRT-RTX FP16 | **5.667** | 5.755 | 5.659 | 5.585 | 5.555 |

### Batch-16 page throughput

Each runner made 33 batch-16 calls, with 11 calls per page. Times
include batch assembly and synchronous host I/O. MPx/s counts source
image pixels. Amortized ms/tile spreads the measured batch-16 wall time
over all 528 tiles. Profiling confirmed RTX placement for the fixed
batch-16 graph. The replay tuning plus gate–pool row comes from the
separate full-FP16 comparison below.

| Runner | Total (s) | MPx/s | ms/tile | page-001 s / MPx/s | page-005 s / MPx/s | page-009 s / MPx/s | Repeated device median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed, split | 1.941 | 4.339 | 3.677 | 0.647 / 4.340 | 0.648 / 4.335 | 0.646 / 4.344 | 56.737 |
| Mixed, replay tuning | 1.938 | 4.347 | 3.670 | 0.646 / 4.345 | 0.646 / 4.344 | 0.645 / 4.354 | 56.703 |
| Mixed, fused gate–pool | 1.920 | 4.388 | 3.636 | 0.641 / 4.381 | 0.639 / 4.398 | 0.641 / 4.384 | 55.983 |
| Mixed, FP16 QKV | 1.919 | **4.390** | **3.634** | 0.642 / 4.374 | 0.639 / 4.395 | 0.638 / 4.402 | 56.065 |
| Full FP16, split | 1.690 | 4.985 | 3.200 | 0.563 / 4.988 | 0.563 / 4.986 | 0.564 / 4.982 | 49.560 |
| Full FP16, replay tuning | 1.693 | 4.977 | 3.206 | 0.564 / 4.982 | 0.564 / 4.978 | 0.565 / 4.971 | 49.514 |
| Full FP16, fused gate–pool | 1.680 | 5.014 | 3.182 | 0.560 / 5.012 | 0.560 / 5.018 | 0.560 / 5.011 | 49.011 |
| Full FP16, replay tuning + fused gate–pool | 1.663 | **5.064** | **3.151** | 0.554 / 5.066 | 0.556 / 5.054 | 0.554 / 5.072 | 48.815 |
| Torch eager FP16 | 5.138 | 1.640 | 9.731 | 1.714 / 1.638 | 1.711 / 1.641 | 1.713 / 1.639 | 152.656 |
| TorchInductor FP16 | 6.834 | 1.233 | 12.943 | 2.281 / 1.231 | 2.272 / 1.236 | 2.281 / 1.231 | 203.172 |
| ORT TensorRT-RTX FP16 | 2.397 | **3.515** | **4.539** | 0.799 / 3.514 | 0.795 / 3.531 | 0.802 / 3.500 | 66.740 |

The instrumented batch-16 profiles put mixed FP16 QKV at 1.519 ms
across the 12 folded projections, down from 2.398 ms. Mixed page
throughput went from 4.339 to 4.390 MPx/s, a gain of about 1.2%.
At the four eligible gate–pool sites, fused operation time moved
from 2.958 to 2.221 ms in mixed precision and from 2.800 to 2.292 ms
in full FP16. Replay tuning alone changed page throughput by less than
0.2% in both precisions.

### Separate SiLU and combined FP16 runs

The first four rows in each table come from the SiLU comparison.
The final two rows compare full-FP16 replay tuning alone with replay
tuning plus gate–pool fusion. Each comparison has its own control.

| Runner | Mean tile (ms) | page-001 (ms) | page-005 (ms) | page-009 (ms) | Repeated device median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed, split | 4.841 | 4.823 | 4.838 | 4.861 | 4.609 |
| Mixed, fused convolution–SiLU | **4.782** | 4.790 | 4.785 | 4.771 | 4.553 |
| Full FP16, split | 4.507 | 4.489 | 4.508 | 4.524 | 4.210 |
| Full FP16, SiLU requested | 4.458 | 4.404 | 4.454 | 4.515 | 4.204 |
| Full FP16, replay tuning | 4.496 | 4.486 | 4.486 | 4.516 | 4.306 |
| Full FP16, replay tuning + fused gate–pool | **4.467** | 4.463 | 4.463 | 4.475 | 4.274 |
4.463 | 4.475 | 4.274 |

| Runner | Total (s) | MPx/s | ms/tile | page-001 s / MPx/s | page-005 s / MPx/s | page-009 s / MPx/s | Repeated device median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed, split | 1.933 | 4.357 | 3.662 | 0.643 / 4.367 | 0.642 / 4.374 | 0.648 / 4.331 | 56.450 |
| Mixed, fused convolution–SiLU | 1.886 | **4.467** | **3.572** | 0.629 / 4.467 | 0.629 / 4.464 | 0.628 / 4.471 | 55.147 |
| Full FP16, split | 1.703 | 4.948 | 3.225 | 0.569 / 4.933 | 0.563 / 4.985 | 0.570 / 4.926 | 49.380 |
| Full FP16, SiLU requested | 1.701 | 4.954 | 3.221 | 0.570 / 4.929 | 0.563 / 4.992 | 0.568 / 4.941 | 49.307 |
| Full FP16, replay tuning | 1.683 | 5.005 | 3.187 | 0.562 / 4.996 | 0.560 / 5.013 | 0.561 / 5.008 | 49.417 |
| Full FP16, replay tuning + fused gate–pool | 1.663 | **5.064** | **3.151** | 0.554 / 5.066 | 0.556 / 5.054 | 0.554 / 5.072 | 48.815 |

Mixed convolution–SiLU fusion ran at 12 sites and improved batch-16
page throughput by 2.5% in its matched comparison. The full-FP16
SiLU request used split kernels because cuDNN reported no compatible
fused plan. Adding gate–pool fusion to full-FP16 replay tuning raised
throughput from 5.005 to 5.064 MPx/s, about 1.2%. Mean individual
tile time changed from 4.496 to 4.467 ms in that comparison.

### VRAM and startup

Warmed VRAM measures device usage above the pre-run context baseline
on the second memory pass. The startup columns show the initial
pass and the populated-cache pass, respectively. Split, replay
tuning and mixed FP16 QKV started with cache misses. SiLU and
gate–pool reused the split library plans already written in the
initial pass. The combined full-FP16 run had its own speed and quality
reports. The native memory sweep covered the nine individual configurations.

Each external baseline ran on its own in a fresh process for each
batch size and pass. The device-wide deltas include retained allocator
and runtime memory. Torch eager shows two independent starts. TorchInductor
and TensorRT-RTX started with empty caches for each graph or shape and
reused them on the second pass. The RTX provider was verified for both
fixed graphs. [Baseline VRAM and startup report](../output/session-bench-20261121-135646/baseline-memory/report.json)

| Runner | Batch 1 warmed MiB | Batch 1 first output (ms) | Batch 16 warmed MiB | Batch 16 first output (ms) |
| --- | ---: | ---: | ---: | ---: |
| Mixed, split | 98 | 539 / 323 | **210** | 959 / 273 |
| Mixed, replay tuning | **86** | 302 / 157 | 212 | 1711 / 252 |
| Mixed, fused gate–pool | **86** | 135 / 140 | 212 | 237 / 240 |
| Mixed, FP16 QKV | **86** | 236 / 144 | 212 | 941 / 259 |
| Mixed, fused convolution–SiLU | **86** | 235 / 242 | **210** | 344 / 352 |
| Full FP16, split | 84 | 623 / 181 | **204** | 5527 / 238 |
| Full FP16, replay tuning | **82** | 567 / 141 | 206 | 6121 / 224 |
| Full FP16, fused gate–pool | **82** | 156 / 149 | 206 | 249 / 230 |
| Full FP16, SiLU requested | 82 | 149 / 155 | 204 | 237 / 234 |
| Torch eager FP16 | **130** | 1350 / 1412 | 434 | 1739 / 1718 |
| TorchInductor FP16 | 166 | 116637 / 12637 | 452 | 113774 / 11221 |
| ORT TensorRT-RTX FP16 | 172 | 16746 / 8343 | **302** | 14857 / 6965 |

Replay tuning selected plans for 13 shapes on a miss at each batch size
and precision. Warmed batch-16 VRAM stayed within 210–212 MiB for
mixed precision and 204–206 MiB for full FP16. The highest sampled
native startup increases at batch 16 were 274 MiB and 268 MiB, respectively.
