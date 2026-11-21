# GateRV3 kernel tuning and full-FP16 experiment

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

- [ ] Replace append-order allocation with an offline packing pass that
  places larger tensors first and assigns offsets using their lifetimes.
  Treat fused launches as simultaneous reads and writes of all participating
  tensors, including the two outputs of residual plus RMSNorm.
- [ ] Validate each offset's alignment and confirm that overlapping lifetimes
  never share memory. Run both precisions against the existing bundle at the
  same shape and precision, with the same kernel plans when checking bitwise
  parity. Remeasure exact runner buffers and warmed device usage separately.

An offline packing of the current schedule reached **8 MiB per tile**.
That corresponds to **128 MiB at batch 16** versus the current 164 MiB
arena, a **36 MiB reduction in activation allocation**. This is an offline
result. GPU output parity and the change in warmed device usage are pending.

### 2. Persist selected kernel plans

- [ ] Add an optional `cache_dir` session setting and `--cache-dir` CLI
  flag. Save the selected cuDNN and cuBLASLt choices and workspace
  requirements. Use the compatibility key and miss behavior in the
  [refactor plan](gaterv3-native-refactor.md#configurable-kernel-plan-cache).
- [ ] Benchmark a first load into an empty cache and a fresh process with
  a populated cache. Record tune time, graph capture time, total time to first
  output, warmed latency and output parity. Include mismatched and corrupt
  cache tests.

Batch-16 setup measured **1.04 s** for mixed and **5.68 s** for full FP16 in
the isolated setup pass. The cache targets repeat setup time. Measure the
remaining model load, allocation and graph capture costs on cache hits.

### 3. Prototype convolution and SiLU fusion

- [ ] Start with the repeated 128×128, 32-channel convolutions in the SISR
  branch. Prototype a convolution with a SiLU epilogue using a compatible
  cuDNN graph plan on the RTX 3050. Keep the current separate kernels as a
  fallback, and check that fused intermediates preserve the intended rounding
  or meet the same-precision fidelity criteria.
- [ ] Compare the fused and split paths with batch 1 and 16, both precisions,
  and identical inputs. Include workspace and startup cost alongside warm
  page throughput. Extend to other shapes after the first candidate is measured.

The batch-16 full-FP16 profile attributed **7.01 ms** to 19 full-resolution
32-to-32 convolutions and **2.98 ms** to 12 SiLU dispatches. Only the
convolutions with a following SiLU are fusion candidates. These are
instrumented operation times. The end-to-end speedup is pending.
[cuDNN graph fusion documentation](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html)
