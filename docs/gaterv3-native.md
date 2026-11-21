# Native GateRV3 implementation and verification

The original GateRV3 paths and commands below document historical
runs. See the [current runtime guide](gaterv3-python.md) for `.ummodel`
and `.umplan` usage.

The target is the 1× GateRV3 checkpoint in this checkout, using RGB 128×128 tiles
at batch 1 for latency and batch 16 for throughput. Keep all 12 latent blocks,
the checkpoint's attention mode, both model branches, and the reference's
boundary handling. Smaller or approximate networks would require a separate
benchmark.

Native inference means a C++/CUDA executable or library can load prepared weights
and run the entire forward pass without Python, PyTorch, ONNX Runtime, or
TensorRT. An offline Python tool can reuse the strict checkpoint loader and
Conv3XC reparameterization. CUDA libraries can provide dense convolutions and
matrix products. The installed system Python remains the baseline and validation
environment.

## Work items

- [x] Locate MSVC, CUDA headers/compiler, and the installed runtime libraries.
  Record the GPU, dependencies, checkpoint hash, and available baseline artifacts.
- [x] Export inference weights and a shape-specialized execution schedule for
  the full graph. Preserve convolution padding, RMSNorm epsilon placement,
  grouped channel ordering, shuffles, attention, skips, and raw RGB output.
- [x] Build one native runtime for both batch sizes. Keep activations on the GPU,
  use Tensor Core library paths for dense layers, and use specialized kernels
  for reductions, attention, channel routing, and pointwise work.
- [x] Verify native output against untouched Torch FP32 before timing. Compare
  unclipped floats, maximum absolute error, finite values, and per-image PSNR.
  Repeat the prepared baseline's SSIMULACRA2 comparison on the same pages and
  add deterministic stress tiles. Use the prepared RTX quality as the initial
  speed-comparison target and report mixed-precision differences explicitly.
- [x] Benchmark actual batch-1 calls and batch-16 throughput. Record warm
  GPU-resident timings separately from synchronized CPU-input/CPU-output
  wall time. Use the same tile data and timing boundary across runners, and
  report the prepared page-level numbers as context.
- [x] Measure the effect of CUDA Graph replay and operation fusion, and profile
  the dense projections and spatial kernels. Record wins, regressions, and
  remaining hotspots or limits before concluding which baselines native
  inference beats.

## Starting reference

The [prepared baseline](gaterv3-baseline.md) reports 6.454 ms mean actual
batch-1 latency and 3.103 MPx/s at batch 16 for ORT TensorRT-RTX FP16.
The low-level comparison will remeasure the baselines because the prepared ORT
adapter copies across the CPU/GPU boundary inside the runner. Model accuracy
and inference speed both need to be measured before promoting an optimization.

## Run log

2026-11-21. Inspected the model source and identified the prepared checkpoint,
sample directory, and fixed batch-1/batch-16 ONNX graphs. This plan was written
before compiling, running inference, or benchmarking.

The RTX 3050 Laptop GPU has 4 GiB VRAM, compute capability 8.6, and driver
610.62. System Python provides CUDA 13 headers, NVRTC, cuBLAS and cuDNN.
MSVC 14.42 is installed. Use NVRTC to compile the device code offline and
MSVC to build the native runner. The checkpoint loader confirmed the spatial
attention path in all twelve blocks. Each 128×128 tile produces 64 latent
tokens with 16 heads and 32 channels per head.

The first native version emitted 440 operations. Its GPU-resident medians
were 5.86 ms at batch 1 and 89.05 ms at batch 16, compared with 5.21 ms and
75.12 ms for TensorRT-RTX in the same run. Shape specialization reduced the
batch-16 median to 73.69 ms in a separate native-only run. An instrumented
run identified dense projections and convolutions as the largest classes,
followed by the local grouped gate and inception mixer at batch 16.

Pairwise vector loads reduce traffic and instructions in the grouped gate,
inception mixer, RMSNorm, and pointwise kernels. The spatial mean and its
small dense projection share a two-stage reduction. The final schedule has
416 operations and a 10.25 MiB activation arena per tile.

## Build and run

The workstation already has the required CUDA runtime DLLs and NVRTC in
system Python. The CUDA runtime wheel lacks the CRT and CCCL header trees
needed by the WMMA and MSVC builds. These commands install the missing headers
under the ignored output directory. [CUDA CRT wheel](https://pypi.org/project/nvidia-cuda-crt/13.4.59/)

```powershell
$py = 'C:\Python314\python.exe'
& $py -m pip install --no-deps --target output/native-toolchain nvidia-cuda-crt==13.4.59
& $py -m pip install --no-deps --target output/native-toolchain/cccl nvidia-cuda-cccl==13.3.5.0.6
& $py scripts/build_native.py
& $py scripts/export_native.py --legacy
```

The build uses MSVC x64 and produces an Ampere `sm_86` CUDA binary. The weight
export reuses the checkpoint loader and fuses Conv3XC in FP32 before packing
FP16 feature weights. The runtime uses FP16 activations, FP32 accumulation
for normalization, attention and dense math, and an FP32 RGB skip. It uses
NHWC internally, cuBLASLt for 1×1 projections and cuDNN for 3×3 convolutions.
The other operations are native CUDA kernels. [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html)
[cuDNN convolution API](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html)

The standalone executable takes raw contiguous little-endian float32 RGB
input in NCHW order, with pixels scaled to `[0, 1]`. It writes a batch of the
same shape. Use batches 1 or 16 at 128×128. The caller handles image decoding,
border tile replication, reassembly, and clamping if needed. The DLL API in
[`native/gaterv3/runner.h`](../native/gaterv3/runner.h) provides synchronous host
I/O and enqueue on a caller-owned CUDA stream.

```powershell
$nvidia = 'C:\Python314\Lib\site-packages\nvidia'
$env:Path = "$nvidia\cu13\bin\x86_64;$nvidia\cudnn\bin;$env:Path"
& .\output\native\gaterv3.exe .\output\native 1 .\input.f32 .\output.f32
```

Standalone inference needs the NVIDIA driver, cuBLASLt, cuDNN and the MSVC
runtime. Distribute the executable or DLL with `kernels.cubin`,
`schedule.bin`, and `weights.bin`. `checkpoint.json` links the generated
files to the source checkpoint for the benchmark harness.

```powershell
& $py scripts/benchmark_native.py --profile --rounds 4 --repeats 8
```

The benchmark uses the two fixed ONNX graphs from `output/baseline/`.
Profiler sessions check placement before the timed RTX sessions are created.
The report under `output/native-validation/` includes raw output metrics,
rendered pages, RTX placement, actual individual tile timings, and
batch-16 per-page timings. The repeated timing test alternates variant order.
Its GPU-resident measurements use prebound device inputs and outputs, and its
host measurements include copies and conversion to each backend's input
dtype. The Torch rerun uses the default cuDNN heuristic selection. Page
timing covers transfers and batch assembly while excluding
image decoding, tiling, stitching, and PNG encoding. The instrumented native
profile uses separate CUDA Graph replays with timing events around each
operation. [CUDA event API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EVENT.html)

## Fidelity results

The raw model outputs were compared with untouched Torch CUDA FP32 before
clipping or PNG quantization. The native checkpoint SHA-256 matches the
prepared baseline.

| Image | Raw PSNR (dB) | Max absolute error | SSIMULACRA2 |
| --- | ---: | ---: | ---: |
| page-001.jpg | 83.80 | 0.000869 | 94.41 |
| page-005.jpg | 88.85 | 0.000829 | 91.08 |
| page-009.jpg | 85.39 | 0.001917 | 91.48 |

Aggregate raw PSNR was **85.55 dB**, versus **72.13 dB** for the prepared
RTX FP16 graph. Mean SSIMULACRA2 was **92.32**, with a minimum of **91.08**.
The prepared RTX graph measured **91.92** mean and **90.59** minimum.
Batch-1 black, white, checkerboard, seeded random and real-image inputs
produced finite outputs with raw PSNR of 100.96, 101.36, 80.97, 79.00 and
89.35 dB, respectively.

The standalone EXE and DLL produced bit-identical outputs on the same
seeded random input for both batches. Standalone raw PSNR versus FP32 was
78.85 dB at batch 1 and 78.98 dB at batch 16. The executable rejected an
unsupported batch of 3. An 81×77 image exercised edge replication, a
partially filled batch, and output cropping, with 79.00 dB raw PSNR.

## Performance results

Measured on the RTX 3050 Laptop GPU with PyTorch 2.15.0+cu132, ONNX Runtime
1.31.1, and the prepared TensorRT-RTX graphs. Both profiled graphs reported
`NvTensorRTRTXExecutionProvider` for executed compute partitions. The IDE
crash interrupted the first timing attempt after fidelity checks. Separate
batch-1 and batch-16 retries used the same verified binary and weights.

| Runner | Mean batch-1 tile (ms) | Batch-16 total (s) | Batch-16 MPx/s |
| --- | ---: | ---: | ---: |
| Native CUDA Graph | **5.420** | **2.448** | **3.441** |
| Torch eager FP16 | 47.574 | 5.695 | 1.479 |
| TorchInductor FP16 | 17.720 | 7.421 | 1.135 |
| ORT TensorRT-RTX FP16 | 5.906 | 2.696 | 3.125 |

The batch-1 average includes 528 individually timed calls per runner.
The native runner measured **8.2% lower mean latency** and **10.1% higher
page throughput** than TensorRT-RTX in the matched rerun. The original
prepared TorchInductor run achieved 12.954 ms and 1.864 MPx/s under its
original settings and timing boundary.

| Image | Runner | Mean batch-1 tile (ms) | Batch-16 time (s) | Batch-16 MPx/s |
| --- | --- | ---: | ---: | ---: |
| page-001.jpg | Native | 5.425 | 0.776 | 3.619 |
| page-001.jpg | Torch eager | 44.428 | 1.950 | 1.440 |
| page-001.jpg | TorchInductor | 17.453 | 2.429 | 1.156 |
| page-001.jpg | TensorRT-RTX | 5.948 | 0.892 | 3.148 |
| page-005.jpg | Native | 5.521 | 0.801 | 3.506 |
| page-005.jpg | Torch eager | 51.494 | 1.860 | 1.510 |
| page-005.jpg | TorchInductor | 17.510 | 2.522 | 1.113 |
| page-005.jpg | TensorRT-RTX | 5.894 | 0.912 | 3.078 |
| page-009.jpg | Native | 5.312 | 0.871 | 3.223 |
| page-009.jpg | Torch eager | 46.800 | 1.886 | 1.489 |
| page-009.jpg | TorchInductor | 18.195 | 2.470 | 1.137 |
| page-009.jpg | TensorRT-RTX | 5.875 | 0.892 | 3.149 |

Repeated-call medians below use 32 samples per runner, shape, and timing
boundary across four alternating-order rounds.

| Runner | Batch 1 on GPU (ms) | Batch 1 host I/O (ms) | Batch 16 on GPU (ms) | Batch 16 host I/O (ms) |
| --- | ---: | ---: | ---: | ---: |
| Native CUDA Graph | 5.320 | 5.653 | 64.732 | 66.394 |
| Native direct dispatch | 12.800 | 9.340 | 66.079 | 66.637 |
| Torch eager FP16 | 62.637 | 86.880 | 160.945 | 166.734 |
| TorchInductor FP16 | 19.858 | 20.458 | 214.525 | 219.475 |
| ORT TensorRT-RTX FP16 | 5.621 | 6.890 | 71.647 | 74.597 |

CUDA Graph replay had its largest effect at batch 1. In the instrumented
batch-16 replay, 1×1 projections and 3×3 convolutions remained the largest
classes at 19.00 ms and 13.75 ms, followed by the inception mixer at 6.01 ms
and the local grouped gate at 5.68 ms. The earlier shape-specialized run
measured 8.98 ms for inception and 9.82 ms for the local gate. These
instrumented timings include the event nodes used for profiling.

The local combined record is `output/native-validation/verified-report.json`.
It contains the fidelity results, bundle hashes, and both timing reports
with every individual tile measurement.
