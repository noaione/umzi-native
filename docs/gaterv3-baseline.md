# GateRV3 CUDA baseline

## Scope

This checkout reconstructs the native PyTorch `GateRV3` module from the raw
checkpoint at `models/1x_umzi_digital_decompress_gaterv3_1.pth`. The baseline
matrix uses 128×128 source tiles and batches 16 tiles. The untouched FP32 CUDA
model is the reference; candidates are eager FP16, TorchInductor FP16, and a
fixed-shape FP16 ONNX graph executed by ONNX Runtime TensorRT-RTX.

The model is 1× (no spatial upscaling), with 40,753,510 parameters and inferred
configuration `dim=32`, encoder `(2,2,4,8)`, decoder `(2,2,2,2)`, 12 latent
blocks, attention enabled, and four SISR blocks. `src/model.py` infers this
configuration from checkpoint keys and loads strictly. `src/gaterv3.py` holds
the native model implementation. Conv3XC reparameterization is fused in FP32
before any optional FP16 cast.

## Environment observed on this workstation

Use the installed global interpreter exactly as requested:

```powershell
$py = 'C:\Python314\python.exe'
& $py --version
& $py -c "import torch, onnx, onnxruntime as ort; print(torch.__version__, torch.cuda.is_available()); print(onnx.__version__, ort.__version__, ort.get_available_providers()); print(torch.cuda.get_device_name(0))"
```

At preparation time this reported Python 3.14, PyTorch `2.15.0+cu132`, CUDA
available on an NVIDIA GeForce RTX 3050 Laptop GPU, ONNX `1.23.2`, ONNX Runtime
`1.31.1`, and the `TensorrtExecutionProvider`, `CUDAExecutionProvider`, and
`CPUExecutionProvider`. The TensorRT-RTX plugin and `tensorrt` Python module are
also importable. Exact driver/runtime versions can change independently; save
the command output with each benchmark report when comparing machines.

## Run the baseline matrix

From the repository root in PowerShell:

```powershell
$py = 'C:\Python314\python.exe'
& $py .\scripts\benchmark.py
```

Defaults are `samples/input`, 128-pixel tiles, batch 16, and output under
`output/baseline/`. The script writes the untouched FP32 reference images and
the eager FP16, Inductor FP16, and TensorRT-RTX FP16 candidate images into
separate subdirectories. The TensorRT-RTX graph has a fixed input shape
`[16,3,128,128]`. ORT telemetry is disabled before session creation, the RTX
provider is registered through the ORT device API, fallback is disabled, and
the run stops if RTX is not active. `--skip-trt` skips ONNX export and runtime;
`--input`, `--output`, `--tile`, and `--batch` can override defaults.

Inductor compiles and warms the exact batch/tile shape before steady inference.
Its compilation cost is separate from the saved image inference times and can
be substantial on first run. A persistent cache can be set before Python starts:

```powershell
$env:TORCHINDUCTOR_CACHE_DIR = (Join-Path $PWD 'output\inductor-cache')
& $py .\scripts\benchmark.py
```

## Quality comparison

Compare each output image against the matching FP32 output (not the input
scan). This isolates numeric/backend differences from restoration quality.
The supplied tool prints one score per pair:

```powershell
$tool = 'C:\Tools\imaging\ssimulacra2.exe'
$reference = '.\output\baseline\torch-fp32'
foreach ($variant in 'torch-fp16','inductor-fp16','trt-rtx-fp16') {
  Get-ChildItem ".\output\baseline\$variant\*.png" | ForEach-Object {
    $ref = Join-Path $reference $_.Name
    if (Test-Path $ref) {
      "{0} {1}" -f $variant, $_.Name
      & $tool $ref $_.FullName
    }
  }
}
```

Keep all individual scores; the mean can hide a poor outlier. The report JSON
records configuration, device, variant wall time, output directories, the ONNX
provider list, and per-image/mean/minimum SSIMULACRA2 scores when the configured
tool exists. SSIMULACRA2 is perceptual and should be read alongside exact
paired-output inspection for any score that looks anomalous.

## Initial run (2026-11-21)

There are two separate timing measurements:

- **Batch-16 throughput:** synchronized time for each image and overall MPx/s
  while running 16 tiles per call. These timings exclude image encoding, ONNX
  export, TensorRT session initialization, and Inductor compilation.
- **Actual tile latency:** every source tile is then run in its own synchronized
  batch-1 call. The JSON stores each sample's elapsed seconds and milliseconds,
  image, tile index, and coordinates. The average is calculated from those
  individual batch-1 samples, never inferred from batch-16 total time. TensorRT-
  RTX uses a separate fixed batch-1 ONNX graph/session for these measurements.

Batch-16 overall results (528 total tiles):

| Variant | Total (s) | MPx/s | Aggregate PSNR (dB) | Mean SSIMULACRA2 | Minimum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Torch CUDA FP32 reference | 12.152 | 0.693 | reference | — | — |
| Torch CUDA eager FP16 | 8.757 | 0.962 | 72.440 | 92.014 | 90.716 |
| TorchInductor CUDA FP16 | 4.520 | 1.864 | 72.408 | 91.988 | 90.766 |
| ONNX Runtime TensorRT-RTX FP16 | 2.715 | 3.103 | 72.133 | 91.922 | 90.589 |

Mean actual batch-1 tile latency across all 528 individual measurements:

| Variant | Average tile time (ms) | Average tile time (s) |
| --- | ---: | ---: |
| Torch CUDA FP32 reference | 42.073 | 0.042073 |
| Torch CUDA eager FP16 | 50.722 | 0.050722 |
| TorchInductor CUDA FP16 | 12.954 | 0.012954 |
| ONNX Runtime TensorRT-RTX FP16 | 6.454 | 0.006454 |

Per-image batch-16 time/throughput and separately measured mean batch-1 tile
latency for that image:

| Image | Variant | Batch-16 time (s) | MPx/s | Mean individual tile time (ms) | PSNR (dB) |
| --- | --- | ---: | ---: | ---: | ---: |
| page-001.jpg | Torch CUDA FP32 | 6.361 | 0.441 | 39.765 | reference |
| page-001.jpg | Torch CUDA eager FP16 | 5.026 | 0.559 | 49.583 | 74.191 |
| page-001.jpg | TorchInductor CUDA FP16 | 1.681 | 1.670 | 13.070 | 73.700 |
| page-001.jpg | TensorRT-RTX FP16 | 0.961 | 2.921 | 6.445 | 73.818 |
| page-005.jpg | Torch CUDA FP32 | 2.932 | 0.958 | 43.989 | reference |
| page-005.jpg | Torch CUDA eager FP16 | 1.863 | 1.507 | 51.974 | 71.899 |
| page-005.jpg | TorchInductor CUDA FP16 | 1.397 | 2.010 | 12.927 | 71.972 |
| page-005.jpg | TensorRT-RTX FP16 | 0.873 | 3.215 | 6.601 | 71.707 |
| page-009.jpg | Torch CUDA FP32 | 2.859 | 0.982 | 42.464 | reference |
| page-009.jpg | Torch CUDA eager FP16 | 1.868 | 1.503 | 50.609 | 71.652 |
| page-009.jpg | TorchInductor CUDA FP16 | 1.442 | 1.948 | 12.864 | 71.796 |
| page-009.jpg | TensorRT-RTX FP16 | 0.880 | 3.192 | 6.317 | 71.276 |

All nine candidate/page perceptual scores were above 90 in this small sample.
PSNR and SSIMULACRA2 describe agreement with the FP32 model outputs, not
ground-truth restoration quality. The RTX session reported
`NvTensorRTRTXExecutionProvider` with CPU fallback provider present; session
fallback is disabled, but provider presence alone does not establish that every
node was assigned to RTX. This is an initial local baseline, not a broad quality
validation.

## Files and artifacts

- `src/gaterv3.py`: native GateRV3 architecture.
- `src/model.py`: checkpoint configuration inference, strict loading, and
  FP32-before-cast evaluation setup.
- `scripts/benchmark.py`: fixed-shape PyTorch/Inductor/ONNX Runtime runner.
- `output/baseline/`: generated ONNX graph, images, report, and compilation
  cache. Generated artifacts are ignored by git; do not commit model weights or
  benchmark image outputs.
- `../umzi-bench`: related optimized exports and previous experiments. Treat
  those as methodological references only; this checkout regenerates a local
  fixed batch-16/tile-128 graph from its own checkpoint.

## Caveats

This initial harness uses non-overlapping tiles and edge replication for the
partial right/bottom border tiles. Compare variants using this same tiling
path. The model's own internal padding is unchanged. Batch 16 is used for
PyTorch and the exported ONNX graph; it is a requested starting point, not a
claim that it is the fastest or fits every GPU. Export/runtime support and
quality must be confirmed from a completed run before treating an FP16 result
as a production candidate. The RTX provider being active does not by itself
prove every ONNX node was assigned to it; inspect ORT profiling if placement
becomes a performance question.
