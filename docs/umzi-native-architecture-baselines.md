# GateRV2, MoESR, and GFISRv2 CUDA baselines

The [architecture expansion plan](umzi-native-architecture-expansion.md) uses
the original Resselt checkpoint forwards in eager CUDA FP32 as the reference.
The benchmark loads a separate copy for each FP16 backend. TF32 and cuDNN
autotuning are disabled. Inductor uses static shapes and requires a full
Dynamo graph. Its complex FFT operations can use ATen kernels within the
compiled graph.

| Checkpoint family | Samples | Scale | Source tile | Throughput batch | Input overlap |
| --- | --- | ---: | ---: | ---: | ---: |
| GateRV2 Dehalftone | `samples/halftone/` | 1× | 128×128 | 16 | 0 px |
| MoESR DWTP DS | `samples/mscale/` | 4× | 256×256 | 2 | 0 px |
| GFISRv2 MangaScale | `samples/mscale/` | 4× | 256×256 | 2 | 32 px |

Dehalftone uses `watanare-page-085.jpg` and `watanare-page-086.jpg`
(1,800×2,565). Both 4× models use `lb-page-003.png`, `lb-page-004.png`,
and `lb-page-005.png` (900×1,350). The page grid is row-major, with
replicated right and bottom padding and a crop at the checkpoint's output
scale. GFISRv2 uses a 224-pixel input stride and linear feathering across
128 output pixels. The final partial throughput batch repeats the last tile.
Page assembly uses only real tiles.

## Run

From the repository root in PowerShell, run each backend in its own process so
its CUDA allocations and compilation state are released before the next run.

```powershell
$py = 'C:\Python314\python.exe'
$env:TORCHINDUCTOR_COMPILE_THREADS = '1'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $PWD 'output\architecture-baselines\inductor-cache-single'

foreach ($family in 'gaterv2', 'moesr', 'gfisrv2') {
    & $py .\scripts\benchmark_architectures.py $family fp32
    & $py .\scripts\benchmark_architectures.py $family inductor
}
& $py .\scripts\benchmark_architectures.py gaterv2 rtx
& $py .\scripts\benchmark_architectures.py moesr supplied
& $py .\scripts\benchmark_architectures.py moesr rtx
& $py .\scripts\benchmark_architectures.py gfisrv2 eager-fp16
```

`--input` selects another directory of source images. `--batch` selects
another throughput batch. Re-run FP32 for that configuration before comparing
candidates. Each variant writes its report, renders and diagnostic files under
`output/architecture-baselines/<family>/batch-<N>/<backend>/`. FP32 also saves
its raw float page and stress outputs. The report records input and checkpoint
SHA-256 digests, setup and environment, and any errors.

Each timed interval is one synchronized device-resident forward call. Transfers,
image assembly, encoding, compilation, and engine warm-up are outside the
interval. Batched throughput uses the original input megapixels divided by the
sum of those calls. A separate pass times every tile individually and records
its input coordinates and elapsed milliseconds. RTX uses separate fixed ONNX
graphs and sessions for the throughput and latency shapes. It disables ORT
telemetry and fallback and checks node placement in a profiling session before
timing. The RTX runs use a 512 MiB workspace limit.

Page PSNR uses raw float outputs after assembly, with peak 1. SSIMULACRA2 uses
the corresponding full RGB PNGs and is recorded separately. The stress check
covers flat black, gray, and white, checkerboard, border impulses, seeded
noise, page crops, rectangles, and distinct batched inputs. For tiny inputs,
the reference records which shapes its reflection padding accepts.

## Measurements

Measured on 2026-11-22 with an NVIDIA GeForce RTX 3050 Laptop GPU (4 GiB),
driver 610.62, Python 3.14.5, PyTorch 2.15.0+cu132, cuDNN 9.26, Resselt 1.4.1,
Triton 3.9.0.post30, ONNX 1.23.2, ONNX Runtime 1.31.1, and RTX plugin 0.5.0.

| Family | Backend | Batched seconds | Input MPx/s | Mean batch-1 tile ms | Raw PSNR dB | SSIMULACRA2 mean/min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GateRV2 | Eager FP32 | 8.929 | 1.034 | 43.658 | — | — |
| GateRV2 | TensorRT-RTX FP16 | 2.338 | 3.949 | 4.880 | 69.730 | 90.361 / 90.058 |
| MoESR | Eager FP32 | 76.963 | 0.047 | 565.989 | — | — |
| MoESR | Inductor FP16 | 18.143 | 0.201 | 258.581 | 67.263 | 87.287 / 87.058 |
| MoESR | TensorRT-RTX FP16 | 11.899 | 0.306 | 125.816 | 60.188 | 86.210 / 85.517 |
| GFISRv2 | Eager FP32 | 19.337 | 0.188 | 262.228 | — | — |
| GFISRv2 | Eager FP16 | 13.346 | 0.273 | 177.040 | 60.166 | 86.385 / 85.496 |
| GFISRv2 | Inductor FP16 | 10.306 | 0.354 | 141.329 | 63.014 | 86.501 / 85.621 |

All three references returned finite outputs on the benchmark tiles, both
rectangles, and distinct batches. GateRV2 rejected all four tiny probes
because the network pads to a multiple of 16 using reflection. MoESR
rejected 1×1 and accepted 2×2, 3×5 and 8×9. GFISRv2 accepted all four.

The first Inductor attempt used four compile threads and hit
`AssertionError: pass_fds not supported on Windows`. The serial retry used
one compile thread and a fresh cache. The first batch-1 and throughput
calls took 121.4 and 123.9 seconds for GateRV2, 78.2 and 81.0 seconds for
MoESR, and 79.8 and 79.7 seconds for GFISRv2. These calls include
compilation and warm-up and are excluded from inference measurements.

### FP32 per-page timings

| Family | Source page | Batched seconds | Input MPx/s | Mean batch-1 tile ms |
| --- | --- | ---: | ---: | ---: |
| GateRV2 | watanare-page-085.jpg | 4.440 | 1.040 | 41.847 |
| GateRV2 | watanare-page-086.jpg | 4.489 | 1.028 | 45.468 |
| MoESR | lb-page-003.png | 25.557 | 0.048 | 566.360 |
| MoESR | lb-page-004.png | 25.686 | 0.047 | 564.877 |
| MoESR | lb-page-005.png | 25.720 | 0.047 | 566.731 |
| GFISRv2 | lb-page-003.png | 6.401 | 0.190 | 262.813 |
| GFISRv2 | lb-page-004.png | 6.432 | 0.189 | 261.302 |
| GFISRv2 | lb-page-005.png | 6.505 | 0.187 | 262.569 |

The distinct-batch check compared every output to its individual call. The
largest FP32 differences were 1.01e-6 for GateRV2 and 1.37e-6 for MoESR.
GFISRv2 matched bit for bit. Peak PyTorch allocations were 600, 657, and
390 MiB respectively.

### FP16 per-page results

| Family | Backend | Source page | Batched seconds | Input MPx/s | Mean batch-1 tile ms | Raw PSNR dB | SSIMULACRA2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| GateRV2 | TensorRT-RTX | watanare-page-085.jpg | 1.173 | 3.935 | 4.801 | 69.262 | 90.665 |
| GateRV2 | TensorRT-RTX | watanare-page-086.jpg | 1.165 | 3.964 | 4.959 | 70.254 | 90.058 |
| MoESR | Inductor | lb-page-003.png | 6.053 | 0.201 | 248.920 | 68.596 | 87.615 |
| MoESR | Inductor | lb-page-004.png | 6.023 | 0.202 | 251.214 | 66.297 | 87.058 |
| MoESR | Inductor | lb-page-005.png | 6.067 | 0.200 | 275.607 | 67.198 | 87.187 |
| MoESR | TensorRT-RTX | lb-page-003.png | 3.941 | 0.308 | 124.853 | 60.852 | 86.731 |
| MoESR | TensorRT-RTX | lb-page-004.png | 4.050 | 0.300 | 126.210 | 59.525 | 86.380 |
| MoESR | TensorRT-RTX | lb-page-005.png | 3.909 | 0.311 | 126.385 | 60.290 | 85.517 |
| GFISRv2 | Eager | lb-page-003.png | 4.404 | 0.276 | 178.137 | 64.067 | 87.086 |
| GFISRv2 | Eager | lb-page-004.png | 4.396 | 0.276 | 176.251 | 58.069 | 86.574 |
| GFISRv2 | Eager | lb-page-005.png | 4.545 | 0.267 | 176.733 | 60.288 | 85.496 |
| GFISRv2 | Inductor | lb-page-003.png | 3.353 | 0.362 | 134.005 | 66.974 | 87.207 |
| GFISRv2 | Inductor | lb-page-004.png | 3.400 | 0.357 | 141.813 | 61.833 | 86.674 |
| GFISRv2 | Inductor | lb-page-005.png | 3.553 | 0.342 | 148.168 | 61.924 | 85.621 |

The GateRV2 batch-1 and batch-16 exports each needed 12 negative transpose
axes normalized to their nonnegative equivalents. Both corrected graphs
passed strict ONNX checking and ran as a single TensorRT-RTX partition per
shape. The checkerboard was the lowest scoring stress case at 35.072 dB,
with a maximum absolute error of 0.09061. The page maximum was 0.00502.

MoESR's matched batch-1 and batch-2 RTX graphs also ran as a single
TensorRT-RTX partition per shape. The largest absolute page error was 0.04539 on
`lb-page-004.png`. The center crop of `lb-page-005.png` was the lowest
scoring stress tile at 51.277 dB.

MoESR Inductor finished all four compiled shapes with finite outputs. Its
lowest stress score was 60.535 dB on the same crop. Its largest absolute
page error was 0.03627 on `lb-page-003.png`.

### GateRV2 Inductor numerical failure

Default Inductor FP16 compiled batch 1 and 16 at 128×128 and batch 1 at both
rectangular sizes. Every value was non-finite on all stress outputs and both
pages. The rerun recorded 13,851,000 non-finite values per page. Its timing
is diagnostic and both fidelity metrics are unavailable.

| Source page | Batch-16 seconds | Input MPx/s | Mean batch-1 tile ms |
| --- | ---: | ---: | ---: |
| watanare-page-085.jpg | 3.936 | 1.173 | 22.543 |
| watanare-page-086.jpg | 3.941 | 1.172 | 21.899 |
| Both pages | 7.877 | 1.172 | 22.221 |

Tracing a black tile found the first non-finite tensor in the fused
attention update of the twelfth latent block. The compiled contraction
returned −64 at all 64 tokens, making the denominator zero. The same
eager CUDA FP16 calculation gave 0.0625 at all 64 tokens and kept all
49,152 output values finite. Eager FP16 scored 33.627 dB on the
checkerboard, close to the RTX score of 35.072 dB. The trace is saved as
`gaterv2-inductor-nonfinite.json`.

### Supplied MoESR ONNX validation

The supplied opset-20 graph retained its dynamic dimensions, FP32 input and
FP16 output. ORT profiled convolutions on its CUDA provider and some
shape-related ops on its CPU provider. All 13 validation cases returned finite
outputs. The 65×79 and 128×160 inputs returned 260×316 and 512×640 at
62.586 and 62.893 dB respectively, confirming the height/width order. The
distinct batch scored 63.267 dB. The lowest stress result was 56.508 dB on
the center crop of `lb-page-005.png`, with a maximum absolute error of
0.04337. Every one of the 733 initializers that shared a checkpoint tensor
name matched that tensor after casting to the initializer's dtype. The
validation report is under `moesr/batch-2/supplied/`.

### GFISRv2 flat-tile fidelity

Eager FP16 scored 5.556 dB on flat 0.5 gray, 5.751 dB on flat 128/255
gray, and 28.080 dB on the checkerboard. Inductor FP16 scored 5.779,
5.791, and 28.201 dB on the same inputs. The raw maximum error on 0.5 gray
was 3.439 in eager and 3.568 in Inductor. The corresponding pre-clipping
page maxima were 0.07545 and 0.05291.
The eager layer trace shows the 0.5-gray RMSE rising from 0.000292 at
block 1 to 0.03924 at block 2 and 0.829 at block 24. Inside block 2, the
frequency-space RMSNorm amplified the 0.5-gray RMSE from 0.000908 to
0.03927 and the checkerboard RMSE from 0.000672 to 0.10070. The FFT and
inverse FFT already run in FP32 in the eager FP16 forward. The traces
are saved as `gfisrv2-flat-trace.json` and `stress-layer-traces.json`.
