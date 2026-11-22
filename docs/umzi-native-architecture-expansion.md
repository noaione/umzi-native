# GateRV2, MoESR, and GFISRv2 native support plan

Reproduce the three checkpoint networks in FP32 and establish numerical
parity before tuning any native kernels. Each family should eventually export
to a single `.ummodel` and use the same Python session API as GateRV3. The
first pass covers the three configurations listed below.

## Checkpoint inventory

The existing CPU checkpoint loader and Resselt 1.4.1 detected and loaded all
three checkpoints without missing or unexpected keys. The configurations below
come from those loaded modules and checkpoint tensors. This inspection did
not execute a model forward pass. [Resselt](https://github.com/rewaifu/resselt)

| Checkpoint in `models/` | Detected model | Configuration to reproduce |
| --- | --- | --- |
| `1x_umzi_dehalftone_gater2_v1_smooth.safetensors` | GateRV2, RGB 1× | 32 base channels, encoder blocks `(2, 2, 4, 8)`, decoder blocks `(2, 2, 2, 2)`, 12 latent attention blocks, and 16-pixel input alignment. |
| `4x_dwtp_ds_moesr_v2.pth` | MoESR, RGB 4× | 64 base channels, nine groups with four full-resolution gated blocks and a multiscale branch per group, and direct pixel shuffle. |
| `4x_wtp_MangaScale_GfisrV2.pth` | GFISRv2, RGB 4× | 48 base channels, 24 Fourier-containing gated blocks, a convolutional tail with a global residual, and direct pixel shuffle. |

## Replicate and check each network

1. **GateRV2 first.** Trace the padded encoder, skip tensors, latent blocks,
   decoder, output convolution, RGB residual, and crop. Match the latent
   attention's Q/K L2 normalization, contractions, and denominator. GateRV2
   uses a different attention calculation from the existing GateRV3 kernel.
   Inspect intermediate tensors on aligned and odd rectangular inputs. Its
   1× path is the first candidate for proving that a second real kernel family
   can use the existing exporter and runtime interfaces.
2. **MoESR second.** Reproduce the full-resolution gated blocks, each
   multiscale down/up branch and its residual, the outer feature residual, and
   the 4× output. Match normalization and Mish, and test pixel shuffle and
   unshuffle order with channel-distinct inputs. Reflect-pad the right or
   bottom by one pixel for each odd axis and crop to `4H × 4W`.
3. **GFISRv2 third.** Reproduce one Fourier unit before assembling the 24-block
   body. Check forward and inverse transforms with orthonormal scaling, exact
   packed real/imaginary channel order, the learned frequency-space layers,
   and the pre/post RMSNorm formula and epsilon placement. Compare traces at
   the first Fourier unit, several points through the body, the residual
   tail, and the final 4× shuffle. The eager FP16 model calls FFTs in FP32
   and casts around the learned operations. Preserve those boundaries for
   the FP16 candidate.

Use the unmodified eager model loaded from each checkpoint as the reference.
Start with a straightforward FP32 execution path and layer-by-layer traces
for the replica. An initial native implementation can use cuDNN and cuBLASLt
for matching primitives, cuFFT for GFISRv2, and small CUDA kernels for the
exact layout, pointwise, and reduction operations. Keep operator-level traces
available until the full model passes. Set absolute and relative acceptance
tolerances from the primitive-level checks before running the held-out pages.
Require finite outputs and investigate the first diverging stage on a failure.

For GFISRv2, use `../umzi-bench/src/gfisrv2.py` and the sibling repository's
`docs/handoff-gfisrv2-trt-performance.md` to confirm semantics and pick stress
inputs. The handoff records an unusual inverse packing and cases where a good
FP16 page score hid large flat-gray errors. Start from the plain graph and
save the optimized backend, custom plugins, and engine partitioning for later
experiments. The handoff's checkpoint SHA-256 matches the local file.

## Reference and baseline matrix

| Backend | GateRV2 | MoESR | GFISRv2 |
| --- | --- | --- | --- |
| Eager Torch CUDA FP32 | Reference | Reference | Reference |
| TorchInductor FP16 | Measure | Measure | Measure |
| ONNX Runtime TensorRT-RTX FP16 | Export and try | Validate supplied ONNX, then measure a matched export | Skip |

- Use `C:\Python314\python.exe`. Keep both the eager FP32 reference and a
  separate FP16 copy for TorchInductor. Record compilation, graph breaks,
  precision boundaries, and any failed shapes. An eager-only fallback does
  not count as an Inductor result.
- The supplied `models/4x_dwtp_ds_moesr_v2_op20_fp16.onnx` uses opset 20 and
  declares dynamic spatial dimensions and batch, a **float32 input**, and a
  **float16 output**. Its metadata has no source checkpoint digest. Validate
  the supplied graph against the checkpoint on identical tiles, including a
  rectangle that checks axis order. For the matched FP16 host I/O comparison,
  export directly from the same checkpoint with FP16 input and output.
  Report the supplied graph separately if measured with its original I/O.
- Reuse `scripts/rtx_session.py` for GateRV2 and MoESR, extending its shape
  checks for 4× output and adapting the fixed profiles to each export's real
  input name. Disable ORT telemetry and fallback, register the TensorRT-RTX
  device, and verify exclusive RTX placement with a separate profiling
  session. Record unsupported exports or operator coverage gaps as blocked.
  GFISRv2 remains outside the RTX matrix because its Fourier path lacked
  full GPU coverage on the tested ORT RTX route. [TensorRT-RTX EP](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html)

## Validation and measurements

### Page benchmark settings

| Model | Sample directory | Input tile | Batch size | Input overlap |
| --- | --- | --- | ---: | ---: |
| Dehalftone (GateRV2) | `samples/halftone/` | 128×128 | 16 | 0 px |
| MangaScale (GFISRv2) | `samples/mscale/` | 256×256 | 2 | 32 px, matching the overlap from the `umzi-bench` handoff |
| DWTP DS (MoESR) | `samples/mscale/` | 256×256 | 2 | 0 px |

For MangaScale, 32 input pixels of overlap gives a 224-pixel input stride
and 128 output pixels of overlap at 4×. Use the same tile grid, edge padding,
feather blending, and crop for every backend and the FP32 reference. The
prior `umzi-bench` handoff used 128×128 tiles with 32 pixels of overlap. Use a
fresh FP32 reference at 256×256 because the Fourier blocks see a different
tile context.

### Fidelity and timing

- Start network validation with batch 1 at the model's benchmark tile size,
  then add 65×79, 128×160, and a multi-image batch with distinct inputs.
  Cover flat black, mid-gray, and white,
  checkerboards, impulses near borders, seeded noise, and the manga sample
  pages. Compare outputs at the same tile size, crop, and scale. For a batch
  check, compare each result with the matching individual result. Check tiny
  inputs against each reference's actual padding constraints.
- Record raw-float PSNR with peak 1, maximum absolute error, and the worst
  stress tile against the same checkpoint's eager CUDA FP32 output before
  clipping or encoding. Compute SSIMULACRA2 separately from rendered PNGs
  with `C:\Tools\imaging\ssimulacra2.exe`. Keep per-image and summary values.
  Investigate every failing stress tile, even when page averages are high.
- Benchmark each model with the settings above and the same input order,
  padding, and submitted batch sizes across backends. Measure synchronized inference after
  session creation, GPU warm-up, compilation, and plan or engine setup.
  Report elapsed time and throughput per page and overall. Use original
  input megapixels for throughput so 4× output size does not inflate it.
  Record VRAM and label out-of-memory cases. If a planned batch cannot run,
  also measure a smaller batch with its actual size in the report.
- Measure actual batch-1 latency in a separate pass with the same tile size
  and overlap, using a fixed batch-1 ONNX graph and RTX session where
  applicable. Save each tile's image, index, coordinates, and synchronized
  call time. Average those calls independently of the batched throughput
  calculation.
- Save checkpoint and graph hashes, reference configuration, runtime versions,
  GPU, shapes, numerical failures, and measurements with each report. Track
  completion by checkpoint and backend in this document before tuning starts.

## Native integration after parity

1. Add one packer and one native `KernelSession` factory per family. Use the
   existing `UMZINPKG` v1 envelope and independent family payload schemas.
   Keep GateRV3 on family ID `1`, use the existing reservation of `2` for
   GFISRv2, and use `3` and `4` for GateRV2 and MoESR. Put family-specific
   payload validation and diagnostics alongside the family code. Reuse the
   existing checksum, staging, and atomic publication in the exporter.
2. Extend `hatch_build.py` to enumerate the family sources and embed CUDA
   images per family. Preserve the CUDA 13 and PTX fallback policy. Plan the
   GFISRv2 cuFFT dependency so existing GateRV3 consumers can still open a
   session without loading it. Verify this during packaging on Windows.
3. Keep `InferenceSession`, `SessionOptions`, `peek_model()`, and the exporter
   CLI as shared entry points. Have each family describe input and output
   shapes independently so 4× runners allocate distinct host and device
   buffers. Use a private FP32 validation path until each family passes.
   Add public precision modes and tuning options only as they pass parity.
4. Validate deterministic exports, wrong-family or wrong-schema packages,
   dynamic shapes, and cleanup. Re-run both GateRV3 checkpoints through the
   existing export and session checks before enabling each new family in a
   wheel. After these checks, profile the reference runner to choose which
   operations to optimize first.

## Progress

The [CUDA baseline runbook and results](umzi-native-architecture-baselines.md)
record the reference and backend runs.
The [GateRV2 handoff](handoff-gaterv2.md) records the validated behavior and
the remaining package and session work.

| Family | CPU checkpoint load | CUDA baselines | Native FP32 parity, export, and session |
| --- | --- | --- | --- |
| GateRV2 | Verified | FP32, RTX FP16, and Inductor FP16 with FP32 attention measured | Package (family 3, schema 1) and C++ session in mixed and FP16 validated on Windows and Linux; first tuning pass done |
| MoESR | Verified, supplied ONNX validated | FP32, Inductor FP16, and RTX FP16 measured | [Private FP32 parity](#moesr-private-fp32-validation), family-4 package and C++ session pending |
| GFISRv2 | Verified, prior handoff reviewed | FP32, eager FP16, and Inductor FP16 measured, flat-tile fidelity failure traced, RTX skipped | Pending |

### GateRV2 private FP32 validation

The private runner loads the same checkpoint as eager FP32 and traces
padding, encoder skips and downsampling, all 12 latent attention updates,
decoder merges, the output convolution, and the cropped RGB output.
GateRV2 shares GateRV3's CUDA RGB padding and crop code. Its new kernels
normalize Q and K across channels and apply the attention denominator. The private runner
uses Torch CUDA primitives for the convolutions, contractions, and
remaining operations. The package and session work follows the private
parity checks.

The CUDA RGB padding and crop matched exactly on four shapes, including
65×79 and 9×9. Standalone attention matched within 8.95e-8 max absolute
error on four batch/token configurations. The first observed rounding
difference was at the first latent attention update on all five traced
inputs. The worst stress output was the checkerboard at 117.920 dB,
6.32e-6 maximum absolute error, and 98.988 SSIMULACRA2. The full output
checks use `atol=1e-5` and `rtol=5e-5`.

Both Dehalftone pages passed that tolerance. SSIMULACRA2 compares the
rendered candidate PNGs with their matching Torch FP32 PNGs using
`C:\Tools\imaging\ssimulacra2.exe`. The report records the paired
Torch reference timing and the CUDA runner's own timings.

| Source page | Batch-16 inference | Input MPx/s | Mean batch-1 tile ms | Raw PSNR dB | SSIMULACRA2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| watanare-page-085.jpg | 4.112 s | 1.123 | 38.735 | 148.706 | 93.100 |
| watanare-page-086.jpg | 4.548 s | 1.015 | 42.256 | 148.964 | 93.167 |
| Both pages | 8.661 s | 1.066 | 40.495 | 148.833 | 93.133 mean / 93.100 min |

The maximum raw page error was 8.05e-7. The rendered PNGs differed in 60
and 52 RGB samples respectively, each by one 8-bit code value.

```powershell
& 'C:\Python314\python.exe' scripts/validate_gaterv2.py --pages --nvrtc-root output/nvrtc-13.0/nvidia/cu13
```

The private validator uses the local CUDA Python bindings and CUDA 13.0
NVRTC toolchain. The report is in `output/gaterv2-native-validation/`.

Both GateRV3 checkpoints exported byte for byte against the existing
packages after extracting the shared CUDA resize code. The rebuilt
Windows DLL passed the aligned and odd-rectangle session checks in
mixed precision and FP16. Both checkpoints matched the CLI bit for
bit and reused the disk cache when reopened.

### GateRV2 native package and session

GateRV2 packages use family ID `3` and payload schema `1`. Section 1 holds
sixteen little-endian 32-bit values: input and output channels, scale,
padding alignment, base channels, level count, latent block count, the
attention epsilon as FP32, four encoder depths, and four decoder depths.
Unused levels are zero. Section 2 is a table of 644 entries, each a 64-bit
byte offset and FP32 element count. Section 3 holds the FP32 parameters at
256-byte offsets in execution order. The Q, K and V projections of each
latent block are concatenated into one projection. Section 4 holds the
checkpoint SHA-256. The [packer](../src/umzi_native/exporter/_gaterv2.py),
the [package reader](../src/umzi_native/_model.py), and the
[session](../native/kernels/gaterv2/runner.cpp) derive the same parameter
list from the architecture and reject any other count, order, or layout.

The session stores NHWC FP16 feature maps. Mixed precision keeps FP32 RGB
input and output and uses FP32 accumulation in cuDNN and the non-attention
cuBLASLt projections. FP16 uses FP16 RGB input and output and FP16
accumulation in those paths. In both modes the Q/K/V projections output
FP32 and the linear attention normalization, contractions and denominator
run in FP32. This protects the flat-tile denominator. The RGB residual uses
the original input at its I/O precision. The [private FP32 runner](../scripts/reference/gaterv2_native.py)
remains available as the network parity control.

The native session supports CUDA Graph and direct execution through host and
device buffers. It accepts batches and rectangular inputs when the required
reflection padding is smaller than the corresponding input axis. The
session uses `SessionOptions.kernel_options` for `tune_plans` and
`workspace_mib`, and `cache_dir` to reuse selected cuBLASLt and cuDNN
algorithms. [GateRV2 session options](quickstart.md#gaterv2-kernel-options)

| Step | Change | Observed effect on RTX 3050 Laptop |
| --- | --- | --- |
| 1 | FP32 C++ port using NCHW and cuDNN FMA | 146.8 dB on the pages against eager FP32 |
| 2 | Mixed and FP16 paths with FP16 features and FP32 attention | Batch-16 totals 3.633/3.525 s and mean batch-1 calls 7.716/7.849 ms for mixed/FP16 |
| 3 | Fuse biases into consumers, parallelize RMSNorm across channels, and parallelize K·V across warps | Mixed batch-16 forward 114.9 → 89.4 ms |
| 4 | NHWC feature maps, biased cuBLASLt GEMMs, timed cuBLASLt and cuDNN plans, and fused custom kernels | About 430 → 290 kernel launches and 89 → 59 ms per mixed batch-16 forward |
| 5 | Reuse register weights and columns in the local gate, parallelize pooling by output, and split attention preparation by value slice | About 52 ms per mixed batch-16 forward and 4.3 ms per batch-1 call |
| 6 | Select one inception branch per eight-channel vector where possible | About 49 ms per mixed batch-16 forward |
| 7 | Pack only the weight copies each operator uses, release unused tuning workspace, size three work buffers by role, and reuse the consumed skip as decoder state | Mixed VRAM fell from 524 to 232 MiB at batch 16 and from 278 to 92 MiB at batch 1 |
| 8 | Specialize the 32-channel 128×128 inception gate with four-channel depthwise branch loads | About 56% less GPU time in the specialized gate in both precisions. [Profile and paired validation](handoff-gaterv2.md#32-channel-inception-gate) |
| 9 | Share activation storage, size plan workspace on demand, and optimize local gate, RMSNorm, and wider inception dispatch | Batch-16 activation buffers fell from 130.8 to 88.5 MiB. [Paired timing and VRAM results](handoff-gaterv2.md#gate-and-memory-optimization) |

Shared-memory tiling for the local and inception gates and a cheaper Mish
formula were evaluated in the earlier experiments. The timing details and
the retained implementations are recorded in the [GateRV2 handoff](handoff-gaterv2.md#improvements-made-in-order).

### GateRV2 memory and setup

These are the handoff's device-wide free-memory deltas after preparing a
single session, excluding CUDA context and caller I/O. The selected cuDNN
and cuBLASLt algorithms determine retained workspace.

| Backend and options | Batch-16 MiB | Batch-1 MiB |
| --- | ---: | ---: |
| Native mixed, before compacting buffers and weights | 524 | 278 |
| Native mixed, after compacting buffers and weights | 232 | 92 |
| Native FP16, after compacting buffers and weights | 237 | 96 |
| Native mixed, with `workspace_mib=0` | 230 | 86 |
| TensorRT-RTX FP16 | 242 | 108 |
| Eager CUDA FP32 | 756 | 188 |

The packed GPU weights total about 60.2 MiB in either mode. Setting
`tune_plans=0` uses the first heuristic plan. In the recorded mixed
batch-16 probe it reduced prepare from 818 to 406 ms with about 3% slower
inference. The zero-workspace mixed setting reduced batch-16 inference
speed by about half in that probe. [Setup and VRAM measurements](handoff-gaterv2.md#latest-measurements)

### GateRV2 validation

The Windows session build passed deterministic export, eleven rejection
checks, fourteen stress and shape cases in both precisions, and host/device
output equality. The current graph and direct runs were bitwise equal on
matching plans. Separately prepared batch-1 runs were within the validator's
3 dB consistency margin against eager FP32. The cleanup run returned free
GPU memory to its starting value after switching shapes. The Windows build
also passed the two GateRV3 package and session regressions, including
deterministic re-export, CLI equality, and cache reuse. [Session validator](../scripts/validate_gaterv2_session.py)

The full WSL build passed with cubins for sm_86, sm_89 and sm_120 and
compute_86 PTX. On a shared 2×3×65×79 input, Windows and Linux output
differed by at most `5.88e-4` in mixed precision and `7.32e-4` in FP16.
Their PSNR against the same eager FP32 output was 75.250 and 75.246 dB
in mixed and 73.105 and 73.076 dB in FP16, respectively.

### GateRV2 plan cache

With `SessionOptions(cache_dir=...)`, GateRV2 saves the selected cuBLASLt
and cuDNN algorithms in `gaterv2_<key>.umplan`. The key covers the model,
CUDA kernel image, precision, padded batch and shape, tuning and workspace
options, GPU, driver, cuDNN, and cuBLASLt versions. The session checks
cached algorithms against their current descriptors and workspace before
reuse. Invalid plans are selected again. Cache read and write failures
remain visible through `cache_status` and `cache_message`.
On a validated cache hit, prepare allocates only the largest workspace
requested by a selected plan. [Workspace measurements](handoff-gaterv2.md#gate-and-memory-optimization)
[Shared cache implementation](../native/common/cuda_plan_cache.h)

Windows measurements cover `session.prepare()` for 128×128 tiles,
including shape allocation, plan selection or reuse, warm-up, and graph
capture. Model opening and weight upload are outside the measured interval.
Each reopened session used zero timed plans and returned bitwise-identical
output to its cold session.

| Precision | Batch | Cache miss prepare ms | Cache hit prepare ms |
| --- | ---: | ---: | ---: |
| Mixed | 1 | 82.3 | 39.5 |
| Mixed | 16 | 465.1 | 130.6 |
| FP16 | 1 | 274.6 | 24.5 |
| FP16 | 16 | 3431.1 | 109.7 |

The validator checked option and shape key isolation, zero-workspace reuse,
checksum-corruption rebuild, and inference when the cache path is
unavailable. The 9×9 FP16 output is sensitive to independently timed
plans on its one latent token. Graph and direct dispatch now share the
same plan file in the stress test, and their outputs matched bitwise for
all fourteen cases. The 9×9 shared-plan check scored 69.643 dB against
eager FP32. [Windows validation report](../output/gaterv2-session/cache-final-verified/report.json)

The rebuilt WSL library also reported a miss followed by a hit for both
precisions on a 2×3×65×79 input. Reopened outputs were bitwise identical
and no plans were timed on a hit. [WSL cache validation](../output/gaterv2-session/cache-final-linux-session.json)

Run the validator with a rebuilt library to reproduce the cache checks.

```powershell
& 'C:\Python314\python.exe' scripts/validate_gaterv2_session.py `
  --library output/gaterv2-session/cache-final-win/umzi_native.dll `
  --output output/gaterv2-session/cache-final-verified --pages
```

### GateRV2 performance against TensorRT-RTX

Three rounds rotated the order of the three backends per page. Each
used a separate warm session for each batch size. Per-tile samples were
timed in individual synchronized batch-1 calls. The table gives medians
across rounds and fidelity against the matching eager CUDA FP32 output.

| Backend | Batch-16 seconds | Input MPx/s | Mean batch-1 tile ms | Raw PSNR dB | SSIMULACRA2 mean / min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native mixed | 2.122 | 4.352 | 4.531 | 77.922 | 90.569 / 90.374 |
| Native FP16 | 1.992 | 4.636 | 4.334 | 73.846 | 90.436 / 90.131 |
| TensorRT-RTX FP16 | 2.323 | 3.975 | 4.931 | 69.730 | 90.361 / 90.058 |

| Backend | 085 batch-16 s | 085 MPx/s | 085 batch-1 ms | 085 PSNR dB / SSIMULACRA2 | 086 batch-16 s | 086 MPx/s | 086 batch-1 ms | 086 PSNR dB / SSIMULACRA2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native mixed | 1.046 | 4.414 | 4.580 | 77.571 / 90.764 | 1.065 | 4.337 | 4.532 | 78.303 / 90.374 |
| Native FP16 | 0.989 | 4.668 | 4.301 | 73.546 / 90.741 | 0.990 | 4.666 | 4.364 | 74.168 / 90.131 |
| TensorRT-RTX FP16 | 1.164 | 3.967 | 4.766 | 69.262 / 90.665 | 1.161 | 3.976 | 4.772 | 70.254 / 90.058 |

The packaged Windows validator also scored the checkerboard and noise
renders against FP32. Mixed scored 95.702 and 97.690 SSIMULACRA2,
respectively, and FP16 scored 94.900 and 97.524. The accompanying raw
PSNR scores were 35.411 and 75.034 dB in mixed and 34.498 and 73.025 dB
in FP16.

Run the following after building the DLL and running the GateRV2 validator
to export the package. The original eager FP32 and RTX baseline reports
provide the reference images and the two matched RTX graphs. The benchmark
checks their hashes and profiles each RTX graph again.

```powershell
& 'C:\Python314\python.exe' scripts/benchmark_gaterv2_backends.py `
  --library output/gaterv2-session/step7-verified-win/umzi_native.dll `
  --model output/gaterv2-session/step7-verified/gaterv2.ummodel `
  --rtx-report output/gaterv2-session/rtx-rerun/gaterv2/batch-16/rtx/report.json `
  --output output/gaterv2-session/step7-paired
```

Related design documents are the [runtime and package plan](umzi-native-runtime-plan.md)
and the [exporter plan](umzi-model-exporter-plan.md).

### MoESR private FP32 validation

The current MoESR checkpoint uses nine groups. Each group has four
full-resolution gated blocks and a branch with three gated blocks at half
the spatial resolution. The full-resolution projection splits into
`[170 gate, 106 identity, 64 inception]`. The reduced-resolution split is
`[96 gate, 32 identity, 64 inception]`. The inception channels split into
`[40 direct, 8 square, 8 row, 8 column]` in both cases. Each gated block
uses channel LayerNorm with centered population variance and epsilon
inside the square root. The multiscale branch adds to the output of its
four full-resolution blocks. The outer feature residual is applied before
the final convolution and 4× RGB shuffle.

The [private runner](../scripts/reference/moesr_native.py) uses CUDA
kernels for padding, LayerNorm, the Mish gate and residual, 2× unshuffle
and shuffle, and 4× shuffle with crop. Torch CUDA still supplies the
checkpoint convolutions. NVRTC compiles the
[private kernels](../native/kernels/moesr/kernels.cu) with fast math
disabled. The untouched Resselt model remains the FP32 reference.

Channel-distinct layout primitives matched exactly. The
[validator](../scripts/validate_moesr.py) compared 149 stages each for
`1×3×32×48` and `2×3×33×47`. It also checked 16 finite stress inputs,
including individual-versus-batched tiles. The `1×1`, `1×2`, and `2×1`
shapes hit the same reflection-padding limit as eager FP32. Valid tiny
inputs included `2×2`, `3×5`, and `8×9`.

| Stress input | Raw PSNR dB | SSIMULACRA2 |
| --- | ---: | ---: |
| Gray 0.5 | 148.920 | 100.000 |
| Gray 128/255 | 148.929 | 96.785 |
| Checkerboard | 143.547 | 95.348 |
| Noise | 138.640 | 98.230 |

The lowest stress PSNR was 137.128 dB on the center tile from
`lb-page-005.png`. The largest stress error was `3.815e-6` on the center
tile from `lb-page-003.png`. The checkpoint SHA-256 is
`d43df68fbd6543c47e41b6dc5ca95da82794521b26108269f75ce533d100ee65`.

The RTX 3050 Laptop GPU run used 256×256 tiles, batch 2, and zero overlap
on the three `samples/mscale/` pages. Each page used 12 synchronized
batch-2 calls. Batch-1 latency came from 24 independent, synchronized
tile calls per page.

| Page | Batch-2 time (s) | Input MPx/s | Mean batch-1 tile (ms) | Raw PSNR dB | SSIMULACRA2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lb-page-003.png` | 23.704 | 0.0513 | 460.898 | 141.971 | 90.050 |
| `lb-page-004.png` | 22.264 | 0.0546 | 444.685 | 139.927 | 89.593 |
| `lb-page-005.png` | 22.309 | 0.0545 | 444.204 | 141.347 | 89.563 |
| All three | 68.277 | 0.0534 | 449.929 | 140.996 | 89.735 mean / 89.563 min |

The maximum raw page error was `4.888e-6`. The compared PNGs differed
in 304, 471, and 205 of 58,320,000 channel values per page, each by one
8-bit code value. The report stores all 72 individual tile latencies with
page name, tile index, and coordinates.
[Local validation report](../output/moesr-native-validation/report.json)

```powershell
$py = 'C:\Python314\python.exe'
$env:PYTHONPATH = 'src;scripts'
& $py scripts/validate_moesr.py --nvrtc-root output/nvrtc-13.0/nvidia/cu13 --pages
```

Page validation uses the saved eager FP32 arrays and PNGs after checking
the checkpoint digest, sample digests, tile size, submitted batch size,
overlap, and scale in the baseline report. The next integration step is
the family-4 packer and C++ session with independent input and 4× output
buffer sizes. Rerun these checks against that session in FP32 before
changing accumulation precision.
