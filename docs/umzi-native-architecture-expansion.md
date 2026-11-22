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
   unshuffle order with channel-distinct inputs. Match the 2-pixel reflection
   padding and crop to `4H × 4W` on odd input sizes.
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
| GateRV2 | Verified | FP32, RTX FP16, and Inductor FP16 with FP32 attention measured | FP32 package (family 3, schema 1) and C++ session validated; FP16 and tuning pending |
| MoESR | Verified, supplied ONNX validated | FP32, Inductor FP16, and RTX FP16 measured | Pending |
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

### GateRV2 FP32 package and session

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

The session keeps NCHW FP32 features. It runs every convolution through
cuDNN with `CUDNN_FMA_MATH`, so TF32 is excluded, then adds the bias in a
separate kernel. The [CUDA kernels](../native/kernels/gaterv2/kernels.cu)
compile without fast math and implement RMSNorm, the grouped local
convolution and gate, pooling and channel attention, the inception
depthwise branches, Mish gating, pixel unshuffle and shuffle, and the
linear attention. The attention adds epsilon to the reduced K vector
before the Q contraction and then adds the token count. The session
supports any batch and any height and width whose reflection padding is
valid, CUDA Graph or direct execution, and host or device buffers. It
accepts only the default precision and rejects kernel options. FP16
modes and plan caching are not implemented yet.

`scripts/validate_gaterv2_session.py` exported the checkpoint twice with
identical bytes. It rejected the wrong schema, the GateRV3 family ID, an
8-pixel alignment, a wrong parameter count, a truncated parameter table,
trailing weights, FP16 precision, a kernel option, an 8×9 input, and a
one-channel input. CUDA Graph and direct execution produced identical
outputs on all 14 stress cases. The cases included 65×79, 128×160, 9×9,
three 31×47 tiles, and a distinct batch of 16. Each batch item was within
1.07e-6 of its batch-1 run. Device enqueue matched eager FP32 on a
128×128 batch of two and a 65×79 tile. Three sessions that each switched
through four shapes returned the device's free memory to its starting value.

Thirteen stress cases met `atol=1e-5` and `rtol=5e-5` against eager FP32,
with at most 1.19e-6 error. The checkerboard reached 1.88e-5 at 108.574 dB,
with two of its 49,152 values outside the tolerance and 98.327 SSIMULACRA2.
That tile drives the attention denominator down to 1.45e-4 against 64
tokens. An FP64 run of the same model shows that eager FP32 is itself
1.68e-5 from FP64 there. The session is 1.80e-5 from FP64. The validator
therefore accepts a tolerance miss only when the session's FP64 error is
within 1.5 times eager FP32's error. Accumulating the denominator in FP64
did not reduce the session's error, so the kernel keeps FP32.

The page benchmark uses the same tiling as the baseline: 128×128 tiles,
batch 16, zero overlap. Session times are synchronized host calls, so they
include the host-to-device and device-to-host RGB copies. The eager
reference runs on device-resident tensors in the same process. Batch-1
latency times each of the 630 real tiles in a separate synchronized call to
a separate session prepared for batch 1.

| Source page | Batch-16 inference | Input MPx/s | Mean batch-1 tile ms | Raw PSNR dB | SSIMULACRA2 | Max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| watanare-page-085.jpg | 3.437 s | 1.343 | 16.232 | 146.658 | 92.918 | 1.01e-6 |
| watanare-page-086.jpg | 3.511 s | 1.315 | 16.337 | 147.029 | 93.263 | 1.73e-6 |
| Both pages | 6.948 s | 1.329 | 16.285 | 146.840 | 93.090 mean / 92.918 min | 1.73e-6 |
| Eager FP32, same run | 9.112 s | 1.013 | 50.844 | — | — | — |

Both pages met the eager tolerance. The rendered PNGs differed from the
FP32 renders in 85 and 70 RGB samples. The eager batch-1 tile latency in this run was slower than the 43.658 ms
baseline report, so compare the session with the paired eager row.

```powershell
$py = 'C:\Python314\python.exe'
& $py scripts/build_native.py --output output/gaterv2-session/native --arch sm_86 --nvrtc-root output/nvrtc-13.0/nvidia/cu13
$env:PYTHONPATH = 'src;scripts'
& $py scripts/validate_gaterv2_session.py --library output/gaterv2-session/native/umzi_native.dll --pages
```

The report is `output/gaterv2-session/report.json`. The same build
re-exported both GateRV3 checkpoints with bytes identical to the previous
packages. Both passed `scripts/check_python_session.py` in mixed precision
and FP16, with bitwise CLI matches and cache reuse on reopen. Those reports
are in `output/gaterv2-session/gaterv3-regression/`.

Related design documents are the [runtime and package plan](umzi-native-runtime-plan.md)
and the [exporter plan](umzi-model-exporter-plan.md).
