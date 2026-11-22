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

| Family | CPU checkpoint load | CUDA baselines | Native FP32 parity, export, and session |
| --- | --- | --- | --- |
| GateRV2 | Verified | FP32 and RTX measured, default Inductor FP16 produces non-finite output | Pending |
| MoESR | Verified, supplied ONNX validated | FP32, Inductor FP16, and RTX FP16 measured | Pending |
| GFISRv2 | Verified, prior handoff reviewed | FP32, eager FP16, and Inductor FP16 measured, flat-tile fidelity failure traced, RTX skipped | Pending |

Related design documents are the [runtime and package plan](umzi-native-runtime-plan.md)
and the [exporter plan](umzi-model-exporter-plan.md).
