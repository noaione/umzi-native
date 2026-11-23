# Vulkan validation and benchmarking plan

**Status:** Test protocol. See the [GateRV3 macOS report](gaterv3-mac.md) for the first Vulkan measurements.
**Companion:** [roadmap](README.md) · [architecture](architecture.md)

## Goals

A Vulkan backend is useful only if it is (1) correct enough for the target restoration/upscaling models, (2) genuinely GPU-accelerated, and (3) faster than selected alternatives **on like-for-like workloads**. No headline speedup should be published without a model/checkpoint, input shape, batch size, hardware/driver, precision, numerical-quality result, and timing boundary.

The three *primary* comparisons are:

1. Vulkan vs **eager PyTorch CUDA** on NVIDIA.
2. Vulkan vs **eager PyTorch MPS** on the same Apple Silicon Mac.
3. Vulkan vs **ONNX Runtime WebGPU** on the same device, when the model is supported and execution placement can be verified.

Secondary/reference comparisons: `torch.compile`/Inductor, TensorRT-RTX, existing native CUDA, MPS compiled paths if available, and any AMD-native runtime introduced later. These are not Vulkan V1 release blockers.

## Existing published repo baselines: context, not Vulkan results

The current CUDA reference data already lives in:

- [`docs/gaterv3-native.md`](../gaterv3-native.md): GateRV3 native CUDA Graph / eager PyTorch / Inductor / TensorRT-RTX latency and throughput.
- [`docs/umzi-native-architecture-baselines.md`](../umzi-native-architecture-baselines.md): GateRV2, MoESR and GFISRv2 PyTorch and alternative baselines.
- [`docs/umzi-native-architecture-expansion.md`](../umzi-native-architecture-expansion.md): final native CUDA measurements, model variants, precision/fidelity, and tiling.

Snapshot of selected **RTX 3050 Laptop** benchmarks from those documents:

| Workload | Eager PyTorch reference | Native CUDA reference | Important caveat |
| --- | --- | --- | --- |
| GateRV3, batch-1 tile | FP16 **47.574 ms** | CUDA Graph **5.420 ms** | Comparable reported FP16-oriented benchmark; see full study |
| GateRV2, batch-1 tile | FP32 **43.658 ms** | FP16 **3.834 ms** | **Different precision**, not a direct like-for-like speedup |
| MoESR, batch-1 tile | FP32 **565.989 ms** | FP16 **98.0 ms** | **Different precision**, also check fidelity |
| GFISRv2, batch-1 tile | FP32 full reference from architecture study | Native mixed/FP16 depends strongly on tile/overlap | See exact study setup; do not combine incompatible tiling results |

These reference measurements are **not** a forecast of Vulkan speed or a substitute for a same-day paired rerun. The earlier PyTorch runs use specified backend settings (e.g. TF32 disabled for some comparisons); document those settings when reproducing.

## 1. Golden fixture format

Create a pinned test manifest per family. Do not store only output screenshots: preserve raw FP32 arrays or a trustworthy reference implementation that can recreate them.

Suggested (schema illustrative):

```json
{
  "family": "gaterv3",
  "variant": "original",
  "model_sha256": "<sha256-of-ummodel>",
  "reference": "pytorch-cuda-fp32",
  "precision": "mixed",
  "input_dtype": "float32",
  "shape": [1, 3, 128, 128],
  "input_sha256": "<sha256-of-tensor-bytes>",
  "reference_output_sha256": "<sha256-of-reference-tensor-bytes>",
  "tile_overlap_px": 0,
  "expected_output_shape": [1, 3, 128, 128]
}
```

Do not treat output SHA equality as the portable GPU correctness criterion; the SHA records provenance. Compare floats with numerical error/quality metrics appropriate to the model and precision.

Required fixtures:

- Flat black, mid-gray and white; single-channel-like repeated RGB; extreme boundary values within `[0,1]`.
- Checkerboard, impulses at corners/edges, gradients, deterministic seeded noise.
- Real page crops and at least one full-page assembly per family.
- Batch 1, typical throughput batch, and **distinct inputs across batch elements** (catches cross-batch indexing bugs).
- Odd/rectangular shapes accepted by the model (and explicit rejection of invalid tiny shapes/padding).
- Repeated same-shape calls, alternating shapes, model reopen, concurrent **independent** sessions, and resource teardown.
- GateRV3 Original and IDC where available; architecture-specific supported variants.

## 2. Numerics and correctness gates

### Stage-by-stage parity

Start with unfused FP32 kernels and add stage dumps around:

- Input pack/pad/crop, RGB layout conversion, filter packing.
- Convolution and GEMM/projection outputs.
- Normalization/reduction and attention numerator/denominator.
- Gates, residuals, pixel shuffle, output projection.
- GFISRv2 FFT input, complex spectrum, post-transform scaling and reconstructed spatial signal.

At each sampled stage, report `max_abs`, `mean_abs`, RMSE, finite-value count, and a few worst indices. Record whether reference and candidate use the same accumulation/rounding policy.

### Model-level checks

- Model output shape, scaling, dtype, and dynamic padding match the model's contract.
- No non-finite outputs on defined valid inputs.
- Check `atol`/`rtol` on raw floats, but **do not rely on them alone** for image reconstruction models.
- Compute raw-float PSNR (peak value documented), PNG-based SSIMULACRA2, and max pixel error for final assembled pages.
- Keep FP32 and mixed/FP16 acceptance thresholds **per model family and precision**, derived from established CUDA or reference-quality distributions. Do not use a single universal PSNR threshold.
- Make quality regressions visible when a faster FP16 arithmetic path is enabled; reject particularly unstable flat/checkerboard stress cases even if average page score looks good.
- In GateRV2, retain FP32-sensitive attention operations: prior all-FP16 paths have produced non-finite results.
- In GFISRv2, check transform direction, real/complex layout, normalizations and the frequency-domain RMS norm before evaluating final output.

**Acceptance:** Preserve or intentionally rebaseline numerical quality with evidence; arbitrary visual similarity is insufficient. In the absence of an agreed threshold, report **not yet validated** rather than mark passed.

## 3. GPU validation and correctness tooling

- Build GLSL → SPIR-V reproducibly and run `spirv-val`.
- Run Vulkan validation layers (`VK_LAYER_KHRONOS_validation`) for resources, pipeline/descriptors, queue synchronization, and command lifetime; enable synchronization validation where supported.
- Run address/bounds-oriented tests on shapes near limits; validate buffer allocation sizes and integer overflow checks.
- Test `prepare_shape()` twice, reuse of command buffers, failed prepares, device out-of-memory errors, and cleanup after exceptions.
- For unavailable half precision, cooperative matrix or subgroup modes, assert a **capability error or reference fallback**, not silent execution of an unsupported shader.
- Exercise multiple Vulkan ICDs where practical; keep the driver name, GPU vendor and exposed extension set in logs.
- Run CPU Vulkan driver smoke tests (e.g. lavapipe) in CI for basic portability, but treat them as correctness only.

## 4. Benchmark fairness rules

1. **Identical model/checkpoint and input tensors.** Log the hashes.
2. **Comparable precision and numerical quality.** An FP16 Vulkan result versus eager FP32 PyTorch is contextual, not a precision-matched win; include FP16 PyTorch if available and compare fidelity against FP32.
3. **Identical tile shapes, overlap, batch, padding and assembly** for each paired test. 4× output size is not 4× input throughput: use **input megapixels/s** consistently.
4. **Warm up** shader pipelines, library autotuning, graph/command recording, and GPU allocations before steady-state timing.
5. **Device-resident path:** preallocate inputs/outputs in GPU memory, measure only execution; exclude transfer, decode, tiling, stitching and output encoding. Note API differences in device buffer ownership.
6. **Host path:** include host→device and device→host transfers and any dtype conversion at the same boundary for each backend; report separately from GPU time.
7. **Page path:** include the same tiling/assembly overhead and exclude image decoder/encoder only if excluded for all candidates. State the boundary.
8. **Synchronization:** wait for completion before stopping host timers and reading results. Use GPU timestamps where available; do not time enqueue-only execution as completed inference.
9. **Paired measurements:** run in rotated order across multiple warmed rounds; report median, p10/p90 or IQR, and sample count. Avoid comparing results from unrelated driver/system configurations.
10. **No unintended fallback:** record ONNX Runtime WebGPU node coverage; log MPS CPU fallbacks if any; verify Vulkan actually dispatches to the selected GPU.
11. **Performance tuning control:** when testing fusion, cooperative matrices, mixed precision or command reuse, change one factor at a time and keep the previous build as control.
12. **GPU power and memory:** record device, driver, thermal/power condition if possible; include peak resident memory or a clearly stated memory measurement method.

### Reported metrics

- Device-execution latency: median batch-1 tile ms and median throughput batch ms.
- Host-call latency: median batch-1 tile ms and total per-page seconds.
- End-to-end page throughput: source input MPx/s.
- Peak GPU allocation and persistent weights/activations/workspace bytes.
- Setup: model load, shader/pipeline construction, shape preparation, first run, cached startup.
- Accuracy: raw PSNR/RMSE/max error and image SSIMULACRA2.
- Optional: per-stage GPU timestamps, dispatch count, barrier overhead, occupancy/memory metrics using vendor profilers.

## 5. Suggested benchmark matrix

| Model | Initial workload | Then test | Comparators |
| --- | --- | --- | --- |
| GateRV3 | Original, 128×128, batch 1 | IDC, batch 16, 65×79, 128×160, full pages | CUDA eager FP16/FP32; CUDA native; ORT WebGPU; MPS on Mac |
| GateRV2 | 128×128, batch 1 | batch 16, accepted rectangles, full dehalftone pages | CUDA eager, Inductor attention-FP32, ORT WebGPU, MPS |
| MoESR | 256×256, batch 1 | batch 2, accepted rectangles, 4× assembled pages | CUDA eager, ORT WebGPU, MPS |
| GFISRv2 | 128×128, batch 1, overlap 32 | 256×256 batch 2 overlap 64, flat stress, full pages | CUDA eager, native CUDA, ORT WebGPU if FFT ops are supported, MPS |

Use repository baseline scripts as a guide; not all comparators will have identical execution or operator support. If ONNX Runtime WebGPU cannot fully run a family, mark it **unsupported/incomparable**, not infinitely slow.

## 6. Platform matrix and CI

| Platform | Initial test type | Must not claim |
| --- | --- | --- |
| Windows + NVIDIA Vulkan | Main development correctness + speed | AMD compatibility |
| Linux + NVIDIA Vulkan | Build/driver portability, speed | Performance identical to Windows |
| macOS Apple Silicon + KosmicKrisp | Capability smoke, quality, MPS benchmark | Feature support across older macOS releases |
| macOS Apple Silicon + MoltenVK | Optional compatibility fallback | Same features/performance as KosmicKrisp |
| Linux + lavapipe | CI smoke/correctness | GPU performance |
| Linux/Windows + AMD Vulkan (remote) | Hardware acceptance, features, correctness, performance | ROCm/HIP compatibility |
| Intel Vulkan GPU (when accessible) | Later smoke and correctness | Tier-1 support before tested |

Record the actual Vulkan implementation, API version, extension list, macOS version and device UUID. KosmicKrisp deployment may require a newer macOS/Metal generation; MoltenVK remains an alternative for configurations it supports.

### CI stages

1. Lint/compile host C++ and GLSL; validate SPIR-V.
2. Run model-file parsing and ABI tests without an installed GPU.
3. Vulkan CPU-driver synthetic primitive smoke (only if environment supports it).
4. NVIDIA Vulkan GPU runner (when available).
5. Apple Silicon Vulkan runner (when available).
6. AMD GPU runner (manual or cloud until automation exists).
7. Emit reports as versioned CI artifacts; make skipped device lanes explicit.

Do not require a GPU for packaging smoke tests such as importing the Python module and inspecting `.ummodel` metadata.

## 7. Proposed result schema

A single summary JSON per `(family, model, precision, backend, device, shape, batch, tiling)` enables later automated comparisons:

```json
{
  "schema": 1,
  "git_commit": "<commit>",
  "family": "gaterv3",
  "model_sha256": "<sha256>",
  "inputs_sha256": "<sha256>",
  "backend": "vulkan",
  "backend_version": "<version>",
  "gpu": {"name": "<device>", "vendor_id": 0, "driver": "<driver>", "vulkan_api": "1.x"},
  "precision": "mixed",
  "shape": [1, 3, 128, 128],
  "batch": 1,
  "overlap_px": 0,
  "iterations": 0,
  "timing_ms": {
    "device_median": null,
    "host_median": null,
    "device_p90": null
  },
  "throughput_input_mpx_s": null,
  "memory_mib": {"weights": null, "activations": null, "workspace": null},
  "quality": {"raw_psnr_db": null, "max_abs": null, "ssimulacra2": null},
  "validation": {"status": "not-run", "details": ""}
}
```

Values are intentionally `null`, not fabricated estimates. Record a link or artifact path to full per-sample timings, stage traces and final renders.

## 8. Performance exit criteria

**P2:** quality/shape/stability gates pass, regardless of speed.  
**P3:** measured speedup over unfused Vulkan baseline at the **same quality**; no notable memory or initialization regression without justification.  
**Initial competitive goal:** stable results faster than eager PyTorch CUDA for a declared GateRV3 workload on the same NVIDIA GPU.  
**Cross-platform goal:** pass correctness on supported Apple/AMD GPUs before advertising them; compare to MPS/WebGPU where directly runnable.

If Vulkan loses a benchmark, keep the measured result. Profile the bottleneck (convolution/GEMM, command overhead, transfers, FFT, bandwidth) and decide whether it merits a specialized path. The goal is a portable, correct backend with measurable competitive value—not a benchmark that is forced to look positive.
