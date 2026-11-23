# Portable GPU backend roadmap (Vulkan first)

**Status:** GateRV3 Vulkan prototype tested on Apple M4
**Last reviewed:** 2026-11-23  
**Repository:** [`noaione/umzi-native`](https://github.com/noaione/umzi-native)

The GateRV3 prototype runs the Original and IDC packages through a standalone Vulkan library.
See the [macOS build, validation, and comparison](gaterv3-mac.md) for the measured configurations.
The remaining phases describe work still to be tested or implemented.

## Objective

Add a native Vulkan inference backend to `umzi-native` without regressing the current CUDA implementation. The immediate aim is **useful cross-vendor GPU inference** with a stable `.ummodel` format and the existing Python-facing `InferenceSession` experience. Winning every CUDA microbenchmark is not a requirement.

**Performance targets (hypotheses to test, not guarantees):**

1. Outperform **eager PyTorch CUDA** on the same NVIDIA GPU, model, shape, and *comparable numerical quality*.
2. Outperform **PyTorch MPS** on the same Apple Silicon Mac where supported, with fair input/output and synchronization boundaries.
3. Outperform **ONNX Runtime WebGPU** when using equivalent hardware and supported model topology; verify there is no unexpected CPU fallback.
4. Collect informative results against `torch.compile`, optimized ONNX execution providers, and native CUDA. These are **comparison references, not launch blockers**.

Cross-vendor support is the primary product goal; initial implementation and performance validation can happen on NVIDIA and Apple Silicon. Hardware-specific AMD verification is required before promising AMD support.

## Existing starting point

- `native/runtime/model_package.{h,cpp}` reads `.ummodel` packages; keep the serialized checkpoint representation backend-neutral.
- `native/runtime/kernel.h` defines `KernelSession`, and `native/runtime/registry.cpp` dispatches model families. **Today these are CUDA-oriented**, especially device enqueue and stream handling.
- `native/include/umzi_native.h` has a stable C ABI, but `UmziStream` is a CUDA stream and `umzi_session_enqueue()` expects CUDA pointers.
- `src/umzi_native/_session.py`, `_abi.py`, and `_loader.py` currently arrange CUDA loading and expose the NumPy API.
- `native/kernels/{gaterv3,gaterv2,moesr,gfisrv2}` each contain a CUDA runner and custom `.cu` kernels.
- `hatch_build.py` compiles CUDA images with NVRTC and links NVIDIA runtime libraries.
- `scripts/benchmark_native.py`, `scripts/benchmark_architectures.py`, model-specific validation scripts, and `docs/umzi-native-architecture-expansion.md` give a good benchmark/quality baseline.

Relevant implementation references: [current API](../../native/include/umzi_native.h), [session interface](../../native/runtime/kernel.h), [CUDA architecture baselines](../umzi-native-architecture-baselines.md), [final architecture results](../umzi-native-architecture-expansion.md).

## Scope and exclusions

### V1 scope

- Windows and Linux Vulkan compute; macOS through a Vulkan-on-Metal implementation where feature support permits.
- GateRV3, GateRV2, MoESR, GFISRv2, introduced incrementally rather than all at once.
- Keep the **same model files** and preserve model family, scale, shapes, precision policy, and postprocessing behavior.
- Host-facing NumPy inference first; shape preparation, persistent allocations, shader/pipeline reuse, and GPU-side execution.
- Portable FP32 correctness reference plus FP16/mixed fast paths when supported and numerically safe.
- A completely separate Vulkan backend that does **not** need CUDA installed or initialized.
- Baseline correctness/throughput tests on NVIDIA; Apple Silicon testing when supported; remote AMD smoke testing prior to AMD support claims.

### Explicitly not V1

- Replacing CUDA or promising Vulkan/CUDA bit-exact results.
- Exposing zero-copy CUDA ↔ Vulkan memory/stream interop.
- Compiling arbitrary ONNX graphs or implementing a general tensor framework.
- Trying to map NVIDIA PTX instructions directly into Vulkan shader code.
- Depending on ROCm/HIP to run the Vulkan backend.
- Optimizing every model family, architecture, and tile size before the first working end-to-end path.

## Proposed phases and acceptance gates

Work in order; each phase should be a mergeable, testable deliverable.

### P0 — Baseline and API decisions

- [ ] Freeze test checkpoints and test-input hashes for all four model families; record model variants (e.g., GateRV3 Original/IDC).
- [ ] Record source CUDA measurements for the **exact** chosen shapes and precision modes; separate device-resident from host-to-host timing.
- [ ] Add a backend capability report (GPU, driver, Vulkan API version, FP16 storage/arithmetic, subgroup size, cooperative matrix shapes, limits).
- [x] Decide shared-library naming and backend discovery, preserving existing CUDA users.
- [x] Define a versioned Vulkan host ABI and retain the CUDA ABI.
- [x] Decide minimum Vulkan API/features and tested macOS implementation(s).

**Exit:** A written compatibility contract, golden test fixtures, benchmark commands, and a Vulkan device enumeration smoke test.

### P1 — Vulkan runtime foundation

- [x] Implement instance/device/compute queue selection, opt-in validation layers, resource lifetime, and error mapping.
- [x] Add staging and device-local storage buffers, persistent model weights, and an activation arena.
- [x] Implement host-copy-in → compute dispatch → host-copy-out without importing CUDA libraries.
- [x] Compile and validate SPIR-V at build time. Load pipelines, descriptor sets, and an in-memory pipeline cache at runtime.
- [x] Record reusable command buffers and intra-graph barriers for prepared shapes.
- [ ] Port a few test primitives (`copy`, FP32→FP16, bias/activation, pixel shuffle) and run in debug validation mode.
- [x] Add Python `backend="vulkan"` routing and retain CUDA as the default.

**Exit:** At least one synthetic graph runs correctly with no CUDA runtime, multiple calls work, reprepare/reopen works, and validation reports no GPU hazards.

### P2 — GateRV3 correctness-first

- [x] Read the existing `.ummodel` GateRV3 schedule and weight metadata.
- [x] Run the required convolutions, projections, padding, normalization, attention, gates, shuffle/concat, skip and output transforms on the Vulkan path.
- [x] Implement portable convolution and projections with tiled GLSL compute.
- [ ] Preserve FP32 accumulation and FP16 rounding boundaries as determined from CUDA/PyTorch stage traces.
- [ ] Verify the Original and IDC variants, batch 1 and throughput batch, odd/rectangular shapes, and malformed inputs.
- [ ] Record model output, stage differences, shader dispatch count, latency, throughput, and peak memory.

**Exit:** GateRV3 completes full image inference and passes agreed correctness/quality gates on NVIDIA Vulkan; at least one Apple Silicon smoke run if supported.

### P3 — Performance passes

- [ ] Reuse recorded compute command buffers and pre-allocate descriptor sets, device buffers, and scratch.
- [ ] Fuse frequent elementwise/norm/gate operations only after the unfused version passes.
- [ ] Tune workgroup/subgroup configurations **per actual device limits**; do not assume 32-lane NVIDIA warps.
- [ ] Add channel-padded convolution/GEMM kernels and optional `VK_KHR_cooperative_matrix` path where advertised and profitable.
- [ ] Benchmark FP16/mixed quality separately; reject modes with non-finite or unacceptable output differences.
- [ ] Add key-isolated tuning/pipeline cache records; never load CUDA-specific `.umplan` payloads as Vulkan plans.

**Exit:** For at least one declared GateRV3 workload, show a stable, repeatable improvement over the unfused Vulkan path; evaluate the three target baselines rather than assuming success.

### P4 — More model families

- [ ] **GateRV2:** preserve Q/K/V and attention's FP32-sensitive arithmetic; implement its pooling/inception/gate operations.
- [ ] **MoESR:** port branch, depthwise, and projection operations; prioritize a correct generic path before AMD/NVIDIA cooperative-matrix acceleration.
- [ ] **GFISRv2:** integrate Vulkan VkFFT R2C/C2R with matching dimensions, batch, layout, normalization and spectrum packing; validate frequency-domain intermediates.
- [ ] Run each family's full-image assembly and stress suite before claiming support.

**Exit:** Every family independently meets its correctness/quality gate. Partial family support is surfaced clearly in capabilities and errors.

### P5 — Platform verification and packaging

- [ ] Windows Vulkan + NVIDIA release smoke tests and Linux Vulkan + NVIDIA tests.
- [ ] macOS Apple Silicon via **KosmicKrisp or MoltenVK** depending on SDK, macOS version and available features; do not rely on one implementation being universally installable.
- [ ] Remote AMD Vulkan smoke tests (at least one supported RDNA GPU); test precision, subgroup behavior, feature negotiation, memory limits, quality and performance.
- [ ] CPU-driver smoke tests (Mesa lavapipe) where appropriate; do not interpret CPU-driver performance as GPU performance.
- [ ] Publish backend-specific build/runtime requirements and a diagnostic command.
- [ ] Add CI matrix for builds, shader validation, CPU-driver tests, and available hardware; treat unavailable hardware tests as **not run**, not passed.

**Exit:** Install/run instructions and explicit support matrix. Existing CUDA wheel/API behavior is unaffected.

### P6 — Optional HIP backend (separate decision)

Only if AMD Vulkan results expose a meaningful remaining performance gap and ROCm device/library support justifies maintenance cost. See [hip-backend-notes.md](hip-backend-notes.md).

## Model-family priority

| Order | Family | Main reason | High-risk component |
| --- | --- | --- | --- |
| 1 | GateRV3 | Existing schedule and strong CUDA-vs-eager reference; good first end-to-end target | cuDNN/cuBLASLt replacements, attention/gates |
| 2 | GateRV2 | Similar convolution/matrix groundwork reused | Numerically sensitive attention |
| 3 | MoESR | Builds on general conv/GEMM/branch infrastructure | Tensor-core-specific fused gate/projection path |
| 4 | GFISRv2 | Requires additional FFT integration and numerical validation | VkFFT integration, spectral FP16 instability |

This order minimizes infrastructure risk; it is **not** a forecast of which model will be fastest.

## Acceptance criteria for release

- [ ] `InferenceSession(..., backend="vulkan")` behaves like the CUDA host API for supported families; errors are actionable for unsupported combinations.
- [ ] Every supported model has reproducible full-image quality reports, finite-value stress tests, and documented precision behavior.
- [ ] No implicit CPU fallback or CUDA library loading on the Vulkan path.
- [ ] Build+smoke runs on at least NVIDIA Vulkan and one supported macOS Vulkan implementation; actual AMD GPU testing required for any AMD support claim.
- [ ] Repeated model prepare/run/destroy cycles have no validation errors, leaks, stale descriptors or cross-session buffer aliasing.
- [ ] Latency and throughput comparisons include methodology and numerical scores, not a headline speedup without context.
- [ ] CUDA regressions are measured and existing ABI/tests remain usable.

## Decisions to revisit

1. **Separate Vulkan library vs one multi-backend binary:** default to a separate backend library to avoid CUDA dependencies; revisit once build/ABI details are measured.
2. **GLSL vs HLSL vs C++ shader transpilation:** default to GLSL compute → SPIR-V for first correctness milestone; consider alternatives after inspecting diagnostics and tooling.
3. **Hand-written convolutions vs borrowing a Vulkan inference engine:** use a small generic baseline first; evaluate ncnn or other mature kernels if the cost of optimization dominates.
4. **Minimum Vulkan features:** require only what the baseline needs; gate `shaderFloat16`, 16-bit storage and cooperative matrices independently.
5. **macOS deployment:** choose a tested Vulkan-on-Metal driver and macOS deployment target rather than assume all Macs expose the same Vulkan feature set.
6. **HIP:** separate project scope and explicit Go/No-Go gate.

## Companion docs

- [Vulkan backend architecture](architecture.md): interfaces, ABI migration, memory model, kernels, build layout and portability decisions.
- [Validation and performance plan](validation.md): fixtures, correctness tests, benchmark fairness, CI/device matrix and reporting format.
- [HIP backend notes](hip-backend-notes.md): AMD-specific alternative, reuse limits, triggers, and test strategy without local AMD hardware.

## Useful upstream references

- [Vulkan specification/reference](https://docs.vulkan.org/)
- [Vulkan cooperative matrix extension](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html)
- [VkFFT](https://github.com/DTolm/VkFFT)
- [MoltenVK](https://github.com/KhronosGroup/MoltenVK) / [KosmicKrisp](https://docs.mesa3d.org/drivers/kosmickrisp.html)
- [ONNX Runtime WebGPU EP](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html)
