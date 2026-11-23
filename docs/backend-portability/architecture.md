# Proposed Vulkan backend architecture

**Status:** Long-term design sketch. See the [GateRV3 prototype](gaterv3-mac.md) for the current ABI, runtime, and tested behavior.
**Companion:** [roadmap](README.md) · [validation](validation.md)

## Current Vulkan source layout

| Path | Responsibility |
| --- | --- |
| `native/vulkan/runtime/` | Host C ABI and model-family registry. Dispatches packages to a Vulkan kernel session. |
| `native/vulkan/common/` | Vulkan devices and buffers, compute commands and pipelines, and reusable GLSL buffer and math helpers. |
| `native/vulkan/kernels/gaterv3/` | GateRV3 runner, option handling, schedule, stage profiling, and shaders. Its CMake file builds the family's embedded SPIR-V. |
| `native/kernels/gaterv3/model_file.h` | GateRV3 package and schedule parsing shared with CUDA. |

Add each Vulkan model family under `native/vulkan/kernels/` and register its factory in `native/vulkan/runtime/registry.cpp`. Keep model-specific push constants and shader dispatch in the family directory. [GateRV3 kernel options](gaterv3-vulkan-optimization.md#gaterv3-kernel-options)

## 1. Architectural principles

1. **Preserve the `.ummodel` contract.** Family IDs, section metadata, checkpoint digests, weights and exported schedules remain readable on CPU without a GPU runtime.
2. **Avoid a CUDA-dependent Vulkan import.** Loading a Vulkan-only session must neither import `torch` nor load NVIDIA DLLs/SOs.
3. **Keep model semantics shared but execution backend-owned.** Each backend owns buffers, dispatches, command recording, precision and implementation-specific plan caches.
4. **Start from host I/O.** The common Python contract accepts NumPy NCHW input and returns NumPy output. Device interop is a future versioned extension, not `reinterpret_cast<uint64_t>(VkBuffer)`.
5. **Provide correct generic kernels first.** All specialized shaders and extensions must have tested fallback paths.
6. **Limit backend coupling.** The existing CUDA runners should not be rewritten into an imagined generic GEMM/conv API before an end-to-end Vulkan proof of concept.

## 2. Existing source ownership

| Existing file | Purpose | Proposed action |
| --- | --- | --- |
| `native/runtime/model_package.{h,cpp}` | `.ummodel` parsing | Reuse unmodified where practical |
| `native/runtime/binary.h` | Packed binary helpers | Reuse |
| `native/runtime/kernel.h` | Abstract session; CUDA-flavored device enqueue | Add a **separate backend-neutral host session interface** or versioned adapter; retain old CUDA behavior |
| `native/runtime/registry.cpp` | Model family → CUDA factory | Split registration into family and backend selection in new code; preserve old entry point |
| `native/runtime/api.cpp` | C ABI | Keep v1; add versioned v2 entry points, not in-place breaking changes |
| `native/include/umzi_native.h` | C ABI types | Introduce v2 types without changing legacy struct layout |
| `native/kernels/*/runner.cpp` | Current CUDA sessions | Leave CUDA implementations intact |
| `native/kernels/*/kernels.cu` | CUDA operations | Implement Vulkan counterparts; do not assume mechanical translation |
| `native/common/tensor_core.cuh` | PTX MMA, `ldmatrix`, `cp.async` | No direct Vulkan reuse; substitute generic or cooperative-matrix shaders |
| `hatch_build.py` | NVRTC/CUDA host build | Add a separate Vulkan build target/toolchain path |
| `src/umzi_native/_session.py` | Python inference API | Add backend selection and backend-specific library loading |
| `src/umzi_native/_loader.py` | Dynamic CUDA libraries | Ensure Vulkan selection bypasses this path |

## 3. Suggested new source layout

This is a proposal, not a requirement to rename current files:

```text
native/
  include/
    umzi_native.h             # existing v1 CUDA ABI; retain
    umzi_native_v2.h          # new backend-neutral ABI
  runtime/
    model_package.*           # existing, reusable
    backend_registry.*        # backend + family resolution
    api_v2.cpp                # new host-facing versioned C API
  vulkan/
    context.{h,cpp}          # physical device, features, logical device, queue
    buffer.{h,cpp}           # allocation, upload/download, staging, arena
    dispatch.{h,cpp}         # pipelines, descriptors, push constants, barriers
    shader_registry.{h,cpp}  # embedded/precompiled SPIR-V manifest
    command_plan.{h,cpp}     # pre-recorded per-shape graph/commands
    cache.{h,cpp}            # Vulkan plan/pipeline cache identity and disk I/O
    ops/                     # reference convolution, GEMM, activation, etc.
    families/
      gaterv3.cpp
      gaterv2.cpp
      moesr.cpp
      gfisrv2.cpp
    shaders/
      common/
      gaterv3/
      gaterv2/
      moesr/
      gfisrv2/
  kernels/                    # existing CUDA files, unchanged initially
src/umzi_native/
  _session.py                # backend option; remains common
  _loader.py                 # backend-aware libraries and diagnostics
scripts/
  validate_vulkan.py         # shape, primitive, and model parity
  benchmark_vulkan.py        # can reuse/extend current benchmark framework
```

Possible deliverables: `umzi_native_cuda.dll/.so` and `umzi_native_vulkan.dll/.so`, or equivalent distinct filenames in the wheel. Backend-specific filenames avoid forcing Vulkan installations to link cuDNN/cuBLASLt. The exact packaging names are a later design choice.

## 4. Backend selection and ABI compatibility

### Python-facing proposal

```python
import umzi_native as umzi

options = umzi.SessionOptions(
    backend="vulkan",       # "cuda" | "vulkan" | "auto" (future: "hip")
    device_id=0,             # index within the selected backend, not universal GPU ID
    precision="mixed",      # subject to backend/model support
    cache_dir="cache/umzi",
)

with umzi.InferenceSession("model.ummodel", sess_options=options) as session:
    (output,) = session.run(None, {"input": tiles})
```

**Compatibility rule:** `SessionOptions.backend` accepts `cuda` and `vulkan`. Omitting it selects CUDA. Any future `auto` policy needs a documented selection order and a way to report the chosen backend and device.

### C ABI proposal

Current `UmziStream = CUstream_st*` and `umzi_session_enqueue()` expose CUDA-specific handles and device pointers. Keep these working for existing callers. Do not redefine `UmziStream` as a Vulkan handle or append fields to existing structs if it would break `struct_size` checks.

Introduce a new exported entry point (name illustrative):

```cpp
// Illustrative API, not code in the repository.
typedef struct UmziSessionOptionsV2 {
    uint32_t struct_size;
    uint32_t backend;       // UMZI_BACKEND_CUDA / VULKAN / (later HIP)
    int32_t device_id;
    uint32_t precision;
    uint32_t flags;         // backend-neutral flags only
    const char* cache_dir;
    const UmziOption* kernel_options;
    uint32_t kernel_option_count;
} UmziSessionOptionsV2;

UMZI_API UmziSession* umzi_session_open_v2(
    const char* path, const UmziSessionOptionsV2* options);
```

Keep v1 `umzi_abi_version()` behavior stable and add an independently queryable v2 capability/version function if needed. An alternative is a backend-specific Vulkan library exposing its own ABI and a Python wrapper that hides the distinction; choose after P0.

V1 Vulkan exposes **host** `run()` and `prepare()`. A later device-input API must use backend-tagged, typed handles (e.g. Vulkan `VkBuffer` plus offset/size and synchronization contract). A Vulkan buffer is **not** a CUDA pointer. If adding interop, specify who owns the device, queue, memory, layout and fences.

## 5. Vulkan execution architecture

The initial data flow:

```text
.ummodel parse + validate  (CPU)
  ├─ read model family, graph/schedule, shapes, FP32 checkpoint weights
  └─ select backend and compatible GPU
               │
               ▼
     Vulkan session creation
  - query features/limits
  - choose shader variants
  - convert/pack weights once
  - upload persistent weights
               │
               ▼
     prepare_shape(N, C, H, W)
  - validate shape & padded shape
  - assign tensor liveness and offsets in activation arena
  - allocate staging, I/O, and optional temporary workspace
  - create per-op pipelines/descriptors
  - record reusable command buffers with barriers
               │
               ▼
        run_host()
  - stage upload to device-local input
  - submit or replay pre-recorded compute commands
  - wait on completion and stage output to CPU
               │
               ▼
       NumPy output
```

### Device capabilities

At session creation, query and record:

- API/version, device UUID and driver version, vendor/device IDs and device name.
- Compute queue availability, workgroup size/count, workgroup-shared memory and storage-buffer limits.
- Storage buffer 16-bit access and `shaderFloat16` **independently**. FP16 storage alone does not imply FP16 arithmetic.
- Subgroup operations and subgroup size; avoid assuming NVIDIA warp width.
- Whether `VK_KHR_cooperative_matrix` exists; enumerate supported shape/type/scope combinations **per device**.
- Limits relevant to Vulkan-on-Metal portability implementations. Unsupported features must result in an explicit fallback/error.

Default to a basic Vulkan compute path that avoids cooperative-matrix requirements. FP32-only hardware can be supported with FP32 reference shaders if the model and memory budget permit; do not advertise FP16 modes unless they work.

### Buffer and command ownership

- All model weights and common activations should be **GPU-resident between runs**.
- Allocate one or a few activation arenas using existing lifetime information; prevent concurrent tensor reuse until prior reads/writes complete.
- Prefer device-local buffers plus host-visible staging unless the selected memory heap makes a different mapping more efficient.
- Use explicit Vulkan memory barriers for write-after-read/read-after-write dependencies; do not assume command recording alone supplies all visibility guarantees.
- Bind descriptors (or descriptor indexing if available) and pipeline objects during `prepare_shape`, not per tile.
- Record command buffers per prepared shape and device. Reuse them safely; changing shape/arena allocation invalidates the old recording.
- Use timestamp queries around GPU work and host clocks around end-to-end calls. Synchronize before reading GPU timings.
- Initial implementation: one compute queue and serial calls per session, matching current session serialization requirements.

### Shader/toolchain plan

- Build GLSL compute shaders into SPIR-V **at build time** with a pinned compiler/toolchain (e.g. `glslangValidator` or shaderc), then run `spirv-val`.
- Maintain a shader manifest mapping `(family, operator, precision, variant)` to SPIR-V assets and required device features.
- Use specialization constants for fixed shapes/workgroup sizes where useful; avoid unbounded shader variants on arbitrary dimensions.
- Stage-level debugging builds should have simple, inspectable operators even if release builds fuse them.
- Keep shader compilation artifacts and plan-cache formats backend/version-specific; do not interpret CUDA cubins/PTX or cuBLASLt plans as Vulkan artifacts.

## 6. Operators and model-specific execution

| Operation | Correctness-first implementation | Optional optimized implementation |
| --- | --- | --- |
| Bias / SiLU / Mish / residual / shuffle | Storage-buffer compute shaders | Fuse adjacent pointwise stages |
| RMSNorm / layer norm / reductions | FP32 accumulation and explicit reductions | Subgroup reductions, fused norm/activation |
| GEMM and 1×1 projections | Generic tiled GEMM in compute | Cooperative-matrix path (feature-gated) |
| Standard convolution | Basic tiled direct or im2col/GEMM convolution | Specialized kernels, prepacked filter layouts |
| Depthwise convolution | Spatially tiled direct shader | Fuse activation/gate/project |
| Attention and gates | Separate, traceable intermediate operations | Model-specific fused attention/gating shaders |
| Pixel shuffle / packing / padding | Simple exact index mapping | Fuse with adjacent projection/output |
| Real FFT / inverse FFT | VkFFT only for GFISRv2 | VkFFT plan tuning; packed-spectrum fusion |

**GateRV3:** Port its schedule faithfully from `gaterv3/model_file.h` and runner. Match input/output scale behavior, 8-lane padded channels, mixed/FP16 policies, and attention/normalization operations. Current cuBLASLt and cuDNN plan selection is *not* reusable; the checkpoint weights and logical operations are.

**GateRV2:** Preserve FP32 Q/K/V projection outputs and attention-sensitive normalization, contractions, key sums and denominators even in FP16 I/O mode until fidelity tests demonstrate a safe replacement. Existing documented all-FP16 attention has exhibited non-finite failures.

**MoESR:** The CUDA fast path uses PTX `cp.async`, `ldmatrix`, and `mma.sync`, notably via `native/common/tensor_core.cuh`. Reimplement semantics as ordinary Vulkan compute first. Recreate the fused tensor-core path only after measuring whether generic kernels miss the performance target.

**GFISRv2:** Integrate VkFFT for batched 2D real-to-complex and complex-to-real transforms; match cuFFT's transform dimensions, batch count, half-spectrum layout, input/output stride, inverse scaling and frequency-space norm/packing. **Do not assume VkFFT and cuFFT use identical packing/normalization defaults.** Validate each Fourier intermediary. Start with FP32 spectral arithmetic and only enable mixed/FP16 after quality testing.

## 7. Caching and tuning

Existing persistent `UMZIPLAN` entries contain opaque CUDA/cuBLASLt/cuDNN choices. They cannot safely be used by Vulkan. New entries should include:

- backend type, backend/plan schema, binary/shader digest, model checkpoint digest;
- vendor/device UUID and driver version;
- selected feature bitset, subgroup/cooperative matrix characteristics;
- precision, model family, shape, relevant kernel settings;
- pipeline/tuning choices (e.g., tile and workgroup sizes), with robust compatibility checks.

Use Vulkan `VkPipelineCache` as appropriate, but assume cache blobs are implementation- and driver-specific. Corrupt or stale cache entries should regenerate cleanly.

## 8. Packaging and developer workflows

- Prototype as an **explicit Vulkan native build** so CUDA toolchains are not needed for a Vulkan smoke test.
- Keep `hatch_build.py` CUDA path working; introduce Vulkan shader build tooling as a separate target before merging packaging logic.
- Avoid runtime shader compiler dependencies in installed wheels where possible; ship precompiled SPIR-V and the Vulkan loader dependency appropriate to the platform.
- `import umzi_native` and `umzi_native read model.ummodel` should not require an installed GPU runtime; choose/loading a backend should be lazy.
- Add `umzi-native devices` (suggested) to list available backends/devices/features.
- Provide explicit errors: backend library missing, no compatible Vulkan compute device, FP16 mode unsupported, Vulkan driver missing, family unsupported, insufficient memory.

## 9. Failure modes and non-goals

- **Pipeline replay hazards:** stale descriptors or missing barriers can look like numerical instability; test with Vulkan validation layers and repeat loops.
- **Non-portable subgroup assumptions:** 32-lane warps are not a universal Vulkan property; dynamically adapt or use a safe fallback.
- **Precision drift:** tiny differences in transcendental functions, contraction, accumulation, and FP16 rounding can be amplified in deeper models; compare stages, not just final PNGs.
- **Platform translation:** Vulkan-on-Metal implementations may expose different feature sets and performance from native Vulkan; report implementation/version in benchmarks.
- **Backend identity:** an AMD Vulkan driver doesn't imply ROCm/HIP availability, and NVIDIA Vulkan doesn't validate AMD-specific behavior.

## 10. Implementation choices deferred until prototype

- Whether device-interoperability APIs should be separate backend entry points or a typed common handle union.
- Whether to use descriptor indexing/device address extensions for performance or remain with straightforward descriptor sets.
- Whether ncnn kernels are suitable enough to reuse without imposing its graph/weight formats.
- Whether to introduce cooperative matrices immediately after GateRV3 correctness or wait for real bottleneck profiles.
- Whether the Vulkan library lives inside the base wheel or in a backend-specific distribution.

A prototype that runs one GateRV3 tile end-to-end and produces a high-quality reference match is the first decision gate. **Do not spend the first iteration implementing an entire general-purpose operator library.**
