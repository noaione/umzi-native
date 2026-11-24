# Multi-kernel Umzi native runtime plan

## Implementation log

- **Container and exporter.** The shared reader and Python writer now
  use `UMZINPKG` v1. GateRV3 exports directly from the checkpoint and
  its decoder validates the existing architecture, schedule and weights
  payload. Two exports matched byte for byte. All four payload sections
  matched the frozen GateRV3 baseline.
- **Plan cache.** GateRV3 now writes its tuning records through the
  shared `UMZIPLAN` envelope as `gaterv3_<hex-key>.umplan`. Backend
  validation of restored tactics remains in the GateRV3 runner. The
  Windows shape check exercised a miss, a hit, rebuilding after
  corruption, and an unavailable cache directory.
- **Native boundary.** Host entry points, registration, and package
  parsing are in `native/runtime/`. GateRV3 lives under
  `native/kernels/gaterv3/`, and the CUDA image selector lives in
  `native/common/`. The small runtime keeps lifecycle and dispatch in
  `api.cpp`; the GateRV3 adapter shares `runner.cpp` with its executor.
- **Python and builds.** Python now binds the versioned Umzi ABI and
  allocates output from the kernel descriptor. The Windows wheel ships
  `umzi_native.dll`. The installed wheel matched the CLI in mixed and
  FP16 with both PyTorch-first and explicit CUDA loading. A standalone
  `sm_89`-only build verified the PTX fallback on the RTX 3050 Laptop.
- **Boundary test.** The standalone `--test-kernel` build supported a
  synthetic 2× output, custom tensor names, and one-channel input through
  the same Python API. The wheel rejected its family ID. Truncation,
  checksum, section bounds, and unsupported schema checks passed.
- **Performance check.** The example and existing benchmarks now target
  the new ABI. The Windows shape suite passed host/device and graph/direct
  parity in both precisions, including batch 16 and odd tiles. The C
  example compiled and ran. The native-only manga rerun used the same
  three pages, zero overlap, and the default optimization flags.

| Precision | Batch-1 measured ms/tile, previous → shared runtime | Batch-16 s/page, previous → shared runtime | Warmed batch-16 VRAM |
| --- | ---: | ---: | ---: |
| Mixed | 5.185 → 5.168 | 0.819 → 0.779 | 210 MiB |
| FP16 | 4.810 → 4.815 | 0.727 → 0.707 | 204 MiB |

The separate runs show no material steady-state regression. Warmed VRAM
is device usage above the pre-session baseline. GateRV3 still keeps one
active shape. Linux validation remains pending. The GFISRv2 adapter is
tracked in the [architecture expansion](umzi-native-architecture-expansion.md#gfisrv2-native-fp32-package-and-session).
Ruff and Pyrefly passed.
[Shape validation](../output/umzi-multi-kernel/validation/shapes.json) ·
[Manga timings](../output/umzi-multi-kernel/manga/report.json) ·
[VRAM](../output/umzi-multi-kernel/memory/report.json)

### Local pickup

```powershell
$py = 'C:\Python314\python.exe'
& $py -m pip wheel . --no-deps -w output/umzi-multi-kernel/final-wheel
& $py scripts/export_native.py --kernel gaterv3 --output output/native/gaterv3.ummodel
```

The verified standalone Windows build used `--nvrtc-root
output/nvrtc-13.0/nvidia/cu13` so the system Python could keep its newer
NVRTC. To check the kernel boundary again, build with `--test-kernel`
in a separate directory and run `scripts/check_runtime_contract.py`
with the production and test DLL paths.

Build one Umzi runtime that loads a `.ummodel` file and selects the
compiled kernel family declared in its header. GateRV3 is the only kernel
family shipped in this rework. The runtime should be able to accept a 4×
kernel through the same API when GFISRv2 is added later.

Today, `native/gaterv3/runner.cpp` owns both the C entry points and
GateRV3 execution. The Python wrapper binds the `gt_session_*` API,
validates RGB inputs itself, and uses one shape and dtype for both input
and output. The GateRV3 model parser also owns the container header,
section table and checksum. Move those responsibilities to the runtime
where they can serve any registered kernel family.

## Target file structure

```text
native/
  include/umzi_native.h              Public C ABI (17 base umzi_* exports)
  common/
    platform.h                       OS services, SHA-256, atomic cache writes
    cuda_module.h                    Cubin selection and PTX fallback
    cuda_helpers.h                   Shared CUDA/cuDNN/cuBLASLt checks, ContextScope, graph capture, FP16 packing
    cuda_plan_cache.h                Shared CUDA .umplan record serialization
    cudnn_conv_plan.h                Shared cuDNN backend graph descriptor, candidate search & timing
    resize_rgb.cuh                   Shared reflection pad/crop device helper
    tensor_core.cuh                  Shared Ampere+ PTX cp.async / ldmatrix / mma helpers
  runtime/
    api.cpp                          C entry points, dispatch, lifecycle and validation
    cli.cpp                          Standalone kernel-agnostic umzi_native CLI (linked into CLI only)
    kernel.h                         Internal kernel contract
    registry.cpp                     Compiled kernel factories and stable IDs
    model_package.h / .cpp           Container parsing and validation
    plan_cache_file.h / .cpp         Shared .umplan envelope and atomic writes
  kernels/
    gaterv3/
      model_file.h                   GateRV3 payload decoder and validator
      runner.cpp                     GateRV3 adapter, execution, shape plans and graphs
      kernels.cu
      fp16.cuh
      plan_cache.h
      silu_plan.h
    gaterv2/
      runner.cpp
      kernels.cu
    moesr/
      runner.cpp
      kernels.cu
    gfisrv2/
      cufft_api.h
      runner.cpp
      kernels.cu
src/umzi_native/
  __init__.py                        Public Python imports
  _loader.py                         Platform library discovery and opt-in CUDA loading
  _abi.py                            ctypes bindings for umzi_session_*
  _session.py                        Generic NumPy inference session
scripts/
  build_native.py                    Standalone DLL/shared-library build
  export_native.py                   Select exporter and write one .ummodel
  exporters/
    package.py                       Shared .ummodel writer
    gaterv3.py                       GateRV3 checkpoint export
  reference/                         Torch models and checkpoint loaders
hatch_build.py                       Compile registered kernels and bundle one runtime
```

Name the library `umzi_native.dll` on Windows and `libumzi_native.so`
on Linux. Keep the active Windows verification scope. Retain the Linux
build path and check it before claiming Linux compatibility for the new
ABI. Each registered family provides its host sources and CUDA source to
the build hook. Generate an embedded image table per family so CUDA
symbols and image selections stay local to that family. Keep the CUDA 13
build, the `sm_86`, `sm_89` and `sm_120` cubins, and the CUDA 13.0
`compute_86` PTX fallback for GateRV3.

## Session API

Keep the Python call site familiar. The runtime reads the package header
to choose the model family. The kernel family supplies named tensor
descriptors and resolves the output shape for the supplied input shape.

```python
import torch
import umzi_native as umzi

options = umzi.SessionOptions(precision="fp16", cache_dir="cache/umzi")
with umzi.InferenceSession("models/restoration.ummodel", options) as session:
    input_name = session.get_inputs()[0].name
    (result,) = session.run(None, {input_name: tiles})
```

- Preserve `InferenceSession`, `SessionOptions`, `get_inputs()`,
  `get_outputs()`, `prepare()`, `run()` and the context manager.
  Return the kernel name, scale and context through session metadata.
  Preserve the GateRV3 names `input` and `output`. The adapter for
  `nao-manga-rls` can use the discovered scale and context.
- Give the C ABI an opaque `UmziSession`, a separate ABI version and
  size-tagged structs. Expose open/close, tensor descriptors, shape
  resolution, prepare, synchronous host inference and session info.
  Keep the existing asynchronous device path through a generic enqueue
  entry point so benchmarks and native clients can supply CUDA pointers
  and a stream. Describe the concrete input and output independently,
  including rank, dimensions, dtype and byte count. Validate the caller
  buffers and keep the current stream and lifetime rules.
- Scope the v1 contract to one image tensor in and one image tensor out.
  GateRV3 keeps its existing mixed and FP16 I/O behavior. Python allocates
  outputs from the kernel's resolved descriptor. C callers query the same
  descriptor to size their buffers. A future 4× model can return
  `(N, C, 4H, 4W)` when its model spec calls for it.
- Put device ID, precision, CUDA Graph and cache directory in the common
  session options. Let each registered family validate its own
  `kernel_options` and expose its own named diagnostics separately from
  shared timings and memory usage. Accept the existing GateRV3 flag fields
  as aliases during migration and reject conflicting or unknown options.
- Keep library lookup in explicit path, installed package and model
  directory order. Keep `import torch` and `load_cuda_libraries(...)`
  as the existing opt-in CUDA dependency paths.

### Internal kernel contract

A registry entry maps a stable family ID to a factory. The factory
validates its payload, advertises its supported precisions and tensor
metadata, and constructs a kernel session. A kernel session resolves I/O
for a shape, prepares execution, and runs host or device buffers against
that prepared shape. The generic session owns package dispatch and the
public lifetime. GateRV3 continues to own its schedules, padding, cuDNN
and cuBLASLt plans, fusions, allocations and graph execution inside its
module. Call the kernel interface at session and batch boundaries.

Keep one active shape in the first rework. Put persistent plan cache
entries directly under the caller's cache directory. The filename
includes the kernel name and cache key.

The manga benchmark measured the cost of switching between full and
tail batch shapes. Preserve the existing repeat-padded batch path for
the benchmark. A bounded in-memory shape cache can be a separate
performance change after the migration, with the full and tail shapes
sharing one copy of the weights and a measured VRAM cost.
[Manga tiling and batch benchmark](gaterv3-manga-tiling.md)

## Portable model package

Use extension `.ummodel`. The first eight bytes are the ASCII magic
`UMZINPKG`, with no terminator. Dispatch from the checked header.
Start the container at version `1`. The separate kernel schema version
allows a family to change the meaning of its payload while the container
and runtime ABI keep their own versions.

### Proposed v1 encoding

All integers are unsigned little-endian. Encoders and decoders use the
specified byte offsets.

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 8 | `UMZINPKG` |
| 8 | 4 | Container version |
| 12 | 4 | Kernel family ID. `1` is GateRV3, `2` is GFISRv2, `3` is GateRV2, `4` is MoESR, `5` is SMoSR, and `6` is FiGSR. |
| 16 | 4 | Kernel payload schema version |
| 20 | 4 | Section count |
| 24 | 8 | Total package size in bytes |
| 32 | 32 | SHA-256 over bytes 0–31 followed by the section table and payloads |

The section table begins at byte 64. Each 24-byte entry contains a
32-bit section ID, a 32-bit reserved field (zero in v1), a 64-bit file
offset and a 64-bit byte length. Entries have unique, increasing IDs.
Payloads are contiguous in table order and account for the full
remaining file. Require the architecture (1), execution data (2) and
weights (3) sections for GateRV3. Allow the original checkpoint SHA-256
as an optional 32-byte provenance section (4). Other families define
their required sections.

The shared reader checks file size, section count, offsets, ordering,
checksum and version before constructing the kernel session. The
kernel reader then validates descriptors, schedule or architecture
data, and weight ranges before GPU allocation. Report malformed
packages, unsupported container versions, absent kernel families and
unsupported payload schemas separately. The digest detects corruption.

The package holds architecture data and weights in one portable file.
The runtime bundles the executable CUDA code. CUDA Graphs, tuned
plans and device-specific artifacts are produced for the local device
and live outside the package.

## Persistent session plan cache

Use `$KERNEL_$HEX-KEY.umplan` inside the configured cache directory.
`$KERNEL` is the registered lowercase kernel name. `$HEX-KEY` is the
64-character lowercase hex encoding of the SHA-256 cache key. For example,
`gaterv3_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.umplan`.
The runtime path uses the trusted registry name. Existing `.gtplan`
entries are ignored and tuning produces fresh `.umplan` entries.

The cache stores the same kind of tuning choices as the current GateRV3
cache. Opening a session still allocates device buffers, restores or
tunes the library plans, and captures the CUDA Graph for the active
shape. CUDA Graph handles, stream handles, and device pointers live
only in the running process.

### Proposed v1 encoding

All integer fields are unsigned little-endian. The shared envelope
starts with the eight ASCII bytes `UMZIPLAN`.

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 8 | `UMZIPLAN` |
| 8 | 4 | Cache container version, initially `1` |
| 12 | 4 | Kernel family ID, shared with `.ummodel` |
| 16 | 4 | Kernel cache payload version, initially `1` for GateRV3 |
| 20 | 4 | Reserved, zero in v1 |
| 24 | 8 | Total file size in bytes |
| 32 | 32 | Raw SHA-256 cache key |
| 64 | 32 | SHA-256 over bytes 0–63 followed by the payload |
| 96 | Remaining bytes | Kernel-specific tuning payload |

The GateRV3 payload starts with three 32-bit fields in order: the GEMM
record count, the convolution record count, and the cuBLASLt algorithm
byte size. The GEMM records follow, then the convolution records.
Each table is sorted by its key and permits one record per key.

| Record | Fields in order |
| --- | --- |
| GEMM | Five 32-bit key fields (input channels, output channels, batch × output pixels, input leading dimension, bias present), 64-bit workspace bytes, then the opaque cuBLASLt algorithm bytes of the declared size |
| Convolution | Five 32-bit key fields (height, width, input channels, output channels, kernel width), 32-bit cuDNN algorithm, 32-bit math mode, 64-bit workspace bytes |

Write fields explicitly with no struct padding. Check the algorithm size
against the compiled type size and check the loaded cuBLASLt version
through the cache key before restoring these bytes. GateRV3 retains its
existing bound of 2,000 records per table and 256 KiB per file for v1.

Compute the key from a deterministic encoding of the cache and kernel
versions, the model digest and payload schema, the runtime build and
selected CUDA image, platform ABI, GPU identity, driver and library
versions, padded batch and tile shape, precision, relevant options, and
workspace limit. Include the family ID in the key so the same model
identity and shape in different families yield distinct entries.

The reader checks bounds, format, family, key, checksum and payload before
passing any choices to the kernel. GateRV3 revalidates restored choices
against the live cuBLASLt and cuDNN descriptors. A missing, incompatible
or corrupt entry triggers normal setup and tuning. Publish a successful
replacement atomically from a temporary file in the same directory.
If the directory cannot be read or written, report cache unavailability
and continue with inference. An unset cache directory disables disk caching.

## Implementation sequence

1. **Container and exporter.** Add the shared reader and writer and
   export GateRV3 directly from its checkpoint into `UMZINPKG` v1.
   Replace `.gtmodel` in the active runtime, scripts and documentation.
   Regenerate local model packages from the checkpoint. Keep the prior
   benchmark results as a performance control.
2. **Runtime and GateRV3 adapter.** Add the registry and generic C ABI.
   Route GateRV3 through it while preserving precision, dynamic shapes,
   and optimizations. Wrap the GateRV3 tuning payload in `.umplan` and
   migrate the cache checks to the new filename. Keep the old fixed-bundle ABI
   under legacy tooling while the existing benchmarks are migrated.
3. **Python and builds.** Bind the new ABI and switch model lookup and
   packaging to the generic library name. Migrate the primary exporter,
   benchmarks and installed-wheel check. Build the wheel and make
   `.ummodel` the documented inference path.
4. **Verify the boundary.** Add a test-only fake family with larger
   output dimensions to exercise registry dispatch and output allocation.
   Ship GateRV3 as the sole registered production family. Add GFISRv2
   in a later effort.

## Acceptance checks

- Export the GateRV3 checkpoint twice. Verify deterministic package
  output and byte parity for the execution data and weights against
  the frozen GateRV3 baseline. Test truncated and corrupt packages,
  invalid sections, unsupported schemas and absent family IDs.
- Verify cold-cache writes, warm-cache hits, kernel isolation and
  rebuilding after a corrupt or incompatible `.umplan`. Confirm that the
  filename key matches the header key and that changing a shape,
  precision or library version selects a different entry.
- Check mixed and full FP16 GateRV3 on batch 1, batch 16, an odd
  rectangular shape and a short final batch. Validate host/device
  output and graph/direct parity against the current runner under the
  same selected plans. Repeat with and without disk caching and cover
  recovery from a rejected allocation.
- Build/install the Windows wheel, test both explicit CUDA loading and
  PyTorch-first loading, and force PTX fallback again. Rerun the manga
  page benchmark against the frozen result with the same model payload
  and options. Compare batch-1 latency, page throughput and VRAM, with
  no material steady-state regression from dispatch and allocation
  changes. Run Ruff and Pyrefly.
