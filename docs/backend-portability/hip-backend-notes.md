# Optional AMD HIP backend — evaluation notes

**Status:** Deferred. HIP is **not** required to implement Vulkan.  
**Companion:** [roadmap](README.md) · [architecture](architecture.md)

## Decision in one paragraph

Implement **Vulkan first** because it can be developed on the existing NVIDIA GPU, works across vendors, and can be checked on Apple Silicon through a Vulkan-on-Metal driver. Evaluate a **separate HIP backend** if remote AMD testing shows that a native HIP implementation could yield enough performance to justify a second codepath. HIP is not a compatibility layer for Vulkan shaders; HIP and Vulkan have different execution, memory, queue and synchronization APIs.

## What can be reused

| Existing component | HIP outlook |
| --- | --- |
| `.ummodel` parsing, model family schemas, binary readers, checkpoint weights | Reuse directly (GPU-independent) |
| Logical model schedules, shape validation and tensor liveness | Reuse algorithms; adapt resource management |
| CUDA kernel math and indexing | Often portable **with source changes** and HIP compilation |
| `half` / `half2`, shuffle/reduction semantics | Adapt, then verify wavefront-width and numerical behavior |
| cuBLASLt matmul | hipBLASLt where available and supported; not ABI/plan compatible |
| cuDNN convolutions | MIOpen where supported; different descriptors, heuristics, tuning and coverage |
| cuFFT R2C/C2R | hipFFT/rocFFT; verify transform semantics and support |
| CUDA Graph captures | HIP Graph behavior is not interchangeable; library support can vary |
| PTX `cp.async`, `ldmatrix`, `mma.sync`, CUDA WMMA | **Rewrite** using portable HIP, rocWMMA/MFMA or alternative kernels |
| CUDA cubin/PTX images, CUBLASLT plan cache | Not reusable |

**Important:** AMD ROCm's supported libraries and platforms evolve independently. In particular, do not assume that seeing a library named in general ROCm release notes guarantees it exists in a chosen Windows HIP SDK build or supports a specific Radeon SKU. Check the exact install distribution, GPU architecture and OS matrix at implementation time.

## When HIP would be justified

All of the following should have evidence:

- [ ] Vulkan path passes AMD numerical and full-image quality checks on actual hardware.
- [ ] Same-device AMD Vulkan benchmarks identify a meaningful bottleneck in conv/GEMM or FFT that HIP libraries are likely to improve.
- [ ] Supported AMD GPU/driver + ROCm/HIP library combination has been confirmed **for the exact target SKU/OS**.
- [ ] An isolated microbenchmark reproduces the potential improvement before a full family port.
- [ ] Added packaging/CI/support overhead is acceptable.

If the generic Vulkan backend is sufficiently fast, skip HIP entirely.

## Recommended HIP proof of concept (if needed)

1. Use Linux AMD ROCm first unless Windows support has been verified for each required library and GPU. Avoid conflating HIP SDK, ROCm Core SDK and library-specific support matrices.
2. Implement basic buffer upload, memory pool, kernel launch and synchronization for a synthetic operator.
3. Port one isolated convolution/GEMM from GateRV3; measure against Vulkan **on the same AMD hardware**.
4. Test numerical parity on flat, noise, checkerboard and real-page fixtures. Check wave32/wave64 subgroup/reduction assumptions.
5. Verify hipBLASLt, MIOpen and hipFFT individually (versions, architectures, supported dtypes/layouts, error handling) before selecting them for production.
6. Explore rocWMMA/MFMA only after basic library-backed performance is measured. Existing PTX tensor-core kernels do not compile into AMD instructions automatically.
7. Treat HIP Graph capture as optional until every invoked library call is confirmed graph-safe on the tested ROCm version.
8. Keep HIP plans/cache and device handles strictly separate from Vulkan and CUDA sessions.

## Testing without an AMD GPU locally

- Develop and validate Vulkan kernels on NVIDIA and supported Apple Silicon configurations.
- Write CPU-backed reference tests that compare shape arithmetic and packed weight generation.
- HIP syntax/build checks may catch source mistakes without a device, but **do not validate kernel execution, wavefront behavior, driver stability, or performance**.
- For AMD-specific claims, use a remotely provisioned Radeon/Instinct machine or a trusted hardware CI runner with suitable GPU access. Confirm access to the actual compute runtime/API you intend to test.
- Save device metadata, hardware architecture (`gfx*`), ROCm release, GPU driver, firmware, library versions and validation logs.

## Open questions for the future

- Is `hipBLASLt` supported and beneficial for the actual model shapes and AMD architecture?
- Would MIOpen or custom kernels be the better path for the many 1×1/3×3/depthwise convolutions?
- Does model performance depend more on library computation or on fused launch/graph scheduling?
- Are graph-capture limitations worth additional effort compared with ordinary HIP stream launches?
- Are separate backend wheels worth the distribution complexity, or is Vulkan-only sufficient for AMD users?

## Source pointers (reverify at implementation time)

- [ROCm release notes](https://rocmdocs.amd.com/en/develop/about/release-notes.html)
- [HIP SDK Windows component support](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html)
- [ROCm graph-safe library support](https://rocmdocs.amd.com/en/develop/reference/graph-safe-support.html)
- [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/)
- [MIOpen](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
- [hipFFT](https://rocm.docs.amd.com/projects/hipFFT/en/latest/)

**Do not open a separate HIP implementation effort until the Vulkan performance profile and a viable AMD test device are available.**
