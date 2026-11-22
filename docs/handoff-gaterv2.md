# GateRV2 native integration handoff

Updated 2026-11-22. The GateRV2 FP32 package (family `3`, schema `1`) and its
C++ `KernelSession` now run through `umzi_native.InferenceSession` and pass
the shipping-path checks below. Pick up with FP16 execution, then profiling
and tuning. [Architecture expansion plan](umzi-native-architecture-expansion.md)

## What is ready

- The reference is the original Resselt 1.4.1 checkpoint in eager CUDA FP32
  with TF32 disabled. The checkpoint is
  `models/1x_umzi_dehalftone_gater2_v1_smooth.safetensors`, SHA-256
  `4330419414db1b5d16d58a682dde8e9378cb912e98f357c891b7a7183ec106e1`.
  It is RGB 1× with 32 base channels, encoder depths `(2, 2, 4, 8)`,
  decoder depths `(2, 2, 2, 2)`, 12 latent attention blocks, and 16-pixel alignment.
- The private runner assembles the full FP32 network and traces 44 stages per
  traced input. The CUDA kernels handle the shared GateRV3 RGB padding and crop,
  GateRV2 Q/K normalization, and the final attention update. Torch CUDA still
  supplies the convolutions, contractions, and remaining primitives. The
  packaged C++ session below replaces it for shipping. [Private runner](../scripts/reference/gaterv2_native.py) · [CUDA kernels](../native/kernels/gaterv2/kernels.cu)
- The default Inductor FP16 graph produced non-finite values in the twelfth
  latent block. On a black 128×128 tile, its attention contraction returned
  −64 at all 64 latent tokens, making the denominator zero. The corrected
  candidate promotes the Q/K/V projection outputs to FP32 for normalization,
  contractions, and division, then casts the attention result back to FP16.
  The projection convolutions and the rest of the network stay FP16. All four
  static graphs compiled with `fullgraph=True`, and both pages and all 12 stress
  cases returned finite outputs. [Mixed-precision attention](../scripts/reference/gaterv2.py) · [Failure analysis and measurements](umzi-native-architecture-baselines.md#gaterv2-inductor-numerical-failure)

## Measured controls

The two Watanare pages in `samples/halftone/` use 128×128 input tiles,
batch 16 and zero overlap. Batched time sums synchronized device-resident
forward calls. Batch-1 latency is measured separately for each of 630 real
tiles. Raw PSNR uses peak 1 before clipping and encoding. SSIMULACRA2 compares
the rendered PNGs with their matching eager FP32 PNGs using
`C:\Tools\imaging\ssimulacra2.exe`.

| Path | Batch-16 seconds | Input MPx/s | Mean batch-1 tile ms | Raw page PSNR dB | Page SSIMULACRA2 mean / min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager FP32 reference | 8.929 | 1.034 | 43.658 | — | — |
| Private FP32 runner | 8.661 | 1.066 | 40.495 | 148.833 | 93.133 / 93.100 |
| Inductor FP16 with FP32 attention | 7.565 | 1.221 | 21.625 | 73.754 | 90.440 / 90.186 |
| TensorRT-RTX FP16 | 2.338 | 3.949 | 4.880 | 69.730 | 90.361 / 90.058 |

| Comparison against eager FP32 | Watanare 085 SSIMULACRA2 | Watanare 086 SSIMULACRA2 | Checkerboard raw PSNR dB | Checkerboard SSIMULACRA2 |
| --- | ---: | ---: | ---: | ---: |
| Private FP32 runner | 93.100 | 93.167 | 117.920 | 98.988 |
| Inductor FP16 with FP32 attention | 90.695 | 90.186 | 35.727 | 95.891 |
| TensorRT-RTX FP16 | 90.665 | 90.058 | 35.072 | — |

Both private FP32 pages passed `atol=1e-5` and `rtol=5e-5`, with a maximum
raw page error of `8.05e-7`. The checkerboard was the hardest stress case at
`6.32e-6` max absolute error in the private runner. Keep the measured
SSIMULACRA2 scores alongside the raw metrics and the pixel tolerances. The
private FP32 renders differed from the references in 60 and 52 RGB samples
respectively, each by one 8-bit code value. [Per-page timings and parity](umzi-native-architecture-expansion.md#gaterv2-private-fp32-validation)

## Reproduce on Windows

Run from the repository root with the global Python, CUDA-enabled Torch, local
CUDA Python bindings, and CUDA 13.0 NVRTC. The private runner's `--nvrtc-root`
below points to the local CUDA 13.0 toolchain used for these runs. Substitute
the corresponding path if it is installed elsewhere.

```powershell
$py = 'C:\Python314\python.exe'
$env:TORCHINDUCTOR_COMPILE_THREADS = '1'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $PWD 'output\architecture-baselines\inductor-cache-single'

& $py scripts/benchmark_architectures.py gaterv2 fp32
& $py scripts/benchmark_architectures.py gaterv2 inductor-attention-fp32
& $py scripts/benchmark_architectures.py gaterv2 rtx
& $py scripts/validate_gaterv2.py --pages --nvrtc-root output/nvrtc-13.0/nvidia/cu13
```

The default Windows compile pool hit `pass_fds not supported on Windows`
with multiple workers. The one-thread configuration is verified. Keep the
FP32 report and raw outputs paired with the candidate. The benchmark checks
checkpoint, image hashes, and tiling before running a candidate.

| Local artifact under `output/` | Evidence |
| --- | --- |
| `architecture-baselines/gaterv2/batch-16/{fp32,inductor,inductor-attention-fp32,rtx}/report.json` | Reference, failed all-FP16 Inductor run, corrected Inductor run, and matched RTX run |
| `architecture-baselines/gaterv2-inductor-nonfinite.json` | First non-finite operation and its denominator |
| `gaterv2-native-validation/report.json` | Primitive checks, stage-by-stage FP32 deltas, all stress outputs, and per-page metrics |
| `gaterv2-gaterv3-regression/{digital,idc}-session.json` | Aligned and odd-rectangle session checks for the rebuilt Windows DLL |

## FP32 package and session

- **Payload.** Section 1 holds the architecture as sixteen 32-bit values.
  Section 2 is a 644-entry table of FP32 parameter offsets and counts.
  Section 3 holds the parameters in execution order, with Q/K/V concatenated
  per latent block. Section 4 holds the checkpoint digest. The
  [packer](../src/umzi_native/exporter/_gaterv2.py), the
  [reader](../src/umzi_native/_model.py), and the
  [session](../native/kernels/gaterv2/runner.cpp) derive the same parameter
  list. `python -m umzi_native export` and `read` handle GateRV2.
- **Session.** It uses NCHW FP32, with cuDNN `CUDNN_FMA_MATH` convolutions
  and no TF32. The [kernels](../native/kernels/gaterv2/kernels.cu) cover the
  remaining operations and compile without fast math. The session takes any
  batch and any valid padded shape. It supports CUDA Graph and direct
  execution, and host and device I/O. Only the default precision is
  accepted, kernel options are rejected, and no plan cache is written.
- **Build.** [The build hook](../hatch_build.py) embeds GateRV2 cubins and the
  PTX fallback next to GateRV3, with fast math disabled for GateRV2 only.
- **Checks.** [The session validator](../scripts/validate_gaterv2_session.py)
  covers deterministic export, ten rejection cases, 14 stress and shape cases
  in both execution modes, batch consistency, device enqueue, cleanup, and
  both pages. Both GateRV3 checkpoints re-exported byte for byte. They also
  passed the session checker bitwise in mixed precision and FP16.

| Path | Batch-16 seconds | Input MPx/s | Mean batch-1 tile ms | Raw page PSNR dB | Page SSIMULACRA2 mean / min |
| --- | ---: | ---: | ---: | ---: | ---: |
| C++ FP32 session (host I/O included) | 6.948 | 1.329 | 16.285 | 146.840 | 93.090 / 92.918 |
| Eager FP32, same run | 9.112 | 1.013 | 50.844 | — | — |

The checkerboard is the only case outside `atol=1e-5`, `rtol=5e-5`: two
values reach 1.88e-5. Its attention denominator falls to 1.45e-4, and eager
FP32 is itself 1.68e-5 from an FP64 run there. The session is 1.80e-5 from
FP64. The validator accepts such a case only within 1.5 times eager FP32's
FP64 error. [Session validation and measurements](umzi-native-architecture-expansion.md#gaterv2-fp32-package-and-session)

```powershell
& $py scripts/build_native.py --output output/gaterv2-session/native --arch sm_86 --nvrtc-root output/nvrtc-13.0/nvidia/cu13
$env:PYTHONPATH = 'src;scripts'
& $py scripts/validate_gaterv2_session.py --library output/gaterv2-session/native/umzi_native.dll --pages
```

## Where to pick up

1. **Add FP16 execution.** Follow the corrected Inductor candidate. Run the
   convolutions and elementwise work in FP16. Keep Q/K/V normalization, the
   contractions, and the denominator in FP32, because the all-FP16 path
   failed on flat tiles. Expose it as `UMZI_FP16` only after it passes the
   stress, page, and SSIMULACRA2 checks against eager FP32. Compare it with
   the Inductor and TensorRT-RTX FP16 rows above.
2. **Profile, then tune.** The session uses straightforward kernels: the
   inception and grouped convolutions, RMSNorm, and the attention reductions
   each use one thread per output. cuDNN chooses its first FMA-math
   algorithm without timing. Profile batch 16 and batch 1 before choosing
   what to fuse or replace. Add a `.umplan` cache only if timed algorithm
   selection pays for itself.
3. **Measure device-resident throughput.** The page timings include the RGB
   copies through host memory. Add a device-buffer timing path through
   `umzi_session_enqueue` for comparison with the eager and RTX rows.
4. **Keep the regression loop.** After native changes, rerun the GateRV2
   validator, both GateRV3 session checks, and the Linux build. The Linux
   build has not been run with GateRV2 yet.

After Python edits, run `ruff format scripts src`, `ruff check scripts src`,
and `pyrefly check` from the repository root.
