# GateRV2 native integration handoff

Updated 2026-11-22. GateRV2 packages (family `3`, schema `1`) run in the C++
`KernelSession` in mixed precision and FP16, like GateRV3. There is no FP32
execution mode. Both modes pass the shipping-path checks on Windows. The
Linux build matches Windows bitwise. Pick up with the next tuning pass. [Architecture expansion plan](umzi-native-architecture-expansion.md)

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

## Native package and session

- **Payload.** Section 1 holds the architecture as sixteen 32-bit values.
  Section 2 is a 644-entry table of FP32 parameter offsets and counts.
  Section 3 holds the parameters in execution order, with Q/K/V
  concatenated per latent block. Section 4 holds the checkpoint digest. The
  [packer](../src/umzi_native/exporter/_gaterv2.py), the
  [reader](../src/umzi_native/_model.py), and the
  [session](../native/kernels/gaterv2/runner.cpp) derive the same parameter
  list. `python -m umzi_native export` and `read` handle GateRV2.
- **Precision.** Both modes store FP16 NCHW features and run cuDNN
  convolutions on an FP16 filter copy. Mixed precision uses FP32 RGB I/O and
  FP32 convolution accumulation. FP16 uses FP16 RGB I/O and FP16
  convolution arithmetic. Q/K normalization, the contractions, the key sum,
  and the denominator stay FP32 in both modes, because all-FP16 attention
  failed on flat tiles. The FP32 parity reference stays in
  [the private runner](../scripts/reference/gaterv2_native.py).
- **Session.** It accepts any batch and any valid padded shape, with CUDA
  Graph or direct execution and host or device I/O. It rejects kernel
  options and writes no plan cache.
- **Build.** [The build hook](../hatch_build.py) embeds GateRV2 cubins and
  PTX, with fast math disabled for GateRV2 only. On Linux, NVRTC 13.0 needs
  `LD_LIBRARY_PATH` to include its wheel's `lib` directory.

| Path | Batch-16 seconds | Input MPx/s | Mean batch-1 tile ms | Raw page PSNR dB | Page SSIMULACRA2 mean / min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager FP32, same run | 9.015 | 1.024 | 47.220 | — | — |
| Native mixed | 3.633 | 2.542 | 7.716 | 74.067 | 90.578 / 90.367 |
| Native FP16 | 3.525 | 2.619 | 7.849 | 70.431 | 90.450 / 90.111 |

Native times are synchronized device-resident enqueue calls. The
checkerboard scores 44.681 dB in mixed precision and 32.923 dB in FP16. The
Inductor candidate scores 35.727 dB and RTX 35.072 dB on that case. All
stress outputs are finite, graph and direct execution match, and cleanup
returns all device memory. [Validation, profile, and Linux details](umzi-native-architecture-expansion.md#gaterv2-native-package-and-session)

```powershell
& $py scripts/build_native.py --output output/gaterv2-session/native --arch sm_86 --nvrtc-root output/nvrtc-13.0/nvidia/cu13
$env:PYTHONPATH = 'src;scripts'
& $py scripts/validate_gaterv2_session.py --library output/gaterv2-session/native/umzi_native.dll --pages
```

## Where to pick up

1. **Tune the custom kernels.** The grouped local convolution and gate
   (19.7% of batch-16 time) and the post-inception gate (9.7%) lead the
   profile. Try fusing the gate into the fc2 input, writing NHWC directly
   to avoid cuDNN's layout conversions, and timing cuDNN algorithm choices.
   A shared-memory tile for the local convolution was slower as written.
2. **Close the gap to TensorRT-RTX.** RTX FP16 runs 2.338 s and 4.880 ms
   against native FP16 at 3.525 s and 7.849 ms. Keep page SSIMULACRA2 at or
   above the RTX scores while tuning.
3. **Add a plan cache only if algorithm timing pays off.** Reuse the GateRV3
   `.umplan` format and keying.
4. **Keep the regression loop.** After native changes, rerun the GateRV2
   validator, both GateRV3 session checks, and the WSL comparison.

After Python edits, run `ruff format scripts src`, `ruff check scripts src`,
and `pyrefly check` from the repository root.
