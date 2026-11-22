# GateRV2 native integration handoff

Updated 2026-11-22. GateRV2 packages (family `3`, schema `1`) run in the C++
`KernelSession` in mixed precision and FP16. The session uses NHWC FP16
features, fused kernels, timed plans, and compact weights. The compact build
beat RTX in three paired page rounds on this workstation. The subsequent
32-channel inception specialization reduced median batch-16 time by 9.9%
in mixed and 6.6% in FP16. GateRV2 reuses tuned algorithms across sessions
when `SessionOptions.cache_dir` is set. [Architecture expansion plan](umzi-native-architecture-expansion.md)

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

- **Payload.** Family `3`, schema `1`. Section 1: sixteen 32-bit
  architecture values. Section 2: 644 FP32 parameter offsets and counts.
  Section 3: FP32 parameters in execution order, Q/K/V concatenated per latent
  block. Section 4: checkpoint digest. The
  [packer](../src/umzi_native/exporter/_gaterv2.py), the
  [reader](../src/umzi_native/_model.py), and the
  [session](../native/kernels/gaterv2/runner.cpp) derive the same list.
- **Precision.** Mixed (FP32 RGB I/O, FP32 accumulation) and FP16 only, like
  GateRV3. Features are NHWC FP16 in both. Q/K/V projection output,
  normalization, contractions, key sum and denominator stay FP32 in both
  modes; all-FP16 attention failed on flat tiles. The FP32 parity reference
  stays in [the private runner](../scripts/reference/gaterv2_native.py).
- **Kernel options:**
  - `tune_plans` (0/1, default 1): time cuBLASLt and cuDNN candidates at
    prepare, or take the first heuristic plan.
  - `workspace_mib` (0–1024, default 64): limit for candidate workspace.
    Prepare allocates it temporarily and keeps only what the chosen plans use.

## Improvements made (in order)

| Step | Change | Effect (RTX 3050 Laptop) |
| --- | --- | --- |
| 1 | FP32 C++ port of the private runner (NCHW, cuDNN FMA, no fast math) | Bitwise-class parity: pages 146.8 dB vs eager FP32 |
| 2 | Mixed and FP16 replace FP32: FP16 NCHW features, FP16 cuDNN filters, FP32 attention | Batch-16 pages 3.63 s / 3.53 s, batch-1 7.7 / 7.8 ms |
| 3 | Biases fused into consumers, channel-parallel RMSNorm, warp-parallel K·V | Batch-16 forward 114.9 → 89.4 ms |
| 4 | NHWC rewrite: 1×1 convs as cuBLASLt GEMMs with fused bias; timed cuBLASLt/cuDNN plan selection; RMSNorm fused into residual/Mish/unshuffle; local 3×3 fused with average-pool sums; inception fused with gate; 3-kernel linear attention; input/output convs fused with pad, crop and RGB residual | ~430 → 290 kernels; batch-16 forward 89 → ~59 ms |
| 5 | local_gate: register weights, 4-row column reuse; pool_project parallel over outputs; attention_prepare split over value slices | Batch-16 ~52 ms, batch-1 ~4.3 ms |
| 6 | inception_gate: one kernel kind per 8-channel vector, mixed vectors handled per channel | Batch-16 ~49 ms |
| 7 | VRAM: FP32 copy only for small kernel parameters (3.9 MB), FP16 for projections/filters (59 MB); tuning workspace freed after prepare; work buffers sized per role (3 instead of 4); decoder state reuses the consumed skip buffer | Batch 16: 524 → 232 MiB; batch 1: 278 → 92 MiB; speed unchanged in an interleaved A/B |
| 8 | Specialize the 32-channel inception gate at 128×128 with fixed four-channel branch loads | About 56% less GPU time in the 32-channel gate at batch 16 and batch 1, in both precisions |

Tried and dropped: shared-memory tile for local_gate (slower), shared-memory
halo for inception_gate (no gain), cheaper Mish formula (no gain).

## Latest measurements

Pages: 128×128 tiles, batch 16, device-resident enqueue, scored against eager
FP32. The first set of numbers was measured right after step 6; the second
after step 7 on the same build of step 7. Run-to-run timing on this laptop
varies by about 10–15% (GPU at 66–69 °C, no throttle flags), so compare
interleaved runs, not separate sessions.

| Path | Batch-16 s | MPx/s | Batch-1 tile ms | Raw PSNR dB | SSIMULACRA2 mean / min |
| --- | ---: | ---: | ---: | ---: | ---: |
| TensorRT-RTX FP16 (baseline run) | 2.338 | 3.949 | 4.880 | 69.730 | 90.361 / 90.058 |
| Native mixed, after step 6 | 1.986 | 4.650 | 4.284 | 77.922 | 90.569 / 90.374 |
| Native FP16, after step 6 | 1.856 | 4.976 | 4.089 | 73.846 | 90.436 / 90.131 |
| Native mixed, after step 7 | 2.314 | 3.990 | 4.905 | 77.922 | 90.569 / 90.374 |
| Native FP16, after step 7 | 2.150 | 4.295 | 4.717 | 73.846 | 90.436 / 90.131 |

The step-7 run was slower, but an interleaved A/B of the step-6 commit
(`555af25`) and step 7 gave the same medians (batch 16: 57.8 vs 57.8 ms
mixed, 54.0 vs 54.1 ms FP16; batch 1: 5.0 vs 5.0 ms mixed, 4.9 vs 4.8 ms FP16).
The difference is the machine state, not the VRAM change. TensorRT-RTX was
re-measured shortly after the step-7 run, in the same slower state
(`output/gaterv2-session/rtx-rerun/gaterv2/batch-16/rtx/report.json`):
1.243 s and 1.258 s per page at batch 16 (2.50 s total), and 5.179 and
5.193 ms per batch-1 tile. Against that rerun, native mixed is about 7%
faster at batch 16 and 6% at batch 1, and native FP16 about 14% and 9%. The
margin was confirmed in the rotated runs below.

### Paired step-7 comparison

The three back-to-back rounds rotate backend order on each page. Each
backend keeps its own warmed batch-16 and batch-1 sessions. The report
contains every tile latency and the per-page SSIMULACRA2 scores. Both RTX
graphs were profiled with exclusive TensorRT-RTX placement.

| Backend | Median batch-16 seconds | Input MPx/s | Median mean batch-1 tile ms | Raw page PSNR dB | Page SSIMULACRA2 mean / min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native mixed | 2.122 | 4.352 | 4.531 | 77.922 | 90.569 / 90.374 |
| Native FP16 | 1.992 | 4.636 | 4.334 | 73.846 | 90.436 / 90.131 |
| TensorRT-RTX FP16 | 2.323 | 3.975 | 4.931 | 69.730 | 90.361 / 90.058 |

Native mixed cut median inference time by 8.7% at batch 16 and 8.1% at
batch 1. Native FP16 cut it by 14.3% and 12.1%. Both were faster on both
measures in all three rounds. [Paired benchmark script](../scripts/benchmark_gaterv2_backends.py) · [Local report](../output/gaterv2-session/step7-paired/report.json)

VRAM (device-wide free-memory delta, excluding context and caller I/O):

| Path | Batch 16 | Batch 1 |
| --- | ---: | ---: |
| Native mixed, before step 7 | 524 MiB | 278 MiB |
| Native mixed, after step 7 | 232 MiB | 92 MiB |
| Native FP16, after step 7 | 237 MiB | 96 MiB |
| Native mixed, `workspace_mib=0` | 230 MiB | 86 MiB, but batch 16 is ~50% slower |
| TensorRT-RTX FP16 | 242 MiB | 108 MiB |
| Eager FP32 | 756 MiB | 188 MiB |

`tune_plans=0` halves prepare at batch 16 (818 → 406 ms) and costs ~3% speed.
FP16 batch-16 prepare took 3.8 s with timing.

Checks after step 7 (all pass): deterministic export; 11 rejections including
the three option errors; 14 stress cases finite in both precisions;
repeated graph and direct runs bitwise stable; direct, graph and batch-1
paths within 3 dB of the graph run's PSNR against eager FP32 (FP16 flat tiles
vary 1.3 dB between plans); host equals device enqueue; all three option sets
above 73 dB on a 65×79 batch; cleanup returns all device memory. ruff,
ruff format and pyrefly are clean.

## Step-7 verification

- The rebuilt Windows library passed both GateRV3 re-exports and session
  checks, including byte-identical packages, CLI equality, and plan-cache
  hits. GateRV2 passed the packaged validator with page metrics and the
  follow-up assertions for heuristic tuning and zero workspace.
- The full WSL Ubuntu build embedded sm_86, sm_89, sm_120 and compute_86
  for both families. On the same 2×3×65×79 input, Windows and Linux
  scored 80.906 dB against each other in mixed and 75.793 dB in FP16.
  Outputs were finite, with respective maxima of `5.88e-4` and
  `7.32e-4` between platforms. Both were within 0.03 dB of each other
  against the same eager FP32 output. The WSL recipe is in
  `output/gaterv2-session/step7-build.sh`. The manylinux NVRTC needs
  `LD_LIBRARY_PATH` to include the wheel's `lib` directory.
- Scratch scripts in `output/gaterv2-session/`: `smoke.py` (quick parity),
  `bench.py [dll]`, `profile.py`, `levels.py`, `ab.py` (interleaved A/B,
  expects `ab/native` from a worktree build), `vram.py <precision> <batch> <json options>`.

```powershell
& $py scripts/build_native.py --output output/gaterv2-session/native --arch sm_86 --nvrtc-root output/nvrtc-13.0/nvidia/cu13
$env:PYTHONPATH = 'src;scripts'
& $py scripts/validate_gaterv2_session.py --library output/gaterv2-session/native/umzi_native.dll --pages
```

## 32-channel inception gate

On the RTX 3050 Laptop, the 32-channel level accounted for roughly 60% of
inception gate GPU time on 128×128 inputs. The table gives the combined
kernel time for the four full-resolution gates in milliseconds per forward,
averaged over eight warmed direct calls with heuristic plans.

| Precision | Batch | 32-channel control ms | 32-channel specialized ms |
| --- | ---: | ---: | ---: |
| Mixed | 16 | 7.092 | 3.142 |
| FP16 | 16 | 6.375 | 2.825 |
| Mixed | 1 | 0.385 | 0.168 |
| FP16 | 1 | 0.390 | 0.170 |

The specialization uses the fixed channel layout to load only the four
features needed for each depthwise square, row, and column branch. It
preserves the FMA order, zero padding, and FP32 Mish. The runner uses the
generic kernel for other channel widths and padded spatial shapes. Noise
and checkerboard outputs matched the control bit for bit in direct
execution at both batch sizes and precisions.
[Local profile](../output/gaterv2-session/inception-fixed32-profile.json)

### Paired page results

Four Windows rounds rotated the order of both builds and both precisions
on each page. The control DLL was built from `ce5410c`. Both builds used
heuristic plans to control for tuning differences, with separate warmed
CUDA Graph sessions for batch 16 and batch 1. Each build and precision had
40 batch-16 calls and 630 individual synchronized batch-1 calls per round.
The table shows medians over the four rounds.

| Precision | Batch-16 seconds, control → specialized | Input MPx/s, control → specialized | Mean batch-1 tile ms, control → specialized |
| --- | ---: | ---: | ---: |
| Mixed | 2.249 → 2.026 | 4.105 → 4.558 | 4.711 → 4.397 |
| FP16 | 2.010 → 1.877 | 4.593 → 4.920 | 4.478 → 4.208 |

Batch-16 time fell by 9.9% in mixed and 6.6% in FP16. Mean batch-1 latency
fell by 6.7% and 6.0%. All four batch-16 rounds improved in each precision.
Mixed batch-1 latency rose by 0.4% in one round and fell in the other three.

| Precision | Page | Batch-16 seconds, control → specialized | Input MPx/s, control → specialized | Mean batch-1 tile ms, control → specialized | Raw PSNR dB | SSIMULACRA2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mixed | 085 | 1.088 → 1.014 | 4.245 → 4.553 | 4.743 → 4.382 | 77.571 | 90.798 |
| Mixed | 086 | 1.125 → 1.004 | 4.105 → 4.600 | 4.574 → 4.340 | 78.301 | 90.285 |
| FP16 | 085 | 1.002 → 0.955 | 4.610 → 4.838 | 4.475 → 4.209 | 73.545 | 90.658 |
| FP16 | 086 | 1.001 → 0.919 | 4.613 → 5.022 | 4.422 → 4.163 | 74.166 | 90.164 |

Both pages matched the control bit for bit within each precision. Against
eager CUDA FP32, pooled raw PSNR was 77.920 dB for mixed and 73.845 dB for
FP16. SSIMULACRA2 mean / minimum was 90.541 / 90.285 and 90.411 / 90.164,
respectively. The report includes DLL and input hashes plus every timed
tile's image, index, coordinates, and elapsed time.
[Local paired report](../output/gaterv2-session/inception-fixed32-paired/report.json)

The packaged validator passed all 15 stress inputs in both precisions,
including flat, tiny, and rectangular inputs. It also passed export,
rejection, host I/O, graph and direct execution, plan-cache, and cleanup
checks. The table gives the stress scores against eager FP32 with the
default tuned plans.

| Input | Mixed raw PSNR dB / SSIMULACRA2 | FP16 raw PSNR dB / SSIMULACRA2 |
| --- | ---: | ---: |
| Checkerboard | 35.460 / 95.338 | 34.542 / 95.050 |
| Noise | 74.895 / 97.741 | 73.193 / 97.620 |

[Local validator report](../output/gaterv2-session/inception-fixed32-validation/report.json)

The two GateRV3 fixtures passed aligned and odd-rectangle session checks,
including cache reopening and bitwise CLI equality. A separate three-round
comparison with default tuned plans confirmed exclusive TensorRT-RTX node
placement for both batch sizes. Median batch-16 totals were 1.936 s for
native mixed, 1.772 s for native FP16, and 2.264 s for TensorRT-RTX FP16.
Median mean batch-1 tile latencies were 4.170, 4.242, and 4.801 ms.
[Local TensorRT-RTX comparison](../output/gaterv2-session/inception-fixed32-rtx/report.json)

```powershell
$py = 'C:\Python314\python.exe'
$env:PYTHONPATH = 'src;scripts'
& $py scripts/benchmark_gaterv2_backends.py `
  --control-library output/gaterv2-session/cache-final-win/umzi_native.dll `
  --library output/gaterv2-session/inception-fixed32-win/umzi_native.dll `
  --model output/gaterv2-session/cache-final-verified/gaterv2.ummodel `
  --output output/gaterv2-session/inception-fixed32-paired
```

## Where to pick up

The [GateRV2 plan-cache checks](umzi-native-architecture-expansion.md#gaterv2-plan-cache)
record prepare latency, key isolation, and recovery from corrupt or
unavailable cache files. Graph and direct stress runs use the same cached
plans so the dispatch comparison excludes differences from tuning.

1. **Next speed work.** In the specialized build's mixed batch-16 profile,
   the local gate accounts for 19.4% of GPU kernel time and the two residual
   RMSNorm kernels together account for 16.3%. Check whether the local gate
   output can be written directly as the next norm input.
2. **Further VRAM:** skips for all levels are kept (about 45 MiB at batch
   16); a lifetime planner like GateRV3's could share them with work buffers.

After Python edits, run `ruff format scripts src`, `ruff check scripts src`,
and `pyrefly check` from the repository root.
