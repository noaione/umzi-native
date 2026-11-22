# MoESR native integration handoff

Updated 2026-11-22. Covers the 4× MoESR checkpoint
`models/4x_dwtp_ds_moesr_v2.pth` (family `4`, payload schema `1`). For
design detail, see the [architecture expansion plan](umzi-native-architecture-expansion.md#moesr-native-package-and-session).

## Status

| Step | State |
| --- | --- |
| FP32 private parity (`scripts/validate_moesr.py`) | Done (committed in `bd53b8e`) |
| Packer `src/umzi_native/exporter/_moesr.py` and Python reader | Done, not committed |
| C++ session `native/kernels/moesr/runner.cpp` (mixed and FP16, graph replay) | Done, not committed |
| Session validation `scripts/validate_moesr_session.py` | Passed on the timed and cached build (2026-11-22) |
| Timed cuDNN plan selection and `workspace_mib` option | Done in this pass |
| Plan cache on disk (`cache_dir`, `moesr_<key>.umplan`) | Done in this pass and validated |
| Optimization pass 1: channel padding, divergence-free gate, backend plans, FP16 accumulation | Done and validated (see below) |
| Optimization pass 2: grouped gate layout, fused residual + LayerNorm, noise-based plan timing, 64-thread gate blocks | Done and validated, small gain (see below) |
| Further work to widen the lead over TensorRT-RTX | **Next** (see next steps) |
| Linux build and validation | Not started |

## Session changes in this pass

The first session build used the first cuDNN heuristic choice under a 64 MiB
workspace cap. For batch 2 at 256×256, that choice was a non-tensor-core
`convolve_common_engine_float_NHWC` kernel for the 64→340 expansion. It took
about 26 ms per call against about 1.4 ms for batch 1, which made batch 2
2.3× slower per tile than batch 1:

| Build | Mode | Batch 1 ms/tile | Batch 2 ms/tile | Batch 4 ms/tile |
| --- | --- | ---: | ---: | ---: |
| Heuristic, 64 MiB | Mixed | 278 | 643 | — |
| Heuristic, 64 MiB | FP16 | 225 | 477 | — |
| Timed, 64 MiB | Mixed | 286 | 717 | 916 |
| Timed, 64 MiB | FP16 | 227 | 479 | 646 |
| Timed, 512 MiB (new default) | Mixed | 278 | 282 | 291 |
| Timed, 512 MiB (new default) | FP16 | 227 | 228 | 230 |

These are quick device-resident random-tile probes (5 calls after a warm-up).
They are not the page benchmark. The fast implicit-GEMM plans need roughly
45 MB of workspace per 256×256 tile, so the cap was the real limit. Timing
alone did not fix it.

The runner now:

- times every heuristic candidate within the workspace cap on the real
  activation buffers, taking one run, then three more only for candidates
  within 2× of the best so far;
- grows tuning scratch on demand, skips candidates that run out of memory,
  and keeps only the selected requirement after prepare;
- accepts kernel option `workspace_mib` (0–4096, default 512) and rejects
  unknown or duplicate options;
- reports diagnostics `timed_plans` and `plan_workspace_bytes`.

Plan timing costs about 0.2–0.4 s in mixed and 1.1–4.4 s in FP16 per new
shape, because the FP16-accumulation candidates include slow kernels.

Timing is not deterministic across sessions. The first rerun of the
validator failed on `tiny-3x5` (mixed) with "graph and direct differ". The
graph and direct sessions had tuned separately and picked different
algorithms for near-tied candidates. Each output was valid, and repeats
within one session were bitwise equal. The session now has a disk plan
cache that reuses `runtime/plan_cache_file` and `common/cuda_plan_cache.h`,
as GateRV2 does:

- the key covers the plan format, family, plan schema, model digest, kernel
  image, precision, `workspace_mib`, device UUID, driver, cuDNN version, and
  padded shape;
- before trusting a saved choice, prepare checks its workspace size and runs
  it once. Otherwise it retimes and reports `rebuilt`;
- `cache_status` and `cache_message` are reported as in GateRV2.

The validator now gives the graph session, direct session, and CLI one
cache directory per precision (`<output>/plans/<precision>`). The direct
session must report `hit`. The new `check_plan_cache` also checks a cold miss
with timing, then a warm hit with `timed_plans == 0` and bitwise-equal
output. It confirms that `workspace_mib` separates the cache keys.

## Optimization pass 1 (2026-11-22)

Build: `output/moesr-session/opt-win`. Validation: `output/moesr-session/validated-opt/report.json`.
Device-resident probe results, in ms per 256×256 tile, on random tiles
(median of three five-call runs):

| Step | Mixed b1 | Mixed b2 | FP16 b1 | FP16 b2 |
| --- | ---: | ---: | ---: | ---: |
| Timed legacy plans (previous pass) | 246 | 248 | 197 | 199 |
| + channel padding and `half2` gate | 210 | 210 | 203 | 207 |
| + divergence-free gate thread order | 195 | 194 | 187 | 188 |
| + cuDNN backend engines (all heuristics A and fallback configurations) | 187 | 186 | 190 | 191 |
| + FP16 accumulation in FP16 mode (backend) | 187 | 190 | **144** | **134** |

What changed:

1. **Channel padding to multiples of 8.** The device copy of the weights is
   rebuilt at session open (`device_weights` in `runner.cpp`). The RGB input is
   packed to 8 channels, the 170-wide projection input to 176 channels, and
   the expansion output to 344 (340 real) channels with zero filters. This
   removed cuDNN's `nhwcAddPaddingKernel`, about 13% of the time. The package
   format is unchanged (still schema 1); only the plan-cache schema was bumped.
   Padding each 170-wide half of the expansion to 176 (352 total) made cuDNN
   choose much slower FP16 kernels, so only the tail is padded.
2. **Gate kernel.** `gate_full_nhwc` and `gate_reduced_nhwc` handle one
   `half2` channel pair per thread and write zero padding. Threads are ordered
   so that each warp runs only plain pairs, only 3×3, only 1×11, or only 11×1
   filters, with no divergence. `fast_mish` uses one `__expf`. The gate share
   fell from about 26% to 13%.
3. **cuDNN backend plans** (`native/kernels/moesr/conv_plan.h`). The legacy
   `cudnnConvolutionForward` API offered 3 algorithms per shape. The backend
   graph API offers 25–44 engine configurations, which are timed on the real
   buffers. Variant packs are cached for each call site. The plan cache stores
   the index of the chosen configuration in the deduplicated list with its
   workspace size, under marker `ConvChoice.math = 0x4d42`. A reused entry must
   still match the workspace size and run once.
4. **FP16 accumulation for `precision="fp16"`.** The best FP16-accumulation
   backend engine is 1.8× faster for the 64→344 expansion (2.5 ms against 4.6
   ms at batch 2). Fidelity matches the earlier FP16 build: pages scored
   59.4–60.8 dB, against TensorRT-RTX at 60.19 dB. Mixed keeps FP32
   accumulation and about 68 dB.

Profile, FP16 batch 2, 268 ms total: the 64→344 expansion takes 37%, the
gate 20%, the 176→64 projection 19%, `gated_residual` 5.5%, LayerNorm 4.3%,
and reduced-branch convolutions and kernels make up the rest. In mixed, the
64→344 expansion uses FP32 accumulation and takes 45% of 437 ms.

Dead ends recorded so it is not retried:

- **Padding the expansion to 352 (176 per half):** FP16 picked the
  `indexed_wo_smem` kernel at 5.6 ms. With the 344 layout this does not occur.
- **FP32 accumulation in FP16 mode with legacy plans:** it matched mixed
  speed, so it did not help. The backend's FP16 engines provided the gain.

## Optimization pass 2 (2026-11-22)

Build: `output/moesr-session/p2-win`. Validation: `output/moesr-session/validated-p2/report.json`
passed all stress, shape, CLI, alternate-stream, cache, and page checks. The
GateRV2 and both GateRV3 regression checks also passed on this DLL.

What changed:

1. **Grouped gate layout.** The expansion's device output channels are
   regrouped into blocks of eight, `[gate x4 | value x4]`, for four hidden
   slots. The 24 depthwise-branch channels come first. One gate thread reads
   a group with a single 16-byte load and writes 8 bytes. The projection's
   input channels are permuted to match the slot order (`hidden_slot` and
   `device_weights` in `runner.cpp`). The expansion is 344 wide (43 groups)
   at full resolution and 192 (24 groups) in the reduced branch.
2. **Fused residual + next LayerNorm** (`residual_norm_nhwc`). Inside each
   chain of gated blocks, the residual of block i and the LayerNorm of block
   i+1 run as one warp-per-pixel kernel, so the state is read once.
3. **Plan timing on random activations** (`fill_noise`) instead of zeros.
   On zero inputs, some tensor-core engines timed faster than they run on
   real data. Candidates within 2× of the best now get five timed runs.
4. **64-thread blocks for the gate kernels.** In an isolated probe this cut
   the batch-2 full-resolution gate from 1.37 to 1.17 ms. The depthwise
   windows stay in L1.

Measured gain, same process, alternating the pass-1 and pass-2 DLLs, batch
2, median ms per call: FP16 263.6 → 254.8 (−3.3%), mixed 379.2 → 367.2
(−3.2%). In the four-way rotated page benchmark, both DLLs used the same
package with timed plans:

| Build | Mode | Batch-2 inference s | Batch-2 input MPx/s | Batch-1 tile latency ms | Raw PSNR dB | SSIMULACRA2 mean / min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Pass 1 | FP16 | 9.463 | 0.3852 | 140.2 | 60.113 | 86.099 / 85.385 |
| **Pass 2** | **FP16** | **9.096** | **0.4007** | **130.6** | 59.993 | 86.094 / 85.420 |
| Pass 1 | Mixed | 13.641 | 0.2672 | 195.3 | 68.134 | 87.293 / 87.097 |
| Pass 2 | Mixed | 13.269 | 0.2747 | 184.8 | 68.133 | 87.263 / 87.095 |

Against TensorRT-RTX, in a three-round rotated run: native FP16 reached
0.3879 MPx/s with 133.8 ms batch-1 latency and 60.09 dB. RTX reached 0.2819
MPx/s with 141.8 ms and 60.19 dB, so native is 1.38× faster in throughput and
6% lower in latency. Native mixed reached 0.2739 MPx/s and 190.0 ms at
68.13 dB, 3% behind RTX in throughput. Reports:
`output/moesr-session/paired-p2-vs-p1/report.json` and
`output/moesr-session/paired-p2-rtx/report.json`.

**Measurement warning.** This laptop GPU runs at about 34 W and 70 °C under
load. Short probes on a cool GPU reported 111 ms per tile for pass 2. Under
sustained alternating load, the same kernels took 127 ms per tile. Trust
only same-process alternating A/B runs or the rotated page benchmark.
`scripts/ab_moesr_builds.py` alternates DLLs:

```powershell
$env:PYTHONPATH = 'src;scripts'
& C:\Python314\python.exe scripts/ab_moesr_builds.py --model output/moesr-session/validated-p2/moesr.ummodel p1=output/moesr-session/opt-win/umzi_native.dll p2=output/moesr-session/p2-win/umzi_native.dll
& C:\Python314\python.exe scripts/benchmark_gaterv2_backends.py --family moesr --library output/moesr-session/p2-win/umzi_native.dll --control-library output/moesr-session/opt-win/umzi_native.dll --model output/moesr-session/validated-p2/moesr.ummodel --output output/moesr-session/paired-p2-vs-p1
```

Batch-2 FP16 profile after pass 2, cool GPU, 220 ms: the 64→344 expansion
(cuDNN, FP16 accumulation) takes 38.5%, the 176→64 projection 20%, the
gate 19%, residual + norm 5.4%, and reduced-branch convolutions about 9%.
In mixed, the expansion takes 48%.

Dead ends in pass 2:

- **cuDNN fused graphs** (conv + bias, or conv + bias + Mish + scale + add)
  with FP16 compute: no engine supported them on this cuDNN/GPU. With FP32
  compute, conv + bias ran 4.76 ms against 4.37 ms for the plain
  convolution, slower than a separate bias. A Mish pointwise chain was not
  available in the frontend graph.
- **A 352-wide expansion** (FP16 accumulation, best backend engine): 2.27 ms
  against 2.10 ms for 344.
- **Pixel-major ordering of depthwise threads** for L1 reuse: the branch
  part of the gate slowed from 0.68 to 0.76 ms, so it was reverted.
- **Gate block sizes 128 and 512:** 1.44 and 1.31 ms, against 1.17 ms for 64.

## Page measurements

### Paired comparison with TensorRT-RTX after pass 1

The script `scripts/benchmark_gaterv2_backends.py --family moesr` ran three
rounds, rotating the backend order. Each backend used warm separate batch-1
and batch-2 sessions, with graph replay for native. RTX used the saved
fixed-shape graphs `t256-b1.onnx` and `t256-b2.onnx`. Values are medians
over rounds. PSNR and SSIMULACRA2 compare against the saved eager FP32 pages.

| Backend | Batch-2 inference s | Batch-2 input MPx/s | Batch-1 tile latency ms | Raw PSNR dB | SSIMULACRA2 mean / min |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Native FP16** | **9.594** | **0.3799** | **135.0** | 60.113 | 86.099 / 85.385 |
| Native mixed | 13.591 | 0.2682 | 190.7 | 68.134 | 87.293 / 87.097 |
| TensorRT-RTX FP16 | 12.899 | 0.2826 | 140.9 | 60.188 | 86.210 / 85.517 |

Native FP16 now has 1.34× the batch-2 throughput of TensorRT-RTX and 4%
lower batch-1 latency, at equal fidelity. Mixed is 5% behind RTX in
throughput but scores about 8 dB higher. RTX measured 0.306 MPx/s in the
earlier baseline run and 0.283 MPx/s here, so this GPU varies between runs
by about 8%; only same-session comparisons are reliable.
Report: `output/moesr-session/paired-opt/report.json`.

```powershell
$env:PYTHONPATH = 'src;scripts'
$b = 'output/moesr-session/opt-win'
& C:\Python314\python.exe scripts/build_native.py --output $b --nvrtc-root output/nvrtc-13.0/nvidia/cu13
& C:\Python314\python.exe scripts/validate_moesr_session.py --library "$b/umzi_native.dll" --cli "$b/umzi_native.exe" --pages --output output/moesr-session/validated-opt
& C:\Python314\python.exe scripts/benchmark_gaterv2_backends.py --family moesr --library "$b/umzi_native.dll" --model output/moesr-session/validated-opt/moesr.ummodel --output output/moesr-session/paired-opt
```

### Before pass 1

On an RTX 3050 Laptop GPU, the three `samples/mscale/` pages used
256×256 tiles, batch 2, no overlap, and graph replay. Per-page tables are in
the [plan](umzi-native-architecture-expansion.md#timed-and-cached-plans).

| Backend | Batch-2 input MPx/s | Batch-1 tile latency ms | Raw PSNR dB | SSIMULACRA2 mean / min |
| --- | ---: | ---: | ---: | ---: |
| Native mixed, timed (current) | 0.1790 | 282.2 | 68.136 | 87.278 / 87.131 |
| Native FP16, timed (current) | 0.2233 | 226.2 | 60.271 | 86.236 / 85.575 |
| Native mixed, heuristic (old) | 0.0838 | 263.6 | 68.131 | 87.268 / 87.109 |
| Native FP16, heuristic (old) | 0.1169 | 201.6 | 59.643 | 86.226 / 85.665 |
| Inductor FP16 | 0.201 | 258.6 | 67.263 | 87.287 / 87.058 |
| TensorRT-RTX FP16 | 0.306 | 125.8 | 60.188 | 86.210 / 85.517 |

The batch-1 plan sets match, yet latency is 7–12% higher than in the
heuristic run. That points to run-to-run variation on the laptop GPU. A
rotated comparison in step 4 should check it.
Report: `output/moesr-session/validated-tuned/report.json`. The GateRV2 and
GateRV3 regression reports are in the same folder.

## Batch-1 profile before pass 1 (mixed, heuristic build, 320 ms device time)

| Stage | Share |
| --- | ---: |
| 64→340 expansion convolutions (CUTLASS 128×128) | 36% |
| `gate_nhwc` (depthwise square/row/column + Mish gate) | 26% |
| 170→64 projection convolutions (CUTLASS 128×64) | 16% |
| cuDNN `nhwcAddPaddingKernel` (channel padding for 170 and 3 inputs) | 13% |
| Residual, LayerNorm, memset, and others | 9% |

## Next steps, in order

1. ~~Validate the timed and cached build.~~ Done.
2. ~~Remove channel padding, speed up the gate, and compare with RTX.~~ Done
   in pass 1. Native FP16 beats RTX.
3. ~~Commit~~ pass 1 (done by the user in `176c581`). **Pass 2 is
   uncommitted.** Validator follow-ups: add corrupt-file rebuild and
   `workspace_mib=0` checks, as in `validate_gaterv2_session.py`.
4. **Pass 3 candidates**, from the post-pass-2 profile:
   - **Custom fused expansion + gate kernel** (largest remaining win). The
     64→344 convolution writes 344 FP16 channels per pixel and the gate
     reads them back. A tensor-core implicit-GEMM kernel (WMMA/mma.sync,
     64 input channels × 9 taps) can apply bias and Mish in its epilogue and
     write only the 176 gated channels. The 24 depthwise channels need a 1-
     or 5-pixel halo, so either keep that part in a small second kernel or
     tile spatially with halos. Expect to save most of the gate's 19%, plus
     about half of the expansion's write traffic.
   - **Projection + residual + next norm:** a custom 176→64 convolution
     with a residual and LayerNorm epilogue would remove `residual_norm`
     (5%) and one state round-trip.
   - **Mixed mode:** the FP32-accumulation expansion takes 48%. Consider
     FP16 accumulation for the expansion only, measuring the fidelity cost
     against the 68 dB target, or a split-K custom kernel.
   - **Batch 4 and larger tiles:** throughput may scale better, so measure
     before writing kernels (`--batch` on the benchmark script).
5. Build and validate on Linux.
6. Rounding drift between engines: mixed `tiny-2x2` scored 92.8 dB at first,
   then 89.2 in pass 1 and 88.3 in pass 2. FP16 checkerboard scored 64.7,
   then 63.4 and 65.6. All remain far above the page scores (about 59–70 dB).
