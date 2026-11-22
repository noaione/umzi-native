# MoESR native integration handoff

Updated 2026-11-22. Covers the 4× MoESR checkpoint
`models/4x_dwtp_ds_moesr_v2.pth` (family `4`, payload schema `1`). For
design detail, see the [architecture expansion plan](umzi-native-architecture-expansion.md#moesr-native-package-and-session).

## Status

| Step | State |
| --- | --- |
| FP32 private parity (`scripts/validate_moesr.py`) | Done (committed in `bd53b8e`) |
| Packer `src/umzi_native/exporter/_moesr.py` and Python reader | Done (committed) |
| C++ session `native/kernels/moesr/runner.cpp` (mixed and FP16, graph replay) | Done (committed) |
| Session validation `scripts/validate_moesr_session.py` | Passed on pass-3 build, including corrupt-cache rebuild, `workspace_mib=0`, `fused_gate=0`, option rejections (2026-11-22) |
| Timed cuDNN plan selection and `workspace_mib` option | Done (committed) |
| Plan cache on disk (`cache_dir`, `moesr_<key>.umplan`) | Done (committed) |
| Optimization pass 1: channel padding, divergence-free gate, backend plans, FP16 accumulation | Done (committed in `176c581`) |
| Optimization pass 2: grouped gate layout, fused residual + LayerNorm, noise-based plan timing, 64-thread gate blocks | Done (committed in `3460ba5`) |
| Optimization pass 3: fused all-channel expansion + Mish gate (`expand_gate_*`), tiled depthwise branch (`gate_branch_tiled_*`), validator follow-ups | Done (committed in `dce9d94`) |
| Optimization pass 4: fused `176→64` / `96→64` projection + Mish + residual + LayerNorm (`project_*`), unified float-smem `gate_branch_tiled_*`, fused `input_bias_norm` / `reduce_norm` / `expand_residual_norm` / `expand_residual_add` | Done and validated, not committed (see below) |
| Linux build, wheel packaging (Windows & Linux), and cross-OS validation | Done and validated (2026-11-22, see below) |

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

**Reading SSIMULACRA2 on these pages.** On the manga pages the scores
saturate at about 86–87 for any image that is not bit-identical. In a check
on `lb-page-004.png`, the FP32 reference itself scored 87.03 after changing
just 0.1% of its 8-bit values by ±1, and 86.56 at 1%. Native mixed differs
in 3.2% of values (0.06% by more than 1 step), and FP16 and RTX in about 7.5–7.8%.
So 86–87 means "visually identical, a few rounding steps differ". The
differences between backends are within the metric's noise, and only a
bit-exact output scores 100. Use raw PSNR to rank fidelity.

## Optimization pass 3 (2026-11-22)

Build: `output/moesr-session/p3-win`. Validation: `output/moesr-session/validated-p3/report.json`
passed all package, stress, tiny/odd shape (`2×2` through `128×160`), CLI,
alternate-stream, host I/O, corrupt-cache rebuild, `workspace_mib=0`,
`fused_gate=0`, option-rejection, and page checks. The GateRV2 and both
GateRV3 regression checks (`gaterv2.json`, `gaterv3-digital.json`,
`gaterv3-idc.json`) also passed on this DLL.

What changed:

1. **Fused all-channel expansion + plain Mish gate** (`expand_gate_full_f16`,
   `expand_gate_full_f32`, `expand_gate_reduced_f16`, `expand_gate_reduced_f32`
   in `native/kernels/moesr/kernels.cu`). Enabled by default via kernel option
   `fused_gate=1` (`fused_gate=0` retains the pass-2 cuDNN expansion path for
   A/B checks). Each block owns an 8×16 pixel tile (10×18 halo = 22.5 KB,
   loaded once into shared memory and reused across all N-passes) with 8 warps
   (256 threads, ≤ 47.6 KB shared memory, 2 active blocks/SM on `sm_86`, zero
   local spills). Weights are pre-packed on the host (`pack_swizzle32_pass` in
   `runner.cpp`) into contiguous swizzle32 K=32 stages:
   - Full-res FP16 (`64→344`): 2 passes (`192 + 160 = 352` ch), 2-stage ring,
     `4×2` warps, double-buffered `k16` fragments.
   - Full-res Mixed (`64→344`, FP32 acc): 4 passes (`4×88 = 352` ch), 3-stage
     ring, `8×1` warps (124 regs, zero spills).
   - Reduced-res FP16 (`64→192`): 1 pass (`192` ch), 2-stage ring, `4×2` warps.
   - Reduced-res Mixed (`64→192`, FP32 acc): 2 passes (`2×96 = 192` ch),
     3-stage ring, `8×1` warps.
   Plain groups (37 full, 18 reduced) are placed at slot offset 0 (32-byte
   aligned), compute `fast_mish` across all 32 warp lanes, and store 16-byte
   `uint4` vectors directly to `gated`. The 48 depthwise-branch channels are
   written in planar-by-type layout to `projected` (three 8-channel value
   planes + one 24-channel gate plane), eliminating both the `64→344` and
   `64→192` cuDNN expansion plans, shrinking `projected` 5.4× (batch-2
   activations 117 MB, cuDNN workspace 0 B), and reducing tuned plans from 8 to 6.
2. **Tiled 16×16 depthwise branch + gate** (`gate_branch_tiled_full`,
   `gate_branch_tiled_reduced`). Loads each 8-channel value plane once per
   16×16 tile into shared memory via coalesced `uint4` loads, computes the
   3×3, 1×11, or 11×1 depthwise stencil and Mish gate, and writes `uint4`
   outputs to the branch slots (`152..175` full, `72..95` reduced). Cut
   branch-gate time by ~23% (0.37 → 0.28 ms/call).
3. **Validator follow-ups** (`scripts/validate_moesr_session.py`). Added
   `workspace_mib=0`, `fused_gate=0` agreement + key isolation, corrupt-file
   cache rebuild (`rebuilt` → `hit`), unavailable cache-directory fallback,
   and invalid kernel-option rejection checks.

Same-process alternating A/B (`scripts/ab_moesr_builds.py`, median ms/tile on
256×256 random tiles):

| Mode | Batch 1 (p2 → p3) | Batch 2 (p2 → p3) | Batch 4 (p2 → p3) |
| --- | ---: | ---: | ---: |
| **FP16** | 126.5 → **105.6** (−16.5%) | 129.6 → **99.8** (−22.9%) | 122.9 → **108.3** (−11.9%) |
| **Mixed** | 179.7 → **164.0** (−8.7%) | 181.5 → **167.0** (−8.0%) | 189.9 → **170.2** (−10.4%) |

Batch 4 did not beat batch 2 per tile on this 4 GB laptop GPU, so batch 2
remains the best operating point.

Four-round rotated page benchmark, pass 3 vs pass 2
(`output/moesr-session/paired-p3-vs-p2/report.json`):

| Build | Mode | Batch-2 inference s | Batch-2 input MPx/s | Batch-1 tile latency ms | Raw PSNR dB | SSIMULACRA2 mean / min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Pass 2 | FP16 | 9.183 | 0.3969 | 129.9 | 59.993 | 86.094 / 85.420 |
| **Pass 3** | **FP16** | **7.959** | **0.4582** | **115.2** | **60.185** | **86.302 / 85.685** |
| Pass 2 | Mixed | 13.105 | 0.2781 | 190.4 | 68.133 | 87.263 / 87.095 |
| **Pass 3** | **Mixed** | **12.704** | **0.2869** | **171.2** | 68.081 | 87.294 / 86.973 |

Three-round rotated page benchmark against TensorRT-RTX
(`output/moesr-session/paired-p3-rtx/report.json`):

| Backend | Batch-2 inference s | Batch-2 input MPx/s | Batch-1 tile latency ms | Raw PSNR dB | SSIMULACRA2 mean / min |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Native FP16 (Pass 3)** | **7.771** | **0.4691** | **109.1** | 60.060 | 86.111 / 85.398 |
| **Native Mixed (Pass 3)** | **11.915** | **0.3059** | 167.7 | **68.086** | **87.327 / 87.203** |
| TensorRT-RTX FP16 | 12.508 | 0.2914 | 134.9 | 60.188 | 86.210 / 85.517 |

Native FP16 is now **1.61× faster** than TensorRT-RTX FP16 in batch-2
throughput with **19.1% lower** batch-1 tile latency at matching fidelity.
Native Mixed (68.09 dB, +7.9 dB over RTX) now also beats TensorRT-RTX FP16 in
batch-2 throughput by **5.0%**.

Post-pass-3 batch-2 FP16 profile (195 ms total): `expand_gate_full_f16` 48%,
`176→64` projection (cuDNN) 22.5%, `residual_norm_nhwc` 6.1%,
`gate_branch_tiled_full` 5.1%, `expand_gate_reduced_f16` 5.1%, remaining
convolutions and elementwise kernels 13%.

Dead ends recorded in pass 3:

- **Large 8×64×64 16-warp single-block-per-SM tile** (100 KB shared memory)
  with 5 separate N-blocks and uncoalesced `half2` stores: 4.07 ms FP16 /
  5.71 ms FP32 (vs 2.14–2.42 ms FP16 / 4.07 ms FP32 for 8-warp 2-blocks/SM
  8×16 with in-block N-passes, pre-packed swizzle32 weights, and `uint4`
  epilogue stores).
- **Keeping `cuDNN 64→48` for the 48 depthwise-branch channels:** fusing all
  344 channels into `expand_gate_full_*` beat `cuDNN 64→48 + expand_gate(296)`
  while eliminating both expansion cuDNN plans and shrinking workspace to 0 B.

## Optimization pass 4 (2026-11-22)

Build: `output/moesr-session/p4-win`. Validation: `output/moesr-session/validated-p4/report.json`
passed all package, stress, tiny/odd shape (`2×2` through `128×160`), CLI,
alternate-stream, host I/O, corrupt-cache rebuild, `workspace_mib=0`,
`fused_gate=0`, option-rejection, and page checks. The GateRV2 and both
GateRV3 regression checks (`gaterv2.json`, `gaterv3-digital.json`,
`gaterv3-idc.json`) also passed on this DLL.

What changed:

1. **Fused projection + bias + Mish + gamma residual (+ next LayerNorm)**
   (`project_full_norm_f16`, `project_full_res_f16`, `project_full_norm_f32`,
   `project_full_res_f32`, and the four `project_reduced_*` counterparts in
   `native/kernels/moesr/kernels.cu`). Enabled alongside `expand_gate_*` under
   `fused_gate=1` (`plan_schema` bumped to `6`). Each 4-warp (128-thread) block
   computes a 16×16 pixel tile (18×18 halo = 324 pixels) across all 64 output
   channels, streaming `K=16` input slices and 3-tap rows of host-packed
   `swizzle16` weights (`pack_swizzle16_project` in `runner.cpp`, zero extra
   weight bytes, 33,024 B shared memory → 3 active blocks/SM in FP16, 2 in
   FP32, zero local spills). Because each warp holds all 64 output channels of
   its rows, the 4 lanes owning each pixel run LayerNorm with two quad-wide
   warp shuffles and zero extra DRAM round-trips. Eliminated all 36 `176→64`
   and 27 `96→64` cuDNN calls, all 45 `residual_norm_nhwc` calls, and all 18
   `gated_residual_nhwc` calls, leaving only 4 tiny cuDNN plans (`first`,
   `down`, `up`, `last`) and 0 B cuDNN workspace.
2. **Unified single-block float-shared-memory `gate_branch_tiled_*`**.
   Replaced the 3-grid (`gridDim.y = 3`) `half`-smem kernel with one 16×16
   block that reuses a single 13 KB `float` shared tile across square (3×3),
   row (1×11), and column (11×1), converts `half→float` and adds `expand_bias`
   once per loaded pixel (zeroing out-of-bounds for branch-free inner loops),
   and writes all 24 gated channels (48 B) to `gated` in one shot. Bitwise
   identical to the pass-3 branch kernel and 10–24% faster under sustained load.
3. **Fused chain-entry LayerNorm and final residual add** (`input_bias_norm`,
   `reduce_norm_nhwc`, `expand_residual_norm_nhwc`, `expand_residual_add_nhwc`).
   Eliminated all 18 standalone `norm_nhwc` launches and the final `add_nhwc`
   launch (bit-exact with the unfused ops).

Four-round rotated page benchmark, pass 4 vs pass 3
(`output/moesr-session/paired-p4-vs-p3/report.json`):

| Build | Mode | Batch-2 inference s | Batch-2 input MPx/s | Batch-1 tile latency ms | Raw PSNR dB | SSIMULACRA2 mean / min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Pass 3 | FP16 | 6.915 | 0.5271 | 97.9 | 60.060 | 86.111 / 85.398 |
| **Pass 4** | **FP16** | **6.168** | **0.5909 (+12.1%)** | **88.3 (−9.8%)** | **60.154** | 86.051 / 85.337 |
| Pass 3 | Mixed | 10.502 | 0.3471 | 150.1 | 68.086 | 87.327 / 87.203 |
| **Pass 4** | **Mixed** | **9.799** | **0.3720 (+7.2%)** | **138.5 (−7.7%)** | **68.215** | 87.287 / 87.152 |

Three-round rotated page benchmark against TensorRT-RTX
(`output/moesr-session/paired-p4-rtx/report.json`):

| Backend | Batch-2 inference s | Batch-2 input MPx/s | Batch-1 tile latency ms | Raw PSNR dB | SSIMULACRA2 mean / min |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Native FP16 (Pass 4)** | **6.778** | **0.5378** | **98.0** | 60.154 | 86.051 / 85.337 |
| **Native Mixed (Pass 4)** | **10.934** | **0.3334** | 154.5 | **68.215** | **87.287 / 87.152** |
| TensorRT-RTX FP16 | 12.441 | 0.2930 | 134.5 | 60.188 | 86.210 / 85.517 |

Native FP16 is now **1.84× faster** than TensorRT-RTX FP16 in batch-2
throughput with **27.1% lower** batch-1 tile latency (sub-100 ms/tile) at
matching fidelity. Native Mixed (68.22 dB, +8.0 dB over RTX) now beats
TensorRT-RTX FP16 in batch-2 throughput by **13.8%**.

### VRAM usage (256×256 tiles)

Runner-allocated device buffers (`buffer_bytes`) and total GPU memory delta
above the CUDA context (`startup_device_used_bytes["captured"] - ["context"]`),
from `output/moesr-session/validated-p4/report.json` against Pass 2
(`output/moesr-session/validated-p2/report.json`):

| Mode & Batch | Pass 2 buffers | Pass 4 buffers | Weights | Activations | Workspace | RGB I/O | GPU delta (`cuMemGetInfo`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 batch 1 | 129.5 MiB | **94.8 MiB** (−27%) | 32.5 MiB | 56.0 MiB | 0.0 MiB | 6.4 MiB | 96.0 MiB |
| FP16 batch 2 | 226.9 MiB | **157.2 MiB** (−31%) | 32.5 MiB | 112.0 MiB | 0.0 MiB | 12.8 MiB | 158.0 MiB |
| Mixed batch 1 | 135.9 MiB | **101.2 MiB** (−26%) | 32.5 MiB | 56.0 MiB | 0.0 MiB | 12.8 MiB | 102.0 MiB |
| Mixed batch 2 | 271.7 MiB | **170.0 MiB** (−37%) | 32.5 MiB | 112.0 MiB | 0.0 MiB | 25.5 MiB | 172.0 MiB |

Replacing the `64→344` cuDNN expansion shrunk `projected` 5.4× (86 → 16 MiB at
batch 2), and replacing both expansion and projection cuDNN plans dropped
retained workspace to 0 MiB (down from 32 MiB in Pass 2 mixed batch 2).

Dead ends recorded in pass 4:

- **`K=32` single-tap `176→64` projection** (54 barriers per call, 49.7 KB
  smem, 2 blocks/SM): 1.72 ms FP16 (only tied with cuDNN + `residual_norm`) vs
  `K=16` 3-tap 4-warp `16×16` (33 barriers, 33.0 KB smem, 3 blocks/SM) at
  1.23–1.41 ms FP16 (+21% faster than unfused).
- **Shared-memory coalesced epilogue on `project_*`**: slower than direct
  quad-shuffle LayerNorm + `half2` writes in both FP16 and FP32 (extra sync
  and smem round-trip).

## Wheel packaging and Linux validation (2026-11-22)

Both platform wheels were built from the pass-4 source tree with embedded
`sm_86`, `sm_89`, `sm_120`, and `compute_86` images:

- **Windows wheel:** `output/moesr-session/wheels-win/umzi_native-0.0.2-py3-none-win_amd64.whl`
  (2.53 MiB). Installed into an isolated target directory and verified via
  `scripts/check_python_session.py` (with `--import-torch` and `--cli`) across
  MoESR, GateRV2, GateRV3 digital, and GateRV3 IDC — all four reported
  `"problems": []` and loaded the bundled `umzi_native.dll` automatically.
  Reports: `output/moesr-session/validated-p4-win-wheel/`.
- **Linux wheel (WSL Ubuntu 24.04, Python 3.12, standalone non-Torch CUDA libs):**
  `output/moesr-session/wheels-linux/umzi_native-0.0.2-py3-none-linux_x86_64.whl`
  (2.57 MiB). Installed into an isolated target directory and verified via
  `scripts/check_python_session.py` (with `--cuda-dir` and the Linux CLI)
  across MoESR, GateRV2, GateRV3 digital, and GateRV3 IDC — all four reported
  `"problems": []`. Additional Linux MoESR checks passed on `rect-65x79`,
  `tiny-2x2`, `tiny-3x5`, and `tile-128` (graph == direct bitwise,
  `workspace_mib=0`, `fused_gate=0` agreement 70.37 dB mixed / 60.77 dB FP16,
  and corrupt plan-file `rebuilt → hit` recovery).
  Reports: `output/moesr-session/validated-p4-linux/` and build/check scripts
  `output/moesr-session/p4-linux-validate.sh`.
- **Cross-OS parity** on the same `2×3×65×79` input
  (`output/moesr-session/validated-p4-win-wheel/cross-os-parity.json`):
  - **FP16:** Windows and Linux outputs are **bitwise identical** (`max_abs = 0`);
    both score **61.422 dB** against eager CUDA FP32.
  - **Mixed:** Windows vs Linux scores **71.453 dB** (`max_abs = 1.71e-3`,
    differing only in cuDNN plan choice on the 4 remaining small convs);
    Windows vs FP32 scores **68.906 dB** and Linux vs FP32 scores **68.909 dB**.
- Note: `scripts/check_python_session.py` was updated to check output shapes
  using `session.scale_factor` (and pass `output_shape` to `cli_output`) so it
  works for non-1× models like MoESR.

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
3. ~~Commit pass 1 and pass 2; add validator follow-ups.~~ Pass 1 (`176c581`)
   and pass 2 (`3460ba5`) committed; validator follow-ups (`corrupt` rebuild,
   `workspace_mib=0`, `fused_gate=0`, option rejections) done in pass 3.
4. ~~Pass 3 & Pass 4: fused expansion + Mish gate, unified tiled depthwise
   branch, fused projection + Mish + residual + LayerNorm, fused chain-entry
   LayerNorm, and batch-4 scaling measurement.~~ Pass 3 committed in `dce9d94`;
   pass 4 done and validated (uncommitted). Native FP16 is **1.84× faster**
   than TensorRT-RTX FP16 (27.1% lower batch-1 latency), and Native Mixed
   (68.22 dB) beats TensorRT-RTX FP16 by **13.8%** in batch-2 throughput.
5. ~~Linux build, wheel packaging (Windows & Linux), and cross-OS parity.~~
   Done and validated on pass 4 (FP16 bitwise identical across Windows and
   Linux; mixed 68.91 dB vs eager FP32 on both).
6. **Commit pass 4**, then move on to GFISRv2 (family `2`) or physical
   `sm_89` / `sm_120` checks if available.
6. Rounding drift between engines: mixed `tiny-2x2` scored 92.8 dB at first,
   then 89.2 in pass 1, 88.3 in pass 2, 89.4 in pass 3, and 83.8 in pass 4.
   FP16 checkerboard scored 64.7, then 63.4, 65.6, 65.6, and 65.3. All remain
   far above the page scores (about 59–70 dB).
