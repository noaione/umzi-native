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
| Further kernel fusion to approach TensorRT-RTX | **Next** |
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
3. **Commit** the MoESR packer, session, backend plans, docs, and scripts.
   All of them are still uncommitted. Also add corrupt-file rebuild and
   `workspace_mib=0` checks to the validator, as in
   `validate_gaterv2_session.py`.
4. **Pass 2 candidates**, from the FP16 batch-2 profile (268 ms):
   - Fuse the expansion bias and Mish gate into the convolution with a
     cuDNN backend pointwise epilogue (conv → add bias), or use a
     CUTLASS-style custom expansion. This needs the depthwise branch to
     stay a separate pass. The expansion takes 37%.
   - Fuse LayerNorm into the previous `gated_residual` or `expand_residual`
     so the state is read once (LayerNorm 4.3% + residual 5.5%). A cuDNN
     backend graph conv → bias → Mish → scale → add could replace
     `gated_residual` for the projection.
   - Gate kernel (20%): stage the 11-tap row and column reads through
     shared memory per tile row.
   - Mixed mode: the FP32-accumulation expansion takes 45%. Check whether
     the backend offers a faster FP32-accumulation engine for a 336+8 split.
5. Build and validate on Linux.
6. Investigate why `tiny-2x2` dropped from 92.8 to 89.2 dB in mixed and why
   the FP16 checkerboard dropped from 64.7 to 63.4 dB. Both remain far above
   the page scores, but different engines round differently.
