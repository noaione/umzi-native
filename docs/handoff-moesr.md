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
| Kernel fusion and tuning to approach TensorRT-RTX | **Next** |
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

## Page measurements

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

## Batch-1 profile (mixed, heuristic build, 320 ms device time)

| Stage | Share |
| --- | ---: |
| 64→340 expansion convolutions (CUTLASS 128×128) | 36% |
| `gate_nhwc` (depthwise square/row/column + Mish gate) | 26% |
| 170→64 projection convolutions (CUTLASS 128×64) | 16% |
| cuDNN `nhwcAddPaddingKernel` (channel padding for 170 and 3 inputs) | 13% |
| Residual, LayerNorm, memset, and others | 9% |

## Next steps, in order

1. ~~Validate the timed and cached build.~~ Done 2026-11-22. Every check
   passed, and the GateRV2 and GateRV3 regressions also passed. To reproduce:

   ```powershell
   $env:PYTHONPATH = 'src;scripts'
   $b = 'output/moesr-session/tuned-win'
   & C:\Python314\python.exe scripts/build_native.py --output $b --nvrtc-root output/nvrtc-13.0/nvidia/cu13
   & C:\Python314\python.exe scripts/validate_moesr_session.py --library "$b/umzi_native.dll" --cli "$b/umzi_native.exe" --pages --output output/moesr-session/validated-tuned
   ```

2. **Commit** the MoESR packer, session, docs, and scripts (all uncommitted).
   Also add corrupt-file rebuild and `workspace_mib=0` checks to the
   validator, as in `validate_gaterv2_session.py`.
3. **Remove the channel-padding kernel.** This is the next optimization. Pad the 170-wide gate output and
   projection filters to 176 channels, which is a multiple of 8, in the packer.
   Pad the RGB input to 8 channels in `pack_rgb`. That removes
   `nhwcAddPaddingKernel` (13%). This needs a schema bump or a packed-layout
   flag.
   Then **speed up `gate_nhwc`**. Read each pixel's channel vector once with
   `half2` loads, tile the 11-tap row/column filters through shared memory,
   and consider fusing LayerNorm into the previous residual kernel.
4. Compare against TensorRT-RTX again, rotating the backend order as in
   the GateRV2 handoff. Re-profile batch 2 with the timed plans first,
   because the profile above covers batch 1 on the heuristic build.
5. Build and validate on Linux.
