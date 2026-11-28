# MoSR handoff

Status: Exporter, package schema, native CUDA/C++ runtime (`native/kernels/mosr/`), validation, and baseline benchmarks (`fp32`, `rtx`, `native-fp32`, `native-mixed`, `native-fp16`) are complete across `mosr`, `mosr-l`, and `mosr-t`. TorchInductor FP16 (`inductor`) hit `WinError 623` inside the Windows AppContainer sandbox when Triton launched `tcc.exe` and remains pending outside the sandbox.

## Workload

- Main checkpoint: `models/4x_umzi_digital_art_mosr_v2.pth` (family `mosr`).
- 4x large checkpoint, used for validation: `models/4x_umzi_digital_art_mosr_l.pth`
  (family `mosr-l`).
- 2x checkpoint, checked for inference and baseline parity:
  `models/2x_umzi_digital_art_mosr_t.pth` (family `mosr-t`).
- Images: the four 900×1350 JPEGs in `samples/4x-test`.
- Batch 1, 512×512 input tiles, zero overlap, replicate padding on edge tiles,
  six tiles per image. The renders are cropped to scale × the source size.
- Native targets: mixed, FP16, and FP32. MoSR is the only family in this
  round that targets native FP32.

## Checkpoint and architecture

Resselt 1.4.1 detects all three files as `mosr`. All three store FP32 weights.

| Checkpoint | Scale | Blocks | Dim | Upsampler | Kernel | Expansion | Parameters |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `4x_umzi_digital_art_mosr_v2.pth` | 4 | 24 | 64 | `gps` | 7 | 1.5 | 4,481,472 |
| `4x_umzi_digital_art_mosr_l.pth` | 4 | 24 | 64 | `gps` | 17 | 1.5 | 4,850,112 |
| `2x_umzi_digital_art_mosr_t.pth` | 2 | 5 | 48 | `dys` | 7 | 1.5 | 592,355 |

MoSR L differs from v2 only in the depthwise kernel: 17×17 instead of 7×7. The
native kernel takes `blocks`, `dim`, `hidden`, `kernel_size`, `scale`,
`upsampler` (`ps`, `gps`, `dys`), and `dysample_groups` from package section 1
rather than hardcoding them.

`conv_ratio` is 1.0 and in/out channels are 3 for all three checkpoints. The body
consists of a 3×3 stem, `blocks` `GatedCNNBlock`s (channel LayerNorm, 3×3
expansion to `2 * hidden`, depthwise `K×K` convolution on `dim` channels, Mish
gate, 3×3 projection + Mish, and `shortcut - 0.5` residual), a 3-conv body tail
(`dim -> 2*dim -> dim -> dim`), and a `ConvBlock` global shortcut added as
`gblocks(x) + (shortcut(x) - 0.5)`. The 4x tail `gps` averages 8
geometric-ensemble predictions before pixel shuffle; the 2x tail `dys` uses
DySample (`offset`, `scope`, bilinear `grid_sample`, `end_conv`).

## Package and native design

- **Package format:** Family `7`, schema `1`. Section 1 packs 10 `uint32`s
  (`in_ch, out_ch, scale, blocks, dim, hidden, kernel_size, upsampler, dysample_groups, flags`).
  Section 3 stores all weights and biases in FP32 (OIHW for convolutions) on
  256-byte boundaries. Section 4 stores the source checkpoint SHA-256.
- **Native FP32 (`precision="fp32"`):** NCHW layout, strict FP32 cuDNN plans
  (filtering out Tensor Core and reduced-precision engines), FP32 pointwise
  LayerNorm/Mish/depthwise/upsample kernels, and FP32 RGB I/O.
- **Native mixed (`precision="mixed"`):** NHWC FP16 feature activations and
  weights with FP32 convolution accumulation, FP32 pointwise math/norms/biases,
  and FP32 RGB I/O.
- **Native FP16 (`precision="fp16"`):** NHWC FP16 feature activations and
  weights with FP16 convolution accumulation (per-shape fallback to FP32
  accumulation only if no half-accumulation plan exists), FP32 pointwise math,
  and FP16 RGB I/O.
- **Options and caching:** Supports `fused_gate` (0 or 1, default 1; enables
  Tensor Core fused `expand_gate` / `project_res_norm` and coalesced planar
  depthwise gating for `(dim=64, hidden=96)` or `(dim=48, hidden=72)` with
  `kernel_size=7` or `17` in mixed and FP16), `workspace_mib` (0–4096, default 512),
  and disk plan caching (`mosr_<key>.umplan` under `cache_dir`, `plan_schema=3`).
  Reports diagnostics `timed_plans`, `plan_workspace_bytes`,
  `peak_plan_workspace_bytes`, and `fp16_fallback_plans`.

## Baseline measurements

Measured on an NVIDIA GeForce RTX 3050 Laptop GPU (4 GiB), driver 610.62,
Python 3.14.5, Torch 2.15.0+cu132, cuDNN 9.26, Resselt 1.4.1, ONNX Runtime
TensorRT-RTX provider 0.5.0. Reports, raw-float `.npy` outputs, and PNG renders
are in `output/architecture-baselines/{mosr,mosr-l,mosr-t}/batch-1/`.

### `mosr` (`4x_umzi_digital_art_mosr_v2.pth`, 4x, k=7, `gps`)

| Backend | Total s | Median ms/image | Median page avg ms/tile | Input MPx/s | Pooled PSNR (dB) | Mean / Min SSIMULACRA2 | Device peak above bg (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Torch CUDA FP32 | 23.575 | 5839.16 | 973.19 | 0.206 | Reference | Reference | 1392.16 |
| TensorRT-RTX FP16 | 6.948 | 1738.83 | 289.80 | 0.699 | 63.28 | 87.12 / 86.11 | 500.44 |
| Native FP32 | 16.046 | 4009.04 | 668.17 | 0.303 | 140.29 | 89.43 / 88.05 | 598.00 |
| Native mixed | 8.992 | 2262.65 | 377.11 | 0.541 | 69.74 | 87.40 / 86.22 | 370.00 |
| Native FP16 | 7.213 | 1790.17 | 298.36 | 0.674 | 64.46 | 87.23 / 86.13 | 372.66 |

Native buffer breakdown at `1×3×512×512`:
- FP32: 17.10 MiB weights, 448.00 MiB activations, 2.35 MiB workspace, 51.00 MiB RGB I/O.
- Mixed: 8.73 MiB weights, 224.00 MiB activations, 0 B workspace, 51.00 MiB RGB I/O.
- FP16: 8.73 MiB weights, 224.00 MiB activations, 0 B workspace, 25.50 MiB RGB I/O.

### `mosr-l` (`4x_umzi_digital_art_mosr_l.pth`, 4x, k=17, `gps`)

| Backend | Total s | Median ms/image | Median page avg ms/tile | Input MPx/s | Pooled PSNR (dB) | Mean / Min SSIMULACRA2 | Device peak above bg (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Torch CUDA FP32 | — | 9038.28 | 1506.38 | 0.135 | Reference | Reference | 1316.81 |
| TensorRT-RTX FP16 | 13.409 | 3364.49 | 560.75 | 0.362 | 62.04 | 86.77 / 85.78 | 533.27 |
| Native FP32 | 27.691 | 6902.59 | 1150.43 | 0.176 | 138.64 | 89.15 / 87.76 | 601.34 |
| Native mixed | 16.487 | 4096.14 | 682.69 | 0.295 | 67.65 | 87.00 / 85.90 | 372.00 |
| Native FP16 | 14.806 | 3700.83 | 616.81 | 0.328 | 63.27 | 86.89 / 85.93 | 370.00 |

### `mosr-t` (`2x_umzi_digital_art_mosr_t.pth`, 2x, k=7, `dys`)

In PyTorch FP32, `mosr-t` reserved 4376 MiB and left 0 bytes free on the 4 GiB card due to the DySample tail at 1024×1024. The native session fuses DySample grid-coord calculation and bilinear sampling into a single pass over the activation arena, cutting peak VRAM above background from 2730.57 MiB to 372.00 MiB (FP32) / 207.75 MiB (FP16). TensorRT-RTX FP16 runs DySample grid coordinates in FP16, dropping pooled PSNR to 41.77 dB (`85.36 / 84.48` SSIMULACRA2), while native FP16 keeps coordinates and interpolation in FP32 (`67.00 dB`, `89.67 / 89.09`).

| Backend | Total s | Median ms/image | Median page avg ms/tile | Input MPx/s | Pooled PSNR (dB) | Mean / Min SSIMULACRA2 | Device peak above bg (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Torch CUDA FP32 | 4.012 | 1002.39 | 167.06 | 1.211 | Reference | Reference | 2730.57 |
| TensorRT-RTX FP16 | 1.282 | 315.56 | 52.59 | 3.792 | 41.77 | 85.36 / 84.48 | 495.84 |
| Native FP32 | 3.003 | 759.46 | 126.58 | 1.618 | 128.26 | 91.52 / 90.48 | 372.00 |
| Native mixed | 1.792 | 449.38 | 74.90 | 2.712 | 70.35 | 89.77 / 89.08 | 212.00 |
| Native FP16 | 1.396 | 352.80 | 58.80 | 3.480 | 66.72 | 89.70 / 89.01 | 207.75 |

## Optimization passes

CUPTI profiling on the baseline build (`output/mosr-session/profile_control.json`,
`1×3×512×512`) showed the 24 GatedBlocks dominated runtime across all three
precisions:
- **FP16 (`305.43 ms`):** `dw_gate_nhwc` `141.52 ms` (46.3%, reading 4 of 192
  channels per block in NHWC stride), `64→192` + `96→64` cuDNN convolutions
  `103.27 ms` (33.8%), `layer_norm_nhwc` `19.57 ms` (6.4%),
  `mish_bias_residual_nhwc` `14.94 ms` (4.9%), and `plain_gate_nhwc` `10.95 ms`
  (3.6%).
- **Mixed (`379.89 ms`):** `64→192` + `96→64` cuDNN `176.61 ms` (46.5%),
  `dw_gate_nhwc` `139.15 ms` (36.6%), `layer_norm_nhwc` `16.84 ms` (4.4%),
  `mish_bias_residual_nhwc` `15.31 ms` (4.0%), and `plain_gate_nhwc` `9.14 ms`.
- **FP32 (`676.97 ms`):** cuDNN Winograd `128×128` `449.73 ms` (66.4%),
  `dw_gate_nchw_f32` `126.86 ms` (18.7%), `layer_norm_nchw_f32` `41.43 ms`,
  `mish_bias_residual_nchw_f32` `31.21 ms`, and `plain_gate_nchw_f32` `14.82 ms`.

Four optimization passes were implemented and benchmarked against the untouched
baseline (`control = output/mosr-session/native-win/umzi_native.dll`):

1. **Pass 1 (`p1-win` — unrolled 8-ch DW gate + vectorized plain gate + fused residual/LayerNorm):**
   Specialized `dw_gate_nhwc_8ch_k{7,17}` (`8` channels/block with `uint4` loads/stores),
   vectorized `plain_gate_nhwc_vec8`, fused `bias_norm_{nhwc,nchw_f32}` and
   `mish_residual_norm_{nhwc,nchw_f32}` across consecutive blocks, and added
   unrolled `dw_gate_nchw_f32_k{7,17}` (`16×16` tiles, later upgraded to 2-row
   128-thread tiling). Also serves as the fallback path (`fused_gate=0` and `mosr-t`).
2. **Pass 2 (`p2-win` — Tensor Core `expand_gate` / `project_res_norm` + coalesced planar DW gate):**
   For `dim=64`, `hidden=96`, `kernel_size=7` or `17` in FP16/mixed (controlled
   by `fused_gate`, default `1`):
   - `mosr_expand_gate_{fp16,mixed}` (`8×16` spatial tile, 384 threads, `swizzle16`
     `cp.async` double-buffering, `m16n8k16`) fuses `fc1` (`64→192`) with
     `plain_gate` directly into dense `[N, H, W, 32]` `plain_gated` (`64 B/px`)
     and writes `fc1`-biased depthwise inputs (`dw_v`) and gates (`dw_g`) in
     coalesced planar 8-channel slices (`[8, N*H*W, 8]`) via warp-quad transposes.
   - `mosr_dw_gate_planar_k{7,17}` loads coalesced `uint4` halos into `half[8]`
     shared memory (no global bias re-reads, zero 192-channel NHWC stride waste),
     evaluates two vertically adjacent pixels per thread with `(K+1)` vertical
     reuse, and writes coalesced `[8, N*H*W, 8]` `dw_gated`.
   - `mosr_project_res_norm_{fp16,mixed}` (and final-block `mosr_project_res_*`)
     runs `fc2` (`96→64`) reading both `plain_gated` and `dw_gated` directly into
     `swizzle16`, fuses `fast_mish(x + bias) + (state - 0.5f)` with quad-transposed
     `uint4` `state` I/O, and computes next-block `LayerNorm` via `quad_sum` in
     registers. Eliminates all block cuDNN plans and retains `0 B` workspace in
     `mosr`/`mosr-l` FP16/mixed. Also upgraded FP32 `dw_gate_nchw_f32_k{7,17}` to
     2 vertically adjacent pixels/thread (`128` threads/block).
3. **Pass 3 (`p3-win` — SM86 occupancy carveout + `16×32` `w_smem` planar DW + smem FP32 norm):**
   Set `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = CU_SHAREDMEM_CARVEOUT_MAX_SHARED`
   for Tensor Core and shared-memory kernels, eliminated double-buffered `b_frag`
   register pressure in mixed accumulation, upgraded `mosr_dw_gate_planar_k7` to a
   `16×32` tile (`256` threads) with `[tap][8]` shared weight caching (`~20–23 ms`
   in `mosr`, ~6–7× faster than control's `141.5 ms`), and replaced 64-element
   per-thread FP32 array spills in `mish_residual_norm_nchw_f32` /
   `bias_norm_nchw_f32` with bank-conflict-free warp-strided shared memory.
4. **Pass 4 (`p4-win` — precision-specific fragment prefetch):**
   Restored double-buffered `b_frag[2][4][4]` for FP16 (`HalfAccumulate=true`)
   while keeping single `b_frag[4]` for mixed (`HalfAccumulate=false`).
5. **Pass 5 (`p5-win` — unified `gated_planar` + `mosr-t` `(dim=48, hidden=72)` tensor-core gating):**
   Unified `plain_gated` and `dw_gated` into a single contiguous planar buffer
   `gated_planar[Hidden/8, N*H*W, 8]` (removing the `ks` branch inside
   `project_res_norm`) and parameterized `mosr_expand_gate` /
   `mosr_project_res_norm` for both `(64, 96)` and `(48, 72)` (`mosr-t`). On
   `dim=48, hidden=72`, the `72`-channel input spans 5 `swizzle16` slices (`9`
   planar 8-channel chunks with the 10th chunk zero-filled by `copy_async`) and
   removes all body cuDNN convolutions across `mosr-t` too.

### Alternating A/B progression (`mosr`, `1×3×512×512`, `scripts/ab_mosr_builds.py`)

Measured in a single process alternating `control` through `p4` (`6 rounds × 3 calls`) and `p4` vs `p5` (`6 rounds × 3 calls`, median ms per batch-1 tile):

| Precision | `control` (ms) | `p1` (ms) | `p2` (ms) | `p3` (ms) | `p4` (ms) | `p5` vs `p4` (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `fp16` | 307.90 | 267.35 | 183.51 | 168.95 | 167.21 | **167.54** (`p4`: 171.96) |
| `mixed` | 382.08 | 339.11 | 265.46 | 253.29 | 265.77 | **239.05** (`p4`: 259.31) |
| `fp32` | 697.10 | 675.26 | 640.35 | **629.95** | 630.91 | — |

On `mosr-t` (`1×3×512×512`, `8 rounds × 3 calls`):
- **`fp16`:** `61.22 (control) → 57.54 (p4) → 31.73 ms (p5)` (**1.93×** speedup vs control)
- **`mixed`:** `72.00 (control) → 63.24 (p4) → 44.85 ms (p5)` (**1.61×** speedup vs control)
- **`fp32`:** `126.69 (control) → 114.70 (p4) → 113.56 ms (p5)` (**1.12×** speedup vs control)

Option parity check (`mosr`, `1×3×64×64`, `p4-win`): `fused_gate=1` vs `fused_gate=0` gives `mse = 1.999e-06` (`max_abs = 8.301e-03`) in FP16 and `mse = 4.753e-07` (`max_abs = 4.700e-03`) in mixed.

## Optimized measurements (`p5-win`)

Reports and outputs are in `output/architecture-baselines/{mosr,mosr-l,mosr-t}/batch-1/native-*-optimized/`.

### `mosr` optimized (`4x_umzi_digital_art_mosr_v2.pth`)

| Backend | Total s | Median ms/image | Median page avg ms/tile | Input MPx/s | Pooled PSNR (dB) | Mean / Min SSIMULACRA2 | Device peak above bg (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native FP32 (`optimized`) | 15.037 | 3750.03 | 625.01 | 0.323 | 140.21 | 89.42 / 88.00 | 602.19 |
| Native mixed (`optimized`) | 5.822 | 1452.23 | 242.04 | 0.835 | 69.85 | 87.41 / 86.23 | 377.82 |
| Native FP16 (`optimized`) | **3.981** | **998.22** | **166.37** | **1.221** | 64.63 | 87.23 / 86.24 | **370.00** |

Relative to the `control` baseline and TensorRT-RTX FP16:
- **Native FP16:** `298.36 → 166.37 ms/tile` (**1.79×** faster than baseline; **1.74×** faster per-image median than TensorRT-RTX FP16's `289.80 ms/tile` while delivering +1.35 dB higher PSNR and ~130 MiB lower peak VRAM).
- **Native mixed:** `377.11 → 242.04 ms/tile` (**1.56×** faster than baseline; **1.20×** faster than TensorRT-RTX FP16 at 69.85 dB PSNR).
- **Native FP32:** `668.17 → 625.01 ms/tile` (**1.07×** full-page, **1.11×** A/B).

### `mosr-l` optimized (`4x_umzi_digital_art_mosr_l.pth`, k=17)

| Backend | Total s | Median ms/image | Median page avg ms/tile | Input MPx/s | Pooled PSNR (dB) | Mean / Min SSIMULACRA2 | Device peak above bg (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native FP32 (`optimized`) | 21.062 | 5261.14 | 876.86 | 0.231 | 138.87 | 89.18 / 87.79 | 605.03 |
| Native mixed (`optimized`) | 7.837 | 1951.50 | 325.25 | 0.620 | 67.91 | 87.01 / 85.89 | 438.00 |
| Native FP16 (`optimized`) | **6.009** | **1501.89** | **250.32** | **0.809** | 62.64 | 86.79 / 85.86 | **415.19** |

- **Native FP16:** `616.81 → 250.32 ms/tile` (**2.46×** faster than baseline; **2.24×** faster per-image median than TensorRT-RTX FP16's `560.75 ms/tile`, with +0.60 dB PSNR, slightly higher SSIMULACRA2, and 118 MiB lower peak VRAM).
- **Native mixed:** `682.69 → 325.25 ms/tile` (**2.10×** faster than baseline; **1.72×** faster than TensorRT-RTX FP16 with +5.87 dB PSNR and 95 MiB lower peak VRAM).
- **Native FP32:** `1150.43 → 876.86 ms/tile` (**1.31×** faster).

### `mosr-t` optimized (`2x_umzi_digital_art_mosr_t.pth`, dim=48, hidden=72, `dys`)

| Backend | Total s | Median ms/image | Median page avg ms/tile | Input MPx/s | Pooled PSNR (dB) | Mean / Min SSIMULACRA2 | Device peak above bg (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native FP32 (`optimized`) | 2.731 | 684.50 | 114.08 | 1.780 | 128.22 | 91.45 / 90.35 | 372.00 |
| Native mixed (`optimized`) | 1.056 | 263.27 | 43.88 | 4.602 | 70.12 | 89.73 / 89.05 | 211.94 |
| Native FP16 (`optimized`) | **0.787** | **196.91** | **32.82** | **6.175** | 66.31 | **89.82 / 89.15** | **208.00** |

- **Native FP16:** `58.80 → 32.82 ms/tile` (**1.79×** faster full-page, **1.93×** A/B vs baseline; **1.60×** faster than TensorRT-RTX FP16's `52.59 ms/tile` while preserving DySample fidelity at **+24.54 dB PSNR**, **89.82 vs 85.36** mean SSIMULACRA2, and **288 MiB** lower peak VRAM).
- **Native mixed:** `74.90 → 43.88 ms/tile` (**1.71×** faster than baseline; **1.20×** faster than TensorRT-RTX FP16 at **70.12 dB** PSNR).
- **Native FP32:** `126.58 → 114.08 ms/tile` (**1.11×** faster).

## Verification commands

```powershell
$env:PYTHONPATH = 'src;scripts'
$py = 'C:\Python314\python.exe'
& $py scripts/build_native.py --arch sm_86 --nvrtc-root output/nvrtc-13.0/nvidia/cu13 --output output/mosr-session/p5-win
& $py scripts/validate_mosr_session.py --library output/mosr-session/p5-win/umzi_native.dll --cli output/mosr-session/p5-win/umzi_native.exe --output output/mosr-session/val-p5
& $py scripts/ab_mosr_builds.py --model output/mosr-session/validation/mosr-t.ummodel --cache-dir output/mosr-session/bench-cache --rounds 8 control=output/mosr-session/native-win/umzi_native.dll p4=output/mosr-session/p4-win/umzi_native.dll p5=output/mosr-session/p5-win/umzi_native.dll
foreach ($f in @('mosr', 'mosr-l', 'mosr-t')) {
    foreach ($b in @('native-fp32', 'native-mixed', 'native-fp16')) {
        & $py scripts/benchmark_architectures.py $f $b --native-model "output/mosr-session/validation/$f.ummodel" --native-library output/mosr-session/p5-win/umzi_native.dll --native-cache-dir output/mosr-session/bench-cache --tag optimized
    }
}
```

## Next

1. Run TorchInductor FP16 (`benchmark_architectures.py mosr inductor`) outside the Windows sandbox so Triton's `tcc.exe` can compile `cuda_utils`.
2. Consider wider `16×32` tiling on `mosr_dw_gate_planar_k17` (`mosr-l`), packed `half2` DW arithmetic, or folding the `8×` GPS ensemble tail weights in mixed/FP16 for additional latency/VRAM margin.
