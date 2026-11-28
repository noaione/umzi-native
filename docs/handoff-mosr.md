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
- **Options and caching:** Supports `workspace_mib` (0–4096, default 512) and
  disk plan caching (`mosr_<key>.umplan` under `cache_dir`). Reports
  diagnostics `timed_plans`, `plan_workspace_bytes`, `peak_plan_workspace_bytes`,
  and `fp16_fallback_plans`.

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
| Native FP32 | 27.691 | 6902.59 | 1150.43 | 0.176 | 138.64 | 89.15 / 87.76 | 601.34 |
| Native mixed | 16.487 | 4096.14 | 682.69 | 0.295 | 67.65 | 87.00 / 85.90 | 372.00 |
| Native FP16 | 14.806 | 3700.83 | 616.81 | 0.328 | 63.27 | 86.89 / 85.93 | 370.00 |

### `mosr-t` (`2x_umzi_digital_art_mosr_t.pth`, 2x, k=7, `dys`)

In PyTorch FP32, `mosr-t` reserved 4376 MiB and left 0 bytes free on the 4 GiB card due to the DySample tail at 1024×1024. The native session fuses DySample grid-coord calculation and bilinear sampling into a single pass over the activation arena, cutting peak VRAM above background from 2730.57 MiB to 372.00 MiB (FP32) / 207.75 MiB (FP16).

| Backend | Total s | Median ms/image | Median page avg ms/tile | Input MPx/s | Pooled PSNR (dB) | Mean / Min SSIMULACRA2 | Device peak above bg (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Torch CUDA FP32 | 4.012 | 1002.39 | 167.06 | 1.211 | Reference | Reference | 2730.57 |
| Native FP32 | 3.003 | 759.46 | 126.58 | 1.618 | 128.26 | 91.52 / 90.48 | 372.00 |
| Native mixed | 1.792 | 449.38 | 74.90 | 2.712 | 70.35 | 89.77 / 89.08 | 212.00 |
| Native FP16 | 1.396 | 352.80 | 58.80 | 3.480 | 66.72 | 89.70 / 89.01 | 207.75 |

## Verification commands

```powershell
$env:PYTHONPATH = 'src;scripts'
$py = 'C:\Python314\python.exe'
& $py scripts/build_native.py --arch sm_86 --nvrtc-root output/nvrtc-13.0/nvidia/cu13 --output output/mosr-session/native-win
& $py scripts/validate_mosr_session.py --library output/mosr-session/native-win/umzi_native.dll --cli output/mosr-session/native-win/umzi_native.exe --output output/mosr-session/validation
foreach ($f in @('mosr', 'mosr-l', 'mosr-t')) {
    foreach ($b in @('native-fp32', 'native-mixed', 'native-fp16')) {
        & $py scripts/benchmark_architectures.py $f $b --native-model "output/mosr-session/validation/$f.ummodel" --native-library output/mosr-session/native-win/umzi_native.dll --native-cache-dir output/mosr-session/bench-cache
    }
}
```

## Next

1. Run TorchInductor FP16 (`benchmark_architectures.py mosr inductor`) outside the Windows sandbox so Triton's `tcc.exe` can compile `cuda_utils`.
2. Profile `mosr` FP16/mixed/FP32 at 512×512 to identify optimization targets (e.g., Tensor Core fused expansion+gate / projection+residual+LayerNorm as in MoESR, and GPS weight folding for mixed/FP16).
