# GateRV3 Vulkan optimization on Apple M4

GateRV3 Original and IDC use the same `.ummodel` files and FP32 NumPy input and output in every Vulkan mode. The optimized path specializes dense 1×1 and 3×3 convolutions, loads adjacent weight pairs together, reuses input addresses across reduction steps, and uses padded shared-memory tiles. It fuses eligible convolution/SiLU and residual/RMSNorm pairs, parallelizes spatial pooling across chunks, and computes local and Inception gates two channels at a time. The fused RMSNorm path checks for 32-lane compute subgroups with arithmetic support. Other devices use the shared-memory RMSNorm.

| Vulkan precision | Intermediate storage | Arithmetic |
| --- | --- | --- |
| `fp32` | FP32 | FP32 throughout, using packed FP16 checkpoint weights |
| `mixed` | FP16 on capable devices | FP32 arithmetic, with FP16 rounding at the same operation boundaries as the reference |
| `fp16` | FP16 on capable devices | Native FP16 fused multiply-add for 16-term partial sums in dense convolutions and projections, merged in FP32 |

The FP16 path requires `shaderFloat16` and keeps normalization, pooling, attention, and gate accumulation in FP32. Native FP16 storage additionally uses `storageBuffer16BitAccess`. Without native half storage, mixed and FP16 modes store rounded intermediates in FP32. The default precision is mixed.

## GateRV3 kernel options

Set any optimization to `0` in `SessionOptions.kernel_options` to disable it for one session. Each option accepts `0` or `1`. `session.get_session_info()["diagnostics"]` reports the effective settings after checking the precision and device capabilities.

| Option | Default | Behavior |
| --- | ---: | --- |
| `fast_convolution` | 1 | Uses the specialized 1×1 and 3×3 shaders. Set to 0 for the reference convolution in mixed or FP32 mode. The FP16 arithmetic path requires the specialized shader. |
| `fp16_storage` | 1 | Packs activations in FP16 in mixed and FP16 modes on devices with half storage support. Set to 0 for FP32 activation storage with FP16 rounding. The session's selected arithmetic mode still applies. |
| `fuse_silu` | 1 | Fuses eligible 3×3 convolutions with SiLU when `fast_convolution` is enabled. |
| `subgroup_norm` | 1 | Uses subgroup RMSNorm on devices with 32-lane compute subgroup arithmetic. Set to 0 for shared-memory RMSNorm. |
| `fuse_residual_norm` | 1 | Fuses eligible residual operations with RMSNorm when `subgroup_norm` is active. |
| `chunked_mean` | 1 | Splits spatial pooling over chunks. Set to 0 for the original spatial reduction. |
| `paired_gates` | 1 | Processes local and Inception gates in channel pairs. Set to 0 for scalar gate shaders. |
| `half2_convolution` | 1 | Uses aligned FP16 pairs and unrolled loops in the direct convolution shader when FP16 arithmetic and storage are active. Set to 0 for the prior direct FP16 kernel. |
| `winograd_convolution` | 0 | Set to 1 to try Winograd for 32-input, 32-output 3×3 operations in FP16 mode with FP16 storage. The kernel needs 128 workgroup invocations and 13 KiB of shared memory. |
| `reference` | 0 | Set to 1 to disable the optimizations as a group and select the original mixed or FP32 path. |
| `profile` | 0 | Set to 1 to collect per-operation GPU timestamps on queues that support them. |

```python
options = SessionOptions(
    backend="vulkan",
    precision="fp16",
    kernel_options={"fuse_silu": 0, "fp16_storage": 0},
)
```

For an A/B benchmark, repeat `--vulkan-option NAME=0` for each option you want to disable. The JSON report saves the requested settings and the effective diagnostics.

## Gains to carry forward

The alternating mixed-mode benchmark measured **1.71–2.19×** speedups across both checkpoints, drivers, and batch sizes. On MoltenVK, the full-page mixed runs roughly halved tile latency and more than doubled batch-16 throughput. Sampled batch-16 GPU allocation fell from **338.3 to 210.7 MiB** for Original and **278.0 to 150.3 MiB** for IDC. Pooled mixed-mode PSNR moved by under **0.01 dB** on both drivers. [Same-precision full-page comparison](#same-precision-speedup-over-the-original-vulkan-kernels) · [Alternating mixed-mode checks](#alternating-mixed-mode-checks)

Native FP16 continued the speed gains and beat WebGPU FP16 in both latency and throughput for both models on both drivers. Mixed mode retained the higher raw PSNR. [Pass 1 full-page results](#full-page-results) · [Pass 2 full-page results](#pass-2-full-page-results)

### What each pass changed

The table shows cumulative gains from the retained MoltenVK builds. Each stage used the same first tile or first 16 tiles from `page-001.jpg`, two warmups, and six synchronous calls measured in alternating build order. All values are host medians in milliseconds per call. Batch-16 values include the whole batch.

| Stage | Precision | Original batch 1 | Original batch 16 | IDC batch 1 | IDC batch 16 |
| --- | --- | ---: | ---: | ---: | ---: |
| Original Vulkan kernels | Mixed | 55.89 | 642.61 | 37.90 | 480.96 |
| Specialized dense convolutions and FP16 activation storage | Mixed | 35.11 | 363.12 | 23.92 | 281.33 |
| Fused convolution/SiLU and residual/RMSNorm, plus chunked spatial pooling | Mixed | 31.73 | 340.39 | 20.98 | 259.42 |
| Paired local and Inception gates | Mixed | 28.44 | 297.57 | 18.45 | 218.96 |
| Same optimized build with native FP16 convolution arithmetic | FP16 | 24.80 | 246.49 | 15.75 | 183.55 |

Specialized convolutions and FP16 storage delivered the largest step, cutting batch-16 call time by about 42–43% on both checkpoints. The raw samples are in `output/vulkan/optimization/pass-breakdown.ndjson`.

## Pass 2 targets

These targets were set after pass 1 for the MoltenVK FP16 path on the same Apple M4, checkpoints, and three pages. The latency column uses independently timed batch-1 tiles. The throughput column uses batch-16 calls.

| Checkpoint | Pass 1 ms/tile | Target ms/tile | Pass 1 MPx/s | Target MPx/s | Pass 1 peak GPU MiB at batch 16 | Pass 1 raw PSNR / SSIMULACRA2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 24.96 | ≤22.00 | 1.032 | ≥1.18 | 210.7 | 80.79 dB / 92.25 |
| IDC | 17.08 | ≤15.00 | 1.398 | ≥1.60 | 150.3 | 68.54 dB / 92.08 |

Start with the dense convolutions. On Original batch 16, 1×1 projections and 3×3 convolutions with fused SiLU accounted for about **177 ms** of the **249 ms** profiled host call. A combined GPU time of **145 ms** is a useful first target if the rest of the call stays near its current cost. Compare tile and workgroup variants against the optimized build at the same precision. [GPU profile](#gpu-profile-and-validation)

For a speed-only change, aim to retain the current sampled memory and quality. Report finite output, per-page PSNR and SSIMULACRA2, and both timing metrics on each driver. Repeat the alternating host comparison with the current optimized build as the baseline. KosmicKrisp's FP16 tile latency is within 3% of MoltenVK's, while its batch-16 throughput is about 20% lower. Use batch 16 to investigate that driver gap. [Full-page results](#full-page-results)

## Pass 2 dense convolutions

The direct FP16 shader now loads pairs of adjacent activations and weights and unrolls its short input, reduction, and output loops. It uses the same 32×32 tile and FP16 accumulation order. The opt-in Winograd F(2×2, 3×3) shader pretransforms the filters for 19 eligible layers per checkpoint once at load time. It reduces the multiply count for those layers and adds 0.59 MiB of transformed filters per session. The FP16 transforms can change numerical output. [Lavin and Gray, Fast Algorithms for Convolutional Neural Networks](https://openaccess.thecvf.com/content_cvpr_2016/html/Lavin_Fast_Algorithms_for_CVPR_2016_paper.html)

The alternating comparison used six synchronized calls after two warmups per variant, with the first tile or first 16 tiles from page-001.jpg. Every value below is the median duration of an entire call.

| Checkpoint | Driver | Batch | Prior direct FP16 (ms) | Direct with half2 (ms) | half2 plus Winograd (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| Original | MoltenVK | 1 | 24.03 | 22.79 | 22.13 |
| Original | MoltenVK | 16 | 245.21 | 235.08 | 224.55 |
| Original | KosmicKrisp | 1 | 25.63 | 24.16 | 23.61 |
| Original | KosmicKrisp | 16 | 295.25 | 282.52 | 276.71 |
| IDC | MoltenVK | 1 | 15.58 | 14.97 | 14.20 |
| IDC | MoltenVK | 16 | 182.23 | 173.98 | 163.48 |
| IDC | KosmicKrisp | 1 | 17.45 | 16.48 | 16.01 |
| IDC | KosmicKrisp | 16 | 223.09 | 212.20 | 206.48 |

In a matched Original batch-16 profile on MoltenVK, combined 1×1 and 3×3 GPU time fell from **173.32 ms** to **160.47 ms** with half2 and **151.26 ms** with Winograd added. The 145 ms target remains open. Wider vector tiles, subgroup shuffles, and repacking weight rows did not beat the prior MoltenVK kernel in the initial probes. The eight-tile Winograd variant gained throughput after transposing and padding its shared operand tile.

Re-run the profile with the current library. Pass `--vulkan-option half2_convolution=0` for the pass 1 kernel and omit the override for the new direct kernel.

```sh
python3 scripts/profile_vulkan.py --model output/vulkan/original.ummodel \
  --library build/vulkan/libumzi_native_vulkan.dylib --batch 16 --precision fp16 \
  --vulkan-option winograd_convolution=1
```

### Pass 2 full-page results

The runs used the same Apple M4, 128×128 tiles, and three sample pages per checkpoint. The batch-16 columns sum 11 synchronized calls per image and compute throughput from source pixels. Batch-1 latency averages 528 separately timed calls per configuration. Timings include copies between host and GPU. The memory columns give the sampled increase in Metal allocations at each batch size. Raw PSNR uses peak 1 before quantization against the matching Torch MPS FP32 output. The SSIMULACRA2 column averages the scores from the rendered PNGs.

| Checkpoint | Driver | FP16 kernel | Batch-16 s | Batch-16 MPx/s | Batch-1 total s | Batch-1 ms/tile | Peak GPU MiB, batch 1 / 16 | Raw PSNR dB | SSIMULACRA2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | MoltenVK | Prior direct | 8.123 | 1.037 | 13.518 | 25.60 | 79.3 / 210.7 | 80.79 | 92.25 |
| Original | MoltenVK | half2 | 7.785 | 1.082 | 12.901 | 24.43 | 79.3 / 210.7 | 80.79 | 92.25 |
| Original | MoltenVK | half2 + Winograd | 7.505 | 1.122 | 11.911 | 22.56 | 79.9 / 211.3 | 80.81 | 92.14 |
| Original | KosmicKrisp | Prior direct | 9.832 | 0.857 | 13.475 | 25.52 | 81.4 / 212.8 | 80.80 | 92.17 |
| Original | KosmicKrisp | half2 | 9.381 | 0.898 | 12.707 | 24.07 | 81.4 / 212.8 | 80.80 | 92.17 |
| Original | KosmicKrisp | half2 + Winograd | 9.550 | 0.882 | 12.551 | 23.77 | 82.0 / 213.3 | 80.81 | 92.21 |
| IDC | MoltenVK | Prior direct | 6.101 | 1.381 | 8.733 | 16.54 | 19.0 / 150.3 | 68.54 | 92.08 |
| IDC | MoltenVK | half2 | 5.973 | 1.410 | 8.663 | 16.41 | 19.0 / 150.3 | 68.54 | 92.08 |
| IDC | MoltenVK | half2 + Winograd | 5.602 | 1.504 | 7.894 | 14.95 | 19.6 / 150.9 | 68.50 | 92.00 |
| IDC | KosmicKrisp | Prior direct | 7.866 | 1.071 | 9.411 | 17.82 | 20.3 / 151.6 | 68.54 | 92.08 |
| IDC | KosmicKrisp | half2 | 7.432 | 1.134 | 8.907 | 16.87 | 20.3 / 151.6 | 68.54 | 92.08 |
| IDC | KosmicKrisp | half2 + Winograd | 7.219 | 1.167 | 8.636 | 16.36 | 20.9 / 152.7 | 68.50 | 92.06 |

The default half2 kernel raised batch-16 throughput by 2.2–5.9% and lowered mean tile latency by 0.8–5.7% across the four model and driver combinations. It matched the previous kernel bit for bit in 20 shape checks, with identical page PNGs, quality metrics, and sampled peak GPU memory.

Winograd gave the fastest MoltenVK results for both models. It met the IDC tile latency target at 14.95 ms. Original reached 22.56 ms against the 22.00 ms target, and both batch-16 throughput targets remain open. On KosmicKrisp, Original throughput peaked with half2 at 0.898 MPx/s. Relative to the prior direct kernel, Winograd changed pooled PSNR by at most 0.05 dB and mean SSIMULACRA2 by up to 0.11 points across the four configurations.

<details>
<summary>Original per image</summary>

| Image | Driver | FP16 kernel | Batch-16 s | Batch-16 MPx/s | Batch-1 total s | Batch-1 ms/tile | Raw PSNR dB | SSIMULACRA2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| page-001.jpg | MoltenVK | Prior direct | 2.712 | 1.035 | 4.491 | 25.51 | 80.37 | 94.30 |
| page-005.jpg | MoltenVK | Prior direct | 2.705 | 1.038 | 4.510 | 25.63 | 84.38 | 90.99 |
| page-009.jpg | MoltenVK | Prior direct | 2.706 | 1.038 | 4.517 | 25.66 | 79.15 | 91.46 |
| page-001.jpg | MoltenVK | half2 | 2.591 | 1.084 | 4.344 | 24.68 | 80.37 | 94.30 |
| page-005.jpg | MoltenVK | half2 | 2.604 | 1.078 | 4.285 | 24.35 | 84.38 | 90.99 |
| page-009.jpg | MoltenVK | half2 | 2.590 | 1.084 | 4.272 | 24.27 | 79.15 | 91.46 |
| page-001.jpg | MoltenVK | half2 + Winograd | 2.478 | 1.133 | 3.952 | 22.45 | 80.39 | 94.28 |
| page-005.jpg | MoltenVK | half2 + Winograd | 2.515 | 1.116 | 3.972 | 22.57 | 84.39 | 90.86 |
| page-009.jpg | MoltenVK | half2 + Winograd | 2.512 | 1.118 | 3.987 | 22.65 | 79.16 | 91.26 |
| page-001.jpg | KosmicKrisp | Prior direct | 3.266 | 0.860 | 4.502 | 25.58 | 80.37 | 94.38 |
| page-005.jpg | KosmicKrisp | Prior direct | 3.287 | 0.854 | 4.491 | 25.52 | 84.38 | 90.81 |
| page-009.jpg | KosmicKrisp | Prior direct | 3.279 | 0.856 | 4.483 | 25.47 | 79.16 | 91.33 |
| page-001.jpg | KosmicKrisp | half2 | 3.121 | 0.900 | 4.235 | 24.06 | 80.37 | 94.38 |
| page-005.jpg | KosmicKrisp | half2 | 3.119 | 0.900 | 4.237 | 24.07 | 84.38 | 90.81 |
| page-009.jpg | KosmicKrisp | half2 | 3.140 | 0.894 | 4.235 | 24.06 | 79.16 | 91.33 |
| page-001.jpg | KosmicKrisp | half2 + Winograd | 3.196 | 0.879 | 4.198 | 23.85 | 80.39 | 94.44 |
| page-005.jpg | KosmicKrisp | half2 + Winograd | 3.186 | 0.881 | 4.169 | 23.69 | 84.40 | 90.85 |
| page-009.jpg | KosmicKrisp | half2 + Winograd | 3.167 | 0.887 | 4.185 | 23.78 | 79.16 | 91.35 |

</details>

<details>
<summary>IDC per image</summary>

| Image | Driver | FP16 kernel | Batch-16 s | Batch-16 MPx/s | Batch-1 total s | Batch-1 ms/tile | Raw PSNR dB | SSIMULACRA2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| page-001.jpg | MoltenVK | Prior direct | 2.031 | 1.382 | 2.917 | 16.58 | 69.32 | 94.14 |
| page-005.jpg | MoltenVK | Prior direct | 2.023 | 1.388 | 2.978 | 16.92 | 68.41 | 90.90 |
| page-009.jpg | MoltenVK | Prior direct | 2.047 | 1.372 | 2.838 | 16.12 | 68.01 | 91.21 |
| page-001.jpg | MoltenVK | half2 | 1.960 | 1.433 | 2.899 | 16.47 | 69.32 | 94.14 |
| page-005.jpg | MoltenVK | half2 | 2.001 | 1.404 | 2.881 | 16.37 | 68.41 | 90.90 |
| page-009.jpg | MoltenVK | half2 | 2.012 | 1.395 | 2.882 | 16.38 | 68.01 | 91.21 |
| page-001.jpg | MoltenVK | half2 + Winograd | 1.845 | 1.522 | 2.751 | 15.63 | 69.29 | 94.05 |
| page-005.jpg | MoltenVK | half2 + Winograd | 1.885 | 1.489 | 2.602 | 14.79 | 68.34 | 90.66 |
| page-009.jpg | MoltenVK | half2 + Winograd | 1.871 | 1.501 | 2.540 | 14.43 | 67.97 | 91.28 |
| page-001.jpg | KosmicKrisp | Prior direct | 2.596 | 1.082 | 3.112 | 17.68 | 69.33 | 94.14 |
| page-005.jpg | KosmicKrisp | Prior direct | 2.622 | 1.071 | 3.144 | 17.86 | 68.39 | 90.86 |
| page-009.jpg | KosmicKrisp | Prior direct | 2.649 | 1.060 | 3.155 | 17.93 | 68.01 | 91.23 |
| page-001.jpg | KosmicKrisp | half2 | 2.459 | 1.142 | 2.940 | 16.70 | 69.33 | 94.14 |
| page-005.jpg | KosmicKrisp | half2 | 2.459 | 1.142 | 2.980 | 16.93 | 68.39 | 90.86 |
| page-009.jpg | KosmicKrisp | half2 | 2.514 | 1.117 | 2.987 | 16.97 | 68.01 | 91.23 |
| page-001.jpg | KosmicKrisp | half2 + Winograd | 2.420 | 1.160 | 2.844 | 16.16 | 69.29 | 94.09 |
| page-005.jpg | KosmicKrisp | half2 + Winograd | 2.388 | 1.176 | 2.887 | 16.40 | 68.33 | 90.83 |
| page-009.jpg | KosmicKrisp | half2 + Winograd | 2.410 | 1.165 | 2.905 | 16.51 | 67.97 | 91.28 |

</details>

The local records are `output/vulkan/optimization/pass2/original/report.json` and `output/vulkan/optimization/pass2/idc/report.json`. They contain each tile's time and coordinates, the per-image quality and timing measurements, sampled allocation peaks, and effective kernel options. The alternating timings and operation profiles are in `production-ab.ndjson` and `production-profile.ndjson` in the same pass directory.

### Reproduce the pass 2 page comparison

Use the same FP16 precision and compare across the three option sets. The command below runs the two Vulkan drivers and produces the matching Torch MPS FP32 reference. Repeat with `half2_convolution=0` for the prior direct kernel or with `winograd_convolution=1` for the optional shader, using a separate output directory for each. For IDC, pass `--checkpoint models/1x_umzi_idc_gaterv3.safetensors` and `--model output/vulkan/idc.ummodel`.

```sh
python3 scripts/benchmark_vulkan.py --model output/vulkan/original.ummodel \
  --library build/vulkan/libumzi_native_vulkan.dylib --backends vulkan \
  --vulkan-device 0 --vulkan-device 1 --vulkan-precision fp16 \
  --ssimulacra2 /opt/homebrew/bin/ssimulacra2 \
  --output output/vulkan/optimization/pass2/reproduction/original-half2
```

## Reproduce

Use the [native build and export commands](gaterv3-mac.md#build-and-run) to prepare the library and both packages. The following runs re-measure eager MPS FP16 and WebGPU FP16 alongside all supported Vulkan modes. Eager MPS FP32 is the numeric reference. The same three sample pages, independent batch-1 calls, and fixed batch-16 throughput boundary are used for both checkpoints. Quality includes raw peak-1 PSNR and PNG SSIMULACRA2 from `/opt/homebrew/bin/ssimulacra2`. Memory is the sampled increase in Metal allocations from a fresh process for each shape. [Measurement details](gaterv3-mac.md#measurement-protocol)

```sh
library=build/vulkan/libumzi_native_vulkan.dylib
python3 scripts/benchmark_vulkan.py --model output/vulkan/original.ummodel \
  --library "$library" --output output/vulkan/optimization/original \
  --vulkan-device 0 --vulkan-device 1 --webgpu-precision fp16 \
  --ssimulacra2 /opt/homebrew/bin/ssimulacra2
python3 scripts/benchmark_vulkan.py --model output/vulkan/idc.ummodel \
  --checkpoint models/1x_umzi_idc_gaterv3.safetensors \
  --library "$library" --output output/vulkan/optimization/idc \
  --vulkan-device 0 --vulkan-device 1 --webgpu-precision fp16 \
  --ssimulacra2 /opt/homebrew/bin/ssimulacra2
```

`profile_vulkan.py` reports the GPU intervals attributable to each operation group and the synchronized host median. Timestamp queries are opt-in and omitted from page benchmarks. Use `--host-only` for a probe without instrumentation. [Vulkan timestamp profiling](https://docs.vulkan.org/guide/latest/profiling.html)

```sh
python3 scripts/profile_vulkan.py --model output/vulkan/original.ummodel \
  --library "$library" --batch 16 --reference
python3 scripts/profile_vulkan.py --model output/vulkan/original.ummodel \
  --library "$library" --batch 16 --precision fp16
```

The complete local records are `output/vulkan/optimization/original/report.json` and `output/vulkan/optimization/idc/report.json`. They contain host timings for every tile, per-image totals and quality, GPU allocation measurements, Vulkan dispatch counts and buffer sizes, and WebGPU graph placement.

## Full-page results

Measured November 23, 2026 on Apple M4 with MoltenVK 1.4.2 and KosmicKrisp build 269a955c89.
The [original comparison](gaterv3-mac.md#measured-results) also includes WebGPU FP32.

| Checkpoint | Backend | Batch-16 s | Batch-16 MPx/s | Batch-1 total s | Batch-1 ms/tile | Peak GPU MiB, batch 1 / 16 | Raw PSNR dB | SSIMULACRA2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | Torch MPS FP32 | 15.273 | 0.552 | 22.842 | 43.26 | 1191.5 / 2234.3 | Reference | Reference |
| Original | Torch MPS FP16 | 12.632 | 0.667 | 19.542 | 37.01 | 123.4 / 1140.8 | 72.39 | 91.98 |
| Original | ORT WebGPU FP16 | 11.731 | 0.718 | 15.119 | 28.63 | 125.8 / 790.9 | 68.54 | 91.86 |
| Original | MoltenVK FP32 | 10.743 | 0.784 | 15.450 | 29.26 | 87.3 / 338.7 | 100.23 | 92.48 |
| Original | MoltenVK mixed | 9.772 | 0.862 | 15.449 | 29.26 | 79.3 / 210.7 | 85.77 | 92.30 |
| Original | MoltenVK FP16 | 8.162 | 1.032 | 13.181 | 24.96 | 79.3 / 210.7 | 80.79 | 92.25 |
| Original | KosmicKrisp FP32 | 17.687 | 0.476 | 22.382 | 42.39 | 89.4 / 341.6 | 100.23 | 92.49 |
| Original | KosmicKrisp mixed | 16.222 | 0.519 | 21.601 | 40.91 | 81.4 / 212.8 | 85.77 | 92.32 |
| Original | KosmicKrisp FP16 | 10.380 | 0.812 | 13.499 | 25.57 | 81.4 / 212.8 | 80.80 | 92.17 |
| IDC | Torch MPS FP32 | 12.937 | 0.651 | 16.940 | 32.08 | 107.3 / 2110.1 | Reference | Reference |
| IDC | Torch MPS FP16 | 10.597 | 0.795 | 14.586 | 27.62 | 67.2 / 1076.6 | 69.72 | 92.16 |
| IDC | ORT WebGPU FP16 | 9.564 | 0.881 | 10.922 | 20.69 | 57.1 / 690.9 | 56.26 | 90.97 |
| IDC | MoltenVK FP32 | 7.986 | 1.055 | 10.873 | 20.59 | 27.0 / 278.3 | 76.10 | 92.27 |
| IDC | MoltenVK mixed | 7.238 | 1.164 | 10.030 | 19.00 | 19.0 / 150.3 | 71.80 | 92.02 |
| IDC | MoltenVK FP16 | 6.024 | 1.398 | 9.018 | 17.08 | 19.0 / 150.3 | 68.54 | 92.08 |
| IDC | KosmicKrisp FP32 | 12.902 | 0.653 | 15.092 | 28.58 | 28.3 / 279.6 | 76.10 | 92.24 |
| IDC | KosmicKrisp mixed | 11.938 | 0.706 | 14.354 | 27.19 | 20.3 / 151.6 | 71.79 | 91.97 |
| IDC | KosmicKrisp FP16 | 7.537 | 1.118 | 9.190 | 17.40 | 20.3 / 151.6 | 68.54 | 92.08 |

### Same-precision speedup over the original Vulkan kernels

| Checkpoint | Backend | Batch-16 throughput ratio | Batch-1 latency ratio | Raw PSNR change (dB) |
| --- | --- | ---: | ---: | ---: |
| Original | MoltenVK FP32 | 1.95× | 1.87× | -0.000 |
| Original | MoltenVK mixed | 2.15× | 1.92× | -0.006 |
| Original | KosmicKrisp FP32 | 1.63× | 1.67× | -0.000 |
| Original | KosmicKrisp mixed | 1.74× | 1.73× | -0.001 |
| IDC | MoltenVK FP32 | 1.99× | 1.90× | +0.000 |
| IDC | MoltenVK mixed | 2.19× | 2.05× | +0.007 |
| IDC | KosmicKrisp FP32 | 1.64× | 1.70× | +0.000 |
| IDC | KosmicKrisp mixed | 1.76× | 1.78× | +0.003 |

### Original per image

| Image | Backend | Batch-16 s | Batch-16 MPx/s | Batch-1 total s | Batch-1 ms/tile | Raw PSNR dB | SSIMULACRA2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| page-001.jpg | Torch MPS FP32 | 5.064 | 0.555 | 7.542 | 42.85 | Reference | Reference |
| page-001.jpg | Torch MPS FP16 | 4.242 | 0.662 | 6.490 | 36.87 | 74.20 | 94.24 |
| page-001.jpg | ORT WebGPU FP16 | 3.881 | 0.724 | 5.043 | 28.65 | 67.20 | 93.98 |
| page-001.jpg | MoltenVK FP32 | 3.566 | 0.787 | 5.079 | 28.86 | 98.23 | 94.50 |
| page-001.jpg | MoltenVK mixed | 3.260 | 0.861 | 5.156 | 29.30 | 83.96 | 94.37 |
| page-001.jpg | MoltenVK FP16 | 2.735 | 1.027 | 4.345 | 24.69 | 80.37 | 94.30 |
| page-001.jpg | KosmicKrisp FP32 | 5.868 | 0.479 | 7.438 | 42.26 | 98.23 | 94.51 |
| page-001.jpg | KosmicKrisp mixed | 5.336 | 0.526 | 7.197 | 40.89 | 83.96 | 94.37 |
| page-001.jpg | KosmicKrisp FP16 | 3.476 | 0.808 | 4.489 | 25.51 | 80.37 | 94.38 |
| page-005.jpg | Torch MPS FP32 | 5.107 | 0.550 | 7.761 | 44.10 | Reference | Reference |
| page-005.jpg | Torch MPS FP16 | 4.190 | 0.670 | 6.540 | 37.16 | 71.79 | 90.67 |
| page-005.jpg | ORT WebGPU FP16 | 3.899 | 0.720 | 5.049 | 28.69 | 70.23 | 90.65 |
| page-005.jpg | MoltenVK FP32 | 3.567 | 0.787 | 5.199 | 29.54 | 102.52 | 91.30 |
| page-005.jpg | MoltenVK mixed | 3.259 | 0.861 | 5.154 | 29.29 | 88.96 | 91.07 |
| page-005.jpg | MoltenVK FP16 | 2.726 | 1.030 | 4.398 | 24.99 | 84.38 | 90.99 |
| page-005.jpg | KosmicKrisp FP32 | 5.899 | 0.476 | 7.417 | 42.14 | 102.52 | 91.29 |
| page-005.jpg | KosmicKrisp mixed | 5.395 | 0.520 | 7.198 | 40.90 | 88.96 | 91.07 |
| page-005.jpg | KosmicKrisp FP16 | 3.447 | 0.815 | 4.495 | 25.54 | 84.38 | 90.81 |
| page-009.jpg | Torch MPS FP32 | 5.102 | 0.550 | 7.540 | 42.84 | Reference | Reference |
| page-009.jpg | Torch MPS FP16 | 4.199 | 0.669 | 6.512 | 37.00 | 71.62 | 91.04 |
| page-009.jpg | ORT WebGPU FP16 | 3.952 | 0.711 | 5.028 | 28.57 | 68.71 | 90.95 |
| page-009.jpg | MoltenVK FP32 | 3.610 | 0.778 | 5.172 | 29.39 | 101.06 | 91.64 |
| page-009.jpg | MoltenVK mixed | 3.253 | 0.863 | 5.138 | 29.20 | 85.75 | 91.46 |
| page-009.jpg | MoltenVK FP16 | 2.701 | 1.040 | 4.438 | 25.22 | 79.15 | 91.46 |
| page-009.jpg | KosmicKrisp FP32 | 5.920 | 0.474 | 7.527 | 42.77 | 101.06 | 91.68 |
| page-009.jpg | KosmicKrisp mixed | 5.491 | 0.511 | 7.206 | 40.94 | 85.75 | 91.51 |
| page-009.jpg | KosmicKrisp FP16 | 3.457 | 0.812 | 4.515 | 25.65 | 79.16 | 91.33 |

### IDC per image

| Image | Backend | Batch-16 s | Batch-16 MPx/s | Batch-1 total s | Batch-1 ms/tile | Raw PSNR dB | SSIMULACRA2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| page-001.jpg | Torch MPS FP32 | 4.309 | 0.652 | 5.587 | 31.75 | Reference | Reference |
| page-001.jpg | Torch MPS FP16 | 3.544 | 0.792 | 4.795 | 27.25 | 69.76 | 94.12 |
| page-001.jpg | ORT WebGPU FP16 | 3.190 | 0.880 | 3.644 | 20.70 | 58.76 | 93.51 |
| page-001.jpg | MoltenVK FP32 | 2.662 | 1.055 | 3.607 | 20.50 | 75.21 | 94.42 |
| page-001.jpg | MoltenVK mixed | 2.419 | 1.161 | 3.243 | 18.43 | 71.25 | 93.99 |
| page-001.jpg | MoltenVK FP16 | 2.012 | 1.395 | 2.983 | 16.95 | 69.32 | 94.14 |
| page-001.jpg | KosmicKrisp FP32 | 4.318 | 0.650 | 5.031 | 28.58 | 75.21 | 94.34 |
| page-001.jpg | KosmicKrisp mixed | 3.953 | 0.710 | 4.769 | 27.09 | 71.24 | 94.02 |
| page-001.jpg | KosmicKrisp FP16 | 2.504 | 1.121 | 3.084 | 17.52 | 69.33 | 94.14 |
| page-005.jpg | Torch MPS FP32 | 4.329 | 0.649 | 5.699 | 32.38 | Reference | Reference |
| page-005.jpg | Torch MPS FP16 | 3.510 | 0.800 | 4.856 | 27.59 | 70.58 | 91.03 |
| page-005.jpg | ORT WebGPU FP16 | 3.204 | 0.876 | 3.639 | 20.67 | 54.95 | 88.81 |
| page-005.jpg | MoltenVK FP32 | 2.661 | 1.055 | 3.649 | 20.73 | 79.27 | 91.07 |
| page-005.jpg | MoltenVK mixed | 2.414 | 1.163 | 3.407 | 19.36 | 73.66 | 90.91 |
| page-005.jpg | MoltenVK FP16 | 2.005 | 1.400 | 3.013 | 17.12 | 68.41 | 90.90 |
| page-005.jpg | KosmicKrisp FP32 | 4.280 | 0.656 | 5.021 | 28.53 | 79.27 | 91.10 |
| page-005.jpg | KosmicKrisp mixed | 4.005 | 0.701 | 4.774 | 27.12 | 73.63 | 90.69 |
| page-005.jpg | KosmicKrisp FP16 | 2.509 | 1.119 | 3.048 | 17.32 | 68.39 | 90.86 |
| page-009.jpg | Torch MPS FP32 | 4.298 | 0.653 | 5.654 | 32.13 | Reference | Reference |
| page-009.jpg | Torch MPS FP16 | 3.543 | 0.793 | 4.935 | 28.04 | 68.96 | 91.33 |
| page-009.jpg | ORT WebGPU FP16 | 3.170 | 0.886 | 3.640 | 20.68 | 55.92 | 90.60 |
| page-009.jpg | MoltenVK FP32 | 2.662 | 1.055 | 3.616 | 20.55 | 74.99 | 91.33 |
| page-009.jpg | MoltenVK mixed | 2.405 | 1.167 | 3.380 | 19.21 | 70.97 | 91.17 |
| page-009.jpg | MoltenVK FP16 | 2.007 | 1.399 | 3.021 | 17.17 | 68.01 | 91.21 |
| page-009.jpg | KosmicKrisp FP32 | 4.304 | 0.652 | 5.041 | 28.64 | 74.99 | 91.28 |
| page-009.jpg | KosmicKrisp mixed | 3.981 | 0.705 | 4.812 | 27.34 | 70.97 | 91.18 |
| page-009.jpg | KosmicKrisp FP16 | 2.525 | 1.112 | 3.057 | 17.37 | 68.01 | 91.23 |

## Alternating mixed-mode checks

Six reference calls and six optimized calls alternated in order after two warmups per path.
Each pair used the same first tile or first 16 tiles from page-001.jpg and the same library build.
Batch 16 is the median latency of a full batch call in this table.

| Checkpoint | Driver | Batch | Reference median ms | Optimized median ms | Speedup | Reference-to-optimized raw PSNR dB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original | MoltenVK | 1 | 54.73 | 27.53 | 1.99× | 100.70 |
| Original | MoltenVK | 16 | 635.07 | 295.98 | 2.15× | 98.77 |
| Original | KosmicKrisp | 1 | 70.69 | 41.35 | 1.71× | 100.82 |
| Original | KosmicKrisp | 16 | 855.07 | 490.97 | 1.74× | 98.73 |
| IDC | MoltenVK | 1 | 37.82 | 18.01 | 2.10× | 78.05 |
| IDC | MoltenVK | 16 | 481.51 | 219.73 | 2.19× | 76.54 |
| IDC | KosmicKrisp | 1 | 48.49 | 27.09 | 1.79× | 78.03 |
| IDC | KosmicKrisp | 16 | 654.78 | 372.10 | 1.76× | 76.52 |

## GPU profile and validation

On MoltenVK, the Original batch-16 profile recorded the following operation-group medians across five profiled calls.
Convolution and projection accounted for about 75% of the reference submission time.

| Operation group | Reference mixed (ms) | Optimized FP16 (ms) |
| --- | ---: | ---: |
| 1×1 convolution | 240.51 | 82.55 |
| 3×3 convolution and SiLU | 254.82 | 94.69 |
| Local gates | 34.77 | 7.87 |
| Inception gates | 32.94 | 14.66 |
| Residuals and RMSNorm | 34.34 | 16.93 |

The original and optimized schedules issued 424 and 361 dispatches per Original call, respectively.
IDC went from 253 to 212. KosmicKrisp returned out-of-order per-dispatch timestamps,
which the profiler discarded. The synchronized host metrics cover both drivers.

Both drivers passed synchronization validation for all three precisions on both checkpoints.
The 60 numeric checks covered square and rectangular input, flat fields, checkerboards, and tiny inputs.
The lowest FP16 raw PSNR in those checks was 71.02 dB for Original and 60.84 dB for IDC.
All timed pages and individual tile calls returned finite outputs. Both WebGPU FP16 graphs for each checkpoint
assigned every node to WebGPU with CPU fallback disabled.
