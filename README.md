# umzi-native

Native CUDA inference runtime for [Umzi](https://github.com/umzi2) image restoration and upscaling models. Checkpoints are packed into a single `.ummodel` file and executed on NVIDIA GPUs through one Python API (`InferenceSession`) backed by custom CUDA kernels, tuned cuBLASLt/cuDNN plans, and CUDA Graph replay.

## Supported models

| Family | Scale | Precision modes | Notes |
| --- | ---: | --- | --- |
| **GateRV3** | 1× | `mixed`, `fp16` | Original (16 px alignment) and IDC (8 px alignment) variants; opt-in kernel fusions |
| **GateRV2** | 1× | `mixed`, `fp16` | Tuned cuBLASLt/cuDNN plans with optional fast cold start |
| **MoESR** | 4× | `mixed`, `fp16` | Custom fused tensor-core gate and depthwise kernels |
| **GFISRv2** | 4× | `fp32`, `mixed`, `fp16` | Requires cuFFT 12 at runtime |
| **SMoSR** | 2× | `fp32`, `mixed`, `fp16` | Direct pixel shuffle with optional SiLU fusion and CUDA Graphs |
| **FiGSR** | 2× | `fp32`, `mixed`, `fp16` | Direct pixel shuffle with Fourier branch; requires cuFFT 12 at runtime |

See [Exporting checkpoints](docs/export.md) for the exact checkpoint topologies accepted by the exporter.

## Requirements

- **Python:** 3.12 or newer
- **GPU:** NVIDIA GPU with compute capability 8.6 or newer (Ampere / RTX 30 series and later)
- **GPU runtime:** CUDA 13-compatible NVIDIA driver, cuBLASLt 13, and cuDNN 9 (plus cuFFT 12 for GFISRv2 and FiGSR)

The simplest way to load the CUDA runtime libraries is to `import torch` from a CUDA-enabled PyTorch install before opening a session. You can also point `SessionOptions(dll_dirs=...)` at standalone NVIDIA library directories without PyTorch — see [CUDA library loading](docs/setup.md#cuda-library-loading).

## Install

### Prebuilt wheels (Windows & Linux x64)

Install a release from the private package index (Windows x64 and Linux `manylinux_2_34_x86_64`, glibc 2.34+ / OpenSSL 3):

```powershell
python -m pip install --extra-index-url https://pypi.serik.at/simple/ umzi-native
```

To export `.safetensors` / `.pth` checkpoints into `.ummodel` files, include the `exporter` extra (adds Resselt and PyTorch for CPU-side conversion; not needed if you only run existing `.ummodel` files):

```powershell
python -m pip install --extra-index-url https://pypi.serik.at/simple/ "umzi-native[exporter]"
```

### Building from source

Install a CI wheel from [GitHub Actions](.github/workflows/build.yml) with `python -m pip install PATH_TO_WHEEL`, or build from a checkout (requires MSVC on Windows or a C++17 compiler + OpenSSL 3 headers on Linux):

```powershell
python -m pip install .
# Or with the checkpoint exporter included:
python -m pip install ".[exporter]"
```

See [Installation and CUDA setup](docs/setup.md) for standalone library loading, editable installs, and build options.

## Quick start

### 1. Export or inspect a `.ummodel`

Skip the export step if you already have a `.ummodel` file.

```powershell
# Convert a supported checkpoint to .ummodel (runs on CPU)
python -m umzi_native export checkpoint.safetensors -o model.ummodel

# Inspect model metadata and validate the package without initializing CUDA
python -m umzi_native read model.ummodel
```

### 2. Run inference in Python

Pass a normalized RGB NumPy batch in **NCHW** layout `(N, 3, H, W)` with values in `[0, 1]`. Setting `cache_dir` saves tuned GPU plans on the first run so later runs skip tuning.

```python
from pathlib import Path

import numpy as np
import torch  # Makes PyTorch's bundled CUDA libraries available.

import umzi_native as umzi

tiles = np.zeros((1, 3, 128, 128), dtype=np.float32)
options = umzi.SessionOptions(
    precision="mixed",  # Use "fp16" with np.float16 input for fastest throughput.
    cache_dir=Path("cache"),
)

with umzi.InferenceSession("model.ummodel", sess_options=options) as session:
    (restored,) = session.run(None, {"input": tiles})
```

- **Full image-to-PNG walkthrough:** [Python inference quickstart](docs/quickstart.md#1-run-your-first-inference)
- **Tuning & options (`precision`, `cache_dir`, `enable_cuda_graph`, `kernel_options`):**
  [Session options](docs/quickstart.md#3-session-options-sessionoptions) ·
  [Model-specific `kernel_options`](docs/quickstart.md#4-model-specific-kernel_options)

## Documentation

### User guide
- [Python inference quickstart & options](docs/quickstart.md)
- [Exporting & inspecting `.ummodel` packages](docs/export.md)
- [Installation, CUDA library loading, and source builds](docs/setup.md)
- [Changelog](CHANGELOG.md)

### Development & benchmarks
- [Adding new model kernels](docs/adding-new-kernels.md)
- [Multi-kernel runtime and `.ummodel` format spec](docs/umzi-native-runtime-plan.md)
- [Built-in model exporter design](docs/umzi-model-exporter-plan.md)
- [GateRV2, MoESR, and GFISRv2 expansion & RTX 3050 summary](docs/umzi-native-architecture-expansion.md#final-timing-and-fidelity-summary-rtx-3050-laptop)
- [Architecture baselines](docs/umzi-native-architecture-baselines.md)
- Model handoffs: [GateRV2](docs/handoff-gaterv2.md) · [MoESR](docs/handoff-moesr.md) · [GFISRv2](docs/handoff-gfisrv2.md) · [SMoSR](docs/handoff-smosr.md) · [FiGSR](docs/handoff-figsr.md)
- GateRV3 notes: [Python/IDC notes](docs/gaterv3-python.md) · [Native build & verification](docs/gaterv3-native.md) · [Optimization notes](docs/gaterv3-native-optimization.md) · [Baseline](docs/gaterv3-baseline.md) · [Manga tiling](docs/gaterv3-manga-tiling.md) · [Refactor plan](docs/gaterv3-native-refactor.md)

## Links

- [umzi](https://github.com/umzi2)
- [enhancr](https://github.com/enhancr)
- [Resselt](https://github.com/rewaifu/resselt)

## License

MPL-2.0
