# umzi_native

Experimental CUDA inference runtime for Umzi models. It currently supports
the original GateRV3 Digital Decompression model and the IDC fine-tune.

## Install

Use Python 3.12 or newer. For a tagged Windows x64 release, install the wheel
from the private package index.

```powershell
python -m pip install --extra-index-url https://pypi.serik.at/simple/ umzi-native
```

For the current development build, install a downloaded GitHub Actions wheel
with `python -m pip install PATH_TO_WHEEL`, or run `python -m pip install .`
from the checkout with MSVC available. [Build workflow](.github/workflows/build.yml) · [Source build instructions](docs/gaterv3-python.md#install)

Inference needs a CUDA 13-compatible NVIDIA driver, cuBLASLt 13, and cuDNN 9.
Import a compatible CUDA PyTorch build or load NVIDIA's libraries explicitly.
[CUDA library loading](docs/gaterv3-python.md#library-loading)

## Usage

If you already have a `.ummodel`, inspect it with `read`. To create one from a
supported SafeTensors or PyTorch checkpoint, install the `exporter` extra.
For the private wheel, use `"umzi-native[exporter]"` in the install command.
Export runs on CPU and loads Resselt and Torch on demand.
[Supported checkpoints and export options](docs/gaterv3-python.md#gaterv3-checkpoints)

```powershell
# From the checkout, if you need to export a checkpoint
python -m pip install ".[exporter]"
python -m umzi_native export checkpoint.safetensors -o model.ummodel
```

```powershell
python -m umzi_native read model.ummodel
```

Open `model.ummodel` and pass a normalized RGB NumPy batch in NCHW order.

```python
import numpy as np
import torch  # Loads the CUDA dependencies when using CUDA PyTorch.

from umzi_native import InferenceSession

tiles = np.zeros((1, 3, 128, 128), dtype=np.float32)
with InferenceSession("model.ummodel") as session:
    (restored,) = session.run(None, {"input": tiles})
```

[Image-to-PNG tutorial](docs/quickstart.md#run-your-first-inference) · [Session and kernel options](docs/quickstart.md#session-options)

## Documentation

- [Changelog](CHANGELOG.md)
- [GateRV3 installation and model inspection](docs/gaterv3-python.md)
- [Native CUDA build, verification, and results](docs/gaterv3-native.md)
- [Kernel tuning, full-FP16 experiment, and VRAM measurements](docs/gaterv3-native-optimization.md)
- [Planned optimizations](docs/gaterv3-native-optimization.md#planned-improvements)
- [Native runtime refactor plan](docs/gaterv3-native-refactor.md)
- [Multi-kernel runtime and `.ummodel` plan](docs/umzi-native-runtime-plan.md)
- [Built-in model exporter plan](docs/umzi-model-exporter-plan.md)
- [Prepared PyTorch and TensorRT-RTX baseline](docs/gaterv3-baseline.md)
- [Manga page tiling and batch-size benchmark](docs/gaterv3-manga-tiling.md)

## Links

- [umzi](https://github.com/umzi2)
- [GateRV3](https://github.com/umzi2/GaterV3)
- [Resselt](https://github.com/rewaifu/resselt)
