# Built-in model exporter plan

Ship a checkpoint exporter in the `umzi-native` wheel. It will use Resselt to
recognize and construct models from SafeTensors and PyTorch checkpoints, then
pack the weights and schedule for a supported native kernel into one `.ummodel`
file. The first implementation covers both GateRV3 configurations already
supported by the runtime. Each future kernel family will supply its own packer.

## Installation and entry points

Add `exporter = ["resselt>=1.4.1"]` to the optional dependencies. Plain
`umzi-native` installs keep their existing runtime dependencies. Installing
`umzi-native[exporter]` makes pip resolve Torch and SafeTensors transitively
through Resselt. A compatible existing Torch installation can satisfy that
requirement. Export runs on CPU, so the extra can use CPU Torch. The wheel's
native build requirements stay separate. [Resselt 1.4.1](https://pypi.org/project/resselt/1.4.1/)

Use the existing `umzi_native.__main__` for both commands. `read` reuses the
current package reader, including multiple input files, `--json`, and exit
status. Keep `python -m umzi_native MODEL [MODEL ...] [--json]` working as an
alias for `read`. `export` delegates to the same `export_model()` function
available to Python callers.

Import Resselt and Torch only when an export begins. Importing `umzi_native`
or `export_model`, calling `peek_model()`, reading packages from the command
line, and opening any help screen should work with the plain install. An
export attempted without Resselt should explain how to install
`umzi-native[exporter]`. Surface import errors from an incomplete or broken
dependency installation so the caller can diagnose them.

```powershell
python -m pip install "umzi-native[exporter]"
python -m umzi_native export --help
python -m umzi_native read --help
python -m umzi_native export models/1x_umzi_idc_gaterv3.safetensors --output output/idc.ummodel
python -m umzi_native read output/idc.ummodel --json
```

```python
from umzi_native.exporter import export_model

info = export_model("models/1x_umzi_idc_gaterv3.safetensors", "output/idc.ummodel")
```

`export_model(checkpoint, output, *, kernel=None)` returns the `ModelInfo` for the
finished file. Both entry points detect the family by default. The optional
`--kernel gaterv3` or `kernel="gaterv3"` asserts the expected family. If Resselt
recognizes the model and there is no matching packer, report the detected model
name and the available native families. If the family matches but its model
configuration is unsupported, identify the incompatible setting.

## Checkpoint loading and packaging

1. Read `.safetensors` on CPU with `safetensors.torch.load_file()`, the same
   reader Resselt uses, and read `.pth`, `.pt`, and `.ckpt` tensor state
   dictionaries with `torch.load(..., map_location="cpu", weights_only=True)`.
   Preserve support for common state-dict wrappers and `module.` prefixes.
   Pass the normalized mapping to Resselt's `load_from_state_dict()` for
   architecture detection and model construction. Loading the mapping first
   lets the packer also check original flags such as GateRV3's `flash`, which
   Resselt's constructed model does not expose. [Resselt registry](https://github.com/rewaifu/resselt/blob/main/resselt/registry.py)
2. Dispatch from the detected Resselt architecture name to a private packer
   registered for the matching native family. The GateRV3 packer checks its
   supported topology, channel widths, scale, and attention mode. Put the model
   in eval mode on CPU and fold Conv3XC weights in FP32 before serializing the
   weight representations the native kernel expects.
3. Write the existing `UMZINPKG` version 1 format, including the family schema,
   architecture metadata, schedule, weights, and source checkpoint SHA-256.
   Stage the package beside the destination and validate it with `peek_model()`
   before replacing the destination. A failed export should leave any existing
   package usable. Use the same output format for both entry points, with
   precision and replay caching configured at inference time as before.

## GateRV3 compatibility

Resselt 1.4.1 recognized both checkpoints in a local CPU probe. With temporary
in-memory adjustments for the current exporter's list and attention-flag
checks, the Resselt models produced identical schedule shapes, operation
records, and packed weight bytes to the local models. The adjustments affected
only what the existing packer accepted during the probe.

| Checkpoint | Padding alignment | Operations |
| --- | ---: | ---: |
| `1x_umzi_digital_decompress_gaterv3_1.pth` | 16 px | 400 |
| `1x_umzi_idc_gaterv3.safetensors` | 8 px | 239 |

The original Digital Decompression model is the attention compatibility case.
The native kernel and current Torch reference use spatial attention. Resselt's
forward uses channel attention for this model. On one seeded 65×79 FP32 CPU
input, the original model differed from the local reference by a maximum of
0.00247, while the IDC model matched bit for bit. The identical packed schedule
and weights show that Resselt can supply the model weights even though its
forward produces different results for the original checkpoint.

The packer should compare split widths by value and select spatial attention
from the supported native topology and checkpoint flags. Preserve the current
spatial default when a supported checkpoint omits the flag. Reject checkpoints
that explicitly request the incompatible attention mode. Keep the existing
Torch reference until a small spatial-attention adapter over the Resselt model,
or an upstream equivalent, passes parity against the current reference.
Then remove the copied GateRV3 architecture from `scripts/reference/`.

## File layout and migration

```text
src/umzi_native/
  __main__.py           CLI dispatch and help for read and export
  _model.py             Package reader and shared format definitions
  exporter/
    __init__.py         Public export_model entry point
    _loader.py          Lazy dependency imports and checkpoint loading
    _package.py         Shared package writer
    _gaterv3.py         GateRV3 validation, schedule, and weight packing
scripts/
  export_native.py      Wrapper for existing repository commands
```

1. Capture the current package sections and FP32 reference outputs for both
   checkpoints. Move the package writer and GateRV3 packer under `src/` and
   reuse the existing format definitions where they describe the same fields.
2. Add the optional dependency, checkpoint loader, family dispatch, and Python
   export API. Add `read` and `export` subcommands to the existing `__main__.py`.
   Keep the existing script flags and legacy split-file export available for
   repository benchmarks. The installed exporter writes `.ummodel`.
3. Verify export and inference, then switch the benchmark model loader to
   Resselt with the spatial-attention compatibility path. Retire the local
   architecture copy once both configurations match the existing FP32 reference.
   Update the user docs to use the installed exporter.

## Acceptance checks

- Install the wheel in a clean environment without the extra. Verify that the
  runtime and exporter API import, `peek_model()` reads a package, and the
  module, `read`, and `export` help screens all work without Torch or Resselt.
  Compare `read` with the existing invocation for text and JSON output,
  multiple packages, and exit status. Verify the missing-extra error by
  attempting an export.
- Install the wheel with the extra and CPU Torch. Export both repository
  checkpoints through the CLI and Python API without a CUDA device. Compare
  output bytes across repeat exports and check that the package sections,
  checkpoint hash, operation count, and padding match the existing exports.
  Exercise PyTorch wrappers, missing and explicitly incompatible attention
  flags, unsupported families and configurations, and failures that must
  preserve an existing destination package.
- Load the new packages on Windows through `InferenceSession` and compare with
  the existing packages and frozen Torch FP32 references. Cover batch sizes 1
  and 16, the 128×128 baseline, and an odd 65×79 input that exercises padding
  for both checkpoint configurations. Verify the reference adapter separately
  before removing the local architecture.
