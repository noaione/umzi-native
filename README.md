# umzi_native

Experimental native reimplementation of the GateRV3 model.

The [Windows build workflow](.github/workflows/build.yml) runs on pull
requests, pushes to `master` or `v*` tags, and manual dispatch from the
Actions tab.
Each successful run uploads a Windows x64 Python wheel and a native SDK
with the DLL, import library, CLI, and C header. The workflow verifies
the installed wheel and its bundled DLL on the hosted runner.

A push of a `v*` tag publishes the verified Windows wheel to the private
PyPI index at `pypi.serik.at`. Set `PYPI_USERNAME` and `PYPI_PASSWORD` in
the repository's `pypi` GitHub Actions environment before the first
release. The uploaded version comes from `project.version` in
`pyproject.toml`.

- [Python inference sessions](docs/gaterv3-python.md)
- [Native CUDA build, verification, and results](docs/gaterv3-native.md)
- [Kernel tuning, full-FP16 experiment, and VRAM measurements](docs/gaterv3-native-optimization.md)
- [Planned optimizations](docs/gaterv3-native-optimization.md#planned-improvements)
- [Native runtime refactor plan](docs/gaterv3-native-refactor.md)
- [Multi-kernel runtime and `.ummodel` plan](docs/umzi-native-runtime-plan.md)
- [Prepared PyTorch and TensorRT-RTX baseline](docs/gaterv3-baseline.md)
- [Manga page tiling and batch-size benchmark](docs/gaterv3-manga-tiling.md)
