"""Build the native GateRV3 session library and CLI without packaging a wheel.

The build itself lives in hatch_build.py at the repository root, which also
runs for every wheel build. Use this script for a bundle directory next to a
packed model, or for benchmarks; use ``pip wheel .`` or ``pip install .`` for
a Python package.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load_build_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("umzi_native_build", ROOT / "hatch_build.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {ROOT / 'hatch_build.py'}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look up the defining module in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    native = load_build_module()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "output/native")
    parser.add_argument(
        "--arch",
        choices=native.CUBIN_TARGETS,
        action="append",
        help="Override the embedded cubin targets. Repeat to include multiple targets.",
    )
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument(
        "--nvrtc-root",
        type=Path,
        help="Use CUDA 13.0 NVRTC from this nvidia/cu13 directory.",
    )
    parser.add_argument("--cudnn-root", type=Path)
    parser.add_argument("--crt-include", type=Path)
    parser.add_argument("--cccl-include", type=Path)
    parser.add_argument("--openssl-include", type=Path, action="append", default=[])
    parser.add_argument("--cuda-driver-lib", type=Path)
    args = parser.parse_args()
    native.build(
        args.output,
        native.BuildSettings(
            arches=args.arch or native.CUBIN_TARGETS,
            cuda_root=args.cuda_root,
            nvrtc_root=args.nvrtc_root,
            cudnn_root=args.cudnn_root,
            crt_include=args.crt_include,
            cccl_include=args.cccl_include,
            openssl_include=args.openssl_include,
            cuda_driver_lib=args.cuda_driver_lib,
        ),
    )


if __name__ == "__main__":
    main()
